# Project Build Report: MCP Server for the FastAPI Docs RAG Assistant

A complete, chronological record of building a Model Context Protocol (MCP) server that exposes an existing hybrid-retrieval RAG system as tools for AI clients, then measuring it, adding a second backend, and containerizing it.

- **Builder:** Shiven
- **Environment:** Windows, VS Code, PowerShell, Python 3.11.8, Node v24.19.0, Docker Desktop (Compose v5.1.3)
- **SDK:** `mcp` 2.2.0 (the v2 API: `MCPServer`, not `FastMCP`)
- **Duration:** Multi-session build, step by step
- **Branch:** `feature/mcp-server` (merge to `main` pending at the time of writing)
- **Caveat on all numbers:** single runs on one Windows laptop and one home network. They show direction and order of magnitude, not benchmarks.

## Table of Contents

1. Project Goal
2. Final Architecture
3. Step-by-Step Build Log
4. Every Issue Encountered and How It Was Fixed
5. Corrections Along the Way
6. Evaluation Results in Full
7. Complete File Inventory
8. Design Decisions and Rationale
9. Known Limitations and Unverified Items
10. Lessons Learned
11. Final Status

---

## 1. Project Goal

Let AI clients (Claude Desktop, Cursor) search the FastAPI documentation through the builder's own RAG system, without building any new data or pipeline, and measure the result instead of describing it. The server reuses the existing corpus (503 chunks from 85 official FastAPI doc files), the existing embeddings, the hybrid BM25 + vector retrieval with Reciprocal Rank Fusion (`search_hybrid.py`), and the grounded Gemini generation (`generate.py`).

The RAG project showed retrieval and generation skills. This project adds the integration layer and the evidence around it:

1. **Does the server change what retrieval returns?** Checked by running the 32 labeled questions through the real stdio server and comparing each one with the original eval results.
2. **What does the extra layer cost?** Latency through the server for two backends, with the protocol overhead measured separately.
3. **Does a model pick the right tool?** A labeled tool-selection eval with the real tool schemas, plus manual observations in two real clients.
4. **Can it be shipped?** A Docker image, a Compose setup, and a client configuration that launches it.

## 2. Final Architecture

```
Claude Desktop / Cursor   (MCP host + client)
        |  JSON-RPC over stdio (the host launches the server as a child process)
        v
mcp_server/server.py      MCPServer "fastapi-docs-rag"
   tools: search_fastapi_docs, ask_fastapi_docs (ask budget: 5 calls per process)
        |  RagBackend interface (mcp_server/rag_backend.py)
        +--------------------------+--------------------------------+
        v                          v
   LocalBackend               HttpBackend (mcp_server/http_backend.py)
   imports src/search_hybrid  calls the running FastAPI service
   and src/generate in the    GET /retrieve, POST /query
   same process
        |                          |
        v                          v
   src/gemini_client.py        src/api.py (service, own container)
   data/processed/*                 |
        |                          v
        +-------------> Gemini API (gemini-embedding-001 for queries,
                                    gemini-3.5-flash-lite for answers)
```

Backend selection: environment variable `RAG_BACKEND=local|http` (default `local`), with `RAG_SERVICE_URL` for the service address. Docker: `Dockerfile.mcp` packages the server alone (no data, no key); `docker-compose.yml` runs the RAG service and starts the MCP server per client session.

## 3. Step-by-Step Build Log

### Step 1: Environment and inspection
Inspected the repo before writing code. Python 3.11.8 in `venv`; Node was missing (needed for MCP Inspector) and was installed (v24.19.0). Read `search_hybrid.py`, `generate.py` and `api.py`. Found: data paths are relative, imports are flat (`src/` as the root), `hybrid_search` rebuilds BM25 and FAISS on every call, and `api.py`'s startup warm-up loads only the chunk list, which `hybrid_search` never reads. `pip show mcp` reported 2.2.0, a major version whose API differs from most tutorials.

### Step 2: Hello-world server
Built `hello_server.py` with two toy tools using `MCPServer`. Tested with an in-memory `Client`. Observed: type hints become a JSON input schema, the docstring becomes the description, and a result carries both text and structured content. Created branch `feature/mcp-server`.

### Step 3: MCP Inspector
`mcp dev` failed (it needs `uv`). Launched Inspector through `npx` with the venv's own Python. All tools worked. A tool that prints to stdout did not break the connection in v2.

### Step 4: The adapter layer
Reproduced the relative-path bug on purpose by running `hybrid_search` from the temp folder (`FileNotFoundError`). Built `rag_backend.py`: a `RagBackend` protocol, `Chunk` and `Answer` dataclasses, and `LocalBackend`, which changes directory to the project root, adds `src/` to the import path, loads `.env` and imports the RAG modules afterward.

### Step 5: Tool 1, `search_fastapi_docs`
Built `server.py` with a lazily created backend. The tool trims and validates the query, clamps `top_k` to 1 to 10 and returns numbered excerpts with source file names. An empty query initially raised `ValueError`, but the client received only a generic message (issue 6).

### Step 6: Tool 2 and error handling
Replaced `ValueError` with `ToolError`. Added `RagBackendError` and a context manager that turns Gemini API errors (code 429 means quota) into plain-language messages. Added `ask_fastapi_docs`, which appends "Files retrieved: ..." (labeled "retrieved", not "used", because a refused question still retrieves files). Real calls: an answerable question returned a cited answer; the Kubernetes question was correctly declined.

### Step 7: Measure, then fix
Timed each stage of a search separately. The BM25/FAISS rebuild was only about 80 to 125 ms, but creating the Gemini client cost about 1 s on every call. Fixes: a cached `get_client()` in `src/gemini_client.py`, a warm-up thread at server start, and quieter logging. Steady-state calls dropped from about 2.4 s to about 0.75 s with identical rankings.

### Step 8: Claude Desktop
Added the server to `claude_desktop_config.json`. On this Windows install the app reads the MSIX-packaged copy under `AppData\Local\Packages\AnthropicPBC.Claude_...`. After a full restart, Settings showed the server as Running.

### Step 9: Cursor
Created `.cursor/mcp.json` with `${workspaceFolder}` paths. The MCP panel showed the server connected with 2 tools.

### Step 10: Automated tests
`pytest.ini` sets the import paths. 8 unit tests with a fake backend and 1 stdio integration test (runs only with `RUN_INTEGRATION=1`). The first run's one failure was a test bug (issue 13).

### Step 11: Evals
- **11a/b: parity and latency through stdio.** `mcp_server/evals/eval_mcp_retrieval.py` runs the 32 labeled questions through the real server (launched from the temp folder), scores with the RAG project's own hit and reciprocal-rank definition, compares each question with `eval_results.json`, and measures cold start, steady-state median and protocol overhead. Result: 32 compared, 0 differ (section 6).
- **11c: tool selection.** `eval_tool_selection.py` gives Gemini the real tool schemas from the live server, one labeled prompt at a time, with automatic function calling disabled, and records which tool it picks without running the tools. Results are saved after every prompt, and a fingerprint of the tool descriptions prevents mixing results from two versions. Result: 23 of 24 correct.

### Step 12: HttpBackend
- Read `api.py`'s actual shapes first: `/retrieve` is a GET with `question` and `k`; `/query` is a POST with `question` and `k` and returns file names and scores but no excerpt text; `question` has a 3-character minimum; errors are not translated.
- `http_backend.py`: one reused `httpx` client, 3 s connect and 60 s read timeouts, parameters built with `params=` and `json=`, and every failure (service down, timeout, error status, unreadable body, unexpected shape) converted to a readable `RagBackendError` that never includes the raw response.
- `RAG_BACKEND` switch in `server.py` (default `local`; an unknown value gives a readable error).
- 16 unit tests with a fake HTTP transport.
- Comparison runs: HTTP backend with a fresh and a warm service, and the local backend again in the same session (section 6).
- Follow-up: client construction cost about 1.2 s; fixed by skipping certificate loading for plain `http://` URLs (issue 16).

### Step 13: Extras
- **13a: a budget for `ask_fastapi_docs`.** In a Cursor test the tools ran without permission prompts, so a model could call the quota-spending tool freely. The server now allows 5 `ask` calls per process (environment variable `ASK_BUDGET`, 0 disables). Failed attempts count, blank questions do not, `search` is never limited, and the error message points to `search_fastapi_docs`. The tool descriptions were deliberately left unchanged so the tool-selection results stay valid. 7 tests.
- **13b: optional resource or prompt.** Skipped: nothing measured suggested it would help.
- **13c: the `docs_src` gap, measured.** Counted on `data/processed/chunks.jsonl` (section 6).

### Step 14: Docker
- **14a: the MCP image.** `Dockerfile.mcp` (python:3.11-slim, `mcp==2.2.0` and `httpx`, three source files, no key or data) with its own `Dockerfile.mcp.dockerignore` (build context 12.79 kB; image about 250 MB). The first import check inside the image failed because `httpx` was missing (issue 15); fixed and rechecked (`imports ok`). A script launching the container with `docker run -i --rm` returned the same top result as every earlier check.
- **14b: Cursor with the Docker server.** Cursor used `search_fastapi_docs` and `ask_fastapi_docs` through the container (in-house server disabled). The service log showed the matching `GET /retrieve` and `POST /query` requests, and `docker ps` showed containers from the image.
- **14c: Compose.** Added a healthcheck to the RAG service and an `mcp-server` service in a Compose profile, launched with `docker compose run --rm -T mcp-server`. This exposed the environment problem in issue 18. After the fix, a script test passed with the service already running and from nothing running (29.9 s to the first answer). A Cursor entry for the compose command was configured and returned an answer; which route served that run was not confirmed.

### Step 15: Documentation
`mcp_server/README.md` (setup, architecture, results with caveats, limitations) and a link from the main README; this report; the merge to `main`.

## 4. Every Issue Encountered and How It Was Fixed

| # | Issue | Root cause | Fix |
|---|---|---|---|
| 1 | `node` and `npx` not recognized | Node.js not installed | Installed Node LTS (v24.19.0), restarted VS Code |
| 2 | `FileNotFoundError` for `data/processed/embeddings.npy` when run from another folder | Relative path resolves against the launch folder, and hosts launch servers from elsewhere | Adapter changes to the project root before loading; RAG code left unedited |
| 3 | Builder-side correction: an early claim that `print()` calls in `generate.py` would corrupt stdout | The prints sit under `if __name__ == "__main__":`, so importing never runs them | Corrected in the log |
| 4 | `mcp dev` failed with "'uv' is not recognized" | `mcp dev` runs the server in a temporary `uv run --with mcp` environment that would also lack FAISS and `rank_bm25` | Launch Inspector with `npx ... .\venv\Scripts\python.exe <script>` |
| 5 | `mcp --help` and `mcp dev` need CLI pieces | Plain `pip install mcp` omits the `cli` extra | `pip install "mcp[cli]"` |
| 6 | Model saw only "Error executing tool search_fastapi_docs" for an empty query | v2 treats ordinary exceptions as crashes and hides their text; only `ToolError` messages reach the client | Raise `ToolError`; wrap backend failures in `RagBackendError` and convert |
| 7 | About 2.2 to 2.7 s per search | `genai.Client(...)` created inside `hybrid_search` and `answer` on every call, about 1 s each | Shared cached client in `src/gemini_client.py` |
| 8 | First tool call 6.7 s | Backend created lazily on the first request | Warm-up thread at server start |
| 9 | `RecursionError: maximum recursion depth exceeded` | A find-and-replace of the client line also rewrote the line inside `gemini_client.py`, making `get_client()` call itself | Restored the file; two unrelated files it had touched were reverted with `git checkout` after reading `git diff` |
| 10 | `git push` failed: no upstream branch | The branch existed only locally | `git push --set-upstream origin feature/mcp-server` |
| 11 | Edit Config targeted a different file than the one the app reads | MSIX-packaged app uses a virtualized config location | Edited the `claude_desktop_config.json` under the `Packages` path with the app closed; validated the JSON |
| 12 | Reload MCP Configuration appeared not to work | Not established; the server later showed Running after a full restart | Full quit from the system tray and reopen |
| 13 | `test_descriptions_steer_the_model` failed | The docstring wraps "does NOT write an answer" across two lines and the test searched for it on one line | Test collapses whitespace before comparing |
| 14 | Prediction was half wrong | Predicted the Gemini network call would be over 80% of per-call time; it was 33 to 48%, with client creation another 48 to 59% | Measured each stage and fixed what the numbers showed |
| 15 | `ModuleNotFoundError: No module named 'httpx'` in the Docker image | `httpx` was assumed to come with `mcp==2.2.0`; it did not | Added `httpx` to the image's install line; the import check inside the built image caught it before any client run |
| 16 | First call about 1.4 s slower against a warm HTTP service | Constructing the default `httpx` client took 1,187 ms (1,301 and 802 ms in a separate test) versus 1 ms with `verify=False`: certificate loading, pointless for plain `http://` | Verification setup skipped only for `http://` URLs; `https://` keeps the default. Construction afterwards measured 113 ms (single run; the remainder is unexplained) |
| 17 | Backend choice set in the eval script did not reach the server | The MCP client forwards only 12 environment variables by default (`PATH`, `USERPROFILE`, `APPDATA` and a few others) | The eval passes `RAG_BACKEND` and `RAG_SERVICE_URL` explicitly. The service log confirmed the HTTP runs reached the service |
| 18 | Launching `docker compose run` from the MCP Python client failed with "Connection closed" | With the client's default environment, `docker compose` was not recognized ("unknown command: docker compose"; docker printed "unknown flag: --rm"), so the process exited before the handshake. With the full environment it worked | The check script passes the full environment through. Plain `docker run -i` is unaffected |
| 19 | A stdio MCP server does not fit a normal long-running Compose service | Stdio is a process transport: the client must own the process's stdin and stdout | Service placed in a Compose profile and launched per session with `docker compose run --rm -T` (`-T` disables pseudo-TTY allocation) |
| 20 | One ground-truth tag in the tool-selection set was wrong | Prompt a06 ("Run the grounded-answer tool...") paraphrases the tool but was tagged `tool-named` | Retagged `paraphrased-tool`; the scores did not change |

## 5. Corrections Along the Way

Claims and predictions that measurements overturned, kept here because they are part of the record.

| Initial claim or prediction | What was measured |
|---|---|
| The BM25/FAISS rebuild per call is the slow part | It costs about 80 to 125 ms; creating the Gemini client cost about 1 s |
| Cold start of the evaluated server would be about 7 to 8 s | 12,742 ms in the first run; 4.6 to 12.7 s across runs, with no reliable difference between backends |
| `httpx` ships with `mcp` | Not installed by `mcp==2.2.0` in a clean image |
| After skipping certificate loading, client construction would take a few milliseconds | 113 ms (single run) |
| The corpus chunks contain no code samples | 168 of 503 chunks (33%) contain a code fence |
| Code fences are mostly console and JSON output | Python was the most common label (90, plus 6 lowercase), ahead of JSON (44, plus 9) and console (42) |
| Expected test totals of 26 collected and 25 passed after adding the HTTP tests | 25 collected and 24 passed (16 new tests) |

## 6. Evaluation Results in Full

### 6a. Retrieval parity through stdio (32 questions, hybrid)
Server launched from the temp folder; recall@5 is a hit rate (any expected file in the top 5); MRR uses the first hit.

| Group | n | MCP recall | MCP MRR | RAG project recall | RAG project MRR |
|---|:---:|:---:|:---:|:---:|:---:|
| Overall | 32 | 1.000 | 0.907 | 1.000 | 0.907 |
| Natural | 14 | 1.000 | 0.824 | 1.000 | 0.824 |
| Keyword | 7 | 1.000 | 0.929 | 1.000 | 0.929 |
| Hard | 11 | 1.000 | 1.000 | 1.000 | 1.000 |

Per-question comparison with `eval_results.json`: 32 compared, **0 differ**, in the local run and in both HTTP runs. Some ranks sit at the edge of the window (for example n03 has a reciprocal rank of 0.200, i.e. rank 5), so recall 1.0 held by a narrow margin on that question.

### 6b. Latency of `search_fastapi_docs` over stdio
Single runs; steady-state excludes the first call (n=31).

| Run | Cold start to first answer | First call | Steady median | Steady min – max |
|---|:---:|:---:|:---:|:---:|
| Local backend (first run) | 12,742 ms | 9,267 ms | 737 ms | 679 – 876 ms |
| Local backend (same session as HTTP) | 7,425 ms | 4,648 ms | 717 ms | 621 – 1,433 ms |
| HTTP backend, fresh service | 8,531 ms | 5,822 ms | 823 ms | 715 – 1,009 ms |
| HTTP backend, warm service | 4,586 ms | 2,229 ms | 821 ms | 674 – 1,019 ms |

- The HTTP backend is about 100 ms slower per call at the median (about 822 ms vs 717 ms). The cause was not identified.
- The 95th percentile is not reported: with 31 samples it is close to the maximum (for example 855 ms against a maximum of 876 ms in the first run, and 1,250 ms against 1,433 ms in the second).
- The two local runs differ by 5.3 s in cold start, so no backend comparison is claimed for cold start.
- In the fresh-service run the server's warm-up request and the first real request reached the service at the same time, so that run mixes two cold costs.

### 6c. Protocol overhead
A trivial `add` tool with no network, 50 calls per run: median 4.28 ms, 5.56 ms, 8.06 ms and 4.48 ms across the four runs (p95 between 4.93 and 9.57 ms). The protocol and stdio add a few milliseconds against about 700 to 800 ms per search; the time is dominated by retrieval.

### 6d. Tool selection (Gemini, 24 prompts)
`gemini-3.5-flash-lite`, default settings, one run per prompt, real tool schemas from the live server, tools never executed.

| Expected | Correct |
|---|:---:|
| `search` (12) | 12 / 12 |
| `ask` (6) | 5 / 6 |
| no tool (6) | 6 / 6 |
| **Overall** | **23 / 24** |

By prompt style (after the retag in issue 20): natural 4/4, exact-term 3/3, cite-source 3/3, not-in-docs 2/2, assistant-named 4/4, tool-named 1/1, paraphrased-tool 0/1, general 3/3, other-code 3/3.

Confusion counts: `ask`→`ask` 5, `ask`→`search` 1, none→none 6, `search`→`search` 12. No prompt that should not call `ask` called it. The one miss was a06, "Run the grounded-answer tool: what is a Pydantic model used for in FastAPI?", where Gemini picked `search`.

Limits: one model, one run each, a small set that is mostly easy (the no-tool prompts are all clearly unrelated to FastAPI, a ceiling effect), only the tool name scored, and the labels follow the policy written in the descriptions, so the eval checks adherence to that policy and not whether the policy is best.

### 6e. Tests
32 collected: 31 passed, 1 skipped (the stdio integration test, which needs `RUN_INTEGRATION=1` and makes real Gemini calls). 8 original unit tests, 16 HTTP-backend tests (including backend selection and the full tool path with a 429), 7 `ask`-budget tests.

### 6f. Docker
- Image about 250 MB; build context 12.79 kB; no key or data inside.
- Smoke tests through the MCP client: plain `docker run -i --rm` and `docker compose run --rm -T mcp-server` both returned the two tool names and a `request-files.md` excerpt first.
- Compose from nothing running (service start, healthcheck wait, MCP container, handshake, one search): 29.9 s to the first answer; single run with images already built.

### 6g. Real clients (manual, not scored)

| # | Prompt | Claude Desktop | Cursor |
|---|---|---|---|
| 1 | "Using the FastAPI docs, how do I read a custom header?" | no tool (inferred, not read from the MCP log) | search |
| 2 | "According to the FastAPI docs, what is root_path used for behind a proxy? Name the source file." | search | search |
| 3 | "Ask the FastAPI docs assistant how to upload a file." | ask | ask |
| 4 | "What is the capital of France?" | no tool | no tool |
| | Permission prompt | yes | no |

Later Cursor runs (model Cursor Grok 4.6 Medium, Dockerized server, in-house server disabled): "Which FastAPI docs page covers python-multipart? Use the documentation tool." produced `search_fastapi_docs` calls with `top_k` 5 and 10; "Ask the FastAPI docs assistant how to upload a file" produced an `ask_fastapi_docs` call. No permission prompts; the tool group was set to "Allow all", and whether that explains it was not tested. These are single conversations with specific models and are not comparable with the Gemini eval.

### 6h. The `docs_src` gap, measured
On `data/processed/chunks.jsonl` (503 chunks, one per line): 275 chunks contain the text `docs_src`; 272 contain a real include marker (54%); 168 contain a code fence (33%); 89 contain both. Fence openings by label: Python 90 (plus 6 lowercase), JSON 44 (plus 9), console 42, mermaid 9, bash 4, jinja 4. These count fence openings, not chunks, and a fence can be split across chunks because chunks are fixed 150-word windows.

One `ask_fastapi_docs` answer in Cursor contained decorators that do not exist in FastAPI (`@app.files`, `@app.upload`). Neither string occurs in any of the 503 chunks, so they were not in the context the model was given; the client model noticed and flagged it. The cause was not verified and this is one observation, not a measured rate. The line `uv add python-multipart` from the same answer does occur in 19 chunks.

## 7. Complete File Inventory

| File | Purpose |
|---|---|
| `mcp_server/server.py` | The MCP server, both tools, backend selection, `ask` budget |
| `mcp_server/rag_backend.py` | `RagBackend` protocol, `LocalBackend`, `RagBackendError`, Gemini error translation |
| `mcp_server/http_backend.py` | `HttpBackend` calling the FastAPI service |
| `mcp_server/check_docker.py` | Smoke test through `docker` or `docker compose` |
| `mcp_server/hello_server.py` | Toy server from step 2, also used to measure protocol overhead |
| `mcp_server/check_*.py`, `profile_stages.py` | Manual check scripts and the per-stage timer from the build steps |
| `mcp_server/evals/eval_mcp_retrieval.py` | Parity and latency eval (`local` or `http`, optional label) |
| `mcp_server/evals/eval_tool_selection.py`, `tool_selection_set.json` | Tool-selection eval and its 24 labeled prompts |
| `mcp_server/evals/results_*.json` | Saved outputs of the runs above |
| `mcp_server/README.md` | Project documentation |
| `src/gemini_client.py` | Cached shared Gemini client |
| `src/search_hybrid.py`, `src/generate.py` | Existing RAG code; one line each now uses `get_client()` |
| `tests/test_mcp_server.py`, `test_http_backend.py`, `test_ask_budget.py`, `pytest.ini` | Test suite and config |
| `Dockerfile.mcp`, `Dockerfile.mcp.dockerignore` | MCP image (server only) |
| `docker-compose.yml` | RAG service with healthcheck, plus the `mcp-server` service in a profile |
| `.cursor/mcp.json` | Cursor server configuration (no secrets) |
| `claude_desktop_config.json` | Lives outside the repo, in the Claude app's data folder |
| `docs/PROJECT_LOG.md`, `docs/PROJECT_REPORT_MCP_server.md` | Running log and this report |

## 8. Design Decisions and Rationale

- **An adapter with a `RagBackend` interface, two implementations.** The adapter paid for itself repeatedly: the relative-path problem, error translation, fake-backend testing and the second backend all went through it. `server.py` never changed to add HTTP.
- **Two tools instead of one.** `search_fastapi_docs` returns excerpts and lets the client model write the answer, using no generation quota. `ask_fastapi_docs` runs the full grounded pipeline for users who want the assistant's own cited answer.
- **Tool descriptions written as prompts**, guarded by a test on their wording, and left unchanged after the tool-selection eval so its results stay valid.
- **Rank numbers instead of RRF scores.** The scores (about 0.03) only encode order and invite misreading as relevance percentages.
- **Readable errors through `ToolError`.** In SDK v2 only these messages reach the model, so every expected failure is converted.
- **A shared Gemini client and a warm-up thread**, each justified by a measurement before and after.
- **A server-side budget for `ask`**, because a client can call it without a permission prompt; counted per process, with failed attempts included.
- **The MCP image holds only the server.** No key, no data, no `.env`; it reaches the RAG service over HTTP, so the two concerns are separate containers.
- **Certificate verification kept for `https://`.** Only plain `http://` skips certificate loading.
- **Evals pass configuration explicitly.** The MCP client does not forward arbitrary environment variables.
- **The RAG project's code and numbers were left alone.** Changes to `generate.py` or the retrieval code would invalidate its evaluation, so limitations are documented instead of patched.

## 9. Known Limitations and Unverified Items

**Limitations**
- **Code samples are largely missing from the corpus** (section 6h). Ingesting `docs_src` would change the corpus and invalidate every number in both projects.
- **`ask_fastapi_docs` can return code that is not in the documents** (one observed case). This is why the descriptions prefer `search_fastapi_docs`.
- **The `ask` budget is per process.** It resets with each server or container start and does not track the daily Gemini quota.
- **HTTP backend differences:** `/query` returns no excerpt text, the service requires questions of at least 3 characters, and service errors are not translated, so a Gemini 429 inside the service is expected to appear as a generic server error (not tested). About 100 ms slower per call; cause not identified.
- **The service rebuilds the BM25 and FAISS indexes on every request.** Its startup warm-up loads only the chunk list, which `hybrid_search` does not use (about 80 to 125 ms per call, measured).
- **Naive chunking** (fixed 150-word windows): excerpts can cut mid-sentence, and doc markup such as `///` appears.
- **Source names are inconsistent:** advanced pages carry an `advanced/` prefix.
- **Tool selection depends on the host and model.** The eval covers one model; client observations are single conversations.
- **Platform:** everything was measured on one Windows machine; Docker was tested on Docker Desktop for Windows only. Client config paths in the examples are Windows-specific.
- **Single runs everywhere.** No repeated runs, so noise (for example the 5.3 s spread in cold start between two local runs) is visible but not characterized.

**Not verified**
1. Whether the quieter-logs change takes effect in a real client's console.
2. Where the stray `print()` from the noisy test went.
3. Whether Cursor's lack of permission prompts is the "Allow all" setting.
4. Whether Claude Desktop's first plain-question run really made no tool call (inferred, not read from the MCP log).
5. Whether Claude Desktop works with the Dockerized server.
6. Which route (compose or plain `docker run`) served the Cursor run with the compose entry configured.
7. The cause of the Gemini SDK's "automatic function calling" message in the service log (hypothesis: hidden in the MCP process by the logging setting).
8. The cause of the roughly 100 ms HTTP gap, and why constructing the HTTP client still took about 113 ms after the certificate fix.

Verified during this project: the `docs_src` marker count (originally unverified), and that the description-wording test and all unit tests pass together.

## 10. Lessons Learned

1. **Read the code before designing around it.** Inspecting the repo exposed the relative paths and the per-call rebuild before any server code existed.
2. **Measure before optimizing, and be ready to be wrong.** The rebuild looked wasteful but cost under 10%; the real cost was a hidden 1 s per call. Several later predictions (cold start, construction time, test counts) were also wrong and are listed in section 5.
3. **Errors are messages to the model.** A hidden error leaves the model unable to recover; a clear one lets it change its input or switch tools.
4. **Test where it will run.** The Docker import check caught a dependency that worked on the development machine only; the MCP client's trimmed environment hid a tool that a normal terminal finds.
5. **A client's environment is not your terminal's.** Anything a server or script launched by a client depends on (variables, plugins, paths) has to be passed or checked explicitly.
6. **Not every server is a service.** A stdio server must be launched by its client, which shaped the Docker design.
7. **A "0 differ" parity result is still worth running.** It turns "the wrapper should not change anything" into evidence, and some ranks sat at the edge of the window.
8. **Report noise honestly.** Two local runs of the same code differed by 5.3 s in cold start, so cold-start comparisons between backends were not claimed, and a 95th percentile at n=31 was not reported.
9. **An eval can hit a ceiling here too.** The tool-selection set was mostly easy, so 23 of 24 shows the descriptions work on clear cases, not that they are robust.
10. **Check claims against numbers.** "No code samples in the chunks" was too strong (168 chunks have code fences), and a claim that the service loads its retriever once at startup does not match the code, which rebuilds the indexes per request.
11. **Descriptions are part of the product.** They steer behavior, and tests can guard their wording. Changing them invalidates an eval that used them, so the fingerprint check matters.
12. **Host behavior varies.** Prompts, permission prompts and tool choice differed between clients with the same server.
13. **Version-specific knowledge goes stale fast.** SDK v2, the Inspector launch path, the Windows config location and the Docker plugin behavior all differed from older tutorials.
14. **Bulk find-and-replace is risky.** One replace rewrote the helper's own definition; `git diff` made the damage visible and reversible.

## 11. Final Status

| Step | Deliverable | Status |
|---|---|---|
| 1–10 | Inspection, adapter, both tools, error handling, latency fixes, Claude Desktop and Cursor, unit and integration tests | Complete |
| 11a/b | Parity and latency eval through stdio | Complete (32/32 identical to the RAG results) |
| 11c | Tool-selection eval (Gemini) | Complete (23/24) |
| 12 | HttpBackend, backend switch, tests, measured comparison | Complete |
| 13a | `ask` budget | Complete |
| 13b | Optional resource or prompt | Skipped |
| 13c | `docs_src` gap measured | Complete |
| 14 | Dockerfile, Compose, Cursor with the Docker server | Complete (compose route configured in Cursor; serving route not confirmed) |
| 15 | README and this report | Complete |
| — | Merge `feature/mcp-server` into `main` | Pending |

The server is built, tested, measured against the original RAG results, offered with two backends, containerized and documented, with its limitations and unverified items listed.

*End of report.*