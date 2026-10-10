# Project Log: MCP Server for the FastAPI Docs RAG Assistant

A running record of building a Model Context Protocol (MCP) server that exposes an existing hybrid-retrieval RAG system as tools for Claude Desktop and Cursor. Covers steps 1 to 10 (build, connect, test). Steps 11 to 15 (evals, HTTP backend, extras, Docker, final report) are listed at the end as pending.

- **Builder:** Shiven
- **Environment:** Windows, VS Code, PowerShell, Python 3.11.8, Node v24.19.0
- **Repo / branch:** `fastapi-rag-tutorial`, branch `feature/mcp-server`
- **SDK:** `mcp` 2.2.0 (the v2 API: `MCPServer`, not `FastMCP`)
- **Status at this log:** steps 1 to 10 complete; step 11 (evals) next
- **Caveat on all numbers:** single runs on one Windows laptop and one home network. They show direction and order of magnitude, not benchmarks.

## Table of Contents

1. Project goal
2. Architecture
3. Design decisions
4. Step-by-step build log (steps 1 to 10)
5. Every issue encountered and how it was fixed
6. Measurements
7. Findings about tool selection across clients
8. Known limitations
9. File inventory
10. Lessons learned
11. Not yet verified
12. Remaining steps

---

## 1. Project goal

Let AI clients (Claude Desktop, Cursor) search the FastAPI documentation through the builder's own RAG system, without building any new data or pipeline. The server reuses the existing corpus (503 chunks from 85 official FastAPI doc files), the existing embeddings, the hybrid BM25 + vector retrieval with Reciprocal Rank Fusion (`search_hybrid.py`), and the grounded Gemini generation (`generate.py`).

Why it matters: the RAG project showed retrieval and generation skills. This project adds the integration layer: making that system usable by any MCP-compatible AI client, with tests and measurements.

## 2. Architecture

```
Claude Desktop / Cursor   (MCP host + client)
        |  JSON-RPC over stdio (the host launches the server as a child process)
        v
mcp_server/server.py      MCPServer "fastapi-docs-rag"
   tools: search_fastapi_docs, ask_fastapi_docs
        |
        v
mcp_server/rag_backend.py RagBackend interface -> LocalBackend
   (switches to project root, puts src/ on the import path,
    translates Gemini errors into messages the model can read)
        |
        v
src/search_hybrid.py  src/generate.py  src/gemini_client.py
data/processed/chunks.jsonl + embeddings.npy
        |
        v
Gemini API (gemini-embedding-001 for queries, gemini-3.5-flash-lite for answers)
```

A second backend (`HttpBackend`, calling the running FastAPI service) is planned for step 12 behind the same `RagBackend` interface.

## 3. Design decisions

| Decision | Why |
|---|---|
| Import the RAG modules directly (option a), behind an adapter; HTTP backend (option b) later | Fewest moving parts for a local stdio server, lowest latency, and the adapter makes swapping cheap. Option b gives a measured comparison later. |
| Same repo, new branch `feature/mcp-server` | The server depends on the RAG code and data; one clone runs everything. Main stays untouched until the merge. |
| Two tools, not one | `search_fastapi_docs` returns excerpts and lets the host model write the answer (no generation quota). `ask_fastapi_docs` runs the full grounded pipeline. |
| Tool descriptions written as prompts | The model reads them fresh each conversation. The `ask` description steers toward `search` and tells the model not to fill gaps silently. |
| Show rank numbers, not RRF scores | RRF scores (about 0.03) only encode order and invite misreading as relevance percentages. |
| `ToolError` at the MCP boundary; `RagBackendError` inside the adapter | The v2 SDK hides the text of ordinary exceptions from the client. Only `ToolError` messages reach the model. |
| Lazy backend creation with a lock, plus a warm-up thread started only under `__main__` | Keeps startup instant, avoids a double build, and keeps tests free of background threads. |
| One shared, cached Gemini client | Measured: creating the client cost about 1 s on every call. |
| No secrets in client config files | The adapter reads the key from the project's `.env`. |
| Cursor config uses `${workspaceFolder}` | No hard-coded user paths, so `.cursor/mcp.json` can live in the repo. |
| Fake backend for unit tests | The `RagBackend` interface lets tests inject known results and failures with no API calls. |

## 4. Step-by-step build log

### Step 1: Environment and inspection
Inspected the repo before writing code. Python 3.11.8 in `venv`; Node was missing (needed for MCP Inspector) and was installed (v24.19.0). Read `search_hybrid.py`, `generate.py`, and `api.py`. Found: data paths are relative (`np.load("data/processed/embeddings.npy")`), imports are flat (`from search import ...` with `src/` as root), `hybrid_search` rebuilds BM25 and FAISS on every call, and `api.py`'s startup warm-up loads only the chunk list into `_state`, which `hybrid_search` never reads. Installed `mcp`; `pip show mcp` reported **2.2.0**, a major version whose API differs from most tutorials. `pip check` was clean.

### Step 2: Hello-world server
Built `hello_server.py` with two toy tools (`add`, `greet`) using `MCPServer`. Tested with an in-memory `Client(mcp)`. Observed: type hints become a JSON input schema, the docstring becomes the description, the return type becomes an output schema, and a result carries both text content and structured content. Installed `mcp[cli]` for the CLI. Created branch `feature/mcp-server`.

### Step 3: MCP Inspector
`mcp dev` failed (see issue 4). Launched Inspector with the venv's own Python via `npx @modelcontextprotocol/inspector .\venv\Scripts\python.exe mcp_server\hello_server.py`. All three tools worked. A `noisy` tool that prints to stdout did **not** break the connection in v2. The first protocol message was `INITIALIZE` and Inspector labeled the session "Legacy", so the 2025-era handshake was in use.

### Step 4: The adapter layer
Reproduced the relative-path bug on purpose by running `hybrid_search` from the TEMP folder (`FileNotFoundError` at `search_hybrid.py` line 23). Built `rag_backend.py`: a `RagBackend` protocol, `Chunk` and `Answer` dataclasses, and `LocalBackend`, which changes directory to the project root, adds `src/` to `sys.path`, loads `.env`, and imports the RAG modules afterward. Re-ran from TEMP: worked, `request-files.md` first for `python-multipart`.

### Step 5: Tool 1, `search_fastapi_docs`
Built `server.py` with a lazily created backend. The tool trims and validates the query, clamps `top_k` to 1 to 10, and returns numbered excerpts with source file names. The SDK passed the whole docstring, including the `Args:` block, as the description. An empty query raised `ValueError`, but the client received only a generic message (issue 6).

### Step 6: Tool 2 and error handling
Replaced `ValueError` with `ToolError`. Added `RagBackendError` and a context manager that turns Gemini API errors (code 429 means quota) into plain-language messages. Added `ask_fastapi_docs`, which appends "Files retrieved: ..." (labeled "retrieved", not "used", because a refused question still retrieves files). Tested with a fake backend for quota errors: messages reached the client, and the `ask` error added a hint toward `search_fastapi_docs`. Real calls: an answerable question returned a cited answer; the Kubernetes question was correctly declined.

### Step 7: Measure, then fix
Timed each stage of a search separately (see section 6). Result: the BM25/FAISS rebuild was only about 80 to 125 ms, but **creating the Gemini client cost about 1 s on every call**. Fixes: (A) `src/gemini_client.py` with a cached `get_client()` used by `search_hybrid.py` and `generate.py`; (B) a warm-up thread at server start; (C) quieter logs. After the fix, steady-state calls dropped from about 2.4 s to about 0.75 s with identical rankings and scores.

### Step 8: Claude Desktop
Added the server to `claude_desktop_config.json`. On this Windows install the app reads the MSIX-packaged copy under `AppData\Local\Packages\AnthropicPBC.Claude_...\LocalCache\Roaming\Claude\`. The file also holds the app's own preferences, so the `mcpServers` block was added at the top and the rest left untouched. After a restart, Settings showed the server as Running. Tested four prompts (section 7).

### Step 9: Cursor
Created `.cursor/mcp.json` with `${workspaceFolder}` paths and `"type": "stdio"`. The app's MCP panel showed `fastapi-docs-rag` connected with 2 tools. Same four prompts as Claude Desktop (section 7).

### Step 10: Automated tests
`pytest.ini` sets `pythonpath = mcp_server src`. `tests/test_mcp_server.py` has 8 unit tests against a fake backend and 1 integration test that launches the real server over stdio from a different folder (runs only with `RUN_INTEGRATION=1`). First run: 7 passed, 1 failed, 1 skipped. The failure was a test bug (issue 13). The integration test passed.

## 5. Every issue encountered and how it was fixed

| # | Issue | Root cause | Fix |
|---|---|---|---|
| 1 | `node` and `npx` not recognized | Node.js not installed | Installed Node LTS (v24.19.0), restarted VS Code |
| 2 | `FileNotFoundError` for `data/processed/embeddings.npy` when run from another folder | Relative path resolves against the launch folder, and hosts launch servers from elsewhere | Adapter changes to the project root before loading; RAG code left unedited |
| 3 | **Builder-side correction:** an early claim that `print()` calls in `generate.py` would corrupt stdout | The prints sit inside `if __name__ == "__main__":`, so importing never runs them | Corrected in the log; v2 also diverts stray stdout to stderr while serving |
| 4 | `mcp dev` failed with "'uv' is not recognized" | `mcp dev` runs the server in a temporary `uv run --with mcp` environment containing only `mcp`, so it would also lack FAISS and `rank_bm25` | Launch Inspector with `npx ... .\venv\Scripts\python.exe <script>` |
| 5 | `mcp --help` and `mcp dev` need CLI pieces | Plain `pip install mcp` omits the `cli` extra | `pip install "mcp[cli]"` |
| 6 | Model saw only "Error executing tool search_fastapi_docs" for an empty query | v2 treats ordinary exceptions as crashes and hides their text; only `ToolError` messages reach the client | Raise `ToolError`; wrap backend failures in `RagBackendError` and convert |
| 7 | About 2.2 to 2.7 s per search | `genai.Client(...)` created inside `hybrid_search` and `answer` on every call, about 1 s each | Shared cached client in `src/gemini_client.py` |
| 8 | First tool call 6.7 s | Backend created lazily on the first request | Warm-up thread at server start |
| 9 | `RecursionError: maximum recursion depth exceeded` | Find-and-replace of the client line also rewrote the line inside `gemini_client.py`, making `get_client()` call itself | Restored the file; the replace had also touched `eval_generation.py` and `eval_retrieval.py`, which were reverted with `git checkout` after reading `git diff` |
| 10 | `git push` failed: no upstream branch | The branch existed only locally (a branch name like `feature/mcp-server` is a label, not a folder) | `git push --set-upstream origin feature/mcp-server` |
| 11 | Edit Config targeted a different file than the one the app reads | MSIX-packaged app uses a virtualized config location; two config files exist (app config and developer settings) | Edited the file named `claude_desktop_config.json` under the `Packages` path, with the app closed, then validated the JSON with `ConvertFrom-Json` |
| 12 | Reload MCP Configuration appeared not to work | Not established; the server later showed Running after a full restart | Full quit from the system tray and reopen is the reliable path |
| 13 | `test_descriptions_steer_the_model` failed | The docstring wraps "does NOT write an answer" across two lines, and the test searched for it on one line | Test collapses whitespace before comparing (`_flat`) |
| 14 | Prediction was half wrong | Predicted the Gemini network call would be over 80% of per-call time; it was 33 to 48%, with client creation another 48 to 59% | Measured each stage and fixed what the numbers showed |

## 6. Measurements

**Per-call stage timing (step 7, before the fix, three runs):**

| Stage | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| load chunks, build BM25, score, load embeddings, build FAISS, search (all together) | about 82 ms | about 125 ms | about 79 ms |
| create Gemini client | 980 ms | 979 ms | 903 ms |
| embed query (Gemini network call) | 985 ms | 541 ms | 877 ms |
| total | 2.05 s | 1.65 s | 1.86 s |

One-time imports: 1.94 s.

**Adapter latency, before and after the shared client:**

| | Before (step 4) | After (step 7) |
|---|---|---|
| Startup (`init`) | 2.52 s | 3.02 s |
| Call 1 (cold) | 2.72 s | 3.21 s |
| Call 2 (same query) | 2.41 s | 0.78 s |
| Call 3 (new query) | 2.25 s | 0.71 s |

Rankings and scores were identical before and after (for example `[0.03227] request-files.md` first for `python-multipart`).

**Inspector protocol timings:**

| | Step 5 | After step 7 |
|---|---|---|
| `initialize` | 4294 ms | 6572 ms |
| first `tools/call` | 6717 ms | 998 ms |

The slower `initialize` is probably the warm-up thread competing for the CPU at startup (a hypothesis from one sample). The saving beyond the roughly 1 s client cost may come from the shared client keeping its connection open; this was not measured separately.

**Tests:** 9 collected; 8 unit tests and 1 stdio integration test. Integration run: 8 passed plus the failure noted in issue 13 on that run, in 11.07 s.

## 7. Findings about tool selection across clients

| # | Prompt | Claude Desktop | Cursor |
|---|---|---|---|
| 1 | "Using the FastAPI docs, how do I read a custom header?" | no tool (very likely; no permission prompt, no tool line) | search |
| 1b | "Use the search_fastapi_docs tool ... Cite the source files." | search | (not run) |
| 2 | "According to the FastAPI docs, what is root_path used for behind a proxy? Name the source file." | search | search |
| 3 | "Ask the FastAPI docs assistant how to upload a file." | ask | ask |
| 4 | "What is the capital of France?" | no tool | no tool |
| | Permission prompt | yes | no |

Observations (single runs; the two hosts use different models, so these show variation by client, not its cause):
- Tool selection varies by client. A hypothesis, untested: requests to name or cite a source file trigger retrieval more reliably.
- Cursor's reasoning for prompt 1 states it would search "rather than using the answer-generation tool", matching the steering text in the `ask` description.
- For "read a custom header", the top-ranked excerpt was the response-headers page. Both hosts' models handled it: Claude labeled it a different case. This is the recall-versus-MRR trade-off from the RAG report playing out in practice.
- Claude Desktop labeled which parts of its answer came from the tool and which from its own knowledge.
- Cursor showed no permission prompts, so a model could call the quota-spending `ask` tool freely. A server-side budget is planned (step 13).

## 8. Known limitations

- **Chunks lack code samples.** FastAPI's markdown pages reference code through `{* ../../docs_src/... *}` include markers, and the `docs_src/` files were not ingested. Claude itself noted the excerpts reference code by file without containing it.
- **Naive chunking** (fixed 150-word windows): excerpts can cut mid-sentence, newlines are flattened, and doc markup such as `///` appears. Kept deliberately in the RAG project.
- **Source names are inconsistent:** advanced pages carry an `advanced/` prefix; tutorial pages do not.
- **No server-side limit on `ask` calls**, which spend Gemini generation quota.
- **Tool selection depends on the host and model**, not only on the descriptions.
- **Config paths are Windows-specific** (`venv/Scripts/python.exe`; macOS and Linux use `venv/bin/python`).
- The SDK passes docstrings with their line breaks and indentation; harmless but untidy.
- The Gemini SDK emits an informational "AFC is enabled" warning on generation calls.

## 9. File inventory

| File | Purpose |
|---|---|
| `mcp_server/server.py` | The MCP server and both tools |
| `mcp_server/rag_backend.py` | `RagBackend` protocol, `LocalBackend`, `RagBackendError`, error translation |
| `mcp_server/hello_server.py` | Toy server from step 2 (also used to measure protocol overhead) |
| `mcp_server/check_hello.py`, `check_backend.py`, `check_server.py`, `check_tools.py` | Manual check scripts from steps 2 to 6 |
| `mcp_server/profile_stages.py` | Per-stage latency timer (step 7) |
| `mcp_server/evals/eval_mcp_retrieval.py` | Step 11a/b script (written, not yet run at this log) |
| `src/gemini_client.py` | Cached shared Gemini client |
| `src/search_hybrid.py`, `src/generate.py` | Existing RAG code; one line each now uses `get_client()` |
| `tests/test_mcp_server.py`, `pytest.ini` | Test suite and config |
| `.cursor/mcp.json` | Cursor server config (no secrets) |
| `claude_desktop_config.json` | Lives outside the repo, in the Claude app's data folder |

## 10. Lessons learned

1. **Read the code before designing around it.** Inspecting the repo exposed the relative paths and the per-call rebuild before any server code existed.
2. **Measure before optimizing, and be ready to be wrong.** The rebuild looked wasteful but cost under 10%. The real cost was a hidden 1 s per call.
3. **Errors are messages to the model.** A hidden error message leaves the model unable to recover; a clear one lets it change its input or switch tools.
4. **The adapter interface paid for itself three times:** relative paths, error translation, and fake-backend testing.
5. **Descriptions are part of the product.** They steer behavior, and tests can guard their wording.
6. **Host behavior varies.** Prompts, permission prompts, and tool choice differed between two clients with the same server.
7. **Version-specific knowledge goes stale fast.** The SDK's v2 API, the Inspector launch path, and the Windows config location all differed from older tutorials. Checking current docs saved time.
8. **Bulk find-and-replace is risky.** One replace rewrote the helper's own definition and two unrelated files; `git diff` made the damage visible and reversible.
9. **A wrong test is still useful.** A too-literal assertion exposed how the SDK passes docstrings through.

## 11. Not yet verified

1. Whether the quieter-logs change takes effect in a real client's log view.
2. Where the stray `print()` from the `noisy` test went (stderr is expected; not observed).
3. A direct check for `docs_src` markers in `data/processed/chunks.jsonl`.
4. Whether Cursor's lack of permission prompts is an auto-run setting.
5. That the whitespace fix in the description test is applied and all unit tests plus the integration test pass together.
6. Whether Claude Desktop's first plain-question run really made no tool call (inferred from the missing tool line and prompt, not from the MCP log).

## 12. Remaining steps

- **11a/b:** run the 32 labeled questions through the real stdio server and compare to `eval_results.json`; measure cold start, steady-state median and p95, and protocol overhead.
- **11c:** tool-selection eval (about 24 labeled prompts) using Gemini with the real tool schemas.
- **12:** `HttpBackend` calling the running FastAPI service, switched by an environment variable, with a measured comparison.
- **13:** server-side budget for `ask`; optional resource (read a full doc file) or prompt template.
- **14:** Dockerfile for the MCP server.
- **15:** README, final project report, and merge to `main`.

## Step 11: Evals
**11a/b: retrieval parity and latency through MCP (stdio).** 32 labeled questions run through the real server, launched from the temp folder; scored with the project's own hit/RR definition.
- Parity: 32 compared, 0 differ from eval_results.json. Overall recall 1.000 / MRR 0.907; natural (n=14) 1.000 / 0.824; keyword (n=7) 1.000 / 0.929; hard (n=11) 1.000 / 1.000.
- Latency (one run): cold start to first answer 12,742 ms; steady median 737 ms, min 679, max 876 (n=31; p95 855 is close to the max and not meaningful at this n). Protocol overhead with a trivial tool: median 4.28 ms.
- Limits: single run, one machine; some ranks sit at the edge (n03 rr=0.200).

**11c: tool selection (Gemini only).** gemini-3.5-flash-lite, default settings, 24 labeled prompts, one run each, real tool schemas, tools never executed. 23/24 correct (search 12/12, ask 5/6, none 6/6). No prompt that should not call `ask` called it.
- Only miss: a06 ("Run the grounded-answer tool..."), a paraphrase of the tool; I tagged it `tool-named` in error and corrected the tag to `paraphrased-tool` afterwards. The result itself was not changed.
- Limits: one model, one run, small and mostly easy set (ceiling effect), no argument checking.

## Step 12: HttpBackend
- Design: same RagBackend interface, new HttpBackend (one reused httpx client; connect timeout 3 s, read timeout 60 s; every failure becomes a readable RagBackendError). Switch: RAG_BACKEND=local|http (default local), RAG_SERVICE_URL.
- api.py facts: /query returns file names and scores but no excerpt text; question min length 3; errors are not translated, so a Gemini 429 is expected to appear as a generic 500 (not tested).
- Tests: 16 new HTTP-backend tests with a fake transport; 24 passed, 1 skipped at that point.
- 12e comparison (32 questions): 0 differ in all runs. Steady median: local 717 ms (earlier session 737), http 823 / 821 ms (fresh / warm service), about +100 ms, cause not identified. Cold start 4.6-12.7 s across runs with no reliable backend difference.
- Finding: constructing the httpx client took 1,187 ms (1,301 / 802 ms in a separate test) versus 1 ms with verify=False; certificate loading is the cause. Fix: skip verification setup only for plain http URLs. After the fix, construction measured 113 ms (single run).
- Finding: the MCP client's default environment passes 12 variables; RAG_BACKEND is not among them, so evals pass it explicitly.

## Step 13: Extras
- ask budget: ask_fastapi_docs allows 5 calls per server process (env ASK_BUDGET; 0 disables). Failed attempts count; blank questions do not; search is never limited. 7 tests; 31 passed, 1 skipped overall. Limit: resets with each server or container start; it does not track the daily Gemini quota.
- docs_src gap, measured on data/processed/chunks.jsonl (503 chunks): 275 contain the text "docs_src", 272 contain a real include marker (54%), 168 contain a code fence (33%), 89 contain both. Fence languages: Python 90 (+6 lowercase), JSON 44 (+9), console 42, mermaid 9, bash 4, jinja 4 (fence openings, not chunks). Some code is therefore present; many runnable examples are not.

## Step 14: Docker
- Dockerfile.mcp: python:3.11-slim, mcp==2.2.0 + httpx, three source files, no key or data; own dockerignore (build context 12.79 kB); image about 250 MB.
- Mistake caught by the import check: I assumed httpx came with mcp; it did not.
- Compose: rag-api (healthcheck on /health) plus mcp-server in a profile, started per client session with `docker compose run --rm -T mcp-server`.
- Finding: with the MCP Python client's default environment, `docker compose` fails ("unknown command: docker compose", surfacing as "Connection closed"); passing the full environment fixes it. Plain `docker run -i` works either way.
- Smoke tests: plain docker run and compose run both returned results via the client. Cold start with nothing running: 29.9 s to the first answer (single run, images cached).
- Cursor (model: Cursor Grok 4.6 Medium): search and ask tools worked with the dockerized server, no permission prompt (tool group set to "Allow all"; whether that explains it is untested). Service log showed matching GET /retrieve and POST /query requests.
- Observation: one ask answer contained non-existent decorators (@app.files, @app.upload). Neither string occurs in any chunk (0 of 503), so they were not in the context the model was given; cause not verified. The "uv add" line occurs in 19 chunks.
- Observation: the service log showed a Gemini SDK message about automatic function calling; not investigated (hypothesis: hidden by the logging setting in server.py).

## Not verified
Quieter logs in a real client console; where the noisy test's stray print went; whether Cursor's no-prompt behavior is the "Allow all" setting; Claude Desktop with the Docker server; which container name/route Cursor used for the compose entry [fill in if you check].
