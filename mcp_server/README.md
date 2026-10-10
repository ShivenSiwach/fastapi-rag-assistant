# FastAPI Docs MCP Server
 
**An MCP server that exposes the FastAPI Docs RAG Assistant as tools for Claude Desktop, Cursor and other MCP clients — with a retrieval-parity eval, a tool-selection eval, two interchangeable backends and a Docker image.**
 
Built on top of the [FastAPI Docs RAG Assistant](../README.md). The RAG system answers questions through Python code and an HTTP API; this server lets an AI client call it as a tool, and measures what that extra layer costs and whether models use it correctly.
 
---
 
## Table of Contents
 
- [Why This Project Exists](#why-this-project-exists)
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Evaluation Results](#evaluation-results)
- [Setup](#setup)
- [Docker](#docker)
- [Configuration](#configuration)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Engineering Challenges Solved](#engineering-challenges-solved)
- [Known Limitations](#known-limitations)
- [What I'd Add Next](#what-id-add-next)
- [Tech Stack](#tech-stack)
---
 
## Why This Project Exists
 
Wrapping a function in an MCP tool is the easy part. This project measures the parts that decide whether the wrapper is any good:
 
1. **Does the server change what retrieval returns?** Checked by running the 32 labeled questions through the real stdio server and comparing every question with the original RAG eval results.
2. **What does the extra layer cost?** Latency through the server for both backends, with the protocol overhead measured separately.
3. **Does a model pick the right tool?** A labeled tool-selection eval using the real tool schemas, without running the tools.
---
 
## What It Does
 
| Tool | Purpose |
|---|---|
| `search_fastapi_docs(query, top_k=5)` | Returns numbered excerpts with their source file names. It does **not** write an answer: the client model reads the excerpts itself. `top_k` is clamped to 1–10. An empty query returns a readable error. |
| `ask_fastapi_docs(question, top_k=5)` | Runs the full pipeline (hybrid retrieval → grounded, cited generation) and appends the list of files retrieved. Limited to 5 calls per server process (see [Configuration](#configuration)). |
 
The tool descriptions steer models toward `search_fastapi_docs`, because it is faster and uses no answer-generation quota (which is small on the free tier).
 
---
 
## Architecture
 
```
Claude Desktop / Cursor  (MCP client)
        │  stdio (JSON-RPC)
        ▼
┌─────────────────────────────────────────────────────────────┐
│  mcp_server/server.py                                          │
│  two tools · input checks · ask budget · quiet logging          │
└─────────────────────────────────────────────────────────────┘
        │  RagBackend interface (mcp_server/rag_backend.py)
        ├───────────────────────────────┐
        ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│ LocalBackend              │   │ HttpBackend                   │
│ imports src/search_hybrid │   │ mcp_server/http_backend.py    │
│ and src/generate in the   │   │ calls the running FastAPI     │
│ same process              │   │ service: /retrieve, /query    │
└──────────────────────────┘   └──────────────────────────────┘
```
 
The tools depend only on the `RagBackend` interface (`retrieve`, `answer`), so switching backends changes nothing in `server.py`.
 
---
 
## Key Engineering Decisions
 
**An adapter between the server and the RAG code.**
`rag_backend.py` is the only place that touches the RAG modules. It also holds the paths, the `.env` loading, and the translation of Gemini API errors (a 429 becomes a readable quota message). The original `src/` code is unchanged apart from sharing one Gemini client.
 
**Only `ToolError` messages are shown to the model.**
In version 2 of the MCP Python SDK, text from any other exception is hidden from the model. Every failure the model should understand (empty query, quota exhausted, service unreachable, budget used up) is therefore converted to a readable `ToolError` or `RagBackendError`.
 
**One shared Gemini client.**
Creating the client cost about 1 s per call. Sharing a cached client brought calls from about 2.4 s to about 0.75 s with identical rankings, and the first tool call from 6.7 s to 1.0 s. Rebuilding BM25 and FAISS per call costs only about 80–125 ms, so that part was left alone.
 
**Search returns excerpts, ranks and file names; not RRF scores.**
Raw fusion scores mean nothing to a model. Excerpts are numbered by rank, and the source file names are printed exactly as the eval set expects them.
 
**A server-side budget for `ask_fastapi_docs`.**
In a Cursor test the tools ran without permission prompts, so a model could call the quota-spending tool freely. The server therefore allows 5 `ask` calls per process by default, counts failed attempts (a failed call may still use quota), and never limits `search`.
 
**No secrets in client configs.**
The Gemini key is read from the project's `.env` by the local backend and the service. The Docker image for the MCP server contains no key and no data.
 
**Certificate loading only for https.**
Constructing the default `httpx` client took about 1.2 s (certificate loading) even for a plain `http://` URL. The HTTP backend now skips it for `http://` and keeps verification for `https://`.
 
---
 
## Evaluation Results
 
### Retrieval parity — through the real stdio server
*32 labeled questions, hybrid retrieval, recall@5 = hit rate (any expected file in the top 5), server launched from a different working folder.*
 
| Group | n | MCP recall | MCP MRR | RAG project recall | RAG project MRR |
|---|:---:|:---:|:---:|:---:|:---:|
| Overall | 32 | 1.000 | 0.907 | 1.000 | 0.907 |
| Natural | 14 | 1.000 | 0.824 | 1.000 | 0.824 |
| Keyword | 7 | 1.000 | 0.929 | 1.000 | 0.929 |
| Hard | 11 | 1.000 | 1.000 | 1.000 | 1.000 |
 
Per-question comparison with `eval_results.json`: **32 compared, 0 differ** — in the local run and in both HTTP-backend runs. The server returns the same ranking as the RAG code on this question set.
 
### Latency — `search_fastapi_docs` over stdio
*Single runs on one Windows machine. Steady-state excludes the first call.*
 
| Run | Cold start to first answer | Steady median | Steady min – max |
|---|:---:|:---:|:---:|
| Local backend (earlier session) | 12,742 ms | 737 ms | 679 – 876 ms |
| Local backend (same session as HTTP) | 7,425 ms | 717 ms | 621 – 1,433 ms |
| HTTP backend, fresh service | 8,531 ms | 823 ms | 715 – 1,009 ms |
| HTTP backend, warm service | 4,586 ms | 821 ms | 674 – 1,019 ms |
 
- Protocol overhead, measured with a trivial tool and no network: about 4–8 ms per call. The time is dominated by retrieval, not by MCP.
- The HTTP backend is about 100 ms slower per call at the median (about 822 ms vs 717 ms). The cause was not identified.
- Docker Compose from nothing running (service start, healthcheck wait, MCP container, handshake, one search): **29.9 s** to the first answer, single run, images already built.
> **Honest caveats:** cold-start numbers vary a lot between runs of the same code (the two local runs differ by 5.3 s), so no backend comparison is claimed for cold start. The 95th percentile is not reported: with 31 samples it is close to the maximum. Everything is a single run on one machine.
 
### Tool selection — does a model pick the right tool?
*Gemini (`gemini-3.5-flash-lite`), default settings, 24 labeled prompts, one run each, real tool schemas from the live server, tools never executed.*
 
| Expected | Correct |
|---|:---:|
| `search` (12 prompts) | 12 / 12 |
| `ask` (6 prompts) | 5 / 6 |
| no tool (6 prompts) | 6 / 6 |
| **Overall** | **23 / 24** |
 
No prompt that should not trigger `ask` called it. The one miss was a prompt that paraphrased the tool ("Run the grounded-answer tool...") instead of naming it.
 
> **Honest caveats:** this measures the tool descriptions for one model only. The labels follow the intended policy in the descriptions, so the eval checks adherence to that policy, not whether the policy is best. The set is small and mostly easy (the no-tool prompts are all clearly unrelated to FastAPI), which is a ceiling effect, and only the tool name was scored, not the arguments.
 
### Real clients (manual, not scored)
 
- **Cursor** (model: Cursor Grok 4.6 Medium) connected to the Dockerized server, used `search_fastapi_docs` for documentation questions and `ask_fastapi_docs` when asked to "ask the docs assistant". No permission prompt appeared; the tool group was set to "Allow all" in Cursor, and whether that setting is the reason was not tested. The service log showed the matching `GET /retrieve` and `POST /query` requests.
- **Claude Desktop** was connected to the local server earlier and used the tool when asked to name a source file; for a plain "how do I" question it appears to have skipped the tool (inferred from the chat, not read from the MCP log).
These are observations of single conversations with specific models. They are not comparable with the Gemini tool-selection numbers above.
 
---
 
## Setup
 
```bash
# from the project root, inside the existing virtual environment
pip install -r requirements-mcp.txt
```
 
The local backend also needs the RAG project's requirements and a `.env` with `GEMINI_API_KEY` (see the [RAG README](../README.md#setup)).
 
**Run the server directly** (it waits silently for an MCP client on stdin/stdout):
 
```bash
python mcp_server/server.py
```
 
**Cursor** — `.cursor/mcp.json`:
 
```json
{
  "mcpServers": {
    "fastapi-docs-rag": {
      "type": "stdio",
      "command": "${workspaceFolder}/venv/Scripts/python.exe",
      "args": ["${workspaceFolder}/mcp_server/server.py"]
    }
  }
}
```
 
**Claude Desktop** — `claude_desktop_config.json` (use absolute paths; no key goes in this file, it is read from `.env`):
 
```json
{
  "mcpServers": {
    "fastapi-docs-rag": {
      "command": "C:\\path\\to\\fastapi-rag-tutorial\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\fastapi-rag-tutorial\\mcp_server\\server.py"]
    }
  }
}
```
 
**HTTP backend** — start the RAG service, then select the backend through the environment:
 
```bash
uvicorn api:app --app-dir src          # terminal 1
RAG_BACKEND=http python mcp_server/server.py   # terminal 2 (PowerShell: $env:RAG_BACKEND="http")
```
 
---
 
## Docker
 
The MCP image holds only the server and its three Python files plus `mcp` and `httpx` (about 250 MB). It contains no key and no data, and calls the RAG service over HTTP.
 
```bash
docker build -f Dockerfile.mcp -t fastapi-docs-mcp .
docker run -i --rm fastapi-docs-mcp
```
 
`-i` is required: a stdio server exits when its input closes. The image's default service address is `http://host.docker.internal:8000`, so the RAG service can run on the host (`uvicorn api:app --app-dir src`).
 
**With Docker Compose** — the RAG service runs as a normal service, and the MCP server is started per client session:
 
```bash
docker compose up -d rag-api                         # service, with a /health healthcheck
docker compose run --rm -T mcp-server                # what an MCP client launches
python mcp_server/check_docker.py compose run --rm -T mcp-server   # smoke test
```
 
Cursor entry for the compose route:
 
```json
"fastapi-docs-rag-docker": {
  "type": "stdio",
  "command": "docker",
  "args": ["compose", "-f", "${workspaceFolder}/docker-compose.yml", "run", "--rm", "-T", "mcp-server"]
}
```
 
`-T` turns off pseudo-TTY allocation so the JSON-RPC messages stay on raw stdin/stdout. The MCP service sits in a Compose profile, so a plain `docker compose up` starts only the RAG service.
 
> **Note:** on the test machine, a script using the MCP Python client with its default environment could not run `docker compose` ("unknown command: docker compose"). The client forwards only a short list of environment variables, and passing the full environment fixed it. Plain `docker run` worked either way. Tested on Docker Desktop for Windows only (Compose v5.1.3).
 
---
 
## Configuration
 
| Variable | Default | Meaning |
|---|---|---|
| `RAG_BACKEND` | `local` | `local` imports the RAG code; `http` calls the service. Any other value gives a readable error. |
| `RAG_SERVICE_URL` | `http://127.0.0.1:8000` (`http://host.docker.internal:8000` in the image, `http://rag-api:8000` in Compose) | Where the HTTP backend finds the service. |
| `ASK_BUDGET` | `5` | Maximum `ask_fastapi_docs` calls per server process; `0` disables the tool. An invalid value falls back to 5. |
| `GEMINI_API_KEY` | from `.env` | Needed by the local backend and by the service; not by the MCP Docker image. |
 
---
 
## Tests
 
```bash
python -m pytest                      # 31 passed, 1 skipped
RUN_INTEGRATION=1 python -m pytest    # also runs the stdio end-to-end test (real Gemini calls)
```
 
The suite covers tool listing, description wording, output formatting, `top_k` clamping, readable errors, the HTTP backend (with a fake transport, no service needed), backend selection, and the `ask` budget.
 
**Evals:**
 
```bash
python mcp_server/evals/eval_mcp_retrieval.py local              # or: http [label]
python mcp_server/evals/eval_tool_selection.py                   # resumable; optional argument = number of prompts to run
```
 
Results are saved as JSON next to the scripts.
 
---
 
## Project Structure
 
```
fastapi-rag-tutorial/
├── mcp_server/
│   ├── server.py                  # MCP server: two tools, ask budget
│   ├── rag_backend.py             # RagBackend interface + LocalBackend
│   ├── http_backend.py            # HttpBackend (calls the FastAPI service)
│   ├── check_docker.py            # smoke test through docker / docker compose
│   └── evals/
│       ├── eval_mcp_retrieval.py       # parity + latency through stdio
│       ├── eval_tool_selection.py      # tool-selection eval
│       ├── tool_selection_set.json     # 24 labeled prompts
│       └── results_*.json              # saved outputs
├── tests/
│   ├── test_mcp_server.py
│   ├── test_http_backend.py
│   └── test_ask_budget.py
├── Dockerfile.mcp                 # MCP image (server only)
├── Dockerfile.mcp.dockerignore
├── docker-compose.yml             # rag-api + mcp-server (profile)
└── .cursor/mcp.json               # Cursor client config
```
 
---
 
## Engineering Challenges Solved
 
- **Hidden error text in MCP SDK v2.** Non-`ToolError` exceptions are not shown to the model, so every expected failure is converted to a readable message.
- **`mcp dev` needs `uv`.** The MCP Inspector was launched through `npx` with the virtual environment's Python instead.
- **Gemini client creation dominated call time** (about 1 s per call); fixed with one cached client, verified with identical rankings before and after.
- **Certificate loading cost about 1.2 s** when constructing the HTTP client for a plain `http://` URL; skipped for `http://`, kept for `https://`.
- **A missing dependency in the Docker image.** `httpx` was assumed to come with `mcp` and did not; caught by an import check inside the built image before any client run.
- **`docker compose` not found from the MCP client's trimmed environment**, surfacing only as "Connection closed"; found by running the same command with the full and the client's default environment.
- **A stdio server cannot be a long-running Compose service.** The client has to own its stdin/stdout, so it is started per session with `docker compose run --rm -T`.
---
 
## Known Limitations
 
- **Code samples are largely missing from the corpus.** The RAG corpus was built from the markdown files only; the `docs_src` folder with the runnable examples was not ingested. Of 503 chunks, 272 (54%) contain an include marker pointing to a `docs_src` file, and 168 (33%) contain a code fence (mostly Python, JSON and console blocks). Fences can be split across chunks because chunks are fixed 150-word windows.
- **`ask_fastapi_docs` can return code that is not in the documents.** In one Cursor run the answer contained decorators that do not exist in FastAPI (`@app.files`, `@app.upload`); neither string occurs in any of the 503 chunks, so they were not in the context the model was given. The client model noticed and flagged it. This is one observation, not a measured rate. It is the reason the tool descriptions prefer `search_fastapi_docs`, where the client model reads the actual excerpts.
- **The `ask` budget is per server process.** It resets whenever the server or container restarts and does not track the daily Gemini quota.
- **HTTP backend differences.** The service's `/query` returns file names and scores but no excerpt text, so `ask` results carry only file names. The service requires questions of at least 3 characters. Service errors are not translated, so a Gemini 429 inside the service is expected to appear as a generic server error (not tested).
- **About 100 ms slower per call over HTTP,** cause not identified.
- **Model-specific results.** The tool-selection eval covers one model; the client observations are single conversations. The no-permission-prompt behavior in Cursor is unexplained.
- **Platform.** Everything was measured on one Windows machine; Docker was tested on Docker Desktop for Windows only.
---
 
## What I'd Add Next
 
- **Ingest `docs_src` and resolve the include markers**, then re-chunk, re-embed and re-run every eval. This changes the corpus, so none of the numbers above would carry over.
- **A persistent daily budget** for `ask_fastapi_docs` that survives restarts and follows the provider's quota reset.
- **Translate Gemini errors inside the service** so the HTTP backend can tell quota exhaustion from other faults.
- **A harder tool-selection set** with FastAPI-adjacent prompts that do not need the docs, repeated runs per prompt, and more than one model.
- **An HTTP-based MCP transport**, so one long-running container could serve several clients.
- **Find the cause of the ~100 ms HTTP gap** by timing the service endpoint directly, outside MCP.
---
 
## Tech Stack
 
**Language & Runtime:** Python 3.11
**Protocol:** Model Context Protocol, Python SDK 2.2.0 (stdio transport)
**Backends:** in-process RAG modules · `httpx` client to the FastAPI service
**Containerization:** Docker · Docker Compose
**Testing & Evaluation:** pytest · custom parity, latency and tool-selection evals
**LLM Provider:** Google Gemini API (via the RAG system)
 
---
 
*Built as an extension of the FastAPI Docs RAG Assistant: the retrieval and generation code stays unchanged, and every claim above is tied to a measurement or marked as untested.*
 