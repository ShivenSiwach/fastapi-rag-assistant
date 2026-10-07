"""11c: which tool does Gemini pick for each labeled prompt? Tools are never executed."""
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp_server"))

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from mcp import Client

import server  # importing does not start the backend (it is created lazily)

SET_PATH = Path(__file__).resolve().parent / "tool_selection_set.json"
OUT_PATH = Path(__file__).resolve().parent / "results_tool_selection.json"
MODEL = "gemini-3.5-flash-lite"
PAUSE_S = 3  # small pause between calls, to stay gentle on rate limits
SHORT = {"search_fastapi_docs": "search", "ask_fastapi_docs": "ask"}


async def load_declarations():
    """Build Gemini tool declarations from the live server's real schemas."""
    async with Client(server.mcp) as client:
        tools = (await client.list_tools()).tools
    decls = []
    for t in tools:
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
        if not schema:
            raise SystemExit(f"Could not find the input schema on tool {t.name!r}")
        decls.append(types.FunctionDeclaration(
            name=t.name, description=t.description, parameters_json_schema=schema))
    return decls


def fingerprint(decls):
    """Hash of everything the model sees, so results can't mix two versions of the tools."""
    blob = json.dumps(
        [(d.name, d.description, d.parameters_json_schema) for d in decls],
        sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def calls_in(resp):
    calls = []
    for cand in resp.candidates or []:
        parts = cand.content.parts if cand.content and cand.content.parts else []
        for part in parts:
            if part.function_call:
                calls.append({"name": part.function_call.name,
                              "args": dict(part.function_call.args or {})})
    return calls


def classify(resp):
    if not resp.candidates:
        return "no_response", []
    calls = calls_in(resp)
    if not calls:
        return "none", []
    return SHORT.get(calls[0]["name"], calls[0]["name"]), calls


def report(items, results):
    done = [i for i in items if i["id"] in results]
    ok = [i for i in done if results[i["id"]]["picked"] == i["expected"]]
    print(f"\n=== Tool selection ({MODEL}, one run per prompt) ===")
    print(f"overall: {len(ok)}/{len(done)} correct")

    for label, keyfn in [("by expected", lambda i: i["expected"]),
                         ("by style", lambda i: i["style"])]:
        groups = defaultdict(list)
        for i in done:
            groups[keyfn(i)].append(i)
        print(f"\n{label}:")
        for g, members in groups.items():
            good = sum(results[m["id"]]["picked"] == m["expected"] for m in members)
            print(f"  {g:<16}{good}/{len(members)}")

    conf = Counter((i["expected"], results[i["id"]]["picked"]) for i in done)
    print("\nconfusion (expected -> picked: count):")
    for (e, p), n in sorted(conf.items()):
        print(f"  {e} -> {p}: {n}")

    wrong = [i for i in done if results[i["id"]]["picked"] != i["expected"]]
    print("\nmismatches:")
    for i in wrong:
        print(f"  {i['id']} expected={i['expected']} picked={results[i['id']]['picked']}: {i['prompt']}")
    if not wrong:
        print("  none")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None  # smoke test: run only N new prompts
    load_dotenv(ROOT / ".env")
    items = json.loads(SET_PATH.read_text(encoding="utf-8"))
    decls = asyncio.run(load_declarations())
    fp = fingerprint(decls)

    state = {"model": MODEL, "tools_fingerprint": fp, "results": {}}
    if OUT_PATH.exists():
        saved = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if saved.get("tools_fingerprint") != fp or saved.get("model") != MODEL:
            raise SystemExit("Saved results used different tools or model. "
                             f"Delete {OUT_PATH.name} to start fresh.")
        state = saved
    results = state["results"]

    client = genai.Client()
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    new_runs = 0
    for item in items:
        if item["id"] in results:
            continue
        if limit is not None and new_runs >= limit:
            break
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=item["prompt"], config=config)
        except errors.APIError as exc:
            print(f"Stopped at {item['id']}: API error {exc.code}. Re-run later to resume.")
            break
        picked, calls = classify(resp)
        results[item["id"]] = {"picked": picked, "calls": calls}
        OUT_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        mark = "ok " if picked == item["expected"] else "MISS"
        print(f"  {item['id']}: expected={item['expected']:<6} picked={picked:<12} {mark}")
        new_runs += 1
        time.sleep(PAUSE_S)

    report(items, results)
    print(f"\n{len(results)}/{len(items)} prompts done. Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()