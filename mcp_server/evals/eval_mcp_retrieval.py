"""11a/11b: run the labeled eval set through the real MCP server (stdio)."""
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]
MCP_DIR = ROOT / "mcp_server"
EVAL_PATH = ROOT / "data" / "eval_set.json"
BASELINE_PATH = ROOT / "eval_results.json"
OUT_PATH = Path(__file__).resolve().parent / "results_mcp_retrieval.json"
K = 5
SEP = "\n\n---\n\n"


def parse_sources(text):
    """Source file of each excerpt, in rank order, from the tool's output."""
    files = []
    for part in text.split(SEP)[1:]:
        first_line = part.split("\n", 1)[0]
        marker = "] source: "
        if marker not in first_line:
            raise ValueError(f"unexpected output format: {first_line!r}")
        files.append(first_line.split(marker, 1)[1].strip())
    return files


def score(files, expected):
    """Same definition as src/eval_retrieval.py: (hit, reciprocal rank)."""
    for pos, f in enumerate(files[:K], start=1):
        if f in expected:
            return 1, 1 / pos
    return 0, 0.0


def summarize(rows):
    n = len(rows)
    return {
        "n": n,
        "recall": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
    }


def percentile95(values):
    return statistics.quantiles(values, n=20)[-1]


async def run_eval():
    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_DIR / "server.py")],
        cwd=tempfile.gettempdir(),  # deliberately not the project folder
    )
    rows = []
    cold_ms = None
    t_launch = time.perf_counter()
    async with Client(stdio_client(params)) as client:
        for item in eval_set:
            t0 = time.perf_counter()
            result = await client.call_tool(
                "search_fastapi_docs", {"query": item["question"], "top_k": K}
            )
            ms = (time.perf_counter() - t0) * 1000
            if cold_ms is None:
                cold_ms = (time.perf_counter() - t_launch) * 1000
            if result.is_error:
                raise SystemExit(f"{item['id']}: tool error: {result.content[0].text}")
            files = parse_sources(result.content[0].text)
            hit, rr = score(files, set(item["expected_sources"]))
            rows.append(
                {"id": item["id"], "type": item["type"], "question": item["question"],
                 "hit": hit, "rr": rr, "ms": round(ms), "files": files}
            )
            print(f"  {item['id']}: hit={hit} rr={rr:.3f} {ms:.0f} ms")
    return rows, cold_ms


async def protocol_overhead(n=50):
    params = StdioServerParameters(
        command=sys.executable, args=[str(MCP_DIR / "hello_server.py")]
    )
    async with Client(stdio_client(params)) as client:
        await client.call_tool("add", {"a": 1, "b": 1})  # warm-up call
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            await client.call_tool("add", {"a": 1, "b": 2})
            times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    print("Running the eval set through the MCP server...")
    rows, cold_ms = asyncio.run(run_eval())

    baseline = {}
    if BASELINE_PATH.exists():
        data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        baseline = {b["id"]: b for b in data["per_question"]}

    print("\n=== Retrieval parity (hybrid, recall@5 = hit rate) ===")
    print(f"{'group':<10}{'n':<4}{'MCP recall':<12}{'MCP MRR':<10}{'RAG recall':<12}{'RAG MRR':<8}")
    summaries = {}
    for group in ["overall", "natural", "keyword", "hard"]:
        sub = rows if group == "overall" else [r for r in rows if r["type"] == group]
        mcp_s = summarize(sub)
        base_rows = [
            {"hit": baseline[r["id"]]["hybrid"]["hit"], "rr": baseline[r["id"]]["hybrid"]["rr"]}
            for r in sub if r["id"] in baseline
        ]
        base_s = summarize(base_rows) if base_rows else {"recall": "-", "mrr": "-"}
        summaries[group] = {"mcp": mcp_s, "rag_project": base_s}
        print(f"{group:<10}{mcp_s['n']:<4}{mcp_s['recall']:<12}{mcp_s['mrr']:<10}"
              f"{base_s['recall']!s:<12}{base_s['mrr']!s:<8}")

    mismatches = [
        r["id"] for r in rows
        if r["id"] in baseline
        and (baseline[r["id"]]["hybrid"]["hit"] != r["hit"]
             or round(baseline[r["id"]]["hybrid"]["rr"], 6) != round(r["rr"], 6))
    ]
    compared = sum(1 for r in rows if r["id"] in baseline)
    print(f"\nPer-question comparison with eval_results.json: "
          f"{compared} compared, {len(mismatches)} differ {mismatches}")

    steady = [r["ms"] for r in rows[1:]]
    latency = {
        "cold_start_to_first_answer_ms": round(cold_ms),
        "steady_n": len(steady),
        "steady_median_ms": round(statistics.median(steady)),
        "steady_p95_ms": round(percentile95(steady)),
        "steady_min_ms": min(steady),
        "steady_max_ms": max(steady),
    }
    print("\n=== Latency through MCP (search_fastapi_docs over stdio) ===")
    for k, v in latency.items():
        print(f"  {k}: {v}")

    print("\nMeasuring protocol overhead (trivial 'add' tool, no network)...")
    times = asyncio.run(protocol_overhead())
    overhead = {
        "n": len(times),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(percentile95(times), 2),
    }
    print(f"  median {overhead['median_ms']} ms, p95 {overhead['p95_ms']} ms (n={overhead['n']})")

    OUT_PATH.write_text(
        json.dumps({"summaries": summaries, "mismatches": mismatches, "latency": latency,
                    "protocol_overhead": overhead, "per_question": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()