import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from generate import answer  # hybrid retrieval + grounded generation (step 8)

load_dotenv()

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_set.json"
RESULTS_PATH = Path(__file__).parent.parent / "eval_generation_results.json"
JUDGE_MODEL = "gemini-2.5-flash"
SECONDS_BETWEEN_QUESTIONS = 13  # keeps two generation calls per question under the free-tier limit

ANSWERABLE_IDS = ["n02", "n04", "n06", "n08", "n10", "n12",
                  "h02", "h04", "h06", "h08", "h09", "h11"]

# Topics the docs do NOT cover. The right behavior is to decline.
UNANSWERABLE = [
    {"id": "u01", "question": "How do I deploy FastAPI to Kubernetes?"},
    {"id": "u02", "question": "How do I add Redis caching to my endpoints?"},
    {"id": "u03", "question": "How do I integrate Stripe payments?"},
    {"id": "u04", "question": "How do I send emails over SMTP from my app?"},
    {"id": "u05", "question": "How do I add Elasticsearch full-text search?"},
]

FAITHFULNESS_PROMPT = """You are a strict evaluator of a documentation Q&A system.

QUESTION:
{question}

RETRIEVED CONTEXT (the only material the system was allowed to use):
{context}

SYSTEM ANSWER:
{answer}

Grade two things.

faithfulness (1-5): Is every factual claim in the answer supported by the retrieved context?
5 = every claim supported. 4 = one minor detail not in the context. 3 = several unsupported
details. 2 = a significant unsupported or contradicted claim. 1 = mostly unsupported.
List each unsupported claim in "unsupported_claims" (empty list if none).

relevancy (1-5): Does the answer directly address the question?
5 = fully answers it. 3 = partially. 1 = off-topic.

Respond with JSON only:
{{"unsupported_claims": ["..."], "faithfulness": <int>, "relevancy": <int>, "reason": "<one sentence>"}}"""

REFUSAL_PROMPT = """You are evaluating a documentation Q&A system on a question its
documentation does NOT cover. The correct behavior is to say the context lacks the information
and NOT present made-up specifics as if they were documented.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER:
{answer}

Decide:
- declined: true if the answer says the context does not contain the requested information.
- fabricated_specifics: true if the answer gives concrete instructions, code, or facts about the
  out-of-scope topic that are not in the context.
(Mentioning related things that ARE in the context, with citations, is not fabrication.)

Respond with JSON only:
{{"declined": <true|false>, "fabricated_specifics": <true|false>, "reason": "<one sentence>"}}"""


def with_retry(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                raise SystemExit(
                    "Daily free-tier quota reached. Progress is saved; "
                    "re-run after the quota resets."
                )
            transient = any(t in msg for t in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not transient or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"    rate limited or busy, waiting {wait}s (attempt {attempt + 1}/{tries})...")
            time.sleep(wait)


def judge(client, prompt):
    def call():
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
        )
        return json.loads(resp.text)
    return with_retry(call)


def build_context(hits):
    return "\n\n---\n\n".join(f"[source: {h['source_file']}]\n{h['text']}" for h in hits)


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(results):
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    eval_set = {q["id"]: q for q in json.loads(EVAL_PATH.read_text(encoding="utf-8"))}

    todo = [("answerable", eval_set[i]) for i in ANSWERABLE_IDS] + \
           [("unanswerable", q) for q in UNANSWERABLE]
    results = load_results()

    for kind, item in todo:
        qid = item["id"]
        if qid in results:
            print(f"  skip {qid} (already done)")
            continue
        print(f"  running {qid}: {item['question']}")

        text, hits = with_retry(lambda: answer(item["question"]))
        context = build_context(hits)
        prompt_tpl = FAITHFULNESS_PROMPT if kind == "answerable" else REFUSAL_PROMPT
        verdict = judge(client, prompt_tpl.format(
            question=item["question"], context=context[:6000], answer=text))

        results[qid] = {"kind": kind, "question": item["question"],
                        "answer": text, "sources": [h["source_file"] for h in hits],
                        "verdict": verdict}
        save_results(results)
        time.sleep(SECONDS_BETWEEN_QUESTIONS)

    # ---- Summary ----
    ans = [r for r in results.values() if r["kind"] == "answerable"]
    una = [r for r in results.values() if r["kind"] == "unanswerable"]

    if ans:
        f = [r["verdict"]["faithfulness"] for r in ans]
        rel = [r["verdict"]["relevancy"] for r in ans]
        print(f"\n=== Answerable questions (n={len(ans)}) ===")
        print(f"avg faithfulness: {sum(f) / len(f):.2f} / 5")
        print(f"avg relevancy:    {sum(rel) / len(rel):.2f} / 5")
        print(f"faithfulness >= 4: {sum(x >= 4 for x in f)}/{len(f)}")
        weak = [r for r in ans if r["verdict"]["faithfulness"] < 5]
        if weak:
            print("\nAnswers with unsupported claims:")
            for r in weak:
                print(f"  - {r['question']}  (faithfulness {r['verdict']['faithfulness']})")
                for c in r["verdict"].get("unsupported_claims", []):
                    print(f"      * {c}")

    if una:
        passed = [r for r in una if r["verdict"]["declined"] and not r["verdict"]["fabricated_specifics"]]
        print(f"\n=== Unanswerable questions (n={len(una)}) ===")
        print(f"correctly declined without fabricating: {len(passed)}/{len(una)}")
        for r in una:
            if r not in passed:
                print(f"  - FAILED: {r['question']} -> {r['verdict']['reason']}")

    print(f"\nFull answers and verdicts saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()