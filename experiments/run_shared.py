"""Run one backend on the shared extractive QA set and export auditable evidence."""

import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

from qwen_backend.prepare_corpus import clean_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def normalized(text):
    return " ".join(re.findall(r"\w+", clean_text(text).casefold()))


def evidence_hit(expected, chunks):
    gold = [value for text in expected if (value := normalized(text))]
    candidates = [f" {normalized(chunk['text'])} " for chunk in chunks]
    return any(f" {value} " in candidate for value in gold for candidate in candidates)


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["bge", "qwen"], required=True)
    parser.add_argument("--strategy", help="fixed/section for BGE or A/B for Qwen")
    parser.add_argument("--rerank", action="store_true", help="rerank 24 Qwen candidates to top five")
    parser.add_argument("--answer", action="store_true", help="also generate answers with PaperScope and DeepSeek")
    parser.add_argument("--cases", type=Path, default=HERE / "shared_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strategies = ("fixed", "section") if args.backend == "bge" else ("A", "B")
    args.strategy = args.strategy or ("section" if args.backend == "bge" else "A")
    if args.strategy not in strategies or (args.rerank and args.backend != "qwen"):
        parser.error("The strategy or reranking option does not match the backend.")
    return args


def main():
    args = parse_options()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    paper_ids = {case["paper_id"] for case in cases}
    if not cases or len({case["question_id"] for case in cases}) != len(cases):
        raise ValueError("The evaluation requires a nonempty set of unique questions.")
    if args.answer and not os.getenv("PAPERSCOPE_API_KEY"):
        raise RuntimeError("Set PAPERSCOPE_API_KEY in your local environment for --answer.")

    if args.backend == "bge":
        from paperscope import PaperScope, load_papers

        papers = load_papers(paper_limit=None, paper_ids=paper_ids)
        if papers.keys() != paper_ids:
            raise ValueError(f"Missing QASPER papers: {sorted(paper_ids - papers.keys())}")
        retriever = PaperScope(papers)
        options = {}
    else:
        from qwen_backend.paperscope_adapter import QwenRetriever
        from qwen_backend.retrieve import DEFAULT_HOST, DEFAULT_MODEL

        retriever = QwenRetriever(
            ROOT / os.getenv("PAPERSCOPE_QWEN_DB", "data/chroma"),
            host=os.getenv("PAPERSCOPE_OLLAMA_HOST", DEFAULT_HOST),
            model=os.getenv("PAPERSCOPE_QWEN_MODEL", DEFAULT_MODEL),
        )
        options = {"rerank_n": 24 if args.rerank else 0}

    rows = []
    for case in cases:
        started = perf_counter()
        chunks = retriever.retrieve(case["paper_id"], case["question"], args.strategy, top_k=5, **options)
        elapsed = perf_counter() - started
        if not chunks or any(chunk["paper_id"] != case["paper_id"] for chunk in chunks):
            raise ValueError(f"Missing or out-of-scope evidence for {case['question_id']}. Check the index.")
        row = {
            "question_id": case["question_id"], "paper_id": case["paper_id"],
            "question": case["question"], "reference": case["gold_answer"],
            "hit_at_5": evidence_hit(case["evidence"], chunks),
            "retrieval_seconds": round(elapsed, 4), "retrieved": chunks,
        }
        if args.answer:
            from dataclasses import asdict
            from paperscope import answer_with_api

            row["answer"] = asdict(answer_with_api(case["question"], chunks))
        rows.append(row)
        print(f"{len(rows)}/{len(cases)} {case['question_id']}", flush=True)

    result = {
        "backend": args.backend, "strategy": args.strategy, "rerank": args.rerank,
        "scope": "selected paper", "top_k": 5,
        "hit_definition": "One top-five chunk contains a complete normalized evidence string.",
        "cases": len(rows), "hits": sum(row["hit_at_5"] for row in rows),
        "hit_rate": mean(row["hit_at_5"] for row in rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result['hits']}/{result['cases']} hits; saved to {args.output}")


if __name__ == "__main__":
    main()
