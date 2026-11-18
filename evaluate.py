"""Reproducible evidence retrieval, QA and sentence-checking evaluation."""

import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from statistics import mean, median
from time import perf_counter

from claim_cases import PAIRS
from paperscope import (
    DEEPSEEK_MODEL, ROOT, STRATEGIES, PaperScope, answer_with_api, load_papers,
    make_api_client, qasper_corpus,
)

RESULTS = ROOT / "results"
TOP_K = 5
ANSWERABLE = 20


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(f".{os.getpid()}.tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def normalized(text):
    return " ".join(text.casefold().split())


def answer_tokens(text):
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.casefold())
    return re.findall(r"\w+", text)


def matches_reference(text, reference):
    tokens = answer_tokens(text)
    expected = answer_tokens(reference)
    return bool(expected) and any(
        tokens[index:index + len(expected)] == expected for index in range(len(tokens))
    )


def paragraph_locations(paper):
    for section_id, (_, paragraphs) in enumerate(paper["sections"], 1):
        offset = 0
        for paragraph in paragraphs:
            yield section_id, offset, paragraph
            offset += len(paragraph) + 2


def build_qa_cases(papers, corpus):
    cases = []
    for paper_id, paper in papers.items():
        paragraphs = list(paragraph_locations(paper))
        for entry in corpus[paper_id]["qas"]:
            annotations = [item["answer"] for item in entry["answers"]]
            if not annotations or any(item["unanswerable"] for item in annotations):
                continue
            references = []
            gold = set()
            for item in annotations:
                spans = item["extractive_spans"]
                if len(spans) != 1 or not 2 <= len(spans[0].strip()) <= 140:
                    continue
                reference = spans[0].strip()
                if re.search(r"BIBREF|TABREF|FIGREF|FLOAT SELECTED", reference, re.I):
                    continue
                selected = {normalized(text) for text in item["evidence"] if isinstance(text, str)}
                locations = [
                    (section_id, offset, offset + len(paragraph))
                    for section_id, offset, paragraph in paragraphs
                    if normalized(paragraph) in selected and normalized(reference) in normalized(paragraph)
                ]
                if locations:
                    gold.update(locations)
                    if reference not in references:
                        references.append(reference)
            if not references:
                continue
            cases.append({
                "id": f"qa-{len(cases) + 1:02}", "paper_id": paper_id,
                "input": entry["question"], "references": references,
                "gold_paragraphs": sorted(gold), "qasper_question_id": entry["question_id"],
            })
            if len(cases) == ANSWERABLE:
                break
        if len(cases) == ANSWERABLE:
            break
    if len(cases) != ANSWERABLE:
        raise RuntimeError(f"Expected {ANSWERABLE} eligible QASPER questions, found {len(cases)}.")
    negative = [
        ("1810.08699", "What is the boiling point of liquid nitrogen at one atmosphere?"),
        ("1609.00425", "What is the orbital period of Neptune in Earth years?"),
        ("1909.09067", "What is the melting point of pure tungsten?"),
        ("1704.06194", "What is the mass of a proton in kilograms?"),
    ]
    for index, (paper_id, question) in enumerate(negative, 1):
        cases.append({
            "id": f"unanswerable-{index:02}", "paper_id": paper_id,
            "input": question, "references": [], "gold_paragraphs": [],
        })
    return cases


def build_claim_cases(papers):
    cases = []
    for index, (paper_id, anchor, positive, modified, verdict) in enumerate(PAIRS, 1):
        for section_id, offset, paragraph in paragraph_locations(papers[paper_id]):
            if anchor in paragraph:
                start = offset + paragraph.index(anchor)
                gold = [(section_id, start, start + len(anchor))]
                break
        else:
            raise ValueError(f"Claim pair {index} has no exact source anchor.")
        for suffix, statement, label in (
            ("a", positive, "SUPPORTED"), ("b", modified, verdict)
        ):
            cases.append({
                "id": f"claim-{index:02}-{suffix}", "paper_id": paper_id,
                "input": statement, "expected": label, "gold_anchor": gold,
                "source_excerpt": anchor,
            })
    return cases


def evidence_coverage(targets, evidence):
    covered = 0
    for section_id, start, stop in targets:
        # Union the retrieved ranges so overlapping chunks do not inflate coverage.
        ranges = sorted(
            (max(start, item["start"]), min(stop, item["stop"]))
            for item in evidence if item["section_id"] == section_id
            and item["start"] < stop and item["stop"] > start
        )
        endpoint, overlap = start, 0
        for left, right in ranges:
            overlap += max(0, right - max(left, endpoint))
            endpoint = max(endpoint, right)
        covered += overlap >= (stop - start) / 2
    return covered / len(targets) if targets else None


def request_case(client, case, strategy, task, evidence, retrieve_time):
    started = perf_counter()
    try:
        answer = answer_with_api(case["input"], evidence, task=task, client=client)
        output = asdict(answer)
        error = ""
    except Exception as exc:
        output = {}
        error = type(exc).__name__
    duration = perf_counter() - started
    row = {
        **case, "strategy": strategy, "task": task, "evidence": evidence,
        "retrieval_seconds": retrieve_time, "generation_seconds": round(duration, 4),
        "output": output, "error": error,
    }
    if task == "qa":
        coverage = evidence_coverage(case["gold_paragraphs"], evidence)
        row["evidence_recall"] = coverage
        row["evidence_hit"] = coverage > 0 if coverage is not None else None
        lines = output.get("lines", [])
        all_text = " ".join(line[0] for line in lines)
        row["correct"] = not error and not output.get("citation_error", False) and (
            any(matches_reference(all_text, ref) for ref in case["references"])
            if case["references"] else not lines
        )
    else:
        row["correct"] = (
            not error and not output.get("citation_error", False)
            and output.get("verdict") == case["expected"]
        )
    return row


def summarize(rows, fingerprint):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['strategy']}/{row['task']}"].append(row)
    metrics = {}
    for label, data in sorted(groups.items()):
        qa = data[0]["task"] == "qa"
        output_ok = [row for row in data if not row["error"]]
        entries = {
            "cases": len(data), "api_errors": len(data) - len(output_ok),
            "citation_validation_errors": sum(row["output"].get("citation_error", False) for row in data),
            "accuracy": mean(row["correct"] for row in data),
            "median_seconds": median(row["retrieval_seconds"] + row["generation_seconds"] for row in data),
            "prompt_tokens": sum(row["output"].get("prompt_tokens", 0) for row in data),
            "completion_tokens": sum(row["output"].get("completion_tokens", 0) for row in data),
        }
        if qa:
            answerable = [row for row in data if row["references"]]
            unanswerable = [row for row in data if not row["references"]]
            entries.update({
                "hit_at_5": mean(row["evidence_hit"] for row in answerable),
                "recall_at_5": mean(row["evidence_recall"] for row in answerable),
                "answerable_span_accuracy": mean(row["correct"] for row in answerable),
                "correct_refusal_rate": mean(row["correct"] for row in unanswerable),
                "answered_cases": sum(bool(row["output"].get("lines")) for row in data),
            })
        else:
            positive = [row for row in data if row["expected"] == "SUPPORTED"]
            negative = [row for row in data if row["expected"] != "SUPPORTED"]
            entries.update({
                "preservation_rate": mean(row["correct"] for row in positive),
                "false_acceptance_rate": mean(
                    row["output"].get("verdict") == "SUPPORTED" for row in negative
                ),
            })
        metrics[label] = entries
    return {"model": os.getenv("PAPERSCOPE_MODEL", DEEPSEEK_MODEL), "top_k": TOP_K,
            "fingerprint": fingerprint, "metrics": metrics}


def run():
    start = perf_counter()
    papers = load_papers()
    qa = build_qa_cases(papers, qasper_corpus())
    claims = build_claim_cases(papers)
    cases = {"qa": qa, "claims": claims}
    save_json(RESULTS / "cases.json", cases)
    config = json.dumps([cases, os.getenv("PAPERSCOPE_MODEL", DEEPSEEK_MODEL), TOP_K], sort_keys=True)
    digest = hashlib.sha256(config.encode("utf-8"))
    for name in ("paperscope.py", "evaluate.py"):
        digest.update((ROOT / name).read_bytes())
    fingerprint = digest.hexdigest()[:16]
    checkpoint_file = ROOT / ".cache" / f"runs-{fingerprint}.json"
    rows = json.loads(checkpoint_file.read_text(encoding="utf-8")) if checkpoint_file.exists() else {}
    rows = {key: row for key, row in rows.items() if not row["error"]}
    print(f"Frozen: {len(qa)} QA cases, {len(claims)} claim cases, {len(papers)} papers.", flush=True)
    scope = PaperScope()
    with make_api_client() as client, ThreadPoolExecutor(max_workers=5) as pool:
        futures = {}
        for strategy in STRATEGIES:
            for case in qa + claims:
                tasks = ("qa",) if case in qa else ("direct", "checked")
                todo = [task for task in tasks if f'{strategy}/{task}/{case["id"]}' not in rows]
                if not todo:
                    continue
                retrieval_start = perf_counter()
                evidence = scope.retrieve(case["paper_id"], case["input"], strategy, TOP_K)
                retrieval_time = round(perf_counter() - retrieval_start, 4)
                for task in todo:
                    key = f'{strategy}/{task}/{case["id"]}'
                    future = pool.submit(request_case, client, case, strategy, task, evidence, retrieval_time)
                    futures[future] = key
        for index, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            rows[key] = future.result()
            if index % 10 == 0 or index == len(futures):
                save_json(checkpoint_file, rows)
                print(f"Completed {index}/{len(futures)} new cases.", flush=True)
    ordered = [rows[key] for key in sorted(rows)]
    save_json(RESULTS / "runs.json", ordered)
    summary = summarize(ordered, fingerprint)
    save_json(RESULTS / "metrics.json", summary)
    print(f"Saved {len(ordered)} results in {perf_counter() - start:.1f}s.", flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--prepare"]:
        papers = load_papers()
        qa = build_qa_cases(papers, qasper_corpus())
        claims = build_claim_cases(papers)
        save_json(RESULTS / "cases.json", {"qa": qa, "claims": claims})
        print(f"{len(qa)} QA cases and {len(claims)} claims verified against the corpus.")
    else:
        run()
