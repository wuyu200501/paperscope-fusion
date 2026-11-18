"""Recompute paired comparisons from the two published benchmark snapshots."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matched(left, right, id_key, fields):
    left_by_id = {row[id_key]: row for row in left}
    right_by_id = {row[id_key]: row for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right) or left_by_id.keys() != right_by_id.keys():
        raise ValueError("The comparison inputs must have exactly the same unique case IDs.")
    for case_id, before in left_by_id.items():
        after = right_by_id[case_id]
        if any(before[field] != after[field] for field in fields):
            raise ValueError(f"Mismatched case data for {case_id}")
        yield before, after


def changes(pairs, key):
    values = Counter((bool(old[key]), bool(new[key])) for old, new in pairs)
    return {
        "before": values[True, True] + values[True, False],
        "after": values[True, True] + values[False, True],
        "gained": values[False, True],
        "lost": values[True, False],
    }


def compare_qwen(saved):
    summaries = []
    for strategy in ("A", "B"):
        for scope in ("paper", "global"):
            key = f"{strategy}_{scope}"
            pairs = list(matched(
                saved[key], saved[f"{key}_rr24"], "question_id", ("paper_id", "question", "qtype"),
            ))
            answerable = [(old, new) for old, new in pairs if old["qtype"] in ("extractive", "abstractive", "boolean")]
            negatives = [(old, new) for old, new in pairs if old["qtype"] in ("unanswerable", "oob")]
            hit = changes(answerable, "hit5")
            refusal = changes(negatives, "refused")
            summaries.append({
                "chunking": {"A": "Structured sections", "B": "Sliding window"}[strategy],
                "scope": scope,
                "answerable": len(answerable),
                "hit_before": hit["before"],
                "hit_after": hit["after"],
                "hits_gained": hit["gained"],
                "hits_lost": hit["lost"],
                "mrr_before": round(mean(old["mrr"] for old, _ in answerable), 3),
                "mrr_after": round(mean(new["mrr"] for _, new in answerable), 3),
                "unanswerable": len(negatives),
                "refused_before": refusal["before"],
                "refused_after": refusal["after"],
                "mean_shared_top5": round(mean(len(set(old["hits"]) & set(new["hits"])) for old, new in answerable), 2),
            })
    return summaries


def compare_paperscope(saved):
    rows = [row for row in saved if row["task"] == "qa" and row["hit"] is not None]
    pairs = list(matched(
        [row for row in rows if row["strategy"] == "fixed"],
        [row for row in rows if row["strategy"] == "section"], "id", (),
    ))
    return {
        "answerable": len(pairs),
        "evidence_hit": changes(pairs, "hit"),
        "strict_answer_match": changes(pairs, "correct"),
        "citation_blocks": {
            "fixed": sum(old["citation_error"] for old, _ in pairs),
            "section": sum(new["citation_error"] for _, new in pairs),
        },
    }


def main():
    results = {
        "method": "Paired reanalysis of archived per-question outputs; no model or API calls.",
        "qwen": compare_qwen(load(HERE / "qwen_frozen.json")),
        "paperscope": compare_paperscope(load(HERE.parent / "results" / "outcomes.json")),
    }
    destination = HERE / "comparison.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
