"""Export a completed PaperScope benchmark as a self-contained HTML replay."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TEMPLATE = ROOT / "replay.template.html"
OUTPUT = ROOT / "PaperScope-Replay.html"
DATA_MARKER = "<!-- REPLAY_DATA -->"


def read_result(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def summarize_run(row):
    output = row["output"]
    return {
        "id": row["id"],
        "strategy": row["strategy"],
        "task": row["task"],
        "correct": row["correct"],
        "error": row["error"],
        "evidence": [
            [chunk["id"], chunk["section_id"], chunk["start"], chunk["stop"], chunk["similarity"]]
            for chunk in row["evidence"]
        ],
        "verdict": output.get("verdict", ""),
        "claims": [
            {"text": text, "source": source, "quote": quote}
            for (text, source), quote in zip(
                output.get("lines", []), output.get("citations", []), strict=True
            )
        ],
        "citation_error": output.get("citation_error", False),
        "seconds": [row["retrieval_seconds"], row["generation_seconds"]],
    }


def load_replay(directory, from_run=False):
    suite = read_result(directory, "cases.json")
    metrics = read_result(directory, "metrics.json")
    cases = {case["id"]: case for case in suite["qa"] + suite["claims"]}
    if len(cases) != len(suite["qa"]) + len(suite["claims"]):
        raise ValueError("The benchmark contains duplicate case IDs.")
    results = defaultdict(list)
    seen = set()
    counts = Counter()
    filename = "runs.json" if from_run else "outcomes.json"
    for row in read_result(directory, filename):
        case_id = row["id"]
        key = (case_id, row["strategy"], row["task"])
        if case_id not in cases or key in seen:
            raise ValueError(f"Missing case or duplicate outcome: {key}")
        if from_run and any(row.get(field) != value for field, value in cases[case_id].items()):
            raise ValueError(f"The run and case definition differ: {case_id}")
        seen.add(key)
        counts[f'{row["strategy"]}/{row["task"]}'] += 1

        result = summarize_run(row) if from_run else row
        evidence_ids = {chunk[0] for chunk in result["evidence"]}
        if any(claim["source"] not in evidence_ids for claim in result["claims"]):
            raise ValueError(f"Outcome cites a missing evidence ID: {key}")
        results[case_id].append(result)

    for case_id, case in cases.items():
        tasks = ("direct", "checked") if "expected" in case else ("qa",)
        expected = {(strategy, task) for strategy in ("fixed", "section") for task in tasks}
        actual = {(row["strategy"], row["task"]) for row in results[case_id]}
        if actual != expected:
            raise ValueError(f"Incomplete results for {case_id}: {actual}")

    if counts != {group: row["cases"] for group, row in metrics["metrics"].items()}:
        raise ValueError("Outcome counts do not match the run summary.")
    return {
        "model": metrics["model"],
        "top_k": metrics["top_k"],
        "fingerprint": metrics["fingerprint"],
        "run_count": len(seen),
        "classroom_presets": not from_run,
        "cases": [{**case, "results": results[case_id]} for case_id, case in cases.items()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, metavar="RESULTS_DIR",
                        help="directory with cases.json, runs.json and metrics.json from one run")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="destination HTML file")
    args = parser.parse_args()
    try:
        replay = load_replay(args.from_run or RESULTS, from_run=args.from_run is not None)
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"Cannot build replay: {error}")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(DATA_MARKER) != 1:
        raise ValueError("Expected one replay data marker in the template.")
    # Keep paper text from closing the HTML script element.
    payload = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.output.write_text(template.replace(DATA_MARKER, payload), encoding="utf-8")
    print(f"{args.output.name}: {len(replay['cases'])} cases, {replay['run_count']} recorded outputs")


if __name__ == "__main__":
    main()
