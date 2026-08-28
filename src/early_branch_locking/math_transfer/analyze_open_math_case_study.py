#!/usr/bin/env python3
"""Build a source-linked case study from one open-math raw trajectory.

The case study deliberately reports exact parser-observed first-calculation
forms rather than claiming that each string is a distinct complete method.
This keeps the problem-level illustration aligned with the paper's
first-calculation entropy proxy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "analysis_results" / "rlvr_passk" / "raw"
OUTPUT_ROOT = ROOT / "data" / "rlvr" / "outputs" / "paper_case_study"
CASE_PROBLEM = "gsm8k_392"
DATASET_PATH = ROOT / "dataset" / "math_eval" / "gsm8k" / "test.jsonl"

RAW_SPECS = (
    (
        "Qwen Base",
        "math_base_7b",
        RAW_ROOT / "expx_trace_diversity_20260503_rlvr_new_metrics"
        / "gsm8k_math_base_7b_official.jsonl",
    ),
    (
        "Qwen SimpleRL",
        "math_simple_rl_7b",
        RAW_ROOT / "expx_trace_diversity_20260503_rlvr_new_metrics"
        / "gsm8k_math_simple_rl_7b_official.jsonl",
    ),
    (
        "OLMo SFT",
        "math_olmo3_sft_7b",
        RAW_ROOT / "expx_trace_diversity_olmo3_20260617"
        / "gsm8k_math_olmo3_sft_7b_official.jsonl",
    ),
    (
        "OLMo DPO",
        "math_olmo3_dpo_7b",
        RAW_ROOT / "expx_trace_diversity_olmo3_20260617"
        / "gsm8k_math_olmo3_dpo_7b_official.jsonl",
    ),
    (
        "OLMo RLVR",
        "math_olmo3_rlvr_7b",
        RAW_ROOT / "expx_trace_diversity_olmo3_20260617"
        / "gsm8k_math_olmo3_rlvr_7b_official.jsonl",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy(labels: list[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if not total:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def load_case_rows(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("problem_id") == CASE_PROBLEM:
                    rows.append(row)
    if len(rows) != 64:
        raise ValueError(f"{path}: expected 64 case rows, found {len(rows)}")
    rows.sort(key=lambda row: int(row["sample_index"]))
    sample_ids = [int(row["sample_index"]) for row in rows]
    if sample_ids != list(range(64)):
        raise ValueError(f"{path}: case sample IDs are not exactly 0..63")
    if len({row.get("ground_truth") for row in rows}) != 1:
        raise ValueError(f"{path}: case ground truth is not stable")
    return rows, {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "rows_in_case": len(rows),
    }


def load_case_question() -> str:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number == 392:
                row = json.loads(line)
                return str(row["question"])
    raise ValueError(f"dataset does not contain index 392: {DATASET_PATH}")


def top_forms(rows: list[dict], limit: int = 5) -> str:
    counts = Counter(str(row["first_calc_branch"]) for row in rows)
    return "; ".join(f"{label}:{count}" for label, count in counts.most_common(limit))


def summarize(label: str, model: str, rows: list[dict], source: dict) -> tuple[dict, list[dict]]:
    all_labels = [str(row["first_calc_branch"]) for row in rows]
    correct_rows = [row for row in rows if bool(row["is_correct"])]
    correct_labels = [str(row["first_calc_branch"]) for row in correct_rows]
    all_counts = Counter(all_labels)
    correct_counts = Counter(correct_labels)
    representative_rows = []
    for branch, _ in all_counts.most_common(5):
        representative = next(row for row in rows if str(row["first_calc_branch"]) == branch)
        representative_rows.append(
            {
                "model_label": model,
                "display_label": label,
                "sample_index": int(representative["sample_index"]),
                "first_calc_branch": branch,
                "is_correct": bool(representative["is_correct"]),
                "answer": representative.get("answer"),
                "completion": representative.get("completion", ""),
            }
        )
    summary = {
        "display_label": label,
        "model_label": model,
        "problem_id": CASE_PROBLEM,
        "ground_truth": rows[0]["ground_truth"],
        "n_samples": len(rows),
        "num_correct": len(correct_rows),
        "pass_at_1": len(correct_rows) / len(rows),
        "pass_at_64": float(any(correct_rows)),
        "all_unique_first_calc_count": len(all_counts),
        "correct_unique_first_calc_count": len(correct_counts),
        "all_first_calc_entropy": entropy(all_labels),
        "correct_first_calc_entropy": entropy(correct_labels),
        "all_top_first_calc": all_counts.most_common(1)[0][0],
        "all_top_first_calc_count": all_counts.most_common(1)[0][1],
        "correct_top_first_calc": correct_counts.most_common(1)[0][0],
        "correct_top_first_calc_count": correct_counts.most_common(1)[0][1],
        "top_forms_all_samples": top_forms(rows),
        "top_forms_correct_samples": top_forms(correct_rows),
        "source_path": source["path"],
        "source_sha256": source["sha256"],
    }
    return summary, representative_rows


def build(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    representatives = []
    sources = []
    question = load_case_question()
    for display_label, model, path in RAW_SPECS:
        rows, source = load_case_rows(path)
        sources.append({"model_label": model, **source})
        summary, examples = summarize(display_label, model, rows, source)
        summaries.append(summary)
        representatives.extend(examples)

    summary_path = output_root / "gsm8k_392_case_summary.csv"
    fieldnames = list(summaries[0])
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    representative_path = output_root / "gsm8k_392_representative_raw_rows.jsonl"
    with representative_path.open("w", encoding="utf-8") as handle:
        for row in representatives:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    metadata = {
        "case_study": "gsm8k_392",
        "dataset": "gsm8k",
        "question": question,
        "interpretation": (
            "first_calc_branch is an exact parser-observed first-calculation form; "
            "it is a breadth proxy and not a claim that strings are distinct complete methods"
        ),
        "source_files": sources,
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "representative_rows": str(representative_path.relative_to(ROOT)),
    }
    metadata_path = output_root / "gsm8k_392_case_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"summary": summaries, "metadata": metadata, "metadata_path": str(metadata_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
