#!/usr/bin/env python3
"""Shared failure taxonomy for failure_taxonomy handoff and C2F outputs.

The taxonomy is descriptive and deterministic. It does not use ground truth
to choose a failure class except for the final success label; branch adherence,
format, parsing, and answer leakage remain separate fields.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def classify_row(row: dict) -> str:
    if row.get("official_correct", row.get("correct", False)):
        return "success"
    completion = str(row.get("full_completion", row.get("completion", "")))
    continuation = str(row.get("continuation", ""))
    if not completion.strip() and not continuation.strip():
        return "empty_completion"
    if row.get("prefix_answer_leakage", row.get("answer_leakage_before_p2", False)):
        return "answer_leakage_before_milestone"
    if row.get("format_valid") is False:
        return "format_failure"
    if row.get("parsed") is False or row.get("parsed_answer") is False:
        return "parse_failure"
    if row.get("branch_switched") is True or row.get("same_branch") is False:
        return "branch_switch"
    if row.get("same_branch") is True:
        return "same_branch_wrong_answer"
    if len(completion.strip()) < 16:
        return "short_or_truncated"
    if re.search(r"(?:timeout|timed out|out of memory|CUDA error)", completion, re.IGNORECASE):
        return "runtime_or_timeout_text"
    return "wrong_or_unclassified"


def classify_rows(rows: Iterable[dict]) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        item["failure_type"] = classify_row(item)
        output.append(item)
    return output


def summarize_failure_rows(rows: Iterable[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = classify_rows(rows)
    if not classified:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(classified)
    group_columns = [column for column in ("stratum", "benchmark", "condition", "target_model", "prefix_milestone") if column in frame.columns]
    group_columns = group_columns or ["failure_type"]
    counts = frame.groupby(group_columns + ["failure_type"], dropna=False).size().reset_index(name="count")
    totals = frame.groupby(group_columns, dropna=False).size().reset_index(name="total")
    table = counts.merge(totals, on=group_columns, how="left")
    table["rate"] = table["count"] / table["total"]
    overall = frame.groupby("failure_type", dropna=False).size().reset_index(name="count")
    overall["rate"] = overall["count"] / len(frame)
    return table, overall


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    table, overall = summarize_failure_rows(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root / "failure_table.csv", index=False)
    overall.to_csv(args.output_root / "failure_overall.csv", index=False)
    (args.output_root / "config.json").write_text(json.dumps({"input": str(args.input), "taxonomy_version": "e8_failure_taxonomy_v1", "ground_truth_used_only_for_success": True}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output_root), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
