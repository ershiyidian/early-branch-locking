#!/usr/bin/env python3
"""Score olmo3_benchmark OLMo base shards with the external official math evaluator.

The generation runner intentionally avoids importing the external evaluator.
This adapter runs in the isolated ``.venv_olmo3_eval`` environment, reconstructs
the evaluator's source examples from each raw record, and writes official
scores plus the repository's trace-diversity diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_EVAL = None
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
evaluate = None
parse_ground_truth = run_execute = PythonExecutor = None

from early_branch_locking.core.math_trace_utils import evaluate_completion, problem_metrics  # noqa: E402


BENCHMARKS = ("gsm8k", "math500", "minerva_math", "olympiadbench", "amc23", "aime24")
KS = (1, 4, 16, 64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluator-root", type=Path, default=None,
                        help="Path to the upstream official math_eval package.")
    return parser.parse_args(argv)


def configure_evaluator(path: Path | None) -> None:
    global evaluate, parse_ground_truth, run_execute, PythonExecutor
    if path is None:
        raise RuntimeError("Pass --evaluator-root pointing to the upstream math_eval package.")
    if not path.is_dir():
        raise FileNotFoundError(f"Official evaluator directory does not exist: {path}")
    sys.path.insert(0, str(path))
    try:
        from evaluate import evaluate as official_evaluate
        from parser import parse_ground_truth as official_parse_ground_truth, run_execute as official_run_execute
        from python_executor import PythonExecutor as official_executor
    except ImportError as exc:
        raise RuntimeError(f"Could not import the official evaluator from {path}: {exc}") from exc
    evaluate = official_evaluate
    parse_ground_truth = official_parse_ground_truth
    run_execute = official_run_execute
    PythonExecutor = official_executor


def load_raw(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line at {line_number} in {path}")
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def build_scoring_sample(row: dict, benchmark: str, executor: PythonExecutor) -> dict:
    source = dict(row.get("source_example") or {})
    source["idx"] = row["idx"]
    source["code"] = list(row["code"])
    source["pred"] = [run_execute(executor, text, "cot", benchmark)[0] for text in source["code"]]
    return source


def official_score(rows: list[dict], benchmark: str) -> tuple[list[dict], dict]:
    if evaluate is None or PythonExecutor is None:
        raise RuntimeError(
            "The upstream official math evaluator is required. Install it or "
            "set EXTERNAL_EVAL to its math_eval directory before scoring."
        )
    os.environ.setdefault("LIMITOFRLVR_MATH_EVAL_WORKERS", "16")
    executor = PythonExecutor(get_answer_from_stdout=True)
    samples = [build_scoring_sample(row, benchmark, executor) for row in rows]
    evaluated, metrics = evaluate(
        data_name=benchmark,
        prompt_type="cot",
        samples=samples,
        execute=True,
    )
    by_idx = {int(row["idx"]): row for row in evaluated}
    merged = []
    for row in rows:
        scored = by_idx[int(row["idx"])]
        merged.append(
            {
                "idx": row["idx"],
                "benchmark": benchmark,
                "question": row["question"],
                "gt": scored.get("gt"),
                "code": row["code"],
                "pred": scored["pred"],
                "score": [bool(value) for value in scored["score"]],
                "finish_reason": row.get("finish_reason", []),
                "completion_tokens": row.get("completion_tokens", []),
            }
        )
    return merged, metrics


def structural_rows(rows: list[dict], benchmark: str) -> list[dict]:
    output = []
    for row in rows:
        # The runner's gt is the fixed source answer string. The official
        # score remains authoritative for correctness; structural metrics use
        # the same completion text and retain their proxy interpretation.
        evals = []
        for text, score in zip(row["code"], row["score"], strict=True):
            parsed = evaluate_completion(text, str(row["gt"]))
            evals.append(replace(parsed, is_correct=bool(score)))
        result = problem_metrics(evals, str(row["gt"]), KS)
        result.update({"benchmark": benchmark, "problem_id": f"{benchmark}_{row['idx']}"})
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_evaluator(args.evaluator_root)
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / f"{args.benchmark}_scored.jsonl"
    metrics_path = args.output_dir / f"{args.benchmark}_official_metrics.json"
    structural_path = args.output_dir / f"{args.benchmark}_structural_per_problem.csv"
    if scored_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {scored_path}")

    rows = load_raw(args.raw)
    os.environ["LIMITOFRLVR_MATH_EVAL_WORKERS"] = str(max(1, args.workers))
    scored, metrics = official_score(rows, args.benchmark)
    structural = structural_rows(scored, args.benchmark)

    with scored_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics["input_raw"] = str(args.raw)
    metrics["scored_output"] = str(scored_path)
    metrics["benchmark"] = args.benchmark
    metrics["n_problems"] = len(scored)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(structural_path, structural)
    print(json.dumps({"scored": str(scored_path), "metrics": str(metrics_path), "structural": str(structural_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
