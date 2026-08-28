#!/usr/bin/env python3
"""Aggregate official olmo3_benchmark benchmark scores after all generation shards finish."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.math_transfer.olmo3_base_evaluation import (
    BENCHMARK_COUNTS,
    EXPERIMENT_ID,
    MODEL_ALIAS,
    validate_raw,
)


BENCHMARKS = tuple(BENCHMARK_COUNTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "rlvr" / "outputs" / "experiments" / "olmo3_full_trajectory_v2")
    parser.add_argument("--n-sampling", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--benchmark-root",
        action="append",
        default=[],
        metavar="BENCHMARK=PATH",
        help="Read this benchmark's raw and score artifacts from PATH instead of --output-root.",
    )
    return parser.parse_args(argv)


def parse_benchmark_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        benchmark, separator, raw_path = value.partition("=")
        if not separator or benchmark not in BENCHMARKS or not raw_path:
            raise ValueError(f"Invalid --benchmark-root {value!r}; expected BENCHMARK=PATH")
        if benchmark in roots:
            raise ValueError(f"Duplicate --benchmark-root for {benchmark}")
        roots[benchmark] = Path(raw_path).resolve()
    return roots


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def latest_raw(benchmark_dir: Path) -> Path | None:
    candidates = sorted(benchmark_dir.glob("records_s*_e*.jsonl"))
    return candidates[-1] if candidates else None


def manifest_path_for_raw(raw: Path) -> Path:
    return raw.with_suffix(".manifest.json")


def load_complete_manifest(raw: Path, benchmark: str) -> tuple[Path, dict]:
    manifest_path = manifest_path_for_raw(raw)
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest={manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {manifest_path}")
    if payload.get("status") != "complete":
        raise ValueError(f"manifest status={payload.get('status')!r}, expected complete")
    if payload.get("benchmark") != benchmark:
        raise ValueError(f"manifest benchmark={payload.get('benchmark')!r}, expected {benchmark!r}")
    recorded_sha = payload.get("raw_sha256")
    if recorded_sha and recorded_sha != sha256_file(raw):
        raise ValueError(f"manifest raw_sha256 mismatch for {raw}")
    return manifest_path, payload


def score_paths(benchmark_dir: Path, benchmark: str) -> tuple[Path | None, Path | None, Path | None]:
    for score_dir in (benchmark_dir / "score", benchmark_dir / "scored"):
        metrics = score_dir / f"{benchmark}_official_metrics.json"
        scored = score_dir / f"{benchmark}_scored.jsonl"
        structural = score_dir / f"{benchmark}_structural_per_problem.csv"
        if metrics.is_file() and scored.is_file():
            return metrics, scored, structural if structural.is_file() else None
    return None, None, None


def mean_csv(path: Path | None, fields: tuple[str, ...]) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        if values:
            output[field] = sum(values) / len(values)
    return output


def aggregate(args: argparse.Namespace) -> dict:
    args.output_root.mkdir(parents=True, exist_ok=True)
    benchmark_roots = parse_benchmark_roots(args.benchmark_root)
    rows = []
    inputs = []
    missing = []
    for benchmark in BENCHMARKS:
        benchmark_dir = benchmark_roots.get(benchmark, args.output_root) / benchmark
        raw = latest_raw(benchmark_dir)
        metrics_path, scored_path, structural_path = score_paths(benchmark_dir, benchmark)
        if raw is None:
            missing.append(f"{benchmark}:raw")
            continue
        try:
            manifest_path, raw_manifest = load_complete_manifest(raw, benchmark)
            validation = validate_raw(raw, expected_problems=BENCHMARK_COUNTS[benchmark], expected_samples=args.n_sampling)
        except Exception as error:
            missing.append(f"{benchmark}:raw_invalid:{error}")
            continue
        if metrics_path is None or scored_path is None:
            missing.append(f"{benchmark}:official_score")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        structural = mean_csv(
            structural_path,
            ("correct_rate", "parse_rate", "first_calc_branch_entropy", "numeric_trace_entropy", "observed_trace_coverage"),
        )
        row = {
            "model": MODEL_ALIAS,
            "benchmark": benchmark,
            "n_problems": validation["problem_count"],
            "n_samples": args.n_sampling,
            "raw_rows": validation["raw_rows"],
            "official_acc_pct": metrics.get("acc"),
            "official_pass_acc_pct": metrics.get("pass_acc"),
            "official_pass_at_1_pct": metrics.get("pass@k", {}).get("1"),
            "official_pass_at_64_pct": metrics.get("pass@k", {}).get("64"),
            "seed": raw_manifest.get("requested", {}).get("seed"),
            "min_new_tokens": raw_manifest.get("requested", {}).get("min_new_tokens"),
            "raw_manifest": str(manifest_path),
            **structural,
        }
        rows.append(row)
        inputs.extend(
            [
                {"path": str(raw), "sha256": sha256_file(raw), "rows": validation["raw_rows"]},
                {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
                {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
                {"path": str(scored_path), "sha256": sha256_file(scored_path)},
            ]
        )

    status = "complete" if not missing and len(rows) == len(BENCHMARKS) else "partial"
    if status != "complete" and not args.allow_partial:
        raise RuntimeError("olmo3_benchmark matrix is not complete: " + ", ".join(missing))
    per_benchmark = args.output_root / "per_benchmark.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with per_benchmark.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "git_commit": git_commit(),
        "status": status,
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL_ALIAS,
        "benchmarks": list(BENCHMARKS),
        "benchmark_counts": BENCHMARK_COUNTS,
        "n_sampling": args.n_sampling,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 16000,
        "seed": None,
        "seed_by_benchmark": {row["benchmark"]: row.get("seed") for row in rows},
        "prompt_type": "cot",
        "apply_chat_template": False,
        "backend": "transformers_hf_per_sequence_stop",
        "benchmark_roots": {benchmark: str(path) for benchmark, path in sorted(benchmark_roots.items())},
        "missing": missing,
        "rows": rows,
        "inputs": inputs,
        "per_benchmark": str(per_benchmark),
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "per_benchmark": str(per_benchmark), "manifest": str(manifest_path), "missing": missing}, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    aggregate(parse_args())
