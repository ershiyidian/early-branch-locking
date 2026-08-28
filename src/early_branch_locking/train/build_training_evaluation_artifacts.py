#!/usr/bin/env python
"""Aggregate and audit the entrance_entropy_training local Countdown control endpoint on CPU.

The endpoint sampler remains ``rollout_collection_collect.py``.  This module deliberately
does not load a model or initialize CUDA: it consumes the sampler's JSONL
rows, derives the semantic natural entrance from the generated continuation,
and writes the entrance_entropy_training curve, problem ledger, family ledger, summaries, and
Good-Turing missing-mass audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import METRICS_DIR, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    enumerate_solution_set,
    pass_at_k,
)
from early_branch_locking.core.entrance_detection import (  # noqa: E402
    find_first_reasoning_entrance,
)


PROTOCOL_VERSION = "s3-local-countdown-endpoint-v1"
DEFAULT_TAG = "entrance_entropy_training_v1"
DEFAULT_PROBLEMS = 150
DEFAULT_SAMPLES = 320
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_SEED = 1729

STEP_RE = re.compile(r"\bstep:(?P<step>\d+)\s+-\s+(?P<body>.*)$")
SCALAR_RE = re.compile(
    r"(?P<key>[A-Za-z][\w./-]*):"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
STEP_NUMBER_RE = re.compile(r"(?:global_step|step)[_-]?(\d+)", re.IGNORECASE)

SUMMARY_METRICS = (
    "pass@1",
    "native_format_rate",
    "natural_entrance_parse_rate",
    "exact_coverage",
    "operator_class_coverage",
    "answer_entrance_family_coverage",
    "natural_entrance_family_coverage",
    "natural_entrance_family_entropy",
    "entrance_family_entropy",
    "feasible_natural_family_entropy",
    "zero_natural_access_family_count",
    "natural_family_fraction_below_0p01",
    "natural_family_fraction_below_0p05",
    "natural_family_gini",
    "good_turing_unseen_solution_mass",
    "good_turing_unseen_solution_mass_valid_conditional",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--raw-path", action="append", type=Path, required=True)
    parser.add_argument("--training-log", action="append", type=Path, default=[])
    parser.add_argument("--protocol-manifest", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=TEST_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--num-problems", type=int, default=DEFAULT_PROBLEMS)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int_list(value: Any, field: str) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not a JSON list: {value!r}") from exc
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{field} must be a list, got {type(value).__name__}")
    return [int(item) for item in value]


def _step_from_label(value: Any) -> int | None:
    match = STEP_NUMBER_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def _model_label(row: dict[str, Any], path: Path, override: str | None) -> str:
    if override:
        return str(override)
    for key in ("model_label", "checkpoint", "model"):
        value = row.get(key)
        if value:
            return Path(str(value).rstrip("/")).name
    return path.stem


def _problem_id(row: dict[str, Any]) -> str:
    for key in ("problem_index", "problem_id", "sample_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError("raw row has no problem_index, problem_id, or sample_id")


def _sample_id(row: dict[str, Any], fallback: int) -> str:
    value = row.get("sample_index", row.get("sample_id", fallback))
    return str(value)


def _natural_family(completion: Any) -> str | None:
    match = find_first_reasoning_entrance(str(completion or ""))
    return match.family if match.found else None


def _expression_family(expression: Any) -> str | None:
    if expression is None or not str(expression).strip():
        return None
    # A canonical solution is not a generated continuation.  Disable the
    # response stop markers so the same semantic detector can read its whole
    # arithmetic expression.
    match = find_first_reasoning_entrance(str(expression), stop_markers=())
    return match.family if match.found else None


def _entropy(counts: Counter[str] | dict[str, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    return float(
        -sum(
            (int(value) / total) * math.log(int(value) / total)
            for value in counts.values()
            if int(value) > 0
        )
    )


def _gini(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = np.maximum(array[np.isfinite(array)], 0.0)
    if not len(array) or float(array.sum()) <= 0.0:
        return 0.0
    array.sort()
    count = len(array)
    weights = np.arange(1, count + 1, dtype=float)
    return float(np.sum((2.0 * weights - count - 1.0) * array) / (count * array.sum()))


def _bootstrap(values: Iterable[float], draws: int, seed: int) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".partial" or path.name.endswith(".partial"):
        raise ValueError(f"refusing partial raw input: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"raw row at {path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"raw input is empty: {path}")
    return rows


def _prepare_rows(
    paths: Sequence[Path],
    *,
    expected_problems: int,
    expected_samples: int,
    model_override: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    input_counts: dict[str, dict[str, Any]] = {}
    for path in paths:
        source_rows = _load_jsonl(path)
        input_counts[str(path)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(source_rows),
        }
        for fallback, source in enumerate(source_rows):
            model = _model_label(source, path, model_override)
            problem_id = _problem_id(source)
            sample_id = _sample_id(source, fallback)
            numbers = _as_int_list(source.get("numbers"), "numbers")
            if source.get("target") is None:
                raise ValueError(f"raw row has no target: {path} problem={problem_id}")
            completion = str(
                source.get(
                    "completion",
                    source.get("response", source.get("generated_text", "")),
                )
                or ""
            )
            if "overall_ok" not in source:
                raise ValueError(
                    f"raw row has no overall_ok; entrance_entropy_training requires rollout_collection local scores: "
                    f"{path} problem={problem_id} sample={sample_id}"
                )
            canonical = source.get("canonical_expr")
            natural_family = _natural_family(completion)
            answer_family = _expression_family(canonical)
            label = str(source.get("checkpoint", model))
            rows.append(
                {
                    "model_label": model,
                    "checkpoint_step": _step_from_label(source.get("checkpoint", label)),
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "numbers": numbers,
                    "target": int(source["target"]),
                    "completion": completion,
                    "overall_ok": _as_bool(source.get("overall_ok")),
                    "native_format_ok": _as_bool(
                        source.get(
                            "native_format_ok",
                            _as_bool(source.get("has_feasible_tag"))
                            and _as_bool(source.get("has_answer_tag")),
                        )
                    ),
                    "canonical_expr": str(canonical) if canonical else None,
                    "natural_family": natural_family,
                    "answer_family": answer_family,
                    "source_path": str(path),
                }
            )

    if not rows:
        raise ValueError("no raw rows were loaded")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_samples: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["model_label"], row["problem_id"])
        sample_key = (*key, row["sample_id"])
        if sample_key in seen_samples:
            raise ValueError(f"duplicate model/problem/sample row: {sample_key}")
        seen_samples.add(sample_key)
        groups[key].append(row)

    models = sorted({key[0] for key in groups})
    if expected_problems > 0:
        for model in models:
            model_groups = [key for key in groups if key[0] == model]
            if len(model_groups) != expected_problems:
                raise ValueError(
                    f"{model} has {len(model_groups)} problems; expected {expected_problems}"
                )
    for key, group in groups.items():
        if expected_samples > 0 and len(group) != expected_samples:
            raise ValueError(
                f"{key[0]} problem {key[1]} has {len(group)} samples; "
                f"expected {expected_samples}"
            )
        first = group[0]
        if any(row["numbers"] != first["numbers"] or row["target"] != first["target"] for row in group):
            raise ValueError(f"problem metadata changes within group {key}")
    return rows, input_counts


def _problem_records(
    rows: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["problem_id"])].append(row)

    problem_records: list[dict[str, Any]] = []
    family_records: list[dict[str, Any]] = []
    good_turing_records: list[dict[str, Any]] = []
    for (model, problem_id), group in sorted(grouped.items()):
        first = group[0]
        numbers = list(first["numbers"])
        target = int(first["target"])
        solution_set = set(enumerate_solution_set(numbers, target))
        solution_family_counts = Counter(
            family
            for expression in solution_set
            if (family := _expression_family(expression)) is not None
        )
        feasible_families = set(solution_family_counts)
        feasible_operator_classes = {family[-1] for family in feasible_families}

        natural_counts = Counter(
            row["natural_family"] for row in group if row["natural_family"] is not None
        )
        answer_counts = Counter(
            row["answer_family"] for row in group if row["answer_family"] is not None
        )
        valid_rows = [
            row
            for row in group
            if row["overall_ok"] and row["canonical_expr"] in solution_set
        ]
        valid_solution_counts = Counter(
            row["canonical_expr"] for row in valid_rows if row["canonical_expr"]
        )
        observed_solutions = set(valid_solution_counts)
        valid_answer_families = {
            family
            for row in valid_rows
            if (family := row["answer_family"]) is not None
        }
        natural_families = set(natural_counts)
        observed_natural_feasible = natural_families & feasible_families
        observed_answer_feasible = valid_answer_families & feasible_families
        natural_feasible_counts = {
            family: count
            for family, count in natural_counts.items()
            if family in feasible_families
        }
        total_rows = len(group)
        feasible_access_rates = [
            natural_counts.get(family, 0) / total_rows
            for family in feasible_families
        ] if total_rows else []

        solution_counts = Counter(valid_solution_counts)
        singleton_count = sum(1 for count in solution_counts.values() if count == 1)
        valid_count = len(valid_rows)
        gt_mass = singleton_count / total_rows if total_rows else float("nan")
        gt_valid_mass = singleton_count / valid_count if valid_count else float("nan")
        exact_coverage = (
            len(observed_solutions) / len(solution_set) if solution_set else float("nan")
        )
        family_coverage = (
            len(observed_natural_feasible) / len(feasible_families)
            if feasible_families
            else float("nan")
        )
        answer_coverage = (
            len(observed_answer_feasible) / len(feasible_families)
            if feasible_families
            else float("nan")
        )
        operator_coverage = (
            len({family[-1] for family in observed_answer_feasible} & feasible_operator_classes)
            / len(feasible_operator_classes)
            if feasible_operator_classes
            else float("nan")
        )
        row_record: dict[str, Any] = {
            "model_label": model,
            "checkpoint_step": first["checkpoint_step"],
            "problem_id": problem_id,
            "numbers": json.dumps(numbers, separators=(",", ":")),
            "target": target,
            "n_samples": total_rows,
            "correct_count": valid_count,
            "pass@1": pass_at_k(total_rows, valid_count, 1),
            "pass_at_1": pass_at_k(total_rows, valid_count, 1),
            "native_format_rate": float(np.mean([row["native_format_ok"] for row in group])),
            "natural_entrance_parse_rate": (
                sum(natural_counts.values()) / total_rows if total_rows else float("nan")
            ),
            "natural_entrance_count": int(sum(natural_counts.values())),
            "natural_family_count": len(natural_families),
            "solution_count": len(solution_set),
            "observed_solution_count": len(observed_solutions),
            "exact_coverage": exact_coverage,
            "feasible_family_count": len(feasible_families),
            "natural_entrance_family_coverage": family_coverage,
            "answer_entrance_family_coverage": answer_coverage,
            "operator_class_coverage": operator_coverage,
            "natural_entrance_family_entropy": _entropy(natural_counts),
            "entrance_family_entropy": _entropy(natural_feasible_counts),
            "feasible_natural_family_entropy": _entropy(natural_feasible_counts),
            "zero_natural_access_family_count": len(feasible_families - natural_families),
            "natural_family_fraction_below_0p01": (
                float(np.mean(np.asarray(feasible_access_rates) < 0.01))
                if feasible_access_rates else float("nan")
            ),
            "natural_family_fraction_below_0p05": (
                float(np.mean(np.asarray(feasible_access_rates) < 0.05))
                if feasible_access_rates else float("nan")
            ),
            "natural_family_gini": _gini(feasible_access_rates),
            "unresolved_solution_family_count": len(solution_set) - sum(solution_family_counts.values()),
            "valid_answer_family_count": len(valid_answer_families),
            "good_turing_singleton_solution_count": singleton_count,
            "good_turing_valid_draw_count": valid_count,
            "good_turing_unseen_solution_mass": gt_mass,
            "good_turing_unseen_solution_mass_valid_conditional": gt_valid_mass,
        }
        problem_records.append(row_record)
        good_turing_records.append(
            {
                "model_label": model,
                "checkpoint_step": first["checkpoint_step"],
                "problem_id": problem_id,
                "n_samples": total_rows,
                "valid_draw_count": valid_count,
                "observed_solution_count": len(observed_solutions),
                "singleton_solution_count": singleton_count,
                "good_turing_unseen_solution_mass": gt_mass,
                "good_turing_unseen_solution_mass_valid_conditional": gt_valid_mass,
            }
        )

        family_union = feasible_families | natural_families | valid_answer_families
        for family in sorted(family_union):
            operator = family[-1] if family else None
            natural_count = int(natural_counts.get(family, 0))
            answer_count = int(answer_counts.get(family, 0))
            valid_family_count = int(
                sum(1 for row in valid_rows if row["answer_family"] == family)
            )
            family_records.append(
                {
                    "model_label": model,
                    "checkpoint_step": first["checkpoint_step"],
                    "problem_id": problem_id,
                    "family": family,
                    "operator": operator,
                    "is_feasible_family": family in feasible_families,
                    "solver_solution_count": int(solution_family_counts.get(family, 0)),
                    "natural_count": natural_count,
                    "natural_rate": natural_count / total_rows if total_rows else float("nan"),
                    "answer_count": answer_count,
                    "valid_solution_count": valid_family_count,
                    "zero_natural_access": family in feasible_families and natural_count == 0,
                }
            )

    return (
        pd.DataFrame(problem_records),
        pd.DataFrame(family_records),
        pd.DataFrame(good_turing_records),
    )


def _summary_frame(per_problem: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in per_problem.groupby("model_label", sort=True):
        first = frame.iloc[0]
        record: dict[str, Any] = {
            "model_label": model,
            "checkpoint_step": first["checkpoint_step"],
            "n_problems": int(len(frame)),
            "n_samples_per_problem": int(frame["n_samples"].mode().iloc[0]),
        }
        for index, metric in enumerate(SUMMARY_METRICS):
            mean, lo, hi = _bootstrap(frame[metric].to_numpy(float), draws, seed + index * 101)
            record[metric] = mean
            record[f"{metric}_ci_lo"] = lo
            record[f"{metric}_ci_hi"] = hi
        record["pass_at_1"] = record["pass@1"]
        record["pass_at_1_ci_lo"] = record["pass@1_ci_lo"]
        record["pass_at_1_ci_hi"] = record["pass@1_ci_hi"]
        rows.append(record)
    return pd.DataFrame(rows)


def _parse_training_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = STEP_RE.search(line)
            if match is None:
                continue
            record: dict[str, Any] = {
                "step": int(match.group("step")),
                "source_log": str(path),
                "source_line": line_number,
            }
            for scalar in SCALAR_RE.finditer(match.group("body")):
                record[scalar.group("key")] = float(scalar.group("value"))
            records.append(record)
    return records


def parse_training_curve(paths: Sequence[Path], tag: str) -> pd.DataFrame:
    all_records: list[dict[str, Any]] = []
    for path in paths:
        all_records.extend(_parse_training_log(path))
    if not all_records:
        return pd.DataFrame(columns=["step", "source_log", "source_line"])
    frame = pd.DataFrame(all_records)
    frame.insert(0, "tag", tag)
    return frame.sort_values(["step", "source_log", "source_line"]).reset_index(drop=True)


def _environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as metadata

        for package in ("numpy", "pandas", "pyarrow"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "unavailable"
    except Exception:
        versions = {}
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "package_versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_commit(),
    }


def _protocol(args: argparse.Namespace, manifest_path: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_scope": "local Countdown test set; public_grpo_replication public/native estimand excluded",
        "num_problems": args.num_problems,
        "samples_per_problem": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "pass_metric": "problem-level sampled pass@1 = c/n",
        "exact_coverage": "unique valid canonical local solutions / solver-enumerated solution set",
        "operator_coverage": "valid answer entrance operators / solver-feasible entrance operators",
        "natural_entrance": "first integer operand plus arithmetic operator in generated continuation",
        "natural_family_coverage": "detected natural families intersected with solver-feasible families / solver-feasible families",
        "natural_entropy": "Shannon entropy over detected natural entrance-family counts",
        "entrance_family_entropy": "Shannon entropy over natural counts restricted to solver-feasible families",
        "zero_natural_access": "solver-feasible families with zero detected natural samples",
        "family_menu_concentration": "feasible-family natural access fractions below 0.01/0.05 and their Gini coefficient",
        "good_turing": "singleton valid-solution species divided by all endpoint draws; conditional-valid variant also reported",
        "statistical_unit": "problem",
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed, "confidence": 0.95},
    }
    if manifest_path is not None:
        payload["training_manifest_path"] = str(manifest_path)
        payload["training_manifest_sha256"] = _sha256(manifest_path)
        try:
            source = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["training_manifest_protocol_version"] = source.get("protocol_version")
            payload["training_manifest_tag"] = source.get("tag")
        except (OSError, json.JSONDecodeError):
            payload["training_manifest_read_status"] = "unreadable"
    return payload


def _write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.partial.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_parquet(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.partial.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _write_json(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.partial.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _default_training_logs(tag: str) -> list[Path]:
    candidates = sorted((ROOT / "logs" / "s3").glob(f"*{tag}*control*.log"))
    return [path for path in candidates if path.is_file()]


def aggregate(args: argparse.Namespace) -> dict[str, str]:
    if args.num_problems < 0 or args.n_samples < 0:
        raise ValueError("problem and sample counts must be non-negative")
    raw_paths = [path.resolve() for path in args.raw_path]
    rows, input_counts = _prepare_rows(
        raw_paths,
        expected_problems=args.num_problems,
        expected_samples=args.n_samples,
        model_override=args.model_label,
    )
    per_problem, family_access, good_turing = _problem_records(rows)
    summary = _summary_frame(per_problem, args.bootstrap_draws, args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_curve": output_dir / f"entrance_entropy_training_training_curve_{args.tag}.csv",
        "endpoint_per_problem": output_dir / f"entrance_entropy_training_endpoint_per_problem_{args.tag}.parquet",
        "endpoint_summary": output_dir / f"entrance_entropy_training_endpoint_summary_{args.tag}.csv",
        "family_access": output_dir / f"entrance_entropy_training_family_access_{args.tag}.parquet",
        "good_turing": output_dir / f"entrance_entropy_training_good_turing_unseen_mass_{args.tag}.csv",
        "manifest": output_dir / f"entrance_entropy_training_artifact_manifest_{args.tag}.json",
    }
    logs = [path.resolve() for path in args.training_log] or _default_training_logs(args.tag)
    curve = parse_training_curve(logs, args.tag)
    _write_csv(curve, paths["training_curve"], args.overwrite)
    _write_parquet(per_problem, paths["endpoint_per_problem"], args.overwrite)
    _write_csv(summary, paths["endpoint_summary"], args.overwrite)
    _write_parquet(family_access, paths["family_access"], args.overwrite)
    _write_csv(good_turing, paths["good_turing"], args.overwrite)

    protocol_manifest = args.protocol_manifest
    if protocol_manifest is None:
        candidate = METRICS_DIR / f"entrance_entropy_training_manifest_{args.tag}.json"
        protocol_manifest = candidate if candidate.is_file() else None
    input_hashes = dict(input_counts)
    for path in logs:
        input_hashes[str(path)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": int(len(curve[curve["source_log"] == str(path)])) if not curve.empty else 0,
        }
    if args.dataset_path.is_file():
        input_hashes[str(args.dataset_path.resolve())] = {
            "path": str(args.dataset_path.resolve()),
            "sha256": _sha256(args.dataset_path),
            "bytes": args.dataset_path.stat().st_size,
            "rows": None,
        }
    if protocol_manifest is not None and protocol_manifest.is_file():
        input_hashes[str(protocol_manifest.resolve())] = {
            "path": str(protocol_manifest.resolve()),
            "sha256": _sha256(protocol_manifest),
            "bytes": protocol_manifest.stat().st_size,
            "rows": None,
        }

    manifest = {
        "experiment_id": "entrance_entropy_training",
        "artifact": "cpu_endpoint_aggregation",
        "protocol_version": PROTOCOL_VERSION,
        "tag": args.tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hashes": input_hashes,
        "protocol": _protocol(args, protocol_manifest),
        "environment": _environment(),
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "confidence": 0.95,
            "statistical_unit": "problem",
        },
        "counts": {
            "raw_rows": len(rows),
            "problem_rows": int(len(per_problem)),
            "family_rows": int(len(family_access)),
            "good_turing_rows": int(len(good_turing)),
            "training_curve_rows": int(len(curve)),
            "models": sorted(per_problem["model_label"].unique().tolist()),
        },
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "training_log_paths": [str(path) for path in logs],
        "paper_status": "S3_endpoint_artifacts_non_paper_until_gate",
    }
    _write_json(manifest, paths["manifest"], args.overwrite)
    return {key: str(value) for key, value in paths.items()}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = aggregate(args)
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
