#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""subtree_retention - Paired structural-retention analysis with leaf-thinning nulls.

Primary question
----------------
Given the set of canonical solutions empirically reachable at a reference
checkpoint (default: global_step_50), does a later RLVR checkpoint retain
those solutions in a structurally clustered way (preferential subtree loss),
or are the surviving solutions at least as structurally dispersed as expected
after ordinary leaf thinning?

Key design choices
------------------
1) Paired temporal null:
   The null is sampled ONLY from the reference checkpoint's empirically
   reachable canonical solutions for the same problem. Newly acquired late
   solutions are reported separately and never used to define retention.

2) Fixed reference cohort:
   All problems solved at least once at the reference checkpoint remain in
   problem-level accounting, including cases with zero retained leaves.
   Structural null comparisons require >=1 retained reference leaf and report
   that sample size explicitly.

3) Order-free structural features:
   We do NOT infer a temporal "first reasoning action" from AST traversal.
   Each canonical expression contributes normalized internal subtree features
   of size k (k original-number leaves). For 4-number Countdown, k=2 captures
   atomic pairwise operations and k=3 captures larger partial computations.
   For 3-number Countdown only k=2 is non-trivial.

4) Operator/direction aware:
   Subtraction and division preserve operand order; addition and
   multiplication are canonicalized under commutativity/associativity.
   Exact rational result values are included in each feature signature.

5) Exact uniform-thinning expectation:
   For each structural feature f appearing in L_f of the L reference leaves,
   sampling r leaves uniformly without replacement hits f with probability
       1 - C(L-L_f, r) / C(L, r).
   Thus the primary null expectation has no Monte-Carlo noise.

6) Optional frequency-biased thinning:
   A secondary Monte-Carlo null samples r unique reference leaves with
   probability proportional to their empirical reference-checkpoint counts.
   This is a simple control for initial leaf frequency, not a fitted survival
   model.

Interpretation
--------------
structural_retention_ratio = observed_cov / null_expected_cov:
    < 1 : retained leaves are more structurally clustered than leaf thinning
          predicts (consistent with preferential structural pruning)
    = 1 : structurally neutral given the number of retained leaves
    > 1 : retained leaves are more structurally dispersed than the null

structural_delta = observed_cov - null_expected_cov:
    < 0 : clustered retention / excess structural loss
    = 0 : leaf-neutral structural retention
    > 0 : dispersed structural retention

This script performs offline analysis only; it does not train or sample models.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import METRICS_DIR, RAW_DIR, TEST_PARQUET
from early_branch_locking.core.countdown_shared import (
    bootstrap_ci_mean,
    enumerate_solution_list,
    extract_ground_truth,
    load_jsonl,
    load_parquet_sorted,
    step_of,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--n_samples", type=int, default=320)
    parser.add_argument("--raw_glob", default="countdown_raw_*_n{N}.jsonl")
    parser.add_argument(
        "--reference_checkpoint",
        default="global_step_50",
        help="Checkpoint defining the empirically reachable reference leaf set.",
    )
    parser.add_argument(
        "--feature_sizes",
        default="2,3",
        help=(
            "Comma-separated subtree sizes (number of original-number leaves). "
            "Sizes >= the problem's input count are skipped as trivial full leaves."
        ),
    )
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument(
        "--weighted_null_draws",
        type=int,
        default=1000,
        help=(
            "Monte-Carlo draws for the optional reference-frequency-biased null. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out_dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--tag", default="v2_paired")
    parser.add_argument(
        "--include_pre_reference",
        action="store_true",
        help=(
            "Also compare checkpoints earlier than the reference checkpoint. "
            "Default analyzes reference and later checkpoints only."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------

_BINOP_SYMBOL = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


def _fraction_from_constant(value) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean constant is not a numeric Countdown atom.")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    raise ValueError(f"Unsupported numeric constant: {value!r}")


def _eval_fraction(node: ast.AST) -> Fraction:
    """Evaluate a Countdown arithmetic AST exactly as a rational number."""
    if isinstance(node, ast.Constant):
        return _fraction_from_constant(node.value)

    if isinstance(node, ast.UnaryOp):
        value = _eval_fraction(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        lhs = _eval_fraction(node.left)
        rhs = _eval_fraction(node.right)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.Div):
            if rhs == 0:
                raise ZeroDivisionError("Division by zero in canonical expression.")
            return lhs / rhs
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _fraction_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _leaf_count(node: ast.AST) -> int:
    """Count numeric atoms underneath this AST node."""
    if isinstance(node, ast.Constant):
        _fraction_from_constant(node.value)
        return 1

    if isinstance(node, ast.UnaryOp):
        return _leaf_count(node.operand)

    if isinstance(node, ast.BinOp):
        return _leaf_count(node.left) + _leaf_count(node.right)

    raise ValueError(f"Unsupported AST node in leaf count: {type(node).__name__}")


def _collect_flat_operands(node: ast.AST, op_type: type[ast.operator]) -> list[ast.AST]:
    """Flatten associative + or * nodes for normalized structural signatures."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
        return (
            _collect_flat_operands(node.left, op_type)
            + _collect_flat_operands(node.right, op_type)
        )
    return [node]


def _normalized_signature(node: ast.AST) -> str:
    """Canonical structural signature.

    + and * are flattened and sorted (associative + commutative).
    - and / preserve left/right direction.
    """
    if isinstance(node, ast.Constant):
        return _fraction_str(_fraction_from_constant(node.value))

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"neg({_normalized_signature(node.operand)})"
        if isinstance(node.op, ast.UAdd):
            return _normalized_signature(node.operand)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if not isinstance(node, ast.BinOp):
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    if isinstance(node.op, (ast.Add, ast.Mult)):
        op_type = type(node.op)
        symbol = _BINOP_SYMBOL[op_type]
        parts = [_normalized_signature(x) for x in _collect_flat_operands(node, op_type)]
        parts.sort()
        return f"{symbol}(" + ",".join(parts) + ")"

    symbol = _BINOP_SYMBOL.get(type(node.op))
    if symbol is None:
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    # Direction matters for subtraction/division.
    return (
        f"{symbol}("
        f"{_normalized_signature(node.left)},"
        f"{_normalized_signature(node.right)})"
    )


def subtree_features(expr: str, feature_size: int) -> set[str]:
    """Return all normalized internal-subtree features with `feature_size` atoms.

    This is deliberately order-free with respect to reasoning chronology.
    For (8-6)*(4+2), both K2 atomic operations are represented; neither is
    arbitrarily called "first".
    """
    try:
        root = ast.parse(expr, mode="eval").body
    except (SyntaxError, TypeError, ValueError):
        return set()

    features: set[str] = set()

    def visit(node: ast.AST) -> None:
        if not isinstance(node, ast.BinOp):
            return

        try:
            k = _leaf_count(node)
        except (ValueError, TypeError):
            return

        if k == feature_size:
            try:
                sig = _normalized_signature(node)
                value = _fraction_str(_eval_fraction(node))
                features.add(f"K{feature_size}::{sig}->{value}")
            except (ValueError, ZeroDivisionError):
                pass

        visit(node.left)
        visit(node.right)

    visit(root)
    return features


# ---------------------------------------------------------------------------
# Null calculations
# ---------------------------------------------------------------------------

def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _prob_feature_hit_uniform(
    total_leaves: int,
    feature_leaf_count: int,
    draw_size: int,
) -> float:
    """P(hit feature at least once) under uniform sampling without replacement."""
    L = total_leaves
    Lf = feature_leaf_count
    r = draw_size

    if r <= 0 or L <= 0 or Lf <= 0:
        return 0.0
    if r >= L:
        return 1.0
    if r > L - Lf:
        return 1.0

    log_no_hit = _log_comb(L - Lf, r) - _log_comb(L, r)
    no_hit = math.exp(log_no_hit)
    return 1.0 - no_hit


def expected_uniform_feature_coverage(
    leaf_to_features: dict[str, set[str]],
    draw_size: int,
) -> float:
    """Exact expected feature coverage from a uniform r-leaf subset."""
    leaves = list(leaf_to_features)
    L = len(leaves)
    if L == 0:
        return math.nan

    universe = set().union(*(leaf_to_features[s] for s in leaves))
    if not universe:
        return math.nan

    feature_counts = Counter()
    for s in leaves:
        for f in leaf_to_features[s]:
            feature_counts[f] += 1

    hit_probs = [
        _prob_feature_hit_uniform(L, feature_counts[f], draw_size)
        for f in universe
    ]
    return float(np.mean(hit_probs))


def weighted_null_feature_coverage(
    leaf_to_features: dict[str, set[str]],
    reference_counts: Counter,
    draw_size: int,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """Monte-Carlo reference-frequency-biased thinning without replacement."""
    if n_draws <= 0:
        return math.nan

    leaves = list(leaf_to_features)
    L = len(leaves)
    if L == 0 or draw_size <= 0:
        return math.nan

    universe = set().union(*(leaf_to_features[s] for s in leaves))
    if not universe:
        return math.nan

    if draw_size >= L:
        return 1.0

    weights = np.asarray(
        [max(0, reference_counts.get(s, 0)) for s in leaves],
        dtype=float,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(L, dtype=float)
    weights /= weights.sum()

    scores = np.empty(n_draws, dtype=float)
    for j in range(n_draws):
        # Sequential PPS sampling without replacement.
        idx = rng.choice(L, size=draw_size, replace=False, p=weights)
        hit = set()
        for ii in idx:
            hit.update(leaf_to_features[leaves[int(ii)]])
        scores[j] = len(hit) / len(universe)

    return float(scores.mean())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _checkpoint_from_path(path: Path, n_samples: int) -> str:
    suffix = f"_n{n_samples}.jsonl"
    prefix = "countdown_raw_"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected raw filename: {name}")
    return name[len(prefix):-len(suffix)]


def _load_checkpoint_counts(
    path: Path,
    num_problems: int,
    allowed_solutions: dict[int, set[str]],
) -> dict[int, Counter]:
    counts: dict[int, Counter] = defaultdict(Counter)

    for row in load_jsonl(path):
        pid = int(row.get("problem_index", -1))
        if pid < 0 or pid >= num_problems:
            continue
        if not row.get("overall_ok"):
            continue

        canonical = row.get("canonical_expr")
        if not canonical:
            continue
        if canonical not in allowed_solutions.get(pid, set()):
            continue

        counts[pid][canonical] += 1

    return counts


def _parse_feature_sizes(spec: str) -> list[int]:
    sizes = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    if not sizes or any(k < 2 for k in sizes):
        raise ValueError("--feature_sizes must contain integers >= 2")
    return sizes


def _boot(
    values: Iterable[float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    arr = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not arr:
        return math.nan, math.nan, math.nan
    return bootstrap_ci_mean(arr, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)
    feature_sizes = _parse_feature_sizes(args.feature_sizes)

    pattern = args.raw_glob.replace("{N}", str(args.n_samples))
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw files matched {RAW_DIR}/{pattern}")

    path_by_checkpoint: dict[str, Path] = {}
    for path in files:
        ckpt = _checkpoint_from_path(path, args.n_samples)
        path_by_checkpoint[ckpt] = path

    if args.reference_checkpoint not in path_by_checkpoint:
        available = ", ".join(sorted(path_by_checkpoint, key=step_of))
        raise FileNotFoundError(
            f"Reference checkpoint {args.reference_checkpoint!r} not found. "
            f"Available: {available}"
        )

    records = load_parquet_sorted(
        TEST_PARQUET,
        n=args.num_problems,
        sort_key="sample_id",
    )

    # Exact solver sets validate generated canonical leaves and provide n_inputs.
    full_solution_sets: dict[int, set[str]] = {}
    n_inputs_by_pid: dict[int, int] = {}

    for pid, record in enumerate(records):
        numbers, target, feasible = extract_ground_truth(record)
        n_inputs_by_pid[pid] = len(numbers)

        if str(feasible).strip().lower() == "yes":
            solutions = enumerate_solution_list(
                [int(x) for x in numbers],
                int(target),
            )
            full_solution_sets[pid] = set(solutions)
        else:
            full_solution_sets[pid] = set()

    # Precompute structural features for every exact solver leaf.
    feature_cache: dict[tuple[int, str, int], set[str]] = {}
    for pid, solutions in full_solution_sets.items():
        n_inputs = n_inputs_by_pid[pid]
        for expr in solutions:
            for k in feature_sizes:
                if k >= n_inputs:
                    continue
                feature_cache[(pid, expr, k)] = subtree_features(expr, k)

    reference_path = path_by_checkpoint[args.reference_checkpoint]
    reference_counts_by_pid = _load_checkpoint_counts(
        reference_path,
        args.num_problems,
        full_solution_sets,
    )
    reference_step = step_of(args.reference_checkpoint)

    # Fixed reference cohort.
    reference_cohort = [
        pid
        for pid in range(args.num_problems)
        if len(reference_counts_by_pid.get(pid, Counter())) > 0
    ]

    checkpoints = sorted(path_by_checkpoint, key=step_of)
    if not args.include_pre_reference:
        checkpoints = [c for c in checkpoints if step_of(c) >= reference_step]

    rng_master = np.random.default_rng(args.seed)

    structural_rows: list[dict] = []
    problem_rows: list[dict] = []

    for checkpoint in checkpoints:
        late_counts_by_pid = _load_checkpoint_counts(
            path_by_checkpoint[checkpoint],
            args.num_problems,
            full_solution_sets,
        )

        ckpt_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(ckpt_seed)

        for pid in reference_cohort:
            ref_counts = reference_counts_by_pid.get(pid, Counter())
            late_counts = late_counts_by_pid.get(pid, Counter())

            ref_leaves = set(ref_counts)
            late_leaves = set(late_counts)

            retained = ref_leaves & late_leaves
            lost = ref_leaves - late_leaves
            gained = late_leaves - ref_leaves

            L = len(ref_leaves)
            r = len(retained)

            base_problem_row = {
                "reference_checkpoint": args.reference_checkpoint,
                "reference_step": reference_step,
                "checkpoint": checkpoint,
                "step": step_of(checkpoint),
                "problem_index": pid,
                "n_inputs": n_inputs_by_pid[pid],
                "reference_unique_leaves": L,
                "late_unique_leaves": len(late_leaves),
                "retained_reference_leaves": r,
                "lost_reference_leaves": len(lost),
                "gained_late_leaves": len(gained),
                "reference_leaf_retention_rate": (r / L) if L else math.nan,
                "late_is_solved": int(len(late_leaves) > 0),
                "zero_reference_retention": int(r == 0),
                "full_support_turnover": int(r == 0 and len(late_leaves) > 0),
            }
            problem_rows.append(base_problem_row)

            # Zero-retention cases remain in problem_rows; they are not silently
            # discarded. Structural shape is undefined when r == 0.
            if r == 0:
                continue

            for k in feature_sizes:
                if k >= n_inputs_by_pid[pid]:
                    continue

                leaf_to_features = {
                    s: feature_cache.get((pid, s, k), set())
                    for s in ref_leaves
                }

                feature_universe = set().union(
                    *(leaf_to_features[s] for s in ref_leaves)
                )
                if not feature_universe:
                    continue

                observed_features = set()
                for s in retained:
                    observed_features.update(leaf_to_features[s])

                observed_cov = len(observed_features) / len(feature_universe)

                uniform_expected = expected_uniform_feature_coverage(
                    leaf_to_features,
                    draw_size=r,
                )
                if not np.isfinite(uniform_expected) or uniform_expected <= 0:
                    continue

                uniform_delta = observed_cov - uniform_expected
                uniform_ratio = observed_cov / uniform_expected
                uniform_log_ratio = (
                    math.log(uniform_ratio)
                    if uniform_ratio > 0
                    else -math.inf
                )

                weighted_expected = weighted_null_feature_coverage(
                    leaf_to_features=leaf_to_features,
                    reference_counts=ref_counts,
                    draw_size=r,
                    n_draws=args.weighted_null_draws,
                    rng=rng,
                )

                if np.isfinite(weighted_expected) and weighted_expected > 0:
                    weighted_delta = observed_cov - weighted_expected
                    weighted_ratio = observed_cov / weighted_expected
                    weighted_log_ratio = (
                        math.log(weighted_ratio)
                        if weighted_ratio > 0
                        else -math.inf
                    )
                else:
                    weighted_delta = math.nan
                    weighted_ratio = math.nan
                    weighted_log_ratio = math.nan

                structural_rows.append({
                    **base_problem_row,
                    "feature_size": k,
                    "reference_feature_universe": len(feature_universe),
                    "observed_retained_features": len(observed_features),
                    "observed_structural_coverage": observed_cov,
                    "uniform_null_expected_coverage": uniform_expected,
                    "uniform_structural_delta": uniform_delta,
                    "uniform_structural_retention_ratio": uniform_ratio,
                    "uniform_log_retention_ratio": uniform_log_ratio,
                    "weighted_null_expected_coverage": weighted_expected,
                    "weighted_structural_delta": weighted_delta,
                    "weighted_structural_retention_ratio": weighted_ratio,
                    "weighted_log_retention_ratio": weighted_log_ratio,
                    "weighted_null_draws": args.weighted_null_draws,
                })

    structural_df = pd.DataFrame(structural_rows)
    problem_df = pd.DataFrame(problem_rows)

    # -----------------------------------------------------------------------
    # Structural summaries
    # -----------------------------------------------------------------------
    summary_rows: list[dict] = []
    if not structural_df.empty:
        for (checkpoint, k), group in structural_df.groupby(
            ["checkpoint", "feature_size"],
            sort=False,
        ):
            step = int(group["step"].iloc[0])

            obs = _boot(
                group["observed_structural_coverage"],
                args.bootstrap_draws,
                args.seed,
            )
            uni = _boot(
                group["uniform_null_expected_coverage"],
                args.bootstrap_draws,
                args.seed,
            )
            delta = _boot(
                group["uniform_structural_delta"],
                args.bootstrap_draws,
                args.seed,
            )
            ratio = _boot(
                group["uniform_structural_retention_ratio"],
                args.bootstrap_draws,
                args.seed,
            )
            log_ratio = _boot(
                group.replace([np.inf, -np.inf], np.nan)[
                    "uniform_log_retention_ratio"
                ],
                args.bootstrap_draws,
                args.seed,
            )

            w_null = _boot(
                group["weighted_null_expected_coverage"],
                args.bootstrap_draws,
                args.seed,
            )
            w_delta = _boot(
                group["weighted_structural_delta"],
                args.bootstrap_draws,
                args.seed,
            )
            w_ratio = _boot(
                group["weighted_structural_retention_ratio"],
                args.bootstrap_draws,
                args.seed,
            )

            summary_rows.append({
                "reference_checkpoint": args.reference_checkpoint,
                "reference_step": reference_step,
                "checkpoint": checkpoint,
                "step": step,
                "feature_size": int(k),
                "n_problems_structural": len(group),

                "observed_structural_coverage_mean": obs[0],
                "observed_structural_coverage_ci_lo": obs[1],
                "observed_structural_coverage_ci_hi": obs[2],

                "uniform_null_expected_coverage_mean": uni[0],
                "uniform_null_expected_coverage_ci_lo": uni[1],
                "uniform_null_expected_coverage_ci_hi": uni[2],

                "uniform_structural_delta_mean": delta[0],
                "uniform_structural_delta_ci_lo": delta[1],
                "uniform_structural_delta_ci_hi": delta[2],

                "uniform_structural_retention_ratio_mean": ratio[0],
                "uniform_structural_retention_ratio_ci_lo": ratio[1],
                "uniform_structural_retention_ratio_ci_hi": ratio[2],

                "uniform_log_retention_ratio_mean": log_ratio[0],
                "uniform_log_retention_ratio_ci_lo": log_ratio[1],
                "uniform_log_retention_ratio_ci_hi": log_ratio[2],

                "weighted_null_expected_coverage_mean": w_null[0],
                "weighted_null_expected_coverage_ci_lo": w_null[1],
                "weighted_null_expected_coverage_ci_hi": w_null[2],

                "weighted_structural_delta_mean": w_delta[0],
                "weighted_structural_delta_ci_lo": w_delta[1],
                "weighted_structural_delta_ci_hi": w_delta[2],

                "weighted_structural_retention_ratio_mean": w_ratio[0],
                "weighted_structural_retention_ratio_ci_lo": w_ratio[1],
                "weighted_structural_retention_ratio_ci_hi": w_ratio[2],

                "bootstrap_draws": args.bootstrap_draws,
                "weighted_null_draws": args.weighted_null_draws,
                "statistical_unit": "problem",
            })

    # -----------------------------------------------------------------------
    # Fixed-cohort problem summaries, including zero-retention problems
    # -----------------------------------------------------------------------
    problem_summary_rows: list[dict] = []
    if not problem_df.empty:
        for checkpoint, group in problem_df.groupby("checkpoint", sort=False):
            retention = _boot(
                group["reference_leaf_retention_rate"],
                args.bootstrap_draws,
                args.seed,
            )
            late_solved = _boot(
                group["late_is_solved"],
                args.bootstrap_draws,
                args.seed,
            )
            zero_ret = _boot(
                group["zero_reference_retention"],
                args.bootstrap_draws,
                args.seed,
            )
            turnover = _boot(
                group["full_support_turnover"],
                args.bootstrap_draws,
                args.seed,
            )

            problem_summary_rows.append({
                "reference_checkpoint": args.reference_checkpoint,
                "reference_step": reference_step,
                "checkpoint": checkpoint,
                "step": int(group["step"].iloc[0]),
                "n_reference_solved_problems": len(group),

                "reference_leaf_retention_rate_mean": retention[0],
                "reference_leaf_retention_rate_ci_lo": retention[1],
                "reference_leaf_retention_rate_ci_hi": retention[2],

                "late_solved_fraction_mean": late_solved[0],
                "late_solved_fraction_ci_lo": late_solved[1],
                "late_solved_fraction_ci_hi": late_solved[2],

                "zero_reference_retention_fraction_mean": zero_ret[0],
                "zero_reference_retention_fraction_ci_lo": zero_ret[1],
                "zero_reference_retention_fraction_ci_hi": zero_ret[2],

                "full_support_turnover_fraction_mean": turnover[0],
                "full_support_turnover_fraction_ci_lo": turnover[1],
                "full_support_turnover_fraction_ci_hi": turnover[2],

                "bootstrap_draws": args.bootstrap_draws,
                "statistical_unit": "problem",
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)

    structural_summary_path = (
        args.out_dir / f"subtree_retention_structural_retention_summary_{args.tag}.csv"
    )
    problem_summary_path = (
        args.out_dir / f"subtree_retention_structural_retention_problem_summary_{args.tag}.csv"
    )
    structural_per_problem_path = (
        args.out_dir / f"subtree_retention_structural_retention_per_problem_{args.tag}.parquet"
    )
    problem_rows_path = (
        args.out_dir / f"subtree_retention_structural_retention_problem_rows_{args.tag}.parquet"
    )

    structural_summary_df = pd.DataFrame(summary_rows)
    if not structural_summary_df.empty:
        structural_summary_df = structural_summary_df.sort_values(
            ["step", "feature_size"]
        )
    structural_summary_df.to_csv(structural_summary_path, index=False)

    problem_summary_df = pd.DataFrame(problem_summary_rows)
    if not problem_summary_df.empty:
        problem_summary_df = problem_summary_df.sort_values("step")
    problem_summary_df.to_csv(problem_summary_path, index=False)

    structural_df.to_parquet(structural_per_problem_path, index=False)
    problem_df.to_parquet(problem_rows_path, index=False)

    print(json.dumps({
        "reference_checkpoint": args.reference_checkpoint,
        "reference_cohort_size": len(reference_cohort),
        "structural_summary": str(structural_summary_path),
        "problem_summary": str(problem_summary_path),
        "structural_per_problem": str(structural_per_problem_path),
        "problem_rows": str(problem_rows_path),
        "n_structural_rows": len(structural_df),
        "n_problem_rows": len(problem_df),
    }, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())