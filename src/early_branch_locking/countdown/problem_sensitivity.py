
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""problem_sensitivity - Problem-structure sensitivity.
Hypothesis: Countdown intervention gains are concentrated on problems whose exact solution space has specific branch structure.
Inputs: Exp A, Exp H and Exp I summaries/per-problem files; Countdown raw trajectories; dataset/test.parquet.
Outputs: data/analysis_results/rlvr_passk/metrics/problem_sensitivity_problem_sensitivity_summary_problem_sensitivity_v1.csv; data/analysis_results/rlvr_passk/metrics/problem_sensitivity_problem_sensitivity_per_problem_problem_sensitivity_v1.parquet
Status: paper-main
"""
from __future__ import annotations
"""Problem-structure sensitivity analysis for Countdown RLVR experiments."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_DATA_ROOT as ANALYSIS_ROOT, METRICS_DIR, TEST_PARQUET  # noqa: E402

from early_branch_locking.core.countdown_shared import extract_ground_truth, load_parquet_sorted  # noqa: E402
from early_branch_locking.core.structure_utils import enumerate_solution_records, summarize_solution_records  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--sets_50_to_200", type=str, default=str(METRICS_DIR / "branch_set_collection_sets_global_step_50_to_global_step_200_n320.json"))
    parser.add_argument("--sets_50_to_275", type=str, default=str(METRICS_DIR / "branch_set_collection_sets_global_step_50_to_global_step_275_n320.json"))
    parser.add_argument("--raw_50_path", type=str, default=str(ANALYSIS_ROOT / "raw" / "countdown_raw_global_step_50_n320.jsonl"))
    parser.add_argument("--raw_200_path", type=str, default=str(ANALYSIS_ROOT / "raw" / "countdown_raw_global_step_200_n320.jsonl"))
    parser.add_argument("--raw_275_path", type=str, default=str(ANALYSIS_ROOT / "raw" / "countdown_raw_global_step_275_n320.jsonl"))
    parser.add_argument("--expi_summary_path", type=str, required=True)
    parser.add_argument("--exph_summary_path", type=str, required=True)
    parser.add_argument("--exph_per_problem_path", type=str, required=True)
    parser.add_argument("--tag", type=str, default="problem_sensitivity_v1")
    return parser.parse_args()


def load_set_flags(path: Path, key: str, num_problems: int) -> Dict[int, int]:
    if not path.exists():
        return {pid: 0 for pid in range(num_problems)}
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = {int(item) for item in data.get(key, [])}
    return {pid: int(pid in ids) for pid in range(num_problems)}


def load_pass64_stats(path: Path, num_problems: int) -> Dict[int, dict]:
    grouped = {pid: {"correct_mass_64": 0.0, "pass64_hit": 0} for pid in range(num_problems)}
    counts = {pid: [] for pid in range(num_problems)}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = int(row.get("problem_index", -1))
            if 0 <= pid < num_problems:
                counts[pid].append(bool(row.get("overall_ok", False)))
    for pid, values in counts.items():
        first64 = values[:64]
        grouped[pid]["correct_mass_64"] = float(np.mean(first64)) if first64 else 0.0
        grouped[pid]["pass64_hit"] = int(any(first64))
    return grouped


def pick_best_expi_row(summary_path: Path) -> dict:
    df = pd.read_csv(summary_path)
    if "tag" not in df.columns:
        fallback_tag = summary_path.stem.replace("candidate_to_final_systematic_ablation_summary_", "")
        df["tag"] = fallback_tag
    c2f = df[df["variant"] == "c2f"].copy()
    if c2f.empty:
        raise ValueError("No c2f rows found in ExpI summary.")
    pass_cols = sorted(
        [col for col in c2f.columns if col.startswith("pass@") and "_ci_" not in col],
        key=lambda name: int(name.split("@")[1]),
    )
    if not pass_cols:
        raise ValueError("No pass@k columns found in ExpI summary.")
    best_pass_col = pass_cols[-1]
    sort_cols = [best_pass_col]
    if "coverage_mean" in c2f.columns:
        sort_cols.append("coverage_mean")
    return c2f.sort_values(sort_cols, ascending=False).iloc[0].to_dict()


def expi_per_problem_path(best_row: dict) -> Path:
    tag = str(best_row["tag"])
    direct = METRICS_DIR / f"candidate_to_final_c2f_per_problem_{tag}.parquet"
    if direct.exists():
        return direct
    draft_ckpt = str(best_row.get("draft_ckpt", ""))
    refine_ckpt = str(best_row.get("refine_ckpt", ""))
    prefix_mode = str(best_row.get("prefix_mode", ""))
    prefix_tokens = int(best_row.get("prefix_tokens", 0) or 0)
    suffix = f"{prefix_mode}{prefix_tokens}" if prefix_mode == "tokens" else prefix_mode
    candidates = [
        METRICS_DIR / f"candidate_to_final_c2f_per_problem_{tag}_{suffix}.parquet",
        METRICS_DIR / f"candidate_to_final_c2f_per_problem_{tag}_{draft_ckpt}_{refine_ckpt}_{suffix}.parquet",
        METRICS_DIR / f"candidate_to_final_c2f_per_problem_{tag}_d{draft_ckpt.split('_')[-1]}_r{refine_ckpt.split('_')[-1]}_{suffix}.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ExpI per-problem parquet found for tag={tag}")


def load_expi_rescue(best_row: dict) -> Dict[int, dict]:
    per_path = expi_per_problem_path(best_row)
    df = pd.read_parquet(per_path)
    pivot = {}
    for pid, sub in df.groupby("problem_index"):
        refine_rows = sub[sub["variant"] == "refine"]
        c2f_rows = sub[sub["variant"] == "c2f"]
        if refine_rows.empty or c2f_rows.empty:
            continue
        refine = refine_rows.iloc[0]
        c2f = c2f_rows.iloc[0]
        pivot[int(pid)] = {
            "c2f_rescue_correct_mass": float(c2f["correct_mass"] - refine["correct_mass"]),
            "c2f_rescue_coverage": float(c2f["coverage"] - refine["coverage"]),
        }
    return pivot


def load_exph_rescue(summary_path: Path, per_problem_path: Path) -> Dict[int, dict]:
    summary = pd.read_csv(summary_path)
    candidates = summary[summary["variant"].str.startswith(("rev_", "inject_R"))].copy()
    if candidates.empty:
        raise ValueError("No reverse/inject_range rows found in ExpH summary.")
    best_variant = candidates.sort_values(["pass@64", "coverage_mean"], ascending=False).iloc[0]["variant"]
    df = pd.read_parquet(per_problem_path)
    pivot = {}
    for pid, sub in df.groupby("problem_index"):
        collapse = sub[sub["variant"] == "pure_collapse"].iloc[0]
        best = sub[sub["variant"] == best_variant].iloc[0]
        pivot[int(pid)] = {
            "best_exph_variant": best_variant,
            "chimera_rescue_correct_mass": float(best["correct_mass"] - collapse["correct_mass"]),
            "chimera_rescue_coverage": float(best["coverage"] - collapse["coverage"]),
        }
    return pivot


def summarize_correlations(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        for target in target_cols:
            sub = df[[feature, target]].dropna()
            if len(sub) < 2:
                continue
            pearson = float(sub[feature].corr(sub[target], method="pearson"))
            spearman = float(sub[feature].corr(sub[target], method="spearman"))
            rows.append(dict(feature=feature, target=target, n=len(sub), pearson=pearson, spearman=spearman))
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    s_loss_200 = load_set_flags(Path(args.sets_50_to_200), "s_loss", args.num_problems)
    s_loss_275 = load_set_flags(Path(args.sets_50_to_275), "s_loss", args.num_problems)
    raw_50 = load_pass64_stats(Path(args.raw_50_path), args.num_problems)
    raw_200 = load_pass64_stats(Path(args.raw_200_path), args.num_problems)
    raw_275 = load_pass64_stats(Path(args.raw_275_path), args.num_problems)
    best_expi = pick_best_expi_row(Path(args.expi_summary_path))
    expi_rescue = load_expi_rescue(best_expi)
    exph_rescue = load_exph_rescue(Path(args.exph_summary_path), Path(args.exph_per_problem_path))

    rows = []
    for pid, rec in enumerate(records):
        numbers, target, _ = extract_ground_truth(rec)
        features = summarize_solution_records(enumerate_solution_records(numbers, target))
        rows.append(dict(
            problem_index=pid,
            s_loss_50_to_200=s_loss_200.get(pid, 0),
            s_loss_50_to_275=s_loss_275.get(pid, 0),
            pass64_drop_50_to_200=raw_50[pid]["pass64_hit"] - raw_200[pid]["pass64_hit"],
            pass64_drop_50_to_275=raw_50[pid]["pass64_hit"] - raw_275[pid]["pass64_hit"],
            correct_mass_drop_50_to_200=raw_50[pid]["correct_mass_64"] - raw_200[pid]["correct_mass_64"],
            correct_mass_drop_50_to_275=raw_50[pid]["correct_mass_64"] - raw_275[pid]["correct_mass_64"],
            **features,
            **expi_rescue.get(pid, {}),
            **exph_rescue.get(pid, {}),
        ))

    per_problem = pd.DataFrame(rows)
    feature_cols = [
        "solution_count", "first_op_entropy", "unique_opseq_count", "unique_tree_count",
        "min_depth", "mean_depth", "requires_division", "requires_parentheses",
        "has_equivalent_path_diversity",
    ]
    target_cols = [
        "s_loss_50_to_200", "s_loss_50_to_275",
        "pass64_drop_50_to_200", "pass64_drop_50_to_275",
        "c2f_rescue_correct_mass", "c2f_rescue_coverage",
        "chimera_rescue_correct_mass", "chimera_rescue_coverage",
    ]
    summary = summarize_correlations(per_problem, feature_cols, target_cols)
    summary["best_expi_tag"] = best_expi["tag"]
    summary["best_exph_variant"] = per_problem["best_exph_variant"].dropna().iloc[0] if "best_exph_variant" in per_problem else ""
    tag = args.tag
    per_path = METRICS_DIR / f"problem_sensitivity_problem_sensitivity_per_problem_{tag}.parquet"
    summary_path = METRICS_DIR / f"problem_sensitivity_problem_sensitivity_summary_{tag}.csv"
    per_problem.to_parquet(per_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[problem_sensitivity] Saved summary → {summary_path}")
    print(f"[problem_sensitivity] Saved per-problem → {per_path}")


if __name__ == "__main__":
    main()
