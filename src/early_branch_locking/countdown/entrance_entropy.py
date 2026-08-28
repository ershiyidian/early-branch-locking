
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""entrance_entropy - Answer and trace entropy diagnostics.
Hypothesis: RLVR increases answer correctness while collapsing answer-level and within-answer trace diversity.
Inputs: Countdown raw JSONL files; optional Exp A sets and branch-survival parquet.
Outputs: data/analysis_results/rlvr_passk/metrics/entrance_entropy_countdown_entropy_n320.csv; data/analysis_results/rlvr_passk/metrics/entrance_entropy_countdown_per_problem_n320.parquet
Status: paper-main
"""
"""
entrance_entropy_entropy_countdown.py

旗舰实验 B：
- 对每个 checkpoint 的 raw jsonl（countdown_raw_{ckpt}_n{N}.jsonl）做：
  1) per-problem rho_i = #correct / N
     输出 mean(rho), var(rho), mass0, mass1
  2) per-problem H(A)（answer_label 熵）、unique_A_count、top1_mass
  3) per-problem H(Y|A) 的离散近似：在每个 A=a 内 trace_label 的熵再加权求和
     （严格可复现，不依赖 embedding 模型）

输出：
- data/analysis_results/rlvr_passk/metrics/entrance_entropy_countdown_entropy_n{N}.csv
- 并可选输出每题明细 parquet（便于你后续画图/做 A 的 S_loss）
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (  # noqa: E402
    COUNTDOWN_DATA_ROOT as ANALYSIS_ROOT,
    METRICS_DIR,
    METRICS_DIR as OUT_DIR,
    RAW_DIR,
    TEST_PARQUET,
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_shared import entropy_from_counts, tree_signature, load_jsonl, step_of  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--glob", type=str, default="countdown_raw_*_n{N}.jsonl",
                   help="raw 文件匹配模式，{N} 会被替换为 n_samples")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pat = args.glob.replace("{N}", str(args.n_samples))
    files = sorted(RAW_DIR.glob(pat))
    if not files:
        raise FileNotFoundError(f"No raw files matched: {RAW_DIR}/{pat}")

    all_rows = []
    per_problem_rows = []

    for fp in files:
        # infer ckpt
        name = fp.name  # countdown_raw_{ckpt}_n{N}.jsonl
        ckpt = name[len("countdown_raw_"):]
        ckpt = ckpt[: ckpt.rfind(f"_n{args.n_samples}.jsonl")]

        # accumulate by problem
        by_prob = defaultdict(list)
        for rec in load_jsonl(fp):
            by_prob[int(rec["problem_index"])].append(rec)

        # per-problem stats
        rhos = []
        mass0 = 0
        mass1 = 0
        HA_list = []
        HYgA_list = []
        HYgA_opseq_list = []
        HYgA_tree_list = []
        uniqA_list = []
        top1_list = []
        uniq_opseq_list = []
        top1_opseq_list = []
        uniq_tree_list = []
        top1_tree_list = []
        fmt_rate_list = []
        HA_fmt_list = []
        HYgA_fmt_list = []
        uniqA_fmt_list = []
        top1_fmt_list = []
        HYgA_opseq_fmt_list = []
        HYgA_tree_fmt_list = []
        uniq_opseq_fmt_list = []
        top1_opseq_fmt_list = []
        uniq_tree_fmt_list = []
        top1_tree_fmt_list = []
        nprobs = len(by_prob)

        for pid, recs in by_prob.items():
            n = len(recs)
            c = sum(1 for r in recs if r.get("overall_ok"))
            rho = c / n if n > 0 else 0.0
            rhos.append(rho)
            if rho == 0.0:
                mass0 += 1
            if rho == 1.0:
                mass1 += 1

            # H(A)
            a_counts = Counter(r.get("answer_label", "INVALID") for r in recs)
            HA = entropy_from_counts(a_counts)
            uniqA = len(a_counts)
            top1 = max(a_counts.values()) / n if n > 0 else 0.0

            # H(Y|A) 离散近似：sum_a p(a) H(trace | a)
            # trace_label 是 hash 后离散，不依赖 embedding
            HYgA = 0.0
            HYgA_opseq = 0.0
            HYgA_tree = 0.0
            for a, ac in a_counts.items():
                # traces within a
                t_counts = Counter(
                    r.get("trace_label", "TRACE::NA")
                    for r in recs
                    if r.get("answer_label", "INVALID") == a
                )
                HYgA += (ac / n) * entropy_from_counts(t_counts)

                op_counts = Counter(
                    r.get("opseq_label", "OPSEQ::")
                    for r in recs
                    if r.get("answer_label", "INVALID") == a
                )
                HYgA_opseq += (ac / n) * entropy_from_counts(op_counts)

                tree_counts = Counter(
                    tree_signature(r.get("canonical_expr"))
                    for r in recs
                    if r.get("answer_label", "INVALID") == a
                )
                HYgA_tree += (ac / n) * entropy_from_counts(tree_counts)

            HA_list.append(HA)
            HYgA_list.append(HYgA)
            HYgA_opseq_list.append(HYgA_opseq)
            HYgA_tree_list.append(HYgA_tree)
            uniqA_list.append(uniqA)
            top1_list.append(top1)

            opseq_counts = Counter(r.get("opseq_label", "OPSEQ::") for r in recs)
            uniq_opseq_list.append(len(opseq_counts))
            top1_opseq_list.append(max(opseq_counts.values()) / n if n > 0 else 0.0)

            tree_counts_all = Counter(tree_signature(r.get("canonical_expr")) for r in recs)
            uniq_tree_list.append(len(tree_counts_all))
            top1_tree_list.append(max(tree_counts_all.values()) / n if n > 0 else 0.0)

            # format-ok 子集（排除 INVALID）上的 H(A), H(Y|A)
            valid_recs = [r for r in recs if r.get("answer_label", "INVALID") != "INVALID"]
            n_valid = len(valid_recs)
            fmt_rate = n_valid / n if n > 0 else 0.0
            fmt_rate_list.append(fmt_rate)

            if n_valid > 0:
                a_counts_fmt = Counter(r.get("answer_label", "INVALID") for r in valid_recs)
                HA_fmt = entropy_from_counts(a_counts_fmt)
                uniqA_fmt = len(a_counts_fmt)
                top1_fmt = max(a_counts_fmt.values()) / n_valid
                HYgA_fmt = 0.0
                HYgA_opseq_fmt = 0.0
                HYgA_tree_fmt = 0.0
                for a, ac in a_counts_fmt.items():
                    t_counts_fmt = Counter(
                        r.get("trace_label", "TRACE::NA")
                        for r in valid_recs
                        if r.get("answer_label", "INVALID") == a
                    )
                    HYgA_fmt += (ac / n_valid) * entropy_from_counts(t_counts_fmt)

                    op_counts_fmt = Counter(
                        r.get("opseq_label", "OPSEQ::")
                        for r in valid_recs
                        if r.get("answer_label", "INVALID") == a
                    )
                    HYgA_opseq_fmt += (ac / n_valid) * entropy_from_counts(op_counts_fmt)

                    tree_counts_fmt = Counter(
                        tree_signature(r.get("canonical_expr"))
                        for r in valid_recs
                        if r.get("answer_label", "INVALID") == a
                    )
                    HYgA_tree_fmt += (ac / n_valid) * entropy_from_counts(tree_counts_fmt)

                opseq_counts_fmt = Counter(r.get("opseq_label", "OPSEQ::") for r in valid_recs)
                uniq_opseq_fmt = len(opseq_counts_fmt)
                top1_opseq_fmt = max(opseq_counts_fmt.values()) / n_valid

                tree_counts_all_fmt = Counter(tree_signature(r.get("canonical_expr")) for r in valid_recs)
                uniq_tree_fmt = len(tree_counts_all_fmt)
                top1_tree_fmt = max(tree_counts_all_fmt.values()) / n_valid
            else:
                HA_fmt = 0.0
                HYgA_fmt = 0.0
                uniqA_fmt = 0
                top1_fmt = 0.0
                HYgA_opseq_fmt = 0.0
                HYgA_tree_fmt = 0.0
                uniq_opseq_fmt = 0
                top1_opseq_fmt = 0.0
                uniq_tree_fmt = 0
                top1_tree_fmt = 0.0

            HA_fmt_list.append(HA_fmt)
            HYgA_fmt_list.append(HYgA_fmt)
            uniqA_fmt_list.append(uniqA_fmt)
            top1_fmt_list.append(top1_fmt)
            HYgA_opseq_fmt_list.append(HYgA_opseq_fmt)
            HYgA_tree_fmt_list.append(HYgA_tree_fmt)
            uniq_opseq_fmt_list.append(uniq_opseq_fmt)
            top1_opseq_fmt_list.append(top1_opseq_fmt)
            uniq_tree_fmt_list.append(uniq_tree_fmt)
            top1_tree_fmt_list.append(top1_tree_fmt)

            if args.save_per_problem:
                per_problem_rows.append(dict(
                    checkpoint=ckpt,
                    problem_index=pid,
                    n=n,
                    correct=c,
                    rho=rho,
                    HA=HA,
                    HYgA=HYgA,
                    HYgA_opseq=HYgA_opseq,
                    HYgA_tree=HYgA_tree,
                    uniqueA=uniqA,
                    top1_mass=top1,
                    unique_opseq=len(opseq_counts),
                    top1_opseq=max(opseq_counts.values()) / n if n > 0 else 0.0,
                    unique_tree=len(tree_counts_all),
                    top1_tree=max(tree_counts_all.values()) / n if n > 0 else 0.0,
                    format_rate=fmt_rate,
                    HA_fmt=HA_fmt,
                    HYgA_fmt=HYgA_fmt,
                    uniqueA_fmt=uniqA_fmt,
                    top1_mass_fmt=top1_fmt,
                    HYgA_opseq_fmt=HYgA_opseq_fmt,
                    HYgA_tree_fmt=HYgA_tree_fmt,
                    unique_opseq_fmt=uniq_opseq_fmt,
                    top1_opseq_fmt=top1_opseq_fmt,
                    unique_tree_fmt=uniq_tree_fmt,
                    top1_tree_fmt=top1_tree_fmt,
                ))

        row = dict(
            checkpoint=ckpt,
            n_samples=args.n_samples,
            num_problems=nprobs,
            rho_mean=float(np.mean(rhos)) if rhos else float("nan"),
            rho_var=float(np.var(rhos)) if rhos else float("nan"),
            rho_mass0=mass0 / nprobs if nprobs else float("nan"),
            rho_mass1=mass1 / nprobs if nprobs else float("nan"),
            HA_mean=float(np.mean(HA_list)) if HA_list else float("nan"),
            HYgA_mean=float(np.mean(HYgA_list)) if HYgA_list else float("nan"),
            HYgA_opseq_mean=float(np.mean(HYgA_opseq_list)) if HYgA_opseq_list else float("nan"),
            HYgA_tree_mean=float(np.mean(HYgA_tree_list)) if HYgA_tree_list else float("nan"),
            uniqueA_mean=float(np.mean(uniqA_list)) if uniqA_list else float("nan"),
            top1_mass_mean=float(np.mean(top1_list)) if top1_list else float("nan"),
            unique_opseq_mean=float(np.mean(uniq_opseq_list)) if uniq_opseq_list else float("nan"),
            top1_opseq_mean=float(np.mean(top1_opseq_list)) if top1_opseq_list else float("nan"),
            unique_tree_mean=float(np.mean(uniq_tree_list)) if uniq_tree_list else float("nan"),
            top1_tree_mean=float(np.mean(top1_tree_list)) if top1_tree_list else float("nan"),
            format_rate_mean=float(np.mean(fmt_rate_list)) if fmt_rate_list else float("nan"),
            HA_fmt_mean=float(np.mean(HA_fmt_list)) if HA_fmt_list else float("nan"),
            HYgA_fmt_mean=float(np.mean(HYgA_fmt_list)) if HYgA_fmt_list else float("nan"),
            uniqueA_fmt_mean=float(np.mean(uniqA_fmt_list)) if uniqA_fmt_list else float("nan"),
            top1_mass_fmt_mean=float(np.mean(top1_fmt_list)) if top1_fmt_list else float("nan"),
            HYgA_opseq_fmt_mean=float(np.mean(HYgA_opseq_fmt_list)) if HYgA_opseq_fmt_list else float("nan"),
            HYgA_tree_fmt_mean=float(np.mean(HYgA_tree_fmt_list)) if HYgA_tree_fmt_list else float("nan"),
            unique_opseq_fmt_mean=float(np.mean(uniq_opseq_fmt_list)) if uniq_opseq_fmt_list else float("nan"),
            top1_opseq_fmt_mean=float(np.mean(top1_opseq_fmt_list)) if top1_opseq_fmt_list else float("nan"),
            unique_tree_fmt_mean=float(np.mean(uniq_tree_fmt_list)) if uniq_tree_fmt_list else float("nan"),
            top1_tree_fmt_mean=float(np.mean(top1_tree_fmt_list)) if top1_tree_fmt_list else float("nan"),
        )
        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    df["step"] = df["checkpoint"].apply(step_of)
    df = df.sort_values("step").reset_index(drop=True)

    out_csv = OUT_DIR / f"entrance_entropy_countdown_entropy_n{args.n_samples}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[entrance_entropy] wrote: {out_csv}")
    print(df[
        [
            "checkpoint",
            "step",
            "rho_mean",
            "rho_var",
            "rho_mass0",
            "rho_mass1",
            "HA_mean",
            "HYgA_mean",
            "HYgA_opseq_mean",
            "HYgA_tree_mean",
            "uniqueA_mean",
            "top1_mass_mean",
            "unique_opseq_mean",
            "top1_opseq_mean",
            "unique_tree_mean",
            "top1_tree_mean",
            "format_rate_mean",
            "HA_fmt_mean",
            "HYgA_fmt_mean",
            "uniqueA_fmt_mean",
            "top1_mass_fmt_mean",
            "HYgA_opseq_fmt_mean",
            "HYgA_tree_fmt_mean",
        ]
    ])

    if args.save_per_problem:
        out_parq = OUT_DIR / f"entrance_entropy_countdown_per_problem_n{args.n_samples}.parquet"
        pd.DataFrame(per_problem_rows).to_parquet(out_parq, index=False)
        print(f"[entrance_entropy] wrote per-problem: {out_parq}")


# ---- merged stratified_sets mode ----
"""Stratify Countdown branch metrics by S_loss/S_both/S_gain sets."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from early_branch_locking.core.countdown_shared import (  # noqa: E402
    entropy_from_counts,
    enumerate_solution_set,
    extract_ground_truth,
    load_jsonl,
    load_parquet_sorted,
    pass_at_k,
    tolerant_parse_completion,
)

OPS = set("+-*/")
PASS_KS = (1, 64, 256)


def parse_args_stratified_sets() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Op1 coverage on S_loss/S_both/S_gain.")
    parser.add_argument("--sets_path", default=str(METRICS_DIR / "branch_set_collection_sets_global_step_50_to_global_step_275_n320.json"))
    parser.add_argument("--raw_paths", nargs="+", default=[str(RAW_DIR / "countdown_raw_global_step_50_n320.jsonl"), str(RAW_DIR / "countdown_raw_global_step_275_n320.jsonl")])
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--survival_path", default=str(METRICS_DIR / "branch_survival_branch_survival_per_problem_branch_survival_v1.parquet"))
    return parser.parse_args()


def first_op(expr: str | None) -> str:
    for char in expr or "":
        if char in OPS:
            return char
    return ""


def load_sets(path: Path) -> Dict[str, List[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: [int(pid) for pid in data.get(name, [])] for name in ("S_loss", "S_both", "S_gain")}


def build_valid_ops(num_problems: int) -> Dict[int, set]:
    records = load_parquet_sorted(TEST_PARQUET, n=num_problems, sort_key="sample_id")
    out = {}
    for pid, record in enumerate(records):
        numbers, target, feasible = extract_ground_truth(record)
        solutions = enumerate_solution_set(numbers, target) if feasible == "yes" else set()
        out[pid] = {first_op(expr) for expr in solutions if first_op(expr)}
    return out


def raw_by_problem(path: Path, num_problems: int) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in load_jsonl(path):
        pid = int(row["problem_index"])
        if pid < num_problems:
            grouped[pid].append(row)
    return grouped


def problem_metrics(rows: Sequence[dict], valid_ops: set) -> dict:
    n = len(rows)
    correct = [row for row in rows if row.get("overall_ok")]
    canons = {row.get("canonical_expr") for row in correct if row.get("canonical_expr")}
    op_counts = Counter(first_op(row.get("canonical_expr")) for row in correct if first_op(row.get("canonical_expr")))
    generated_ops = generated_op_counts(rows)
    op_hit = set(op_counts)
    generated_op_hit = set(generated_ops) & valid_ops
    return {
        "n_samples": n,
        **{f"pass{k}": pass_at_k(n, len(correct), k) for k in PASS_KS},
        "format_ok_rate": sum(row.get("parse_status") == "OK" for row in rows) / n if n else math.nan,
        "exact_unique": len(canons),
        "exact_coverage": math.nan,
        "op1_unique": len(op_hit),
        "op1_coverage": len(op_hit) / len(valid_ops) if valid_ops else math.nan,
        "op1_entropy": entropy_from_counts(op_counts) if op_counts else 0.0,
        "generated_op1_coverage": len(generated_op_hit) / len(valid_ops) if valid_ops else math.nan,
        "generated_op1_unique": len(generated_op_hit),
        "generated_op1_entropy": entropy_from_counts(generated_ops) if generated_ops else 0.0,
        "correct_rate": len(correct) / n if n else math.nan,
    }


def generated_op_counts(rows: Sequence[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        answer = tolerant_parse_completion(row.get("completion", "")).get("answer_block", "")
        op = first_op(answer)
        if op:
            counts[op] += 1
    return counts


def add_exact_coverage(metric: dict, rows: Sequence[dict], solution_count: int) -> dict:
    if solution_count <= 0:
        metric["exact_coverage"] = math.nan
        return metric
    canons = {row.get("canonical_expr") for row in rows if row.get("overall_ok") and row.get("canonical_expr")}
    metric["exact_coverage"] = len(canons) / solution_count
    return metric


def summarize_checkpoint(path: Path, sets: Dict[str, List[int]], valid_ops: Dict[int, set], solution_counts: Dict[int, int]) -> List[dict]:
    grouped = raw_by_problem(path, max(valid_ops) + 1)
    checkpoint = path.name.removeprefix("countdown_raw_").removesuffix("_n320.jsonl")
    rows = []
    for set_name, pids in sets.items():
        per_problem = []
        for pid in pids:
            metric = problem_metrics(grouped.get(pid, []), valid_ops.get(pid, set()))
            per_problem.append(add_exact_coverage(metric, grouped.get(pid, []), solution_counts.get(pid, 0)))
        rows.append(aggregate_rows(checkpoint, set_name, len(pids), per_problem))
    return rows


def aggregate_rows(checkpoint: str, set_name: str, n_problems: int, metrics: List[dict]) -> dict:
    row = {"checkpoint": checkpoint, "problem_set": set_name, "num_problems": n_problems}
    keys = [
        "pass1", "pass64", "pass256", "format_ok_rate", "correct_rate",
        "exact_coverage", "exact_unique", "op1_coverage", "op1_unique",
        "op1_entropy", "generated_op1_coverage", "generated_op1_unique",
        "generated_op1_entropy",
    ]
    for key in keys:
        values = [item[key] for item in metrics if not pd.isna(item[key])]
        row[f"{key}_mean"] = float(np.mean(values)) if values else math.nan
    return row


def solution_counts(num_problems: int) -> Dict[int, int]:
    records = load_parquet_sorted(TEST_PARQUET, n=num_problems, sort_key="sample_id")
    counts = {}
    for pid, record in enumerate(records):
        numbers, target, feasible = extract_ground_truth(record)
        counts[pid] = len(enumerate_solution_set(numbers, target)) if feasible == "yes" else 0
    return counts


def survival_summary(path: Path, sets: Dict[str, List[int]]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    rows = []
    for set_name, pids in sets.items():
        sub = df[df["problem_index"].isin(pids)]
        for (model, mode), group in sub.groupby(["model_name", "prefix_mode"], sort=False):
            rows.append(
                {
                    "checkpoint": model,
                    "problem_set": set_name,
                    "prefix_mode": mode,
                    "survival_any_valid_mean": float(group["any_valid_survival_rate"].mean()),
                    "survival_same_branch_mean": float(group["same_branch_survival_rate"].mean()),
                }
            )
    return pd.DataFrame(rows)


def run_stratified_sets() -> None:
    args = parse_args_stratified_sets()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sets = load_sets(Path(args.sets_path))
    valid = build_valid_ops(args.num_problems)
    sol_counts = solution_counts(args.num_problems)
    rows = []
    for raw_path in args.raw_paths:
        rows.extend(summarize_checkpoint(Path(raw_path), sets, valid, sol_counts))
    summary = pd.DataFrame(rows)
    survival = survival_summary(Path(args.survival_path), sets)
    out = METRICS_DIR / "entrance_entropy2_sloss_sboth_sgain_op1_comparison.csv"
    summary.to_csv(out, index=False)
    if not survival.empty:
        survival.to_csv(METRICS_DIR / "entrance_entropy2_sloss_sboth_sgain_survival_comparison.csv", index=False)
    print(summary.to_string(index=False), flush=True)

def _run_selected():
    selector = None
    selector_index = None
    flag = "--stratify-sets"
    for index, argument in enumerate(sys.argv):
        if argument == flag:
            selector_index = index
            selector = sys.argv[index + 1] if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--") else "__flag__"
            break
        if argument.startswith(flag + "="):
            selector_index = index
            selector = argument.split("=", 1)[1]
            break
    if selector_index is not None:
        if selector == "__flag__":
            sys.argv.pop(selector_index)
        else:
            del sys.argv[selector_index:selector_index + 2]
        return run_stratified_sets()
    return main()

if __name__ == "__main__":
    _run_selected()
