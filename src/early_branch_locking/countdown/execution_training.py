#!/usr/bin/env python3
"""Collect the conditional-execution training curve.

Hypothesis: once a solution-family entrance is supplied, L3/L4 execution is
retained or strengthened through training, while the wrong-value countdown_rlvr control
stays low. Inputs: the fixed Countdown test order, retry-prefix ledger, and
actor checkpoints at steps 25/100/200/250. Outputs: an execution_training raw JSONL matrix,
``exec_training_curve_v1.csv``, ``fig1_fork_data_v1.csv``, a specialized
ledger, and a manifest. Log: ``internal experiment log`` under the execution_training entry.
Status: GPU collection plus CPU aggregation; harness failures remain
diagnostic and never delete sampled rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR, METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    build_prompt_text,
    evaluate_countdown_completion,
    extract_ground_truth,
    get_prompt_content,
    load_jsonl,
    load_parquet_sorted,
    pass_at_k,
)
from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion  # noqa: E402
from early_branch_locking.countdown.state_staircase import _render_rung, raw_answer_family  # noqa: E402


CHECKPOINTS = (25, 100, 200, 250)
RUNGS = ("L0", "L3", "L4", "countdown_rlvr")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 1729


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def bootstrap_mean(values, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "aggregate"), default="prepare")
    parser.add_argument("--num-problems", type=int, default=150)
    parser.add_argument("--ledger-path", type=Path, default=METRICS_DIR / "state_staircase_ledger_main_ends.jsonl")
    parser.add_argument("--e2-ledger-path", type=Path, default=METRICS_DIR / "state_staircase_ledger_e2_retry_v1.jsonl")
    parser.add_argument("--raw-out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--checkpoint", type=int, choices=CHECKPOINTS, default=25)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--n-continuations", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-ledger-rows", type=int, default=0)
    parser.add_argument("--tag", default="v1",
                        help="provenance suffix for isolated reruns; v1 preserves maintained filenames")
    return parser.parse_args(argv)


def tagged_suffix(tag):
    return "v1" if str(tag) == "v1" else str(tag)


def prepare(args):
    rows = list(load_jsonl(args.ledger_path))
    candidates = [
        row for row in rows
        if str(row.get("scaffold_type")) == "retry" and str(row.get("rung")) == "L0"
    ]
    selected = {}
    for row in sorted(candidates, key=lambda item: (int(item["problem_index"]), str(item["target_family"]), str(item["prefix_id"]))):
        selected.setdefault(int(row["problem_index"]), row)
    if len(selected) < args.num_problems:
        missing = sorted(set(range(args.num_problems)) - set(selected))
        raise ValueError(f"execution_training ledger lacks retry L0 rows for problems: {missing}")

    output = []
    for pid in range(args.num_problems):
        base = dict(selected[pid])
        for rung in RUNGS:
            row = dict(base)
            row["rung"] = rung
            row["rung_text"] = _render_rung(rung, base["witness_render"], base["numbers"])
            row["prefix_id"] = f"{pid}_{base['target_family']}_retry_{rung}_e2"
            row["state_dose_tokens"] = len((row["scaffold_text"] + row["rung_text"]).split())
            output.append(row)
    args.e2_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with args.e2_ledger_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    manifest = {
        "artifact": "execution_training",
        "stage": "prepare",
        "checkpoints": list(CHECKPOINTS),
        "rungs": list(RUNGS),
        "scaffold_policy": "one deterministic retry scaffold and one target family per problem",
        "n_problems": args.num_problems,
        "ledger_rows": len(output),
        "source_ledger": str(args.ledger_path),
        "tag": args.tag,
        "status": "prepared",
    }
    (args.out_dir / "e2_execution_manifest_v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(args.e2_ledger_path), "rows": len(output), "problems": len(selected)}, sort_keys=True))


def _model_path(args):
    return args.model_path or (COUNTDOWN_ACTOR_DIR / f"global_step_{args.checkpoint}")


def run(args):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = _model_path(args)
    label = args.model_label or f"global_step_{args.checkpoint}"
    ledger = list(load_jsonl(args.e2_ledger_path))
    if args.max_ledger_rows > 0:
        ledger = ledger[:args.max_ledger_rows]
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_ids = {
        pid: tokenizer.encode(build_prompt_text(get_prompt_content(record), tokenizer), add_special_tokens=False)
        for pid, record in enumerate(records)
    }
    requests = []
    for row in ledger:
        prefix = str(row["scaffold_text"]) + ("\n" if row["scaffold_text"] and row["rung_text"] else "") + str(row["rung_text"])
        requests.append({
            "prompt_token_ids": prompt_ids[int(row["problem_index"])]
            + tokenizer.encode(prefix, add_special_tokens=False)
        })
    llm = LLM(model=str(model), tensor_parallel_size=1, gpu_memory_utilization=0.88,
              trust_remote_code=True, seed=args.seed, max_model_len=4096)
    params = SamplingParams(n=args.n_continuations, temperature=args.temperature,
                            top_p=args.top_p, max_tokens=args.max_new_tokens,
                            seed=args.seed,
                            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None)
    outputs = llm.generate(requests, params)
    path = args.raw_out_dir / f"state_staircase_raw_e2_{label}_{tagged_suffix(args.tag)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, result in zip(ledger, outputs):
            prefix = str(row["scaffold_text"]) + ("\n" if row["scaffold_text"] and row["rung_text"] else "") + str(row["rung_text"])
            for index, sequence in enumerate(result.outputs):
                continuation = sequence.text or ""
                full = prefix + continuation
                evaluation = evaluate_countdown_completion(
                    full, row["numbers"], row["target"], row["feasible_label"],
                    parse_countdown_completion, evaluate_countdown_expression,
                )
                family, family_source = raw_answer_family(full)
                item = {
                    **row,
                    "checkpoint": int(args.checkpoint),
                    "model_label": label,
                    "continuation_index": index,
                    "continuation": continuation,
                    "generated_tokens": len(getattr(sequence, "token_ids", ()) or ()),
                    "sampling_seed": args.seed,
                    "sampling_mode": "vllm_batch_n16",
                    "any_valid": bool(evaluation.overall_ok),
                    "canonical_expr": evaluation.canonical_expr or "",
                    "observed_family": family,
                    "family_source": family_source,
                    "in_family": bool(evaluation.overall_ok and family == str(row["target_family"])),
                }
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    if hasattr(llm, "shutdown"):
        llm.shutdown()
    config = {
        "artifact": "execution_training",
        "checkpoint": args.checkpoint,
        "model": str(model),
        "raw_path": str(path),
        "n_problems": args.num_problems,
        "raw_rows": len(ledger) * args.n_continuations,
        "sampling": {"n": args.n_continuations, "temperature": args.temperature,
                     "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                     "seed": args.seed},
    }
    config["tag"] = args.tag
    config["harness_gate"] = {
        "status": "deferred_to_aggregate",
        "reason": "L0 retry-prefix diagnostic is computed after all checkpoint cells are available",
    }
    (args.out_dir / f"e2_config_step{args.checkpoint}_{tagged_suffix(args.tag)}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, sort_keys=True))


def _load_e2_raw(args):
    frames = []
    paths = []
    for checkpoint in CHECKPOINTS:
        path = args.raw_out_dir / f"state_staircase_raw_e2_global_step_{checkpoint}_{tagged_suffix(args.tag)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
        frame = pd.read_json(path, lines=True)
        frame["any_valid"] = frame["any_valid"].map(parse_bool)
        frame["in_family"] = frame["in_family"].map(parse_bool)
        frame["n_numbers"] = frame["numbers"].map(len)
        frame["generated_tokens"] = pd.to_numeric(frame["generated_tokens"], errors="coerce").fillna(0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), paths


def _reference_pass1(checkpoint):
    path = RAW_DIR / f"countdown_raw_global_step_{checkpoint}_n320.jsonl"
    frame = pd.read_json(path, lines=True)
    valid = frame["overall_ok"].map(parse_bool)
    return float(valid.groupby(frame["problem_index"]).mean().mean())


def _a1_shared_anchor():
    """Load A1's maintained shared-curve anchor for continuity checks."""
    path = METRICS_DIR / "state_staircase_dose_curve_shared_v1.csv"
    if not path.exists():
        return {"path": str(path), "available": False}
    frame = pd.read_csv(path)
    return {
        "path": str(path),
        "available": True,
        "intersection_n": int(frame["intersection_n"].iloc[0]),
        "rungs": sorted(frame["rung"].astype(str).unique().tolist()),
        "checkpoints": sorted(frame["checkpoint"].astype(int).unique().tolist()),
    }


def _shared_e2_problem_ids(frame):
    """Return the complete all-cell intersection, separately by number stratum."""
    cells = set(zip(frame["checkpoint"].astype(int), frame["rung"].astype(str)))
    required = {(checkpoint, rung) for checkpoint in CHECKPOINTS for rung in RUNGS}
    if cells != required:
        raise ValueError(f"execution_training cells do not match required matrix: missing={sorted(required - cells)}")
    ids = {}
    for number_policy, subset in [
        ("all_numbers", frame),
        ("primary_4_number", frame[frame["n_numbers"].eq(4)]),
    ]:
        sets = []
        for checkpoint, rung in required:
            group = subset[(subset.checkpoint.eq(checkpoint)) & subset.rung.eq(rung)]
            sets.append(set(group.problem_index.astype(int).unique()))
        ids[number_policy] = set.intersection(*sets) if sets else set()
    return ids


def aggregate(args):
    frame, raw_paths = _load_e2_raw(args)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per = frame.groupby(["checkpoint", "rung", "problem_index"], as_index=False).agg(
        in_family=("in_family", "mean"), any_valid=("any_valid", "mean"),
        n=("in_family", "size"), n_numbers=("n_numbers", "first"),
    )
    l0 = per[per.rung.eq("L0")][["checkpoint", "problem_index", "in_family"]].rename(
        columns={"in_family": "l0_in_family"})
    per = per.merge(l0, on=["checkpoint", "problem_index"], how="left", validate="many_to_one")
    per["excess_over_own_L0"] = per["in_family"] - per["l0_in_family"]
    shared_ids = _shared_e2_problem_ids(per.assign(n_numbers=per["n_numbers"]))
    per["shared_all_cells"] = per.apply(
        lambda row: int(int(row.problem_index) in shared_ids[
            "primary_4_number" if int(row.n_numbers) == 4 else "all_numbers"
        ]), axis=1,
    )
    curve_rows = []
    policy_frames = [
        ("primary_4_number", per[per.n_numbers.eq(4)]),
        ("all_numbers", per),
    ]
    for policy, policy_frame in policy_frames:
        for (checkpoint, rung), group in policy_frame.groupby(
            ["checkpoint", "rung"], sort=False
        ):
            row = {"checkpoint": int(checkpoint), "rung": rung, "number_policy": policy,
                   "n_problems": int(group.problem_index.nunique()),
                   "n_sampled_rows": int(group.n.sum()), "min_continuations_per_problem": int(group.n.min()),
                   "max_continuations_per_problem": int(group.n.max())}
            for metric in ("in_family", "any_valid", "excess_over_own_L0"):
                mean, lo, hi = bootstrap_mean(group[metric], rng)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_lo"] = lo
                row[f"{metric}_ci_hi"] = hi
            curve_rows.append(row)
    curve = pd.DataFrame(curve_rows)
    curve["shared_intersection_n"] = curve.apply(
        lambda row: len(shared_ids[str(row.number_policy)]), axis=1
    )
    curve["intersection_policy"] = "all checkpoint/rung cells; reported strata-specific shared IDs"
    cost = pd.read_csv(METRICS_DIR / "state_staircase_tf_cost_normalized_v1.csv")
    cost = cost.groupby("checkpoint", as_index=False).agg(
        entrance_per_token=("entrance_per_token", "mean"),
        execution_per_token=("execution_per_token", "mean"),
    )
    curve = curve.merge(cost, on="checkpoint", how="left")
    curve["cost_source"] = "state_staircase_tf_cost_normalized_v1.csv"
    curve.to_csv(args.out_dir / "exec_training_curve_v1.csv", index=False)
    per.to_csv(args.out_dir / "exec_training_per_problem_v1.csv", index=False)

    fork_rows = []
    for row in cost.itertuples():
        fork_rows.extend([
            {"panel": "entrance_cost", "series": "entrance_per_token", "checkpoint": int(row.checkpoint), "x": int(row.checkpoint), "y": float(row.entrance_per_token), "source": "state_staircase_tf_cost_normalized_v1.csv"},
            {"panel": "entrance_cost", "series": "execution_per_token", "checkpoint": int(row.checkpoint), "x": int(row.checkpoint), "y": float(row.execution_per_token), "source": "state_staircase_tf_cost_normalized_v1.csv"},
        ])
    plot_curve = curve[(curve.number_policy == "all_numbers") & curve.rung.isin(["L3", "L4", "countdown_rlvr"])]
    for row in plot_curve.itertuples():
        fork_rows.append({"panel": "conditional_execution", "series": f"{row.rung}_excess", "checkpoint": int(row.checkpoint), "x": int(row.checkpoint), "y": float(row.excess_over_own_L0_mean), "ci_lo": float(row.excess_over_own_L0_ci_lo), "ci_hi": float(row.excess_over_own_L0_ci_hi), "source": "exec_training_curve_v1.csv"})
    pd.DataFrame(fork_rows).to_csv(args.out_dir / "fig1_fork_data_v1.csv", index=False)

    gate = {}
    for checkpoint in CHECKPOINTS:
        l0 = frame[(frame.checkpoint.eq(checkpoint)) & frame.rung.eq("L0")]
        observed = float(l0.groupby("problem_index").any_valid.mean().mean())
        reference = _reference_pass1(checkpoint)
        gate[str(checkpoint)] = {"observed_l0_retry_pass1": observed, "reference_unconditioned_pass1": reference,
                                 "difference": observed - reference, "tolerance": 0.03,
                                 "passed": abs(observed - reference) <= 0.03}
    manifest = {
        "artifact": "execution_training", "stage": "aggregate", "tag": args.tag,
        "checkpoints": list(CHECKPOINTS), "rungs": list(RUNGS),
        "sampling": {"n": args.n_continuations, "temperature": args.temperature, "top_p": args.top_p,
                     "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "raw_paths": [str(path) for path in raw_paths], "raw_rows": int(len(frame)),
        "eligible_problem_count": int(frame.problem_index.nunique()),
        "four_number_problem_count": int(frame.loc[frame.n_numbers.eq(4), "problem_index"].nunique()),
        "intersection_policy": "all selected checkpoint/rung cells; one retry target-family prefix per problem",
        "shared_intersection": {
            "all_numbers_n": len(shared_ids["all_numbers"]),
            "primary_4_number_n": len(shared_ids["primary_4_number"]),
            "problem_ids_not_imputed": True,
        },
        "a1_anchor": _a1_shared_anchor(),
        "continuity_check": {
            "reference": "A1 shared curve; execution_training uses a different checkpoint matrix and retry scaffold, so no numeric identity is asserted",
            "reproduction_tolerance": 0.03,
            "status": "not_applicable_without_shared checkpoints 50/150/275 and shared A1 raw cells",
        },
        "harness_gate": gate,
        "status": "complete; L0 gate is diagnostic because it is retry-prefix conditioned",
    }
    (args.out_dir / "exec_training_manifest_v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Keep the registry's maintained manifest synchronized with aggregate mode;
    # prepare mode may have written an earlier stage marker to this path.
    (args.out_dir / "e2_execution_manifest_v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"curve_rows": len(curve), "raw_rows": len(frame), "harness_gate": gate}, sort_keys=True))


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
