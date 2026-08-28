#!/usr/bin/env python3
"""State-staircase intervention for plan re-entry.

Hypothesis: restoring external state (from a cue through a completed first
calculation) produces a dose-response curve whose half-recovery price moves
with RLVR training. Inputs: test.parquet, prefix_recovery scaffold/menu files, and
checkpoint models. Outputs: a solver/template ledger, per-instance raw CSV,
summary/curve CSVs, and optional vLLM raw JSONL. Log: internal experiment log
section anchored by ``state_staircase-state-staircase``. Status: prepare/aggregate are
CPU-complete; run/score are GPU modes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from early_branch_locking._repo import METRICS_DIR, RAW_DIR, COUNTDOWN_ACTOR_DIR, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.countdown_shared import (build_prompt_text, enumerate_solution_set,
    evaluate_countdown_completion, extract_ground_truth, get_prompt_content, load_jsonl,
    load_parquet_sorted, pass_at_k, tolerant_parse_completion)  # noqa: E402
from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion  # noqa: E402
from early_branch_locking.countdown.prefix_splice_recovery import (  # noqa: E402
    answer_family_with_source,
    branch_menu,
    minimal_render,
    FIRST_TRIAL_RE,
)

RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L6b", "L7", "countdown_rlvr", "C4")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "run", "score", "aggregate"), default="prepare")
    p.add_argument("--num-problems", type=int, default=150)
    p.add_argument("--scaffold-path", type=Path, default=METRICS_DIR / "prefix_recovery_scaffolds_splice_v1.jsonl")
    p.add_argument("--menu-path", type=Path, default=METRICS_DIR / "prefix_recovery_menu_splice_v1.csv")
    p.add_argument("--ledger-path", type=Path, default=METRICS_DIR / "state_staircase_ledger_v1.jsonl")
    p.add_argument("--target-ledger", type=Path, default=None,
                   help="CSV/JSONL of explicit (problem, family, checkpoint) targets for targeted sampling")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="checkpoint row to select from --target-ledger")
    p.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    p.add_argument("--raw-out-dir", type=Path, default=RAW_DIR)
    p.add_argument("--model-path", type=Path, default=COUNTDOWN_ACTOR_DIR / "global_step_275")
    p.add_argument("--model-label", default="global_step_275")
    p.add_argument("--checkpoints", default="50,150,275", help="comma-separated checkpoint steps for score mode")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--n-continuations", type=int, default=16)
    p.add_argument("--max-ledger-rows", type=int, default=0, help="positive value limits GPU smoke runs")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--tag", default="v1")
    p.add_argument("--raw-tag", default=None, help="raw filename tag when aggregate output uses a separate tag")
    p.add_argument("--raw-path", type=Path, action="append", default=[], help="explicit raw JSONL input; repeat to combine checkpoint/control files")
    p.add_argument("--rungs", default=",".join(RUNGS), help="comma-separated rung subset for prepare mode")
    p.add_argument("--scaffold-types", default="retry,neutral",
                   help="comma-separated scaffold subset for prepare mode")
    p.add_argument("--gate-scores", type=Path, default=None,
                   help="optional T2 gate CSV keyed by prefix_id; lexical template gate is used otherwise")
    p.add_argument("--refamily-from-raw", action=argparse.BooleanOptionalAction, default=True,
                   help="recompute family from raw answer text (default: enabled)")
    p.add_argument("--problem-intersection", action="store_true", help="retain problems gated pass at every selected rung")
    return p.parse_args(argv)


def _numbers_in(text):
    return re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", str(text))


def parse_bool(value) -> bool:
    """Parse booleans from JSON values and pandas/string CSV fields."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _render_rung(rung, witness, numbers):
    ent = re.match(r"\s*(\(?\d+\)?\s*[+\-*/])", witness)
    entrance = ent.group(1).strip() if ent else ""
    vals = _numbers_in(witness)
    first = vals[0] if vals else str(numbers[0])
    second = vals[1] if len(vals) > 1 else str(numbers[1])
    op = re.search(r"[+\-*/]", entrance).group(0) if re.search(r"[+\-*/]", entrance) else "+"
    try:
        value = int(eval(f"{first}{op}{second}", {"__builtins__": {}}, {}))
    except Exception:
        value = 0
    if rung == "L0": return ""
    if rung == "L1": return "Let me try a different combination:"
    if rung == "L2": return f"Let me try starting with {first}:"
    if rung == "L3": return f"Let me try: {first} {op}"
    if rung == "L4": return f"Let me try: {first} {op} {second} = {value}."
    if rung == "L5": return f"Let me try: {first} {op} {second} = {value}, and then with the remaining numbers:"
    if rung == "L6": return "Plan: combine a pair first, then apply the remaining operations one by one."
    if rung == "L6b": return f"Plan: combine {first} and {second}, then finish using the remaining numbers."
    if rung == "countdown_rlvr": return f"Let me try: {first} {op} {second} = {value + 1}."
    if rung == "C4": return f"Plan: combine the numbers in a different order, then undo the first operation."
    return witness


def _gate(text, witness, target_family):
    norm = re.sub(r"\s+", "", text).lower()
    lexical = re.sub(r"\s+", "", witness).lower() in norm if witness else False
    return "fail" if lexical else "pass"


def _expr_family(expr):
    match = re.search(r"(?<!\d)(\d+)\s*([+\-*/])", str(expr or ""))
    return f"{int(match.group(1))}{match.group(2)}" if match else ""


def raw_answer_family(full_text):
    """Extract the first trial family from the answer text."""
    family, source = answer_family_with_source(full_text)
    if family is None:
        return "", source
    return f"{family[0]}{family[1]}", source


def prepare(args):
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    scaffolds = list(load_jsonl(args.scaffold_path))
    menu = pd.read_csv(args.menu_path)
    by_pid = {}
    for row in scaffolds:
        by_pid.setdefault(int(row["problem_index"]), row)
    rows = []
    selected_rungs = tuple(x.strip() for x in args.rungs.split(",") if x.strip())
    selected_scaffolds = tuple(x.strip() for x in args.scaffold_types.split(",") if x.strip())
    unknown = sorted(set(selected_rungs) - set(RUNGS))
    if unknown:
        raise ValueError(f"unknown rungs: {unknown}")
    unknown_scaffolds = sorted(set(selected_scaffolds) - {"retry", "neutral"})
    if unknown_scaffolds:
        raise ValueError(f"unknown scaffold types: {unknown_scaffolds}")
    gate_map = {}
    if args.gate_scores and args.gate_scores.exists():
        gate_frame = pd.read_csv(args.gate_scores)
        if "prefix_id" not in gate_frame or "gate_status" not in gate_frame:
            raise ValueError("--gate-scores must contain prefix_id and gate_status")
        gate_map = {str(row.prefix_id): str(row.gate_status) for row in gate_frame.itertuples()}
    if args.target_ledger is not None:
        if args.checkpoint_step is None:
            raise ValueError("--target-ledger requires --checkpoint-step")
        if not args.target_ledger.exists():
            raise FileNotFoundError(args.target_ledger)
        if args.target_ledger.suffix.lower() == ".csv":
            target_rows = pd.read_csv(args.target_ledger).to_dict("records")
        else:
            target_rows = list(load_jsonl(args.target_ledger))
        target_rows = [
            row for row in target_rows
            if int(row.get("checkpoint_step", -1)) == int(args.checkpoint_step)
        ]
        if not target_rows:
            raise ValueError(f"no target rows for checkpoint {args.checkpoint_step} in {args.target_ledger}")
        for row in target_rows:
            pid = int(row["problem_index"])
            if pid < 0 or pid >= len(records) or pid not in by_pid:
                raise ValueError(f"target row has no matching test/scaffold record: {row}")
            numbers, target, feasible = extract_ground_truth(records[pid])
            if str(feasible).lower() != "yes" or len(numbers) not in (3, 4):
                raise ValueError(f"target row is not a feasible 3/4-number problem: {row}")
            scaffold = by_pid[pid]
            target_family = str(row["target_family"])
            witness = str(row["witness_render"])
            valid_menu = menu[(menu.problem_index == pid)
                              & (menu.branch.astype(str) == target_family)
                              & menu.feasible.astype(str).str.lower().eq("true")]
            if valid_menu.empty:
                raise ValueError(f"target family is absent from feasible menu: pid={pid}, family={target_family}")
            neutral = "The numbers are " + ", ".join(map(str, numbers)) + ". Let me think about which numbers to combine first."
            scaffold_values = {
                "retry": str(scaffold.get("prefix_text", "")),
                "neutral": neutral,
            }
            scaffold_type = str(row["scaffold_type"])
            rung = str(row["rung"])
            if scaffold_type not in scaffold_values or rung not in selected_rungs:
                raise ValueError(f"target row outside selected scaffold/rung set: {row}")
            scaffold_text = scaffold_values[scaffold_type]
            rung_text = _render_rung(rung, witness, numbers)
            prefix_id = f"{pid}_{target_family}_{scaffold_type}_{rung}"
            lexical_gate = _gate(scaffold_text + rung_text, witness, target_family)
            solutions = [minimal_render(expr) for expr in enumerate_solution_set(numbers, target)]
            rows.append({"prefix_id": prefix_id, "problem_index": pid,
                         "numbers": list(map(int, numbers)), "target": int(target), "feasible_label": str(feasible),
                         "target_family": target_family, "witness_render": witness, "target_witness": witness,
                         "solution_renders": solutions, "scaffold_type": scaffold_type,
                         "scaffold_text": scaffold_text, "rung": rung, "rung_text": rung_text,
                         "checkpoint_step": int(args.checkpoint_step),
                         "gate_status": (str(row.get("gate_status"))
                                         if str(row.get("gate_status", "")).lower() in {"pass", "soft", "fail"}
                                         else lexical_gate),
                         "gate_method": "target_ledger_or_lexical_template_fallback",
                         "state_dose_tokens": len((scaffold_text + rung_text).split()),
                         "chance": float(row.get("chance", valid_menu.iloc[0].get("chance", 0.0)))})
        if len(rows) != len(target_rows):
            raise AssertionError(f"target row expansion mismatch: {len(rows)} != {len(target_rows)}")
    else:
        for pid, rec in enumerate(records):
            numbers, target, feasible = extract_ground_truth(rec)
            if str(feasible).lower() != "yes" or len(numbers) not in (3, 4):
                continue
            fams = menu[(menu.problem_index == pid) & (menu.feasible.astype(str).str.lower() == "true")]
            if fams.empty or pid not in by_pid:
                continue
            fams = fams.sort_values(["n_solutions_in_family", "branch"], ascending=[False, True])
            # One observed-at-50 proxy and one alternative family. The raw sets
            # are deliberately not used to choose separate problem collections.
            chosen = fams.head(2).to_dict("records")
            scaffold = by_pid[pid]
            retry = str(scaffold.get("prefix_text", ""))
            neutral = "The numbers are " + ", ".join(map(str, numbers)) + ". Let me think about which numbers to combine first."
            for fam in chosen:
                witness = str(fam["witness_render"])
                target_family = str(fam["branch"])
                all_scaffolds = (("retry", retry), ("neutral", neutral))
                for scaffold_type, scaffold_text in all_scaffolds:
                    if scaffold_type not in selected_scaffolds:
                        continue
                    for rung in selected_rungs:
                        text = _render_rung(rung, witness, numbers)
                        prefix_id = f"{pid}_{target_family}_{scaffold_type}_{rung}"
                        lexical_gate = _gate(scaffold_text + text, witness, target_family)
                        rows.append({"prefix_id": prefix_id, "problem_index": pid,
                                     "numbers": list(map(int, numbers)), "target": int(target), "feasible_label": str(feasible),
                                     "target_family": target_family, "witness_render": witness, "scaffold_type": scaffold_type,
                                     "scaffold_text": scaffold_text, "rung": rung, "rung_text": text,
                                     "gate_status": gate_map.get(prefix_id, lexical_gate),
                                     "gate_method": "t2_judge_join" if prefix_id in gate_map else "lexical_template_fallback",
                                     "state_dose_tokens": len((scaffold_text + text).split()),
                                     "chance": float(fam.get("chance", 0.0))})
    args.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger_path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "problems": len({r['problem_index'] for r in rows}), "ledger": str(args.ledger_path)}, sort_keys=True))


def run(args):
    # Respect a controller's launch-time GPU isolation.  Reassigning an
    # inherited visibility mask inside a worker can collapse independently
    # launched GPU0/GPU1 jobs onto the same physical device.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    ledger = list(load_jsonl(args.ledger_path))
    if args.max_ledger_rows > 0:
        ledger = ledger[:args.max_ledger_rows]
    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_ids = {
        pid: tok.encode(build_prompt_text(get_prompt_content(rec), tok), add_special_tokens=False)
        for pid, rec in enumerate(records)
    }
    prompts = []
    for row in ledger:
        prefix = row["scaffold_text"] + ("\n" if row["scaffold_text"] and row["rung_text"] else "") + row["rung_text"]
        prompts.append(prompt_ids[int(row["problem_index"])] + tok.encode(prefix, add_special_tokens=False))
    llm = LLM(model=str(args.model_path), tensor_parallel_size=1, gpu_memory_utilization=0.88, trust_remote_code=True, seed=1729)
    requests = [{"prompt_token_ids": ids} for ids in prompts]
    out = llm.generate(requests, SamplingParams(n=args.n_continuations, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens))
    raw_path = args.raw_out_dir / f"state_staircase_raw_{args.tag}_{args.model_label}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as f:
        for row, result in zip(ledger, out):
            for j, seq in enumerate(result.outputs):
                # Score the same scaffold+rung transcript that was sent to the model.
                full = row["scaffold_text"] + row["rung_text"] + (seq.text or "")
                ev = evaluate_countdown_completion(full, row["numbers"], row["target"], row["feasible_label"], parse_countdown_completion, evaluate_countdown_expression)
                family, family_source = raw_answer_family(row["scaffold_text"] + row["rung_text"] + (seq.text or ""))
                f.write(json.dumps({**row, "model_label": args.model_label, "continuation_index": j, "continuation": seq.text or "", "any_valid": bool(ev.overall_ok), "canonical_expr": ev.canonical_expr, "in_family": bool(ev.overall_ok and family == str(row["target_family"])), "observed_family": family, "family_source": family_source}) + "\n")
    if hasattr(llm, "shutdown"): llm.shutdown()
    print(json.dumps({"raw": str(raw_path), "rows": len(ledger) * args.n_continuations}, sort_keys=True))


def aggregate(args):
    raw_tag = args.raw_tag or args.tag
    paths = list(args.raw_path) or sorted(args.raw_out_dir.glob(f"state_staircase_raw_{raw_tag}_*.jsonl"))
    rows = [r for p in paths for r in load_jsonl(p)]
    if not rows: raise FileNotFoundError("no state_staircase raw files; run --mode run")
    frame = pd.DataFrame(rows)
    required = {"continuation", "any_valid", "target_family", "problem_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"incomplete state_staircase raw input missing {missing}; use v3/v4 raw files")
    frame["continuation"] = frame["continuation"].fillna("").astype(str)
    frame["any_valid"] = frame["any_valid"].map(parse_bool)
    # A historical runner wrote only ledger metadata and defaulted every score
    # to false. Refuse that file instead of publishing a fabricated zero curve.
    nonempty = frame["continuation"].str.strip().ne("")
    has_answer = frame["continuation"].str.contains(r"<answer>", case=False, regex=True)
    if not nonempty.any() or (not has_answer.any() and not frame["any_valid"].any()):
        raise ValueError("incomplete state_staircase raw input has no model answer text; refusing zero aggregate")
    if args.gate_scores and args.gate_scores.exists():
        gate = pd.read_csv(args.gate_scores, usecols=["prefix_id", "gate_status"])
        gate = gate.rename(columns={"gate_status": "gate_status_t2"})
        frame = frame.drop(columns=["gate_status_t2"], errors="ignore").merge(gate, on="prefix_id", how="left")
        frame["gate_status"] = frame["gate_status_t2"].fillna(frame["gate_status"])
        frame["gate_method"] = np.where(frame["gate_status_t2"].notna(), "t2_judge_join", "raw_ledger")
    if args.refamily_from_raw:
        fams = frame.apply(lambda r: raw_answer_family(str(r.get("scaffold_text", "")) + str(r.get("rung_text", "")) + str(r.get("continuation", ""))), axis=1)
        frame["observed_family_old"] = frame.get("observed_family", "")
        old_family = frame["in_family"] if "in_family" in frame else pd.Series(False, index=frame.index)
        frame["in_family_old"] = old_family.map(parse_bool)
        frame["observed_family_raw"] = [x[0] for x in fams]
        frame["family_source"] = [x[1] for x in fams]
        frame["in_family_raw"] = frame["any_valid"].astype(bool) & (frame["observed_family_raw"] == frame["target_family"].astype(str))
        frame["in_family"] = frame["in_family_raw"]
    else:
        if "in_family" not in frame:
            raise ValueError("state_staircase raw input has no in_family column and refamily is disabled")
        frame["in_family"] = frame["in_family"].map(parse_bool)
    # Compute continuation-level pass@k first, then average those values over
    # gated prefixes/problems. This preserves the experiment's problem-level
    # aggregation and makes pass@64 exact when the raw file has fewer than 64
    # continuations (as in the existing 8/16-sample campaigns).
    group_cols = ["model_label", "problem_index", "scaffold_type", "rung", "gate_status"]
    per_rows = []
    for keys, group in frame.groupby(group_cols, sort=False, dropna=False):
        any_values = group["any_valid"].astype(bool).tolist()
        family_values = group["in_family"].astype(bool).tolist()
        n = len(group)
        row = dict(zip(group_cols, keys))
        row.update({
            "in_family_valid_rate": float(np.mean(family_values)) if n else 0.0,
            "any_valid_rate": float(np.mean(any_values)) if n else 0.0,
            "pass@1": pass_at_k(n, sum(any_values), 1),
            "pass@64": pass_at_k(n, sum(any_values), 64),
            "in_family_pass@1": pass_at_k(n, sum(family_values), 1),
            "in_family_pass@64": pass_at_k(n, sum(family_values), 64),
            "n": n,
            "state_dose_tokens": group["state_dose_tokens"].iloc[0],
            "chance": group["chance"].iloc[0],
        })
        per_rows.append(row)
    per = pd.DataFrame(per_rows)
    summary = per[per.gate_status == "pass"].groupby(["model_label", "scaffold_type", "rung"], as_index=False).agg(
        in_family_valid_rate=("in_family_valid_rate", "mean"), any_valid_rate=("any_valid_rate", "mean"),
        **{"pass@1": ("pass@1", "mean"), "pass@64": ("pass@64", "mean"),
           "in_family_pass@1": ("in_family_pass@1", "mean"),
           "in_family_pass@64": ("in_family_pass@64", "mean")},
        n_problems=("problem_index", "nunique"), state_dose_tokens=("state_dose_tokens", "mean"), chance=("chance", "mean"))
    # Gate diagnostics must not erase sampled rows.  Preserve the
    # preregistered pass-only curve and also write the complete sampled matrix
    # for sensitivity reporting when a judge marks a prefix fail/soft.
    summary_all = per.groupby(["model_label", "scaffold_type", "rung"], as_index=False).agg(
        in_family_valid_rate=("in_family_valid_rate", "mean"), any_valid_rate=("any_valid_rate", "mean"),
        **{"pass@1": ("pass@1", "mean"), "pass@64": ("pass@64", "mean"),
           "in_family_pass@1": ("in_family_pass@1", "mean"),
           "in_family_pass@64": ("in_family_pass@64", "mean")},
        n_problems=("problem_index", "nunique"), state_dose_tokens=("state_dose_tokens", "mean"), chance=("chance", "mean"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(args.out_dir / f"state_staircase_per_instance_{args.tag}.csv", index=False)
    summary.to_csv(args.out_dir / f"state_staircase_summary_{args.tag}.csv", index=False)
    summary.to_csv(args.out_dir / f"state_staircase_curves_{args.tag}.csv", index=False)
    summary_all.to_csv(args.out_dir / f"state_staircase_summary_{args.tag}_all_gates.csv", index=False)
    if args.refamily_from_raw:
        confusion = pd.crosstab(frame["in_family_old"], frame["in_family_raw"], dropna=False).reset_index()
        confusion.to_csv(args.out_dir / f"state_staircase_family_confusion_{args.tag}.csv", index=False)
        audit = frame[(frame["gate_status"] == "pass") & frame["any_valid"].astype(bool)].head(30)
        audit_rows = audit[["problem_index", "target_family", "observed_family_raw", "family_source", "continuation"]].to_dict("records")
        if args.menu_path.exists():
            menu = pd.read_csv(args.menu_path)
            for item in menu[menu["witness_render"].notna()].itertuples():
                # A menu witness is a bare expression rather than a model
                # completion, so parse it directly for the extractor audit.
                witness_match = FIRST_TRIAL_RE.search(str(item.witness_render))
                observed = (
                    f"{int(witness_match.group(1))}{witness_match.group(2)}"
                    if witness_match else ""
                )
                source = "witness" if witness_match else "missing"
                audit_rows.append({"audit_kind": "menu_witness", "problem_index": int(item.problem_index), "target_family": str(item.branch), "observed_family_raw": observed, "family_source": source, "witness_match": bool(observed == str(item.branch))})
        pd.DataFrame(audit_rows).to_json(args.out_dir / f"state_staircase_family_audit_{args.tag}.jsonl", orient="records", lines=True)
    if args.problem_intersection:
        passing = per[per.gate_status == "pass"]
        needed = set(passing.rung.unique())
        shared = passing.groupby(["model_label", "problem_index", "scaffold_type"]).rung.nunique()
        shared_ids = shared[shared == len(needed)].index
        shared_per = passing.set_index(["model_label", "problem_index", "scaffold_type"]).loc[shared_ids].reset_index()
        shared_per.to_csv(args.out_dir / f"state_staircase_per_instance_{args.tag}_shared.csv", index=False)
        shared_per.groupby(["model_label", "scaffold_type", "rung"], as_index=False).agg(
            in_family_valid_rate=("in_family_valid_rate", "mean"), any_valid_rate=("any_valid_rate", "mean"),
            **{"pass@1": ("pass@1", "mean"), "pass@64": ("pass@64", "mean"),
               "in_family_pass@1": ("in_family_pass@1", "mean"),
               "in_family_pass@64": ("in_family_pass@64", "mean")},
            n_problems=("problem_index", "nunique"), state_dose_tokens=("state_dose_tokens", "mean"), chance=("chance", "mean")
        ).to_csv(args.out_dir / f"state_staircase_summary_{args.tag}_shared.csv", index=False)
    print(json.dumps({"raw_files": len(paths), "per_rows": len(per), "summary": str(args.out_dir / f'state_staircase_summary_{args.tag}.csv')}, sort_keys=True))


def score(args):
    """Teacher-force witness remainder and split entrance/execution costs."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ledger = list(load_jsonl(args.ledger_path))
    if args.max_ledger_rows > 0:
        ledger = ledger[:args.max_ledger_rows]
    steps = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    for step in steps:
        model_path = COUNTDOWN_ACTOR_DIR / f"global_step_{step}"
        if not model_path.exists():
            continue
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
        prompt_text = {
            pid: build_prompt_text(get_prompt_content(rec), tokenizer)
            for pid, rec in enumerate(records)
        }
        model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        for start in range(0, len(ledger), args.batch_size):
            batch = ledger[start:start + args.batch_size]
            prompts, targets = [], []
            for item in batch:
                prompts.append(prompt_text[int(item["problem_index"])] + item["scaffold_text"] + ("\n" if item["scaffold_text"] and item["rung_text"] else "") + item["rung_text"])
                targets.append(str(item["witness_render"]))
            full = tokenizer([p + t for p, t in zip(prompts, targets)], return_tensors="pt", padding=True, add_special_tokens=False)
            encp = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            ids, mask = full.input_ids.to(device), full.attention_mask.to(device)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask).logits.float()
            lp = F.log_softmax(logits[:, :-1], dim=-1); labels = ids[:, 1:]
            for j, item in enumerate(batch):
                seq_len = int(mask[j].sum()); plen = int(encp.attention_mask[j].sum())
                begin, end = max(plen - 1, 0), seq_len - 2
                if end < begin: continue
                vals = lp[j, begin:end + 1].gather(-1, labels[j, begin:end + 1].unsqueeze(-1)).squeeze(-1)
                target_encoded = tokenizer(targets[j], add_special_tokens=False,
                                           return_offsets_mapping=True)
                target_ids = list(target_encoded["input_ids"])
                offsets = list(target_encoded["offset_mapping"])
                # Use character offsets because Qwen may merge a leading space
                # with the operator token (so encoding("+") is not sufficient).
                op_match = re.search(r"(?<![0-9])\s*[+*/-]\s*(?=[0-9(])", targets[j])
                cut = None
                if op_match is not None:
                    op_start = op_match.start() + len(op_match.group(0)) - len(op_match.group(0).lstrip())
                    for k, (start, end) in enumerate(offsets):
                        if start <= op_start < end or (start >= op_start and end > start):
                            cut = k
                            break
                if cut is None:
                    cut = max(0, len(target_ids) - 1)
                entrance = float(-vals[:cut].sum().item()) if cut else 0.0
                execution = float(-vals[cut:].sum().item()) if cut < len(vals) else 0.0
                rows.append({"checkpoint": step, "problem_index": item["problem_index"], "prefix_id": item["prefix_id"], "scaffold_type": item["scaffold_type"], "rung": item["rung"], "gate_status": item["gate_status"], "total_cost": float(-vals.sum().item()), "entrance_cost": entrance, "execution_cost": execution, "target_tokens": len(vals)})
        del model
        if device.startswith("cuda"): torch.cuda.empty_cache()
    out = args.out_dir / f"state_staircase_tf_cost_{args.tag}.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({"rows": len(rows), "output": str(out)}, sort_keys=True))


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "prepare": prepare(args)
    elif args.mode == "run": run(args)
    elif args.mode == "aggregate": aggregate(args)
    else: score(args)


if __name__ == "__main__": main()
