#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""prefix_recovery - Splice-at-retry: solver entrances grafted into the model's own traces.

Modified for universal trajectory processing (failed and successful).

Leak protocol:
  - the scaffold prefix comes from ALL samples (failed or successful);
  - the entrance comes only from solver enumeration;
  - prefix is rejected if it contains any full solver solution or answer/feasible tags.
"""

from __future__ import annotations

import argparse
import ast as pyast
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR, METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.countdown_utils import (  # noqa: E402
    evaluate_countdown_expression,
    parse_countdown_completion,
)
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    bootstrap_ci_mean,
    build_prompt_text,
    enumerate_solution_set,
    evaluate_countdown_completion,
    extract_ground_truth,
    get_prompt_content,
    load_jsonl,
    load_parquet_sorted,
    pass_at_k,
    tolerant_parse_completion,
)

OPS = "+-*/"
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
SYM = {pyast.Add: "+", pyast.Sub: "-", pyast.Mult: "*", pyast.Div: "/"}
ENTRANCE_RE = re.compile(r"^(\(*)\s*(\d+)\s*([+\-*/])")
FIRST_TRIAL_RE = re.compile(r"\(?\s*(\d+)\s*([+\-*/])")
CUE_RE = re.compile(r"(?i)\b(let'?s\s+try|let\s+me\s+try|i\s+will\s+try|i'?ll\s+try|now\s+try|next\s+try)\b")
TRIAL_RESULT_RE = re.compile(r"=\s*(-?\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "logit", "aggregate"), default="prepare")
    parser.add_argument("--raw-path", type=Path, default=RAW_DIR / "countdown_raw_global_step_275_n320.jsonl")
    parser.add_argument("--model-path", type=Path, default=COUNTDOWN_ACTOR_DIR / "global_step_275")
    parser.add_argument("--num-problems", type=int, default=150)
    parser.add_argument("--scaffolds-per-problem", type=int, default=1)
    parser.add_argument("--arms", default="feasible,empty",
                        help="comma set from {feasible,empty,infeasible}")
    parser.add_argument("--max-infeasible-per-problem", type=int, default=2)
    parser.add_argument("--n-continuations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--sets-path", type=Path,
                        default=METRICS_DIR / "branch_set_collection_sets_global_step_50_to_global_step_275_n320.json")
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", default="splice_v1")
    parser.add_argument("--neutral-scaffold", action="store_true",
                        help="prepare a matched neutral problem-restatement scaffold instead of retry text")
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--raw-out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--scaffolds-input", type=Path, default=None,
                        help="reuse an existing scaffold JSONL while writing outputs under a new tag")
    parser.add_argument("--menu-input", type=Path, default=None,
                        help="reuse an existing solver menu CSV while writing outputs under a new tag")
    return parser.parse_args(argv)


def paths_of(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scaffolds": args.scaffolds_input or args.out_dir / f"prefix_recovery_scaffolds_{args.tag}.jsonl",
        "menu": args.menu_input or args.out_dir / f"prefix_recovery_menu_{args.tag}.csv",
        "eligibility": args.out_dir / f"prefix_recovery_eligibility_{args.tag}.csv",
        "per_instance": args.out_dir / f"prefix_recovery_per_instance_{args.tag}.csv",
        "raw": args.raw_out_dir / f"prefix_recovery_raw_{args.tag}.jsonl",
        "summary": args.out_dir / f"prefix_recovery_summary_{args.tag}.csv",
        "access": args.out_dir / f"prefix_recovery_access_{args.tag}.csv",
        "factorization": args.out_dir / f"prefix_recovery_factorization_{args.tag}.csv",
        "logit": args.out_dir / f"prefix_recovery_logit_access_{args.tag}.csv",
        "manifest": args.out_dir / f"prefix_recovery_manifest_{args.tag}.json",
    }


def json_ready(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def parse_bool(value) -> bool:
    """Parse booleans from JSON/pandas/CSV without treating ``"False"`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


# ---------------------------------------------------------------------------
# Solver-side machinery
# ---------------------------------------------------------------------------

def _mrender(node) -> tuple[str, int]:
    if isinstance(node, pyast.Expression):
        return _mrender(node.body)
    if isinstance(node, pyast.Constant):
        value = node.value
        return (str(int(value)) if float(value).is_integer() else str(value)), 3
    if isinstance(node, pyast.UnaryOp) and isinstance(node.op, pyast.USub):
        text, _ = _mrender(node.operand)
        return f"(-{text})", 3
    if isinstance(node, pyast.BinOp):
        op = SYM[type(node.op)]
        p = PREC[op]
        ltext, lp = _mrender(node.left)
        rtext, rp = _mrender(node.right)
        if lp < p:
            ltext = f"({ltext})"
        if rp < p or (rp == p and op in "-/"):
            rtext = f"({rtext})"
        return f"{ltext} {op} {rtext}", p
    raise ValueError(f"unsupported node {type(node).__name__}")


def minimal_render(canonical_expr: str) -> str:
    return _mrender(pyast.parse(canonical_expr, mode="eval"))[0]


def entrance_of(render: str) -> tuple[int, str, str] | None:
    match = ENTRANCE_RE.match(render.replace(" ", ""))
    if match is None:
        return None
    parens, number, op = match.group(1), match.group(2), match.group(3)
    graft = f"{parens}{number} {op}"
    return int(number), op, graft


def _apply(left: Fraction, op: str, right: Fraction) -> Fraction | None:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if right == 0:
        return None
    return left / right


def _completions(seq: tuple[int, ...]):
    if len(seq) == 1:
        yield Fraction(seq[0]), None
        return
    for i in range(1, len(seq)):
        for lv, lop in _completions(seq[:i]):
            for rv, _ in _completions(seq[i:]):
                for op in OPS:
                    if lv is None or rv is None:
                        yield None, (lop if lop is not None else op)
                        continue
                    yield _apply(lv, op, rv), (lop if lop is not None else op)


def chance_baseline(numbers: list[int], target: int, n0: int, op0: str) -> tuple[float, int]:
    rest = list(numbers)
    rest.remove(n0)
    hits = total = 0
    for perm in sorted(set(permutations(rest))):
        for value, first_op in _completions((n0, *perm)):
            if first_op != op0 or value is None:
                continue
            total += 1
            hits += int(value == Fraction(target))
    return (hits / total if total else 0.0), total


def branch_menu(numbers: list[int], target: int) -> list[dict]:
    families: dict[tuple[int, str], list[str]] = defaultdict(list)
    for expr in sorted(enumerate_solution_set(numbers, target)):
        render = minimal_render(expr)
        ent = entrance_of(render)
        if ent is not None:
            families[(ent[0], ent[1])].append(render)
    rows = []
    for (n0, op), renders in sorted(families.items()):
        renders.sort(key=lambda item: (item.startswith("("), len(item), item))
        graft = entrance_of(renders[0])[2]
        chance, denom = chance_baseline(numbers, target, n0, op)
        rows.append({
            "first_number": n0, "first_op": op, "branch": f"{n0}{op}",
            "graft_text": graft, "n_solutions_in_family": len(renders),
            "witness_render": renders[0], "chance": chance, "chance_denominator": denom,
            "feasible": True,
        })
    return rows


def infeasible_entrances(numbers: list[int], target: int, menu: list[dict], limit: int) -> list[dict]:
    feasible = {(row["first_number"], row["first_op"]) for row in menu}
    rows = []
    for n0 in sorted(set(int(v) for v in numbers)):
        for op in OPS:
            if (n0, op) in feasible:
                continue
            chance, denom = chance_baseline(numbers, target, n0, op)
            if chance > 0:
                continue
            rows.append({
                "first_number": n0, "first_op": op, "branch": f"{n0}{op}",
                "graft_text": f"{n0} {op}", "n_solutions_in_family": 0,
                "witness_render": "", "chance": 0.0, "chance_denominator": denom,
                "feasible": False,
            })
    rows.sort(key=lambda row: (row["first_number"], row["first_op"]))
    return rows[:limit]


# ---------------------------------------------------------------------------
# Scaffold extraction
# ---------------------------------------------------------------------------

def trial_positions(text: str, target: int) -> list[int]:
    """Return ends of intermediate trials that demonstrably miss the target."""
    out = []
    for match in TRIAL_RESULT_RE.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value != float(target):
            out.append(match.end())
    return out


def retry_cut(text: str, target: int) -> dict | None:
    positions = trial_positions(text, target)
    if not positions:
        return None
    # We choose the first position (like failed samples) to maintain consistency
    # regardless of whether the whole sample was correct or not.
    anchor = positions[0]
    cue = CUE_RE.search(text, anchor)
    if cue is not None:
        return {"cut": cue.end(), "joiner": " ", "cut_kind": "cue"}
    fail_sentence = re.search(
        rf"(?i)(?:not\s+equal|is\s+not\s+{target}\b|doesn'?t\s+(?:equal|work)|does\s+not\s+work)",
        text[anchor:],
    )
    if fail_sentence is None:
        return None
    anchor_end = anchor + fail_sentence.end()
    terminator = re.search(r"[.!]\s|\n", text[anchor_end:])
    if terminator is None:
        return None
    return {"cut": anchor_end + terminator.end(), "joiner": "\n\n", "cut_kind": "sentence"}


def prefix_leak_ok(prefix: str, target: int, solution_renders: list[str]) -> bool:
    lower = prefix.lower()
    if "<answer>" in lower or "<feasible>" in lower:
        return False
    if re.search(rf"=\s*{target}(?!\d)", prefix):
        return False
    normalized = re.sub(r"\s+", "", prefix)
    for render in solution_renders:
        if re.sub(r"\s+", "", render) in normalized:
            return False
    return True


def run_prepare(args: argparse.Namespace) -> None:
    paths = paths_of(args)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    by_pid: dict[int, list[dict]] = defaultdict(list)
    for row in load_jsonl(args.raw_path):
        pid = int(row.get("problem_index", -1))
        if 0 <= pid < args.num_problems:
            by_pid[pid].append(row)

    scaffolds, menu_rows, eligibility = [], [], []
    for pid, record in enumerate(records):
        numbers, target, feasible_label = extract_ground_truth(record)
        numbers = [int(v) for v in numbers]
        if str(feasible_label).strip().lower() != "yes":
            eligibility.append({"problem_index": pid, "status": "solver_infeasible_problem"})
            continue
        menu = branch_menu(numbers, target)
        if not menu:
            eligibility.append({"problem_index": pid, "status": "no_entrance_families"})
            continue
        renders = [row["witness_render"] for row in menu]
        renders += [minimal_render(expr) for expr in enumerate_solution_set(numbers, target)]
        renders += sorted(enumerate_solution_set(numbers, target))

        if args.neutral_scaffold:
            neutral = "The numbers are " + ", ".join(map(str, numbers)) + ". Let me think about which numbers to combine first."
            scaffolds.append({"problem_index": pid, "sample_index": -1, "prefix_text": neutral,
                              "prefix_chars": len(neutral), "joiner": "\n", "cut_kind": "neutral",
                              "numbers": numbers, "target": int(target), "feasible_label": str(feasible_label),
                              "original_overall_ok": False})
            for row in menu + infeasible_entrances(numbers, target, menu, args.max_infeasible_per_problem):
                menu_rows.append({"problem_index": pid, **row})
            eligibility.append({"problem_index": pid, "status": "eligible_neutral", "n_samples": len(by_pid.get(pid, [])), "n_scaffolds": 1, "n_feasible_families": len(menu)})
            continue
        # We take ALL samples, not just failed ones
        attempts = [row for row in by_pid.get(pid, []) if row.get("completion")]
        candidates = []
        for row in attempts:
            cut = retry_cut(str(row["completion"]), target)
            if cut is None:
                continue
            prefix = str(row["completion"])[: cut["cut"]]
            if not prefix_leak_ok(prefix, target, renders):
                continue
            candidates.append({
                "problem_index": pid, "sample_index": int(row.get("sample_index", -1)),
                "prefix_text": prefix, "prefix_chars": len(prefix),
                "joiner": cut["joiner"], "cut_kind": cut["cut_kind"],
                "numbers": numbers, "target": int(target), "feasible_label": str(feasible_label),
                "original_overall_ok": bool(row.get("overall_ok"))
            })
        if not candidates:
            eligibility.append({"problem_index": pid, "status": "no_usable_retry_point",
                                "n_samples": len(attempts)})
            continue
        # Sort by length/cue presence to pick the most representative scaffolds
        candidates.sort(key=lambda c: (c["cut_kind"] != "cue", c["prefix_chars"], c["sample_index"]))
        chosen = candidates[: args.scaffolds_per_problem]
        scaffolds.extend(chosen)
        for row in menu + infeasible_entrances(numbers, target, menu, args.max_infeasible_per_problem):
            menu_rows.append({"problem_index": pid, **row})
        eligibility.append({"problem_index": pid, "status": "eligible",
                            "n_samples": len(attempts), "n_scaffolds": len(chosen),
                            "n_feasible_families": len(menu)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with paths["scaffolds"].open("w", encoding="utf-8") as handle:
        for row in scaffolds:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
    pd.DataFrame(menu_rows, columns=["problem_index", "first_number", "first_op", "branch",
                                     "graft_text", "n_solutions_in_family", "witness_render",
                                     "chance", "chance_denominator", "feasible"]).to_csv(
                                         paths["menu"], index=False)
    pd.DataFrame(eligibility).to_csv(paths["eligibility"], index=False)
    manifest = {
        "experiment_id": "prefix_recovery", "tag": args.tag, "stage": "prepare",
        "source_raw": str(args.raw_path), "scaffold_source": "all trajectories (failed and successful)",
        "entrance_source": "solver enumeration, minimal render, cut after first operator",
        "leak_rules": ["no '= target' in prefix", "no full solution substring (normalized)",
                       "no answer/feasible tags", "final operation of any solution never grafted"],
        "n_scaffolds": len(scaffolds), "n_menu_rows": len(menu_rows),
        "n_eligible_problems": sum(row["status"] in {"eligible", "eligible_neutral"} for row in eligibility),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_scaffolds", "n_menu_rows", "n_eligible_problems")}, sort_keys=True))
    print(pd.DataFrame(eligibility)["status"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Instance construction and evaluation
# ---------------------------------------------------------------------------

def build_instances(args: argparse.Namespace) -> list[dict]:
    paths = paths_of(args)
    scaffolds = list(load_jsonl(paths["scaffolds"]))
    menu = pd.read_csv(paths["menu"]).to_dict("records")
    menu_by_pid: dict[int, list[dict]] = defaultdict(list)
    for row in menu:
        menu_by_pid[int(row["problem_index"])].append(row)
    arms = {item.strip() for item in args.arms.split(",") if item.strip()}
    instances = []
    for scaffold in scaffolds:
        pid = int(scaffold["problem_index"])
        # Pass the original status flag to the instance
        orig_ok = scaffold["original_overall_ok"]
        if "empty" in arms:
            instances.append({**scaffold, "arm": "empty", "branch": "", "graft_text": "",
                              "chance": 0.0, "n_solutions_in_family": 0,
                              "prefix_full": scaffold["prefix_text"],
                              "original_overall_ok": orig_ok})
        for row in menu_by_pid.get(pid, []):
            # pandas reads CSV booleans as either bools or strings depending on
            # mixed/empty columns; bool("False") would silently invert the arm.
            arm = "feasible" if parse_bool(row.get("feasible", False)) else "infeasible"
            if arm not in arms:
                continue
            graft = str(row["graft_text"])
            instances.append({**scaffold, "arm": arm, "branch": str(row["branch"]),
                              "first_number": int(row["first_number"]), "first_op": str(row["first_op"]),
                              "graft_text": graft, "chance": float(row["chance"]),
                              "n_solutions_in_family": int(row["n_solutions_in_family"]),
                              "prefix_full": scaffold["prefix_text"] + scaffold["joiner"] + graft,
                              "original_overall_ok": orig_ok})
    return instances


def answer_family_with_source(full_text: str) -> tuple[tuple[int, str] | None, str]:
    """Return the first operation family from the answer, with a safe fallback."""
    parsed = tolerant_parse_completion(str(full_text or ""))
    answer = str(parsed.get("answer_block", "") or "")
    match = FIRST_TRIAL_RE.search(answer)
    if match is not None:
        return (int(match.group(1)), match.group(2)), "answer"

    # Only use an actual think block as fallback; never scan the scaffold,
    # because scaffold text may contain unrelated failed trials.
    think = str(parsed.get("think_block", "") or "")
    if "<think>" in str(full_text).lower() and think:
        matches = list(FIRST_TRIAL_RE.finditer(think))
        if matches:
            match = matches[-1]
            return (int(match.group(1)), match.group(2)), "think_fallback"
    return None, "missing"


def answer_family(full_text: str) -> tuple[int, str] | None:
    return answer_family_with_source(full_text)[0]


def evaluate_instance(instance: dict, continuations: list[str]) -> tuple[dict, list[dict]]:
    n = len(continuations)
    any_valid = in_family = adhere = 0
    first_trial_counts: Counter = Counter()
    answer_family_counts: Counter = Counter()
    raw_rows = []
    grafted = (int(instance.get("first_number", -1)), str(instance.get("first_op", "")))
    for index, continuation in enumerate(continuations):
        full = instance["prefix_full"] + continuation
        result = evaluate_countdown_completion(
            full, instance["numbers"], instance["target"], instance["feasible_label"],
            parse_countdown_completion, evaluate_countdown_expression,
        )
        valid = bool(result.overall_ok)
        any_valid += int(valid)
        family, family_source = answer_family_with_source(full)
        if family is not None:
            answer_family_counts[f"{family[0]}{family[1]}"] += 1
        if instance["arm"] in ("feasible", "infeasible"):
            adhere += int(bool(re.match(r"\s*\(?\s*\d", continuation)))
            if valid and family == grafted:
                in_family += 1
        else:
            trial = FIRST_TRIAL_RE.search(continuation)
            if trial is not None:
                first_trial_counts[f"{int(trial.group(1))}{trial.group(2)}"] += 1
        raw_rows.append({
            "problem_index": instance["problem_index"], "scaffold_sample_index": instance["sample_index"],
            "arm": instance["arm"], "branch": instance["branch"], "continuation_index": index,
            "continuation": continuation, "overall_ok": valid,
            "observed_family": "" if family is None else f"{family[0]}{family[1]}",
            "family_source": family_source,
            "in_family": bool(valid and family == grafted),
            "original_overall_ok": instance["original_overall_ok"],
            "canonical_expr": result.canonical_expr, "parse_status": result.parse_status,
        })
    row = {
        "problem_index": instance["problem_index"], "scaffold_sample_index": instance["sample_index"],
        "arm": instance["arm"], "branch": instance["branch"], "graft_text": instance["graft_text"],
        "cut_kind": instance["cut_kind"], "prefix_chars": instance["prefix_chars"],
        "n_solutions_in_family": instance["n_solutions_in_family"], "chance": instance["chance"],
        "original_overall_ok": instance["original_overall_ok"],
        "n": n, "any_valid_rate": any_valid / n if n else 0.0,
        "in_family_valid_rate": in_family / n if n else 0.0,
        # Standard runs have n=16. If n<64, pass@64 is the exact
        # at-least-one result over the available continuations.
        "pass@1": pass_at_k(n, any_valid, 1),
        "pass@64": pass_at_k(n, any_valid, 64),
        "in_family_pass@1": pass_at_k(n, in_family, 1),
        "in_family_pass@64": pass_at_k(n, in_family, 64),
        "excess_over_chance": (in_family / n - instance["chance"]) if n else 0.0,
        "adherence_rate": adhere / n if n else 0.0,
        "first_trial_family_counts": json.dumps(dict(first_trial_counts), sort_keys=True),
        "answer_family_counts": json.dumps(dict(answer_family_counts), sort_keys=True),
    }
    return row, raw_rows


def run_generation(args: argparse.Namespace) -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    paths = paths_of(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_out_dir.mkdir(parents=True, exist_ok=True)
    instances = build_instances(args)
    if not instances:
        raise RuntimeError("no prefix_recovery instances; run --mode prepare first")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_ids = {pid: tokenizer.encode(build_prompt_text(get_prompt_content(rec), tokenizer),
                                        add_special_tokens=False)
                  for pid, rec in enumerate(records)}
    requests = [{"prompt_token_ids": prompt_ids[int(inst["problem_index"])]
                 + tokenizer.encode(inst["prefix_full"], add_special_tokens=False)}
                for inst in instances]
    llm = LLM(model=str(args.model_path), tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
              seed=args.seed, dtype=args.dtype, max_model_len=args.max_model_len,
              enforce_eager=args.enforce_eager)
    params = SamplingParams(n=args.n_continuations, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_new_tokens,
                            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None)
    outputs = llm.generate(requests, params)
    per_instance, raw_rows = [], []
    for instance, output in zip(instances, outputs):
        row, raw = evaluate_instance(instance, [seq.text or "" for seq in output.outputs])
        per_instance.append(row)
        if args.save_raw:
            raw_rows.extend(raw)
    pd.DataFrame(per_instance).to_csv(paths["per_instance"], index=False)
    if args.save_raw:
        with paths["raw"].open("w", encoding="utf-8") as handle:
            for row in raw_rows:
                handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
    if hasattr(llm, "shutdown"):
        llm.shutdown()
    del llm
    gc.collect()
    print(json.dumps({"instances": len(per_instance), "raw_rows": len(raw_rows),
                      "per_instance": str(paths["per_instance"])}, sort_keys=True))


# ---------------------------------------------------------------------------
# Logit access
# ---------------------------------------------------------------------------

def run_logit(args: argparse.Namespace) -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = paths_of(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scaffolds = list(load_jsonl(paths["scaffolds"]))
    menu = pd.read_csv(paths["menu"])
    menu = menu[menu["feasible"].map(parse_bool)]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True).to("cuda")
    model.eval()
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_ids = {pid: tokenizer.encode(build_prompt_text(get_prompt_content(rec), tokenizer),
                                        add_special_tokens=False)
                  for pid, rec in enumerate(records)}
    rows = []
    with torch.inference_mode():
        for scaffold in scaffolds:
            pid = int(scaffold["problem_index"])
            base = prompt_ids[pid] + tokenizer.encode(scaffold["prefix_text"], add_special_tokens=False)
            families = menu[menu["problem_index"] == pid]
            logps = {}
            for _, fam in families.iterrows():
                ent_ids = tokenizer.encode(scaffold["joiner"] + str(fam["graft_text"]),
                                           add_special_tokens=False)
                ids = torch.tensor([base + ent_ids], device="cuda")
                logits = model(input_ids=ids).logits[0].float()
                logprobs = torch.log_softmax(logits[:-1], dim=-1)
                targets = ids[0, 1:]
                span = slice(len(base) - 1, len(base) - 1 + len(ent_ids))
                token_lp = logprobs[span].gather(1, targets[span].unsqueeze(1)).squeeze(1)
                logps[str(fam["branch"])] = {
                    "entrance_logprob": float(token_lp.sum().item()),
                    "entrance_tokens": len(ent_ids),
                    "entrance_logprob_per_token": float(token_lp.mean().item()),
                }
            if logps:
                raw = np.array([value["entrance_logprob"] for value in logps.values()])
                shares = np.exp(raw - raw.max())
                shares = shares / shares.sum()
                for (branch, value), share in zip(logps.items(), shares):
                    rows.append({"problem_index": pid, "scaffold_sample_index": scaffold["sample_index"],
                                 "original_overall_ok": parse_bool(scaffold["original_overall_ok"]),
                                 "branch": branch, **value,
                                 "access_logit_share_within_menu": float(share)})
    model.cpu()
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(paths["logit"], index=False)
    print(json.dumps({"rows": len(rows), "output": str(paths["logit"])}, sort_keys=True))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _problem_bootstrap(values_by_pid: dict[int, list[float]], draws: int, seed: int):
    means = [float(np.mean(v)) for v in values_by_pid.values() if v]
    return bootstrap_ci_mean(means, n_boot=draws, seed=seed) if means else (float("nan"),) * 3


def run_aggregate(args: argparse.Namespace) -> None:
    paths = paths_of(args)
    frame = pd.read_csv(paths["per_instance"])
    required = {"n", "any_valid_rate", "in_family_valid_rate", "arm", "problem_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"incomplete prefix_recovery per-instance input missing {missing}")
    # Empty/infeasible controls legitimately have no target family. A feasible
    # arm with valid answers must nevertheless have a nonzero family count;
    # this catches the old family-scoring tables without rejecting controls.
    feasible = frame[frame["arm"].eq("feasible")]
    if (not feasible.empty and (feasible["any_valid_rate"] > 0).any()
            and (feasible["in_family_valid_rate"] > 0).sum() == 0):
        raise ValueError("prefix_recovery per-instance input has valid feasible answers but zero family matches; regenerate with current family parser")
    # Keep aggregation backward-compatible with per-instance files written
    # before pass@k columns were added. Rates plus n recover the observed
    # success counts exactly for the stored integer-count runs.
    if "pass@1" not in frame.columns:
        any_counts = np.rint(frame["n"] * frame["any_valid_rate"]).astype(int)
        family_counts = np.rint(frame["n"] * frame["in_family_valid_rate"]).astype(int)
        frame["pass@1"] = [pass_at_k(int(n), int(c), 1) for n, c in zip(frame["n"], any_counts)]
        frame["pass@64"] = [pass_at_k(int(n), int(c), 64) for n, c in zip(frame["n"], any_counts)]
        frame["in_family_pass@1"] = [pass_at_k(int(n), int(c), 1) for n, c in zip(frame["n"], family_counts)]
        frame["in_family_pass@64"] = [pass_at_k(int(n), int(c), 64) for n, c in zip(frame["n"], family_counts)]
        # Keep the corrected per-instance artifact in sync with the summary.
        frame.to_csv(paths["per_instance"], index=False)
    sets = {}
    if args.sets_path and Path(args.sets_path).exists():
        data = json.loads(Path(args.sets_path).read_text(encoding="utf-8"))
        for name in ("S_loss", "S_both", "S_gain"):
            for pid in data.get(name, []):
                sets[int(pid)] = name
    frame["problem_set"] = frame["problem_index"].map(lambda pid: sets.get(int(pid), "unknown"))

    summary_rows = []
    # By grouping by original_overall_ok, we can analyze successful vs failed trajectories
    for (arm, pset, orig_ok), group in frame.groupby(["arm", "problem_set", "original_overall_ok"], sort=True):
        row = {"arm": arm, "problem_set": pset, "original_overall_ok": orig_ok,
               "n_instances": len(group), "n_problems": group["problem_index"].nunique(),
               "chance_mean": float(group["chance"].mean())}
        for metric in (
            "any_valid_rate",
            "in_family_valid_rate",
            "pass@1",
            "pass@64",
            "in_family_pass@1",
            "in_family_pass@64",
            "excess_over_chance",
            "adherence_rate",
        ):
            by_pid = defaultdict(list)
            for _, item in group.iterrows():
                by_pid[int(item["problem_index"])].append(float(item[metric]))
            mean, lo, hi = _problem_bootstrap(by_pid, args.bootstrap_draws, args.seed)
            row[f"{metric}_mean"], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = mean, lo, hi
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False)

    access_rows = []
    empty = frame[frame["arm"] == "empty"]
    access_by_pid: dict[int, dict[str, float]] = {}
    for _, item in empty.iterrows():
        counts = Counter(json.loads(item["first_trial_family_counts"]))
        total = sum(counts.values())
        pid = int(item["problem_index"])
        shares = {branch: count / total for branch, count in counts.items()} if total else {}
        access_by_pid[pid] = shares
        for branch, share in sorted(shares.items()):
            access_rows.append({"problem_index": pid, "branch": branch,
                                "original_overall_ok": item["original_overall_ok"],
                                "access_first_trial_share": share,
                                "empty_any_valid_rate": float(item["any_valid_rate"])})
    pd.DataFrame(access_rows).to_csv(paths["access"], index=False)

    fact_rows = []
    feas = frame[frame["arm"] == "feasible"]
    exec_by = {(int(item["problem_index"]), str(item["branch"])): float(item["in_family_valid_rate"])
               for _, item in feas.iterrows()}
    for _, item in empty.iterrows():
        pid = int(item["problem_index"])
        shares = access_by_pid.get(pid, {})
        measured = {branch: exec_by[(pid, branch)] for branch in shares if (pid, branch) in exec_by}
        predicted = sum(shares[branch] * measured[branch] for branch in measured)
        fact_rows.append({"problem_index": pid, "problem_set": sets.get(pid, "unknown"),
                          "original_overall_ok": item["original_overall_ok"],
                          "predicted_solve_rate": predicted,
                          "observed_empty_any_valid": float(item["any_valid_rate"]),
                          "access_mass_on_measured_families": float(sum(shares[b] for b in measured)),
                          "n_families_measured": len(measured)})
    fact = pd.DataFrame(fact_rows)
    # Feasible-only paired runs intentionally omit the empty arm.  Keep their
    # factorization artifacts well-formed and mark the missing observed-empty
    # comparison as unavailable instead of crashing the aggregate mode.
    if fact.empty:
        fact = pd.DataFrame(columns=["problem_index", "problem_set", "original_overall_ok",
                                     "predicted_solve_rate", "observed_empty_any_valid",
                                     "access_mass_on_measured_families", "n_families_measured"])
    fact.to_csv(paths["factorization"], index=False)
    # Store the regression and residual-depth views used by the revised paper
    # narrative instead of leaving these as an ephemeral headline statistic.
    regression_rows = []
    def add_regression(label, group):
        g = group.dropna(subset=["predicted_solve_rate", "observed_empty_any_valid"])
        if len(g) < 3 or float(g["predicted_solve_rate"].var()) == 0.0:
            regression_rows.append({"stratum": label, "n": int(len(g)), "slope": np.nan,
                                    "intercept": np.nan, "corr": np.nan,
                                    "slope_ci_lo": np.nan, "slope_ci_hi": np.nan})
            return
        x, y = g.predicted_solve_rate.to_numpy(float), g.observed_empty_any_valid.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        rng = np.random.default_rng(args.seed)
        boots = []
        for _ in range(min(args.bootstrap_draws, 10000)):
            idx = rng.integers(0, len(g), len(g))
            xb, yb = x[idx], y[idx]
            if np.var(xb) > 0:
                boots.append(float(np.polyfit(xb, yb, 1)[0]))
        regression_rows.append({"stratum": label, "n": int(len(g)), "slope": float(slope),
                                "intercept": float(intercept), "corr": float(np.corrcoef(x, y)[0, 1]),
                                "slope_ci_lo": float(np.quantile(boots, .025)) if boots else np.nan,
                                "slope_ci_hi": float(np.quantile(boots, .975)) if boots else np.nan})
    if {"predicted_solve_rate", "observed_empty_any_valid"}.issubset(fact.columns):
        add_regression("all", fact)
        for key, group in fact.groupby("original_overall_ok", dropna=False):
            add_regression(f"original_overall_ok={key}", group)
        for key, group in fact.groupby("problem_set", dropna=False):
            add_regression(f"problem_set={key}", group)
    else:
        regression_rows.append({"stratum": "feasible_only_no_empty_arm", "n": 0,
                                "slope": np.nan, "intercept": np.nan, "corr": np.nan,
                                "slope_ci_lo": np.nan, "slope_ci_hi": np.nan})
    pd.DataFrame(regression_rows).to_csv(args.out_dir / f"prefix_recovery_factorization_regression_{args.tag}.csv", index=False)

    # Residual depth is represented by the number of solver solutions in the
    # grafted family; report problem-level means in 3/4-number strata.
    num_by_pid = {}
    try:
        for pid, rec in enumerate(load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")):
            num_by_pid[pid] = len(extract_ground_truth(rec)[0])
    except Exception:
        num_by_pid = {}
    feas_depth = frame[frame["arm"] == "feasible"].copy()
    feas_depth["n_numbers"] = feas_depth.problem_index.map(num_by_pid)
    feas_depth["depth_bin"] = pd.cut(feas_depth["n_solutions_in_family"],
                                      bins=[-1, 1, 3, 8, np.inf],
                                      labels=["1", "2-3", "4-8", ">8"])
    residual = feas_depth.groupby(["n_numbers", "depth_bin", "original_overall_ok"], dropna=False, observed=True, as_index=False).agg(
        n_instances=("problem_index", "size"), n_problems=("problem_index", "nunique"),
        chance_mean=("chance", "mean"), in_family_valid_rate=("in_family_valid_rate", "mean"),
        excess_over_chance=("excess_over_chance", "mean"))
    residual.to_csv(args.out_dir / f"prefix_recovery_residual_depth_{args.tag}.csv", index=False)
    headline = {
        "summary": str(paths["summary"]),
        "factorization_corr": float(fact["predicted_solve_rate"].corr(fact["observed_empty_any_valid"]))
        if len(fact) > 2 else float("nan"),
    }
    print(json.dumps(headline, sort_keys=True))


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "run":
        run_generation(args)
    elif args.mode == "logit":
        run_logit(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
