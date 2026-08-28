
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for op1-mechanism analysis on Countdown (format-free).
Also contains shared probe/patch helpers (op token ids, entropy, KL, device).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from early_branch_locking.core.countdown_shared import tolerant_parse_completion, extract_ground_truth, load_parquet_sorted


OP_RE = re.compile(r"[+\-*/]")
ALLOWED_EXPR_CHARS = set("0123456789+-*/() ")


# ----------------------------
# Shared probe/patch helpers
# ----------------------------

def get_device(gpu_id: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_op_token_ids(tokenizer) -> List[int]:
    """Return single-token ids for +, -, *, /."""
    op_ids = []
    for op in ["+", "-", "*", "/"]:
        ids = tokenizer.encode(op, add_special_tokens=False)
        if len(ids) == 1:
            op_ids.append(ids[0])
        else:
            ids2 = tokenizer.encode(f" {op}", add_special_tokens=False)
            if len(ids2) == 1:
                op_ids.append(ids2[0])
            else:
                raise ValueError(f"Cannot find single-token id for operator '{op}'")
    return op_ids


def entropy_from_probs(p: np.ndarray) -> np.ndarray:
    """Element-wise entropy from a probability array (last axis is distribution)."""
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) along the last axis."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def summarise_op_probs(p: np.ndarray) -> Dict[str, float]:
    """Summary statistics for an (N, 4) operator probability array."""
    ent = entropy_from_probs(p)
    top1 = p.max(axis=-1)
    top2 = np.partition(p, -2, axis=-1)[:, -2]
    gap = top1 - top2
    return {
        "op1_entropy_mean": float(ent.mean()),
        "op1_top1_mean": float(top1.mean()),
        "op1_gap_mean": float(gap.mean()),
    }


def apply_final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the model's final layer norm before the lm_head."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden)
    return hidden


def get_layers_container(model):
    """Return the list/ModuleList of transformer layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model; cannot locate layers container.")


def build_clean_prompt(numbers: List[int], target: int) -> str:
    nums = " ".join(str(n) for n in numbers)
    return (
        "Use the numbers to make the target with + - * / and parentheses.\n"
        f"Numbers: {nums}\n"
        f"Target: {target}\n"
        "Use each number exactly once. Output only the expression.\n"
        "Expression: "
    )


def extract_answer_text(completion: str) -> str:
    parsed = tolerant_parse_completion(completion or "")
    ans = (parsed.get("answer_block") or "").strip()
    return ans


def prefix_before_op1(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    m = OP_RE.search(answer_text)
    if not m:
        return None
    return answer_text[: m.start()]


def extract_expression_from_output(text: str) -> str:
    if text is None:
        return ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    if line.lower().startswith("expression:"):
        line = line.split(":", 1)[1].strip()
    filtered = "".join(ch for ch in line if ch in ALLOWED_EXPR_CHARS)
    return filtered.strip()


def load_raw_indexed(path: Path) -> Dict[int, List[dict]]:
    by_prob = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_prob[int(rec["problem_index"])].append(rec)
    return by_prob


def pick_shortest_success_completion(by_prob: Dict[int, List[dict]], pid: int) -> Optional[str]:
    recs = by_prob.get(pid, [])
    succ = [r for r in recs if r.get("overall_ok")]
    if not succ:
        return None
    succ.sort(key=lambda r: len(r.get("completion", "")))
    return succ[0].get("completion", "")


def solved_flags(by_prob: Dict[int, List[dict]]) -> Dict[int, bool]:
    return {pid: any(bool(r.get("overall_ok")) for r in recs) for pid, recs in by_prob.items()}


def load_problem_ids_from_sets(sets_path: Path, set_name: str) -> Optional[List[int]]:
    if not sets_path.exists():
        return None
    data = json.loads(sets_path.read_text(encoding="utf-8"))
    ids = data.get(set_name)
    if not isinstance(ids, list):
        return None
    return [int(x) for x in ids]


def pick_diverse_success_completions(
    by_prob: Dict[int, List[dict]],
    pid: int,
    max_per_solution: int = 3,
) -> List[dict]:
    """
    For problem *pid*, group successful completions by canonical_expr,
    assign each group a local solution_id (0, 1, 2, …), and return up to
    *max_per_solution* completions per solution class.

    Each returned dict contains:
        completion       – raw completion text
        canonical_expr   – canonical expression string
        opseq_label      – operator-sequence label
        solution_id      – int, solution class index within this problem
    """
    recs = by_prob.get(pid, [])
    succ = [r for r in recs if r.get("overall_ok")]
    if not succ:
        return []

    # group by canonical_expr
    from collections import OrderedDict
    groups: OrderedDict[str, List[dict]] = OrderedDict()
    for r in succ:
        canon = r.get("canonical_expr")
        if not canon:
            continue
        groups.setdefault(canon, []).append(r)

    results: List[dict] = []
    for sol_id, (canon, group_recs) in enumerate(groups.items()):
        # prefer shorter completions within each group
        group_recs_sorted = sorted(group_recs, key=lambda r: len(r.get("completion", "")))
        for r in group_recs_sorted[:max_per_solution]:
            results.append(dict(
                completion=r.get("completion", ""),
                canonical_expr=canon,
                opseq_label=r.get("opseq_label", "OPSEQ::"),
                solution_id=sol_id,
            ))
    return results


def get_solution_class_count(by_prob: Dict[int, List[dict]], pid: int) -> int:
    """Return the number of distinct canonical solutions observed for *pid*."""
    recs = by_prob.get(pid, [])
    canons = set()
    for r in recs:
        if r.get("overall_ok") and r.get("canonical_expr"):
            canons.add(r["canonical_expr"])
    return len(canons)


def build_prefix_records(
    teacher_raw_path: Path,
    num_problems: int,
    teacher_only: bool = False,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    by_teacher = load_raw_indexed(teacher_raw_path)
    prefix_by_pid = {}
    answer_by_pid = {}
    for pid in range(num_problems):
        y_succ = pick_shortest_success_completion(by_teacher, pid)
        if not y_succ:
            continue
        ans = extract_answer_text(y_succ)
        if not ans:
            continue
        pref = prefix_before_op1(ans)
        if pref is None:
            continue
        prefix_by_pid[pid] = pref
        answer_by_pid[pid] = ans
    return prefix_by_pid, answer_by_pid


def build_format_free_inputs(
    test_parquet: Path,
    prefix_by_pid: Dict[int, str],
    num_problems: int,
) -> Tuple[List[int], List[str], List[Tuple[List[int], int]]]:
    records = load_parquet_sorted(test_parquet, n=num_problems, sort_key="sample_id")
    pids = []
    prompts = []
    gts = []
    for pid, rec in enumerate(records):
        if pid not in prefix_by_pid:
            continue
        numbers, target, _ = extract_ground_truth(rec)
        prompt = build_clean_prompt(numbers, target)
        pids.append(pid)
        prompts.append(prompt)
        gts.append((numbers, target))
    return pids, prompts, gts


def ckpt_name_from_path(path: str) -> str:
    """Extract a checkpoint name from a model path."""
    p = Path(path)
    if p.name.startswith("global_step_"):
        return p.name
    if "global_step_" in path:
        return path.split("global_step_")[-1]
    return p.name

