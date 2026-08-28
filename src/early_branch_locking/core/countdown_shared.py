
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_countdown_shared.py

Countdown-Feasible 任务共享工具：
- 读 parquet / 对齐 sample_id
- 构建 prompt（chat messages -> chat_template）
- completion 解析/容错
- correctness 分解：feasible_ok / expr_ok / overall_ok
- pass@k 估计、bootstrap CI
- 轨迹/答案的离散化 label，用于 H(A)、H(Y|A)
"""

from __future__ import annotations

import ast
import json
import math
import re
import random
import hashlib
import glob
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from early_branch_locking._repo import (
    COUNTDOWN_DATA_ROOT,
    FIGURES_DIR,
    METRICS_DIR,
    RAW_DIR,
    TEST_PARQUET,
)

ANALYSIS_ROOT = COUNTDOWN_DATA_ROOT


# ----------------------------
# 数据读取与对齐
# ----------------------------

def load_parquet_sorted(path, n: int, sort_key: str = "sample_id") -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    if sort_key in df.columns:
        df = df.sort_values(sort_key)
    else:
        df = df.sort_index()
    if n is not None and n > 0:
        df = df.head(n)
    return df.to_dict("records")


def extract_ground_truth(rec: dict) -> Tuple[List[int], int, str]:
    """
    兼容你的 parquet 结构：rec['reward_model']['ground_truth']
    """
    rm_data = rec.get("reward_model", {})
    if isinstance(rm_data, str):
        try:
            rm_data = json.loads(rm_data)
        except Exception:
            rm_data = {}

    gt_data = (rm_data or {}).get("ground_truth", {})
    if isinstance(gt_data, str):
        try:
            gt_data = json.loads(gt_data)
        except Exception:
            gt_data = {}

    numbers = gt_data.get("numbers", [])
    target = gt_data.get("target", 0)
    feasible_label = str(gt_data.get("feasible_label", "yes")).lower()
    return numbers, target, feasible_label


def get_prompt_content(rec: dict):
    """
    取 parquet 的 prompt 字段，兼容 numpy/list
    """
    prompt_content = rec.get("prompt")
    if hasattr(prompt_content, "tolist"):
        prompt_content = prompt_content.tolist()
    return prompt_content


def build_prompt_text(prompt_content, tokenizer) -> str:
    """
    把 prompt_content 变成最终可送给 vLLM 的 prompt 文本。
    - 如果是 chat messages list，则用 apply_chat_template(add_generation_prompt=True)
    - 否则直接 str()
    """
    if isinstance(prompt_content, list):
        return tokenizer.apply_chat_template(
            prompt_content, tokenize=False, add_generation_prompt=True
        )
    return str(prompt_content)


def build_conditioned_prompt_text(prompt_content, prefix_text: str, tokenizer) -> str:
    """
    关键：prefix-conditioning
    - 如果 prompt_content 是 chat messages list：先构建 add_generation_prompt 的 prompt，
      然后把 prefix_text 直接追加到 assistant 同一轮的开头，实现“强制前缀后续写”。
    - 否则：退化为文本拼接（不保证等价，但至少可跑）；强烈建议 prompt 是 list 才可靠。
    """
    if isinstance(prompt_content, list):
        base_prompt = tokenizer.apply_chat_template(
            prompt_content, tokenize=False, add_generation_prompt=True
        )
        return base_prompt + prefix_text
    # fallback：纯文本 prompt
    return str(prompt_content) + "\n\n" + prefix_text


# ----------------------------
# pass@k 与 bootstrap CI
# ----------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Chen et al. 无偏估计：
        pass@k = 1 - C(n-c,k)/C(n,k)
    """
    if n <= 0 or k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def bootstrap_ci_mean(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """
    对题级（problem-level）values 做 bootstrap 置信区间
    返回 (mean, lo, hi)
    """
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    arr = values[:]
    m = float(np.mean(arr))
    bs = []
    for _ in range(n_boot):
        sample = [arr[rng.randrange(len(arr))] for _ in range(len(arr))]
        bs.append(float(np.mean(sample)))
    bs.sort()
    lo = bs[int((alpha / 2) * n_boot)]
    hi = bs[int((1 - alpha / 2) * n_boot) - 1]
    return m, lo, hi


# ----------------------------
# completion 解析（容错）
# ----------------------------

TAG_FEASIBLE_RE = re.compile(r"<feasible>\s*(yes|no)\s*</feasible>", re.IGNORECASE | re.DOTALL)
TAG_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TAG_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def tolerant_parse_completion(text: str) -> Dict[str, str]:
    """
    容错解析：即便标签不闭合也尽量提取
    返回：
      feasible_pred: "yes"/"no"/""
      answer_block:  str 或 ""
      think_block:   str 或 ""
    """
    out = {"feasible_pred": "", "answer_block": "", "think_block": ""}

    m = TAG_FEASIBLE_RE.search(text)
    if m:
        out["feasible_pred"] = (m.group(1) or "").strip().lower()

    m = TAG_ANSWER_RE.search(text)
    if m:
        out["answer_block"] = (m.group(1) or "").strip()

    m = TAG_THINK_RE.search(text)
    if m:
        out["think_block"] = (m.group(1) or "").strip()

    # 轻度容错：如果没有闭合 answer，但出现 <answer>，取到结尾
    if not out["answer_block"]:
        idx = text.lower().find("<answer>")
        if idx != -1:
            out["answer_block"] = text[idx + len("<answer>"):].strip()

    if not out["feasible_pred"]:
        # 容错：找 "<feasible>yes" 之类
        idx = text.lower().find("<feasible>")
        if idx != -1:
            tail = text[idx + len("<feasible>"): idx + len("<feasible>") + 10].lower()
            if "yes" in tail:
                out["feasible_pred"] = "yes"
            elif "no" in tail:
                out["feasible_pred"] = "no"

    return out


# ----------------------------
# Countdown correctness 分解
# ----------------------------

@dataclass
class CountdownEvalResult:
    feasible_pred: str
    answer_block: str
    think_block: str
    feasible_ok: bool
    expr_ok: bool
    overall_ok: bool
    # 额外：方便后续 B 实验
    answer_label: str  # 离散化后的 A
    trace_label: str   # 离散化后的 Y（简化版：hash/截断文本）
    expr_status: str
    canonical_expr: Optional[str]
    opseq_label: str
    parse_status: str
    has_feasible_tag: bool
    has_answer_tag: bool
    tag_order_ok: Optional[bool]


def make_trace_label(text: str, max_chars: int = 256) -> str:
    """
    用于 H(Y|A) 的简单轨迹离散化：取 think_block（优先）或全体文本的前 max_chars，做稳定 hash。
    """
    t = (text or "").strip().replace("\r\n", "\n")
    t = t[:max_chars]
    h = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"TRACE::{h}"


def make_answer_label(feasible_pred: str, answer_block: str, expr_status: str, canonical_key: Optional[str]) -> str:
    """
    A 的离散化 label：
    - INVALID：缺关键字段
    - NO_SOLUTION：声明无解
    - BAD_EXPR：有 answer 但表达式解析/执行失败
    - EXPR::<canonical>：有表达式
    """
    if not feasible_pred or not answer_block:
        return "INVALID"

    if feasible_pred == "no":
        # 无解
        return "NO_SOLUTION" if answer_block.strip().upper().startswith("NO_SOLUTION") else "NO_SOLUTION_BAD"
    # feasible_pred == "yes"
    if expr_status != "OK":
        return f"BAD_EXPR::{expr_status}"
    if canonical_key is None:
        return "EXPR::CANON_NONE"
    return f"EXPR::{canonical_key}"


_EXPR_ALLOWED_RE = re.compile(r"^[0-9+\-*/().\s]+$")


def _normalize_number(value: float | int) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(int(value))


def _flatten_commutative(node: ast.AST, op_type: type) -> List[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
        return _flatten_commutative(node.left, op_type) + _flatten_commutative(node.right, op_type)
    return [node]


def _canonicalize_ast(node: ast.AST) -> Tuple[str, List[str]]:
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in (ast.Add, ast.Mult):
            op_symbol = "+" if op_type is ast.Add else "*"
            items = []
            for child in _flatten_commutative(node, op_type):
                c_str, c_seq = _canonicalize_ast(child)
                items.append((c_str, c_seq))
            items.sort(key=lambda x: x[0])
            combined = f"({op_symbol.join(s for s, _ in items)})"
            seq = []
            for _, s in items:
                seq.extend(s)
            seq.append(op_symbol)
            return combined, seq
        if op_type in (ast.Sub, ast.Div):
            op_symbol = "-" if op_type is ast.Sub else "/"
            left_str, left_seq = _canonicalize_ast(node.left)
            right_str, right_seq = _canonicalize_ast(node.right)
            return f"({left_str}{op_symbol}{right_str})", left_seq + right_seq + [op_symbol]
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise ValueError("Only unary minus is supported")
        child_str, child_seq = _canonicalize_ast(node.operand)
        return f"(-{child_str})", child_seq + ["neg"]
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return _normalize_number(node.value), []
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def canonicalize_expression(expr: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Canonicalize an arithmetic expression under commutativity/associativity (+,*),
    returning (canonical_expr, opseq_label). Returns (None, None) on parse failure.
    """
    if expr is None:
        return None, None
    s = expr.strip()
    if not s:
        return None, None
    if not _EXPR_ALLOWED_RE.match(s):
        return None, None
    try:
        tree = ast.parse(s, mode="eval")
        canon, seq = _canonicalize_ast(tree.body)
        opseq = "OPSEQ::" + "".join(seq) if seq else "OPSEQ::"
        return canon, opseq
    except Exception:
        return None, None

def evaluate_countdown_completion(
    text: str,
    numbers: List[int],
    target: int,
    feasible_label: str,
    parse_countdown_completion,
    evaluate_countdown_expression,
) -> CountdownEvalResult:
    """
    - parse：容错
    - feasible_ok：与 gt feasible_label 是否一致
    - expr_ok：表达式是否用全数字且等于 target（或 NO_SOLUTION）
    """
    parsed = tolerant_parse_completion(text)
    feasible_pred = (parsed.get("feasible_pred") or "").lower().strip()
    answer_block = parsed.get("answer_block") or ""
    think_block = parsed.get("think_block") or ""

    text_lower = (text or "").lower()
    has_feasible_tag = "<feasible>" in text_lower
    has_answer_tag = "<answer>" in text_lower
    feasible_idx = text_lower.find("<feasible>")
    answer_idx = text_lower.find("<answer>")
    tag_order_ok = None
    if feasible_idx != -1 and answer_idx != -1:
        tag_order_ok = feasible_idx < answer_idx

    if not has_feasible_tag and not has_answer_tag:
        parse_status = "MISSING_BOTH_TAGS"
    elif not has_feasible_tag:
        parse_status = "MISSING_FEASIBLE_TAG"
    elif not has_answer_tag:
        parse_status = "MISSING_ANSWER_TAG"
    elif not feasible_pred:
        parse_status = "MALFORMED_FEASIBLE_TAG"
    elif not answer_block:
        parse_status = "MALFORMED_ANSWER_TAG"
    else:
        parse_status = "OK"

    gt_feasible = (feasible_label or "").lower().strip()
    feasible_ok = bool(gt_feasible == "" or feasible_pred == gt_feasible)

    expr_ok = False
    expr_status = "NO_PARSE"
    canonical_expr = None
    opseq_label = "OPSEQ::"
    canonical_key = None

    if feasible_pred == "yes":
        # 用你已有 evaluator（强验证）
        try:
            eval_res = evaluate_countdown_expression(answer_block, numbers, target)
            expr_ok = bool(
                eval_res.get("uses_all_numbers")
                and eval_res.get("matches_target")
                and eval_res.get("error") is None
            )
            expr_status = "OK" if expr_ok else "WRONG_OR_INCOMPLETE"
            # 只有表达式整体可解析时才做 canonical（避免噪声）
            if eval_res.get("error") is None:
                canonical_expr, opseq_label = canonicalize_expression(answer_block)
                canonical_key = canonical_expr
        except Exception:
            expr_ok = False
            expr_status = "EVAL_EXCEPTION"
    elif feasible_pred == "no":
        expr_ok = answer_block.strip().upper().startswith("NO_SOLUTION")
        expr_status = "OK" if expr_ok else "BAD_NO_SOLUTION"
    else:
        expr_ok = False
        expr_status = "NO_FEASIBLE_TAG"

    overall_ok = bool(feasible_ok and expr_ok)
    answer_label = make_answer_label(feasible_pred, answer_block, expr_status, canonical_key)
    trace_label = make_trace_label(think_block if think_block else text)

    return CountdownEvalResult(
        feasible_pred=feasible_pred,
        answer_block=answer_block,
        think_block=think_block,
        feasible_ok=feasible_ok,
        expr_ok=expr_ok,
        overall_ok=overall_ok,
        answer_label=answer_label,
        trace_label=trace_label,
        expr_status=expr_status,
        canonical_expr=canonical_expr,
        opseq_label=opseq_label,
        parse_status=parse_status,
        has_feasible_tag=has_feasible_tag,
        has_answer_tag=has_answer_tag,
        tag_order_ok=tag_order_ok,
    )

# ----------------------------
# 信息熵工具
# ----------------------------

def entropy_from_counts(counts: Dict[str, int]) -> float:
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / tot
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return float(h)


# ----------------------------
# JSONL 通用加载
# ----------------------------

def load_jsonl(path: Path):
    """Yield parsed dicts from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ----------------------------
# checkpoint 工具
# ----------------------------

def step_of(ckpt: str) -> int:
    """Extract integer step from a checkpoint name like 'global_step_50'."""
    if ckpt.startswith("global_step_"):
        try:
            return int(ckpt.split("_")[-1])
        except Exception:
            return 0
    return 0


def collect_model_paths(
    base_model_path: Optional[str] = None,
    include_base: bool = False,
    actor_dir: Optional[str] = None,
    only_steps: Optional[str] = "",
    explicit_paths: Optional[List[str]] = None,
) -> List[str]:
    """
    Collect model paths from actor_dir (global_step_*), base model, and
    explicit paths. Deduplicates while preserving order.
    """
    paths: List[str] = []
    if include_base and base_model_path:
        paths.append(base_model_path)
    if explicit_paths:
        paths.extend(explicit_paths)

    only_steps_set: set = set()
    if only_steps and only_steps.strip():
        for s in only_steps.split(","):
            s = s.strip()
            if s:
                only_steps_set.add(int(s))

    if actor_dir:
        actor_dir_path = Path(actor_dir)
        if actor_dir_path.exists():
            def _step_key(p: Path) -> int:
                try:
                    return int(p.name.split("_")[-1])
                except ValueError:
                    return 10 ** 12

            for d in sorted(actor_dir_path.glob("global_step_*"), key=_step_key):
                if not d.is_dir():
                    continue
                if only_steps_set:
                    try:
                        st = int(d.name.split("_")[-1])
                    except ValueError:
                        continue
                    if st not in only_steps_set:
                        continue
                paths.append(str(d))

    seen: set = set()
    uniq: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ----------------------------
# 数学工具
# ----------------------------

def logsumexp(vals: List[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not vals:
        return float("-inf")
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def gini_from_probs(probs: List[float]) -> float:
    """Gini coefficient from a probability vector."""
    if not probs:
        return 0.0
    arr = np.array(sorted(probs), dtype=float)
    if arr.sum() <= 0:
        return 0.0
    n = len(arr)
    cum = np.cumsum(arr)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


# ----------------------------
# 统一解枚举
# ----------------------------

def _is_target_fraction(val: Fraction, target: int) -> bool:
    return val == Fraction(target, 1)


def enumerate_solution_set(numbers: List[int], target: int) -> set:
    """
    Enumerate all distinct canonical solutions for a Countdown problem.
    Returns a set of canonical expression strings.
    """
    items = [(Fraction(n), str(n)) for n in numbers]
    solutions: set = set()

    def search(items_local: List[Tuple[Fraction, str]]):
        if len(items_local) == 1:
            val, expr = items_local[0]
            if _is_target_fraction(val, target):
                canon, _ = canonicalize_expression(expr)
                if canon:
                    solutions.add(canon)
            return
        n_items = len(items_local)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                a_val, a_expr = items_local[i]
                b_val, b_expr = items_local[j]
                rest = [items_local[k] for k in range(n_items) if k not in (i, j)]

                search(rest + [(a_val + b_val, f"({a_expr}+{b_expr})")])
                search(rest + [(a_val * b_val, f"({a_expr}*{b_expr})")])
                search(rest + [(a_val - b_val, f"({a_expr}-{b_expr})")])
                search(rest + [(b_val - a_val, f"({b_expr}-{a_expr})")])
                if b_val != 0:
                    search(rest + [(a_val / b_val, f"({a_expr}/{b_expr})")])
                if a_val != 0:
                    search(rest + [(b_val / a_val, f"({b_expr}/{a_expr})")])

    search(items)
    return solutions


def enumerate_solution_list(numbers: List[int], target: int) -> List[str]:
    """Same as enumerate_solution_set but returns a sorted list."""
    return sorted(enumerate_solution_set(numbers, target))


def tree_signature(expr: str) -> str:
    """
    Produce a structural signature from an arithmetic expression AST,
    ignoring concrete numeric values but preserving operator/tree shape.
    Canonical across the codebase (used by entrance_entropy, auxiliary_entropy, representation_geometry).
    """
    if not expr:
        return "TREE::NA"
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return "TREE::PARSE_FAIL"

    def _sig(node) -> str:
        if isinstance(node, ast.Expression):
            return _sig(node.body)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type is ast.Add:
                op = "+"
            elif op_type is ast.Sub:
                op = "-"
            elif op_type is ast.Mult:
                op = "*"
            elif op_type is ast.Div:
                op = "/"
            else:
                op = "?"
            return f"({op}{_sig(node.left)}{_sig(node.right)})"
        if isinstance(node, ast.UnaryOp):
            return f"(neg{_sig(node.operand)})"
        if isinstance(node, ast.Constant):
            return "N"
        return "?"

    return "TREE::" + _sig(tree)
