"""
Utility helpers shared by both analysis branches for the countdown-feasible task.

The GSM8K specific logic that previously lived inside each analysis script is
replaced with the abstractions defined here.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

HF_MIRROR_URL = "https://hf-mirror.com"


def ensure_hf_mirror_env(endpoint: str | None = None) -> None:
    """
    Ensure that every Hugging Face call is automatically routed through the
    mirror endpoint requested by the user environment.
    """
    target = endpoint or HF_MIRROR_URL
    for key in ("HF_ENDPOINT", "HF_HUB_BASE_URL"):
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = target


# ---------------------------------------------------------------------------
# Prompt parsing utilities
# ---------------------------------------------------------------------------

_QWEN_BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\s*(.*?)\s*<\|im_end\|>", re.S)


def parse_qwen_prompt_blocks(raw_prompt: str) -> Dict[str, str]:
    """
    Split a Qwen-style prompt (with <|im_start|> tokens) into role-specific blocks.
    """
    blocks: Dict[str, str] = {}
    last_end = 0
    for match in _QWEN_BLOCK_RE.finditer(raw_prompt):
        role = match.group(1).strip().lower()
        blocks[role] = match.group(2).strip()
        last_end = match.end()

    # The assistant prefill is not closed with <|im_end|> because generation
    # starts right afterwards. Capture it if present.
    trailing = raw_prompt[last_end:]
    assistant_preface = re.search(r"<\|im_start\|>assistant\s*(.*)", trailing, flags=re.S)
    if assistant_preface:
        blocks["assistant_preface"] = assistant_preface.group(1).strip()
    return blocks


_USER_SEGMENT_ORDER = [
    ("task_numbers", r"(Using the numbers .*?equals .*?\.)"),
    ("allowed_operations", r"(You may use .*?once\.)"),
    ("private_reasoning", r"(First, .*?<think></think>.*?\.)"),
    ("feasible_gate", r"(Then output .*?</feasible>.*?\.)"),
    ("answer_if_feasible", r"(If the task is feasible, .*?</answer>.*?\.)"),
    ("answer_if_infeasible", r"(If the task is impossible, .*?</answer>.*?\.)"),
]


def _extract_user_segments(user_text: str) -> List[Dict[str, str]]:
    """
    Break the countdown user instruction into human-interpretable segments that
    downstream scripts can reason about individually.
    """
    remaining = user_text
    segments: List[Dict[str, str]] = []

    for name, pattern in _USER_SEGMENT_ORDER:
        match = re.search(pattern, remaining, flags=re.S | re.I)
        if not match:
            continue
        chunk = match.group(1).strip()
        start, end = match.span(1)
        remaining = remaining[:start] + remaining[end:]
        segment_type = "anchor" if name == "task_numbers" else "rules"
        if name in {"private_reasoning", "feasible_gate"}:
            segment_type = "process"
        if name.startswith("answer_if"):
            segment_type = "verification"

        segments.append(
            {
                "name": name,
                "text": chunk,
                "type": segment_type,
            }
        )

    leftover = remaining.strip()
    if leftover:
        segments.append({"name": "user_misc", "text": leftover, "type": "rules"})
    return segments


def build_structured_segments(raw_prompt: str) -> List[Dict[str, str]]:
    """
    Produce a canonical ordered list of prompt segments.
    """
    blocks = parse_qwen_prompt_blocks(raw_prompt)
    structured: List[Dict[str, str]] = []

    system_text = blocks.get("system", "").strip()
    if system_text:
        structured.append(
            {"name": "system_instructions", "text": system_text, "type": "system"}
        )

    user_text = blocks.get("user", "").strip()
    if user_text:
        structured.extend(_extract_user_segments(user_text))

    assistant_preface = blocks.get("assistant_preface")
    if assistant_preface:
        structured.append(
            {
                "name": "assistant_preface",
                "text": assistant_preface,
                "type": "assistant",
            }
        )

    return structured


# ---------------------------------------------------------------------------
# Dataclasses and dataset loading
# ---------------------------------------------------------------------------


@dataclass
class CountdownSample:
    """
    Lightweight container with the metadata required by both analysis branches.
    """

    prompt_messages: List[Dict[str, str]]
    raw_prompt: str
    structured_segments: List[Dict[str, str]]
    numbers: List[int]
    target: int | None
    feasible_label: str
    has_solution: bool
    sample_type: str
    split: str
    sample_id: str
    source_index: int
    extra_metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def structured_context(self) -> str:
        parts = []
        for seg in self.structured_segments:
            cleaned = seg["text"].strip()
            if not cleaned:
                continue
            parts.append(f"[{seg['name']}] {cleaned}")
        return "\n\n".join(parts)

    def anchor_text(self) -> str:
        parts = [
            seg["text"].strip()
            for seg in self.structured_segments
            if seg["type"] in {"system", "anchor", "rules"}
        ]
        return "\n".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Helper: robust field extraction from parquet / jsonl records
# ---------------------------------------------------------------------------


def _is_nan_like(x: Any) -> bool:
    """Return True if x is NaN / None-ish."""
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    return False


def _maybe_json_load(x: Any) -> Any:
    """如果是 str 且看起来像 JSON，就尝试 json.loads，否则直接返回原值。"""
    if isinstance(x, str):
        s = x.strip()
        if s and (s[0] in "{[" and s[-1] in "}]"):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def _to_list_if_array(x: Any) -> Any:
    """如果是 numpy array，把它转成 Python list。"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _extract_gt_from_record(data: Dict[str, Any], default_index: int):
    """
    从多种格式中提取 GT：
      - train/test.parquet:   data['reward_model']['ground_truth']
      - model_predictions:    顶层 numbers / target / feasible_label / has_solution
    """
    # ---- reward_model / ground_truth ----
    rm = data.get("reward_model", None)
    rm = _maybe_json_load(rm)
    if _is_nan_like(rm):
        rm = {}
    if not isinstance(rm, dict):
        rm = {}

    gt = rm.get("ground_truth", {})
    gt = _maybe_json_load(gt)
    if _is_nan_like(gt) or not isinstance(gt, dict):
        gt = {}

    # ---- numbers ----
    numbers = data.get("numbers", None)
    if _is_nan_like(numbers):
        numbers = gt.get("numbers", [])
    numbers = _to_list_if_array(numbers)
    if numbers is None:
        numbers = []
    if not isinstance(numbers, (list, tuple)):
        numbers = [numbers]
    # 尽量转成 int 列表
    cleaned_numbers: List[int] = []
    for v in numbers:
        try:
            cleaned_numbers.append(int(v))
        except Exception:
            continue

    # ---- target ----
    target = data.get("target", None)
    if _is_nan_like(target):
        target = gt.get("target", None)
    try:
        if not _is_nan_like(target):
            target = int(target)
        else:
            target = None
    except Exception:
        target = None

    # ---- feasible_label ----
    feasible_label_val = data.get("feasible_label", None)
    if _is_nan_like(feasible_label_val):
        feasible_label_val = gt.get("feasible_label", "")
    feasible_label = (str(feasible_label_val) or "").strip().lower()

    # ---- has_solution ----
    has_solution_raw = None
    if "has_solution" in data and not _is_nan_like(data.get("has_solution")):
        has_solution_raw = data.get("has_solution")
    elif "has_solution" in gt and not _is_nan_like(gt.get("has_solution")):
        has_solution_raw = gt.get("has_solution")

    # 展平单元素列表 / 数组
    if isinstance(has_solution_raw, (list, tuple, np.ndarray)) and len(has_solution_raw) == 1:
        has_solution_raw = has_solution_raw[0]

    if isinstance(has_solution_raw, (bool, int)):
        has_solution = bool(has_solution_raw)
    else:
        has_solution = False

    # ---- meta 信息 ----
    extra_info = data.get("extra_info", {}) or {}
    extra_info = _maybe_json_load(extra_info)
    if not isinstance(extra_info, dict):
        extra_info = {}

    try:
        source_index = int(extra_info.get("index", default_index))
    except Exception:
        source_index = default_index

    sample_identifier = data.get("sample_id", None)
    if sample_identifier is None:
        sample_identifier = f"{default_index}_{source_index}"

    sample_type = extra_info.get("sample_type", data.get("sample_type", ""))
    split = extra_info.get("split", data.get("split", ""))

    return {
        "numbers": cleaned_numbers,
        "target": target,
        "feasible_label": feasible_label,
        "has_solution": has_solution,
        "sample_type": sample_type,
        "split": split,
        "sample_id": str(sample_identifier),
        "source_index": int(source_index),
    }


def _build_sample_from_dict(data: Dict[str, Any], line_idx: int) -> CountdownSample:
    """
    将一条 dict 记录（来自 jsonl 或 parquet）转成 CountdownSample。
    """
    prompt_obj = data.get("prompt", None)

    # 某些 parquet 记录里，prompt 字段会变成 numpy.ndarray（元素仍是 dict）。
    # 这时若直接 str(prompt_obj) 会得到 "[{'content': ...}]" 形式的文本，
    # 模型看到的其实是“字典字符串”，导致 Chat 模板失效。
    # 因此一旦能安全地转成 list[dict]，就优先转换。
    if "np" in globals():
        try:
            if isinstance(prompt_obj, np.ndarray):
                prompt_obj = prompt_obj.tolist()
            elif hasattr(prompt_obj, "tolist") and not isinstance(prompt_obj, (list, tuple, dict, str)):
                maybe = prompt_obj.tolist()
                if isinstance(maybe, list):
                    prompt_obj = maybe
        except Exception:
            # tolist() 失败则保持原样
            pass
    raw_prompt: str
    prompt_messages: List[Dict[str, str]]

    if _is_nan_like(prompt_obj) or prompt_obj is None:
        # 没有 prompt，就退回到整个 dict 的字符串版
        raw_prompt = str(data)
        prompt_messages = [{"role": "user", "content": raw_prompt}]
    else:
        # 兼容三种常见情况：
        #   1) Qwen-style 完整 prompt 字符串
        #   2) list[{"role","content"}]
        #   3) 其他对象（比如 tuple / numpy array）：都转成 str
        if isinstance(prompt_obj, list) and all(
            isinstance(m, dict) and "content" in m for m in prompt_obj
        ):
            # 按 messages 恢复 Qwen chat 模板；若 content 已包含 <|im_start|> 块，则视为完整 prompt。
            concatenated = ""
            for msg in prompt_obj:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content.strip().startswith("<|im_start|>"):
                    snippet = content
                else:
                    snippet = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                if not snippet.endswith("\n"):
                    snippet += "\n"
                concatenated += snippet
            raw_prompt = concatenated
            prompt_messages = prompt_obj
        elif isinstance(prompt_obj, str):
            raw_prompt = prompt_obj
            prompt_messages = [{"role": "user", "content": raw_prompt}]
        else:
            # numpy array / tuple / 其它结构
            raw_prompt = str(prompt_obj)
            prompt_messages = [{"role": "user", "content": raw_prompt}]

    structured = build_structured_segments(raw_prompt)
    gt_info = _extract_gt_from_record(data, line_idx)

    return CountdownSample(
        prompt_messages=prompt_messages,
        raw_prompt=raw_prompt,
        structured_segments=structured,
        numbers=gt_info["numbers"],
        target=gt_info["target"],
        feasible_label=gt_info["feasible_label"],
        has_solution=gt_info["has_solution"],
        sample_type=gt_info["sample_type"],
        split=gt_info["split"],
        sample_id=gt_info["sample_id"],
        source_index=gt_info["source_index"],
        extra_metadata=data,
    )


def load_countdown_jsonl(path: str | Path, limit: int | None = None) -> List[CountdownSample]:
    """
    Load countdown-feasible samples from a JSONL or Parquet file.

    支持三类文件：
      - *.jsonl : 每行一个 JSON（train.jsonl / test.jsonl / model_predictions_*.jsonl）
      - *.json  : 单一 JSON 行（几乎不用）
      - *.parquet : 官方 Countdown train/test parquet

    行为：
      - 若文件名包含 "train" 且样本数 > 10000，会固定 seed=42 抽样 10000 条；
      - limit 若不为 None，会在抽样后再做切片。
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Countdown dataset not found at {json_path}")

    suffix = json_path.suffix.lower()
    dataset: List[CountdownSample] = []

    # -------- 1) JSONL / JSON 路径 --------
    if suffix in {".jsonl", ".json"}:
        with json_path.open("r", encoding="utf-8") as reader:
            all_lines = [line.strip() for line in reader if line.strip()]

        TARGET_TRAIN_SIZE = 10000
        name_lower = json_path.name.lower()

        if "train" in name_lower and len(all_lines) > TARGET_TRAIN_SIZE:
            print(
                f"[INFO] Detected TRAIN jsonl ({json_path.name}). "
                f"Sampling N={TARGET_TRAIN_SIZE} with seed=42."
            )
            random.seed(42)
            all_lines = random.sample(all_lines, TARGET_TRAIN_SIZE)

        if limit is not None:
            all_lines = all_lines[:limit]

        for line_idx, line in enumerate(all_lines):
            data = json.loads(line)
            sample = _build_sample_from_dict(data, line_idx)
            dataset.append(sample)

        return dataset

    # -------- 2) Parquet 路径 --------
    if suffix == ".parquet":
        print(f"[INFO] Loading Countdown parquet from {json_path} ...")
        df = pd.read_parquet(json_path)
        records = df.to_dict("records")

        TARGET_TRAIN_SIZE = 10000
        name_lower = json_path.name.lower()

        if "train" in name_lower and len(records) > TARGET_TRAIN_SIZE:
            print(
                f"[INFO] Detected TRAIN parquet ({json_path.name}). "
                f"Sampling N={TARGET_TRAIN_SIZE} with seed=42."
            )
            random.seed(42)
            indices = list(range(len(records)))
            indices = random.sample(indices, TARGET_TRAIN_SIZE)
            records = [records[i] for i in indices]

        if limit is not None:
            records = records[:limit]

        for idx, row in enumerate(records):
            data: Dict[str, Any] = dict(row)
            sample = _build_sample_from_dict(data, idx)
            dataset.append(sample)

        return dataset

    raise ValueError(f"Unsupported countdown dataset format: {json_path}")


# ---------------------------------------------------------------------------
# Completion parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_FEASIBLE_RE = re.compile(r"<feasible>\s*(yes|no)\s*</feasible>", re.S | re.I)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)


def parse_countdown_completion(text: str) -> Dict[str, object]:
    """
    Extract structured information from a countdown model completion.
    Supports implicit reasoning blocks where <think> tags are missing or unclosed.
    """
    think_match = _THINK_RE.search(text)
    feasible_match = _FEASIBLE_RE.search(text)
    answer_match = _ANSWER_RE.search(text)

    if think_match:
        # 1. 完美匹配到 <think>...</think>
        think_block = think_match.group(1).strip()
    else:
        # 2. 回退策略：如果找不到完整的 think 块
        lower_text = text.lower()

        # 寻找推理的起始点
        start_idx = 0
        think_start_tag = lower_text.find("<think>")
        if think_start_tag != -1:
            start_idx = think_start_tag + 7  # len("<think>")

        # 寻找推理的终止点 (通常是 <feasible> 或 <answer> 的开始)
        end_idx = len(text)

        # 查找后续标签的位置
        feasible_idx = lower_text.find("<feasible>")
        answer_idx = lower_text.find("<answer>")

        candidates = []
        if feasible_idx != -1:
            candidates.append(feasible_idx)
        if answer_idx != -1:
            candidates.append(answer_idx)

        if candidates:
            end_idx = min(candidates)

        # 截取中间内容作为 reasoning
        extracted = text[start_idx:end_idx].strip()

        # 简单的启发式过滤：如果提取出来的内容包含了 <|im_end|> 等特殊token，去掉它
        if "<|im_end|>" in extracted:
            extracted = extracted.split("<|im_end|>")[0].strip()

        think_block = extracted

    feasible_pred = (feasible_match.group(1).lower() if feasible_match else "")
    answer_block = answer_match.group(1).strip() if answer_match else ""

    think_lines = [ln.strip() for ln in think_block.splitlines() if ln.strip()]
    review_lines = [
        ln
        for ln in think_lines
        if re.search(r"\b(check|verify|confirm|test|recompute)\b", ln.lower())
    ]

    return {
        "think_block": think_block,
        "think_line_count": len(think_lines),
        "review_line_count": len(review_lines),
        "feasible_pred": feasible_pred,
        "answer_block": answer_block,
        "has_private_reasoning": bool(think_block),
    }


def summarize_behavior(sample: CountdownSample, completion_text: str) -> Dict[str, object]:
    parsed = parse_countdown_completion(completion_text)
    feasible_correct = (
        parsed["feasible_pred"] == sample.feasible_label if sample.feasible_label else None
    )
    return {
        "sample_id": sample.sample_id,
        "target": sample.target,
        "numbers": sample.numbers,
        "feasible_label": sample.feasible_label,
        "feasible_pred": parsed["feasible_pred"],
        "feasible_correct": feasible_correct,
        "think_line_count": parsed["think_line_count"],
        "review_line_count": parsed["review_line_count"],
    }


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}


def evaluate_countdown_expression(expression: str, numbers: List[int], target: int) -> Dict[str, object]:
    """
    Evaluate a candidate Countdown expression and check:
      - 是否只使用了给定的 numbers，且每个数字恰好一次（按数值多重集比较）
      - 表达式值是否等于 target（浮点容差 1e-6）
    """
    constants: List[int] = []

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            return _ALLOWED_BIN_OPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = _eval(node.operand)
            return _ALLOWED_UNARY[op_type](operand)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
            value = float(node.value)
            if isinstance(node.value, int):
                constants.append(int(abs(node.value)))
            return value
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    result_value = None
    error = None
    try:
        tree = ast.parse(expression, mode="eval")
        result_value = _eval(tree)
    except Exception as exc:
        error = str(exc)

    expected = sorted(numbers)
    observed = sorted(constants)
    uses_all_numbers = observed == expected

    matches_target = False
    if error is None and result_value is not None:
        matches_target = abs(result_value - target) < 1e-6

    return {
        "uses_all_numbers": uses_all_numbers,
        "value": result_value,
        "matches_target": matches_target,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Countdown brute-force feasibility solver
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _can_reach_target_cached(nums_tuple: tuple[int, ...], target: int) -> bool:
    """
    递归暴力求解 Countdown 可行性：
      - nums_tuple: 排好序的整数元组（允许重复）
      - target:    目标整数

    算法：
      - 取任意一对数字 (a, b)，尝试 +, -, 反向 -, *, /, 反向 /
      - 每次生成一个新数字 r，并递归地在剩余数字 + {r} 上继续搜索
      - 限制在整数运算：除法仅当整除时才允许
    """
    nums = list(nums_tuple)
    n = len(nums)
    if n == 0:
        return False
    if n == 1:
        return nums[0] == target

    # 为避免重复状态，枚举有序对索引 i < j
    for i in range(n):
        for j in range(i + 1, n):
            a = nums[i]
            b = nums[j]
            rest = [nums[k] for k in range(n) if k not in (i, j)]

            results: list[int] = []

            # 加法和乘法对称，只考虑一种顺序即可
            results.append(a + b)
            results.append(a * b)

            # 减法：两种方向都算
            results.append(a - b)
            results.append(b - a)

            # 除法：只在整除时考虑，且避免 0 做除数
            if b != 0 and a % b == 0:
                results.append(a // b)
            if a != 0 and b % a == 0:
                results.append(b // a)

            for r in results:
                new_nums = rest + [r]
                new_tuple = tuple(sorted(new_nums))
                if _can_reach_target_cached(new_tuple, target):
                    return True
    return False


def countdown_has_solution(numbers: List[int] | tuple[int, ...], target: int | None) -> bool:
    """
    对外封装的 Countdown 可行性判定函数。

    Args:
        numbers: 初始数字列表（或元组），例如 [83, 87, 88]
        target:  目标数字；若为 None 或无效，直接返回 False

    Returns:
        bool: 是否存在使用每个数字恰好一次、仅用 + - * / 的表达式，
              使得表达式值等于 target。
    """
    if target is None:
        return False
    try:
        t = int(target)
    except Exception:
        return False

    cleaned: List[int] = []
    for v in numbers:
        try:
            cleaned.append(int(v))
        except Exception:
            continue

    if not cleaned:
        return False

    nums_tuple = tuple(sorted(cleaned))
    return _can_reach_target_cached(nums_tuple, t)
