#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for non-enumerable math trace-diversity experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

TRACE_NUMBER_LIMIT = 24
ANSWER_PLACEHOLDER = "<unparsed>"
TRACE_PLACEHOLDER = "<no_trace>"
BRANCH_PLACEHOLDER = "<no_calc>"
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
CALC_RE = re.compile(
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"([+\-*/×÷])\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)
BOXED_RE = re.compile(r"\\boxed")
GSM_HASH_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.\d+)?)")


@dataclass(frozen=True)
class SampleEval:
    answer: str
    numeric_trace: str
    first_calc_branch: str
    is_correct: bool
    parsed: bool
    completion_chars: int


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or c <= 0 or k <= 0:
        return 0.0
    capped_k = min(k, n)
    if n - c < capped_k:
        return 1.0
    return 1.0 - (math.comb(n - c, capped_k) / math.comb(n, capped_k))


def shannon_entropy(labels: Iterable[str]) -> float:
    counts = Counter(label for label in labels if label)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def top_mass(labels: Iterable[str]) -> float:
    counts = Counter(label for label in labels if label)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(max(counts.values()) / total)


def normalized_number(text: str) -> str:
    value = text.strip().replace(",", "").replace("$", "")
    if value.endswith("%"):
        value = value[:-1] + "%"
    return value


def extract_numeric_trace(text: str) -> str:
    numbers = [normalized_number(match.group(0)) for match in NUMBER_RE.finditer(text)]
    if not numbers:
        return TRACE_PLACEHOLDER
    return "|".join(numbers[:TRACE_NUMBER_LIMIT])


def extract_first_calc_branch(text: str) -> str:
    match = CALC_RE.search(text)
    if not match:
        return BRANCH_PLACEHOLDER
    left = normalized_number(match.group(1))
    op = match.group(2).replace("×", "*").replace("÷", "/")
    right = normalized_number(match.group(3))
    return f"{left}{op}{right}"


def last_boxed_answer(text: str) -> str | None:
    last = None
    for match in BOXED_RE.finditer(text):
        last = match.start()
    if last is None:
        return None
    open_idx = text.find("{", last)
    if open_idx < 0:
        return None
    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        if char == "}":
            depth -= 1
        if depth == 0:
            return text[open_idx + 1 : idx].strip()
    return None


def extract_final_answer(text: str) -> str:
    boxed = last_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)
    hashed = GSM_HASH_RE.findall(text)
    if hashed:
        return normalize_answer(hashed[-1])
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])
    return ANSWER_PLACEHOLDER


def normalize_answer(text: str) -> str:
    value = text.strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "").replace(",", "")
    value = value.replace("\\%", "%").replace("%", "")
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", value)
    value = _normalize_frac(value)
    value = re.sub(r"\s+", "", value)
    value = value.strip(".。,:;，；")
    return value.lower() if value else ANSWER_PLACEHOLDER


def _normalize_frac(value: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while True:
        updated = pattern.sub(r"(\1)/(\2)", value)
        if updated == value:
            return updated
        value = updated


def answers_equivalent(prediction: str, ground_truth: str) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    non_finite_literals = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    if pred in non_finite_literals or gold in non_finite_literals:
        return False
    if pred == gold:
        return True
    pred_num = _decimal_or_none(pred)
    gold_num = _decimal_or_none(gold)
    if pred_num is None or gold_num is None or not pred_num.is_finite() or not gold_num.is_finite():
        return False
    try:
        return abs(pred_num - gold_num) <= Decimal("1e-8")
    except (InvalidOperation, ValueError):
        return False


def _decimal_or_none(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def evaluate_completion(completion: str, ground_truth: str) -> SampleEval:
    answer = extract_final_answer(completion)
    parsed = answer != ANSWER_PLACEHOLDER
    return SampleEval(
        answer=answer,
        numeric_trace=extract_numeric_trace(completion),
        first_calc_branch=extract_first_calc_branch(completion),
        is_correct=parsed and answers_equivalent(answer, ground_truth),
        parsed=parsed,
        completion_chars=len(completion),
    )


def majority_correct(evals: Sequence[SampleEval], ground_truth: str) -> float:
    parsed_answers = [item.answer for item in evals if item.parsed]
    if not parsed_answers:
        return 0.0
    answer, _ = Counter(parsed_answers).most_common(1)[0]
    return float(answers_equivalent(answer, ground_truth))


def problem_metrics(evals: Sequence[SampleEval], ground_truth: str, ks: Sequence[int]) -> dict:
    n = len(evals)
    correct = sum(item.is_correct for item in evals)
    answer_labels = [item.answer for item in evals]
    trace_labels = [item.numeric_trace for item in evals]
    branch_labels = [item.first_calc_branch for item in evals]
    correct_traces = [item.numeric_trace for item in evals if item.is_correct]
    correct_branches = [item.first_calc_branch for item in evals if item.is_correct]
    row = _base_metric_row(evals, ground_truth, n, correct)
    for k in ks:
        if k <= n:
            row[f"pass@{k}"] = pass_at_k(n, correct, k)
    row.update(_diversity_metrics(answer_labels, trace_labels, branch_labels))
    row.update(_correct_diversity_metrics(correct_traces, correct_branches))
    return row


def _base_metric_row(evals: Sequence[SampleEval], ground_truth: str, n: int, correct: int) -> dict:
    return {
        "n_samples": n,
        "num_correct": correct,
        "correct_rate": float(correct / n) if n else 0.0,
        "parse_rate": float(sum(item.parsed for item in evals) / n) if n else 0.0,
        "majority_accuracy": majority_correct(evals, ground_truth),
        "avg_completion_chars": _mean([item.completion_chars for item in evals]),
    }


def _diversity_metrics(answer_labels: Sequence[str], trace_labels: Sequence[str], branch_labels: Sequence[str]) -> dict:
    n = len(answer_labels)
    return {
        "answer_entropy": shannon_entropy(answer_labels),
        "top_answer_mass": top_mass(answer_labels),
        "numeric_trace_entropy": shannon_entropy(trace_labels),
        "observed_trace_coverage": _unique_fraction(trace_labels, n),
        "first_calc_branch_entropy": shannon_entropy(branch_labels),
        "unique_answer_count": len(set(answer_labels)),
        "unique_trace_count": len(set(trace_labels)),
        "unique_first_calc_branch_count": len(set(branch_labels)),
    }


def _correct_diversity_metrics(trace_labels: Sequence[str], branch_labels: Sequence[str]) -> dict:
    return {
        "correct_numeric_trace_entropy": shannon_entropy(trace_labels),
        "correct_first_calc_branch_entropy": shannon_entropy(branch_labels),
        "unique_correct_trace_count": len(set(trace_labels)),
        "unique_correct_first_calc_branch_count": len(set(branch_labels)),
    }


def _unique_fraction(labels: Sequence[str], denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(len(set(labels)) / denominator)


def summarize_problem_rows(rows: Sequence[dict], ks: Sequence[int]) -> dict:
    summary = {"num_problems": len(rows)}
    metric_names = _summary_metric_names(ks)
    for name in metric_names:
        values = [float(row[name]) for row in rows if name in row]
        if values:
            summary[f"{name}_mean"] = _mean(values)
    for threshold in (0.10, 0.25, 0.50):
        key = f"cover@{threshold:.2f}_mean"
        summary[key] = _mean([row["correct_rate"] >= threshold for row in rows])
    summary["self_consistency_gain_mean"] = (
        summary.get("majority_accuracy_mean", 0.0) - summary.get("pass@1_mean", 0.0)
    )
    return summary


def _summary_metric_names(ks: Sequence[int]) -> list[str]:
    names = [f"pass@{k}" for k in ks]
    names.extend(
        [
            "correct_rate",
            "parse_rate",
            "majority_accuracy",
            "answer_entropy",
            "top_answer_mass",
            "numeric_trace_entropy",
            "correct_numeric_trace_entropy",
            "observed_trace_coverage",
            "first_calc_branch_entropy",
            "correct_first_calc_branch_entropy",
            "unique_answer_count",
            "unique_trace_count",
            "unique_first_calc_branch_count",
            "unique_correct_trace_count",
            "avg_completion_chars",
        ]
    )
    return names


def _mean(values: Iterable[float | int | bool]) -> float:
    arr = [float(value) for value in values]
    if not arr:
        return 0.0
    return float(sum(arr) / len(arr))
