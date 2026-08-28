"""Strict native evaluator for the public Philschmid Countdown protocol.

The public run does not use the repository's ``<feasible>`` gate.  A generated
completion is therefore evaluated after reconstructing the synthetic opening
``<think>`` tag.  This module keeps that evaluator separate from the local
Countdown parser so the two prompt contracts cannot silently merge.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from early_branch_locking.core.countdown_shared import canonicalize_expression, enumerate_solution_set
from early_branch_locking.core.entrance_detection import find_first_reasoning_entrance


THINK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(?P<body>.*?)</answer>", re.IGNORECASE | re.DOTALL)
INTEGER_RE = re.compile(r"^[+-]?\d+$")
EXPRESSION_CHARS_RE = re.compile(r"^[0-9+*/().\s-]+$")
FIRST_ANSWER_RE = re.compile(r"(?<![\w.])(?P<first>-?\d+)\s*(?P<op>[+\-*/])")


@dataclass(frozen=True)
class NativeEvalResult:
    native_format_ok: bool
    has_think_tag: bool
    has_answer_tag: bool
    answer_expr: str
    answer_rhs: str | None
    uses_all_numbers: bool
    matches_target: bool
    overall_ok: bool
    canonical_expr: str | None
    opseq_label: str
    answer_entrance_family: str | None
    think_entrance_family: str | None
    parse_status: str
    value: str | None
    right_hand_side_matches_target: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_format_ok": self.native_format_ok,
            "has_think_tag": self.has_think_tag,
            "has_answer_tag": self.has_answer_tag,
            "answer_expr": self.answer_expr,
            "answer_rhs": self.answer_rhs,
            "uses_all_numbers": self.uses_all_numbers,
            "matches_target": self.matches_target,
            "overall_ok": self.overall_ok,
            "canonical_expr": self.canonical_expr,
            "opseq_label": self.opseq_label,
            "answer_entrance_family": self.answer_entrance_family,
            "think_entrance_family": self.think_entrance_family,
            "parse_status": self.parse_status,
            "value": self.value,
            "right_hand_side_matches_target": self.right_hand_side_matches_target,
        }


def _first_answer_family(expression: str) -> str | None:
    match = FIRST_ANSWER_RE.search(str(expression or ""))
    if match is None:
        return None
    return f"{int(match.group('first'))}{match.group('op')}"


def _split_equation(answer_block: str) -> tuple[str, str | None, str | None]:
    count = str(answer_block).count("=")
    if count > 1:
        return "", None, "MULTIPLE_EQUALS"
    if count == 0:
        return str(answer_block).strip(), None, None
    left, right = str(answer_block).split("=", 1)
    left, right = left.strip(), right.strip()
    if not left or not right or not INTEGER_RE.fullmatch(right):
        return "", right or None, "MALFORMED_EQUATION"
    return left, right, None


def _evaluate_exact_expression(expression: str, numbers: list[int], target: int) -> dict[str, Any]:
    """Evaluate the restricted arithmetic grammar with exact rationals."""

    constants: list[int] = []

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            constants.append(int(node.value))
            return Fraction(int(node.value), 1)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        raise ValueError(f"unsupported AST node: {type(node).__name__}")

    expression = str(expression or "").strip()
    result: Fraction | None = None
    error: str | None = None
    if not expression or not EXPRESSION_CHARS_RE.fullmatch(expression):
        error = "DISALLOWED_EXPRESSION_CHARACTERS"
    else:
        try:
            result = visit(ast.parse(expression, mode="eval"))
        except Exception as exc:  # evaluator boundary: return a structured failure
            error = f"{type(exc).__name__}: {exc}"
    observed = sorted(abs(value) for value in constants)
    expected = sorted(int(value) for value in numbers)
    uses_all_numbers = error is None and Counter(observed) == Counter(expected)
    matches_target = result is not None and result == Fraction(int(target), 1)
    return {
        "uses_all_numbers": uses_all_numbers,
        "matches_target": bool(matches_target),
        "value": str(result) if result is not None else None,
        "error": error,
    }


def evaluate_native_countdown(completion: str, numbers: list[int], target: int) -> NativeEvalResult:
    """Score one completion emitted after the public prompt's ``<think>`` prefill."""

    generated = str(completion or "")
    full = generated if generated.lstrip().lower().startswith("<think>") else "<think>" + generated
    think_matches = list(THINK_RE.finditer(full))
    answer_matches = list(ANSWER_RE.finditer(full))
    has_think_tag = len(think_matches) == 1
    has_answer_tag = len(answer_matches) == 1
    think_block = think_matches[0].group("body") if has_think_tag else ""
    answer_block = answer_matches[0].group("body").strip() if has_answer_tag else ""
    think_end = think_matches[0].end() if has_think_tag else -1
    answer_start = answer_matches[0].start() if has_answer_tag else -1
    native_format_ok = bool(has_think_tag and has_answer_tag and think_end <= answer_start)

    answer_expr, answer_rhs, split_error = _split_equation(answer_block) if has_answer_tag else ("", None, "MISSING_ANSWER_TAG")
    expression_eval = _evaluate_exact_expression(answer_expr, numbers, target) if not split_error else {
        "uses_all_numbers": False,
        "matches_target": False,
        "value": None,
        "error": split_error,
    }
    # `_split_equation` preserves malformed RHS text for diagnostics. Do not
    # let that diagnostic value escape the evaluator as an int conversion.
    rhs_matches = (
        None
        if answer_rhs is None
        else bool(INTEGER_RE.fullmatch(str(answer_rhs)) and int(answer_rhs) == int(target))
    )
    matches_target = bool(expression_eval["matches_target"] and (rhs_matches is not False))
    overall_ok = bool(native_format_ok and expression_eval["uses_all_numbers"] and matches_target)
    canonical_expr, opseq_label = (None, "OPSEQ::")
    if expression_eval["error"] is None:
        canonical_expr, opseq_label = canonicalize_expression(answer_expr)
    think_entrance = find_first_reasoning_entrance(think_block)
    status = "OK" if overall_ok else (expression_eval["error"] or "FORMAT_OR_WRONG_EXPRESSION")
    return NativeEvalResult(
        native_format_ok=native_format_ok,
        has_think_tag=has_think_tag,
        has_answer_tag=has_answer_tag,
        answer_expr=answer_expr,
        answer_rhs=answer_rhs,
        uses_all_numbers=bool(expression_eval["uses_all_numbers"]),
        matches_target=matches_target,
        overall_ok=overall_ok,
        canonical_expr=canonical_expr,
        opseq_label=opseq_label,
        answer_entrance_family=_first_answer_family(answer_expr),
        think_entrance_family=think_entrance.family if think_entrance.found else None,
        parse_status=status,
        value=expression_eval["value"],
        right_hand_side_matches_target=rhs_matches,
    )


__all__ = ["NativeEvalResult", "evaluate_native_countdown", "enumerate_solution_set"]
