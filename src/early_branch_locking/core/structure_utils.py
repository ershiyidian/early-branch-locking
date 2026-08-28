
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact-solver structure features for Countdown problems."""

from __future__ import annotations

import ast
import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from early_branch_locking.core.countdown_shared import canonicalize_expression, tree_signature, enumerate_solution_set


def enumerate_solution_records(numbers: List[int], target: int) -> List[dict]:
    """
    Enumerate all distinct canonical solutions with structural metadata.
    Uses the shared enumerate_solution_set for the core search, then
    enriches each canonical expression with metadata.
    """
    canonical_set = enumerate_solution_set(numbers, target)
    records: Dict[str, dict] = {}
    for canonical_expr in canonical_set:
        if canonical_expr not in records:
            records[canonical_expr] = _build_record(canonical_expr)

    # Also run the full recursive search to capture opseq_label properly
    # (since enumerate_solution_set only returns canonical exprs)
    items = tuple((Fraction(num, 1), str(num)) for num in numbers)
    _enrich_records(items, target, records)

    return list(records.values())


def _build_record(canonical_expr: str) -> dict:
    _, opseq_label = canonicalize_expression(canonical_expr)
    return {
        "canonical_expr": canonical_expr,
        "opseq_label": opseq_label or "OPSEQ::",
        "tree_signature": tree_signature(canonical_expr),
        "first_op": first_operator(canonical_expr),
        "depth": expression_depth(canonical_expr),
        "requires_division": "/" in canonical_expr,
        "requires_parentheses": "(" in canonical_expr,
    }


def _enrich_records(items, target, records: Dict[str, dict]) -> None:
    """Run the recursive search to fill in any missing opseq_label details."""
    def search(entries):
        if len(entries) == 1:
            value, expr = entries[0]
            if value == Fraction(target, 1):
                add_record(expr, records)
            return
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                rest = tuple(entries[k] for k in range(len(entries)) if k not in (i, j))
                left_val, left_expr = entries[i]
                right_val, right_expr = entries[j]
                combos = [
                    (left_val + right_val, f"({left_expr}+{right_expr})"),
                    (left_val * right_val, f"({left_expr}*{right_expr})"),
                    (left_val - right_val, f"({left_expr}-{right_expr})"),
                    (right_val - left_val, f"({right_expr}-{left_expr})"),
                ]
                if right_val != 0:
                    combos.append((left_val / right_val, f"({left_expr}/{right_expr})"))
                if left_val != 0:
                    combos.append((right_val / left_val, f"({right_expr}/{left_expr})"))
                for combo in combos:
                    search(rest + (combo,))
    search(items)


def add_record(expr: str, records: Dict[str, dict]) -> None:
    canonical_expr, opseq_label = canonicalize_expression(expr)
    if canonical_expr is None or canonical_expr in records:
        return
    records[canonical_expr] = {
        "canonical_expr": canonical_expr,
        "opseq_label": opseq_label or "OPSEQ::",
        "tree_signature": tree_signature(canonical_expr),
        "first_op": first_operator(canonical_expr),
        "depth": expression_depth(canonical_expr),
        "requires_division": "/" in canonical_expr,
        "requires_parentheses": "(" in canonical_expr,
    }


def first_operator(expr: str) -> str:
    for char in expr:
        if char in "+-*/":
            return char
    return "none"


def expression_depth(expr: str) -> int:
    try:
        tree = ast.parse(expr, mode="eval").body
    except Exception:
        return 0
    return node_depth(tree)


def node_depth(node) -> int:
    if isinstance(node, ast.BinOp):
        return 1 + max(node_depth(node.left), node_depth(node.right))
    if isinstance(node, ast.UnaryOp):
        return 1 + node_depth(node.operand)
    return 1


def summarize_solution_records(records: List[dict]) -> dict:
    first_op_counts: Dict[str, int] = {}
    for record in records:
        op = record["first_op"]
        first_op_counts[op] = first_op_counts.get(op, 0) + 1
    depths = [record["depth"] for record in records]
    return {
        "solution_count": len(records),
        "first_op_entropy": entropy(first_op_counts),
        "unique_opseq_count": len({record["opseq_label"] for record in records}),
        "unique_tree_count": len({record["tree_signature"] for record in records}),
        "min_depth": min(depths) if depths else 0,
        "mean_depth": float(sum(depths) / len(depths)) if depths else 0.0,
        "requires_division": int(any(record["requires_division"] for record in records)),
        "requires_parentheses": int(any(record["requires_parentheses"] for record in records)),
        "has_equivalent_path_diversity": int(len({record["opseq_label"] for record in records}) > 1),
    }


def entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            value -= prob * math.log(prob + 1e-12)
    return float(value)

