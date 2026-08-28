
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continuation prefix utilities for Countdown experiments."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


FEASIBLE_TAG = "<feasible>"
FEASIBLE_CLOSE_TAG = "</feasible>"
ANSWER_TAG = "<answer>"
ANSWER_CLOSE_TAG = "</answer>"
OPS = re.compile(r"[+\-*/]")
PREFIX_MODES = (
    "think_end",
    "answer_start",
    "op1_before",
    "after_op1",
    "after_op2",
    "after_op3",
    "tokens",
)


def truncate_to_think_prefix(text: str) -> str:
    idx = (text or "").lower().find(FEASIBLE_TAG)
    if idx == -1:
        return text or ""
    return text[:idx]


def answer_span(text: str) -> Optional[Tuple[int, int]]:
    full_text = text or ""
    lower_text = full_text.lower()
    start = lower_text.find(ANSWER_TAG)
    if start == -1:
        return None
    start += len(ANSWER_TAG)
    end = lower_text.find("</answer>", start)
    if end == -1:
        end = len(full_text)
    return start, end


def operator_positions(text: str) -> List[int]:
    span = answer_span(text)
    if span is None:
        return []
    start, end = span
    answer_text = text[start:end]
    return [start + match.start() for match in OPS.finditer(answer_text)]


def prefix_char_positions(text: str) -> Dict[str, Optional[int]]:
    span = answer_span(text)
    positions: Dict[str, Optional[int]] = {
        "think_end": len(truncate_to_think_prefix(text)),
        "answer_start": None,
        "op1_before": None,
        "after_op1": None,
        "after_op2": None,
        "after_op3": None,
    }
    if span is None:
        return positions
    positions["answer_start"] = span[0]
    ops = operator_positions(text)
    if ops:
        positions["op1_before"] = ops[0]
    for index, label in enumerate(("after_op1", "after_op2", "after_op3")):
        if index < len(ops):
            positions[label] = ops[index] + 1
    return positions


def extract_prefix_text(text: str, mode: str) -> Optional[str]:
    if mode == "tokens":
        raise ValueError("tokens mode requires token-based truncation.")
    if mode == "think_end":
        return truncate_to_think_prefix(text)
    positions = prefix_char_positions(text)
    pos = positions.get(mode)
    if pos is None:
        return None
    return (text or "")[:pos]


def prefix_text_map(text: str) -> Dict[str, str]:
    prefixes: Dict[str, str] = {}
    for mode in PREFIX_MODES:
        if mode == "tokens":
            continue
        prefix = extract_prefix_text(text, mode)
        if prefix is not None:
            prefixes[mode] = prefix
    return prefixes


def available_prefix_modes(text: str, modes: Sequence[str]) -> Dict[str, str]:
    prefixes: Dict[str, str] = {}
    for mode in modes:
        prefix = extract_prefix_text(text, mode)
        if prefix is not None:
            prefixes[mode] = prefix
    return prefixes


# ----------------------------
# Milestone extraction (merged from branch_set_collection)
# ----------------------------

_NUMBER_RE_CACHE: Dict[int, re.Pattern] = {}


def _number_pattern(n: int) -> re.Pattern:
    if n not in _NUMBER_RE_CACHE:
        _NUMBER_RE_CACHE[n] = re.compile(rf"(?<!\d){re.escape(str(n))}(?!\d)")
    return _NUMBER_RE_CACHE[n]


def _first_number_pos(text: str, numbers: List[int]) -> Optional[int]:
    best = None
    for n in numbers:
        m = _number_pattern(n).search(text)
        if m:
            pos = m.start()
            if best is None or pos < best:
                best = pos
    return best


def _first_op_pos(text: str) -> Optional[int]:
    m = re.search(r"[+\-*/]", text)
    return m.start() if m else None


def _first_lparen_pos(text: str) -> Optional[int]:
    idx = text.find("(")
    return idx if idx != -1 else None


def _first_balanced_paren_end(text: str) -> Optional[int]:
    start = text.find("(")
    if start == -1:
        return None
    bal = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            bal += 1
        elif text[i] == ")":
            bal -= 1
            if bal == 0:
                return i + 1
    return None


def _tag_end_pos(text: str, tag: str) -> Optional[int]:
    m = re.search(tag, text, flags=re.IGNORECASE)
    return m.end() if m else None


def _tag_start_content_pos(text: str, tag: str) -> Optional[int]:
    m = re.search(tag, text, flags=re.IGNORECASE)
    return m.end() if m else None


def _first_number_pos_in_span(text: str, numbers: List[int], span: Optional[Tuple[int, int]]) -> Optional[int]:
    if not span:
        return None
    start, end = span
    sub = text[start:end]
    pos = _first_number_pos(sub, numbers)
    if pos is None:
        return None
    return start + pos


def _first_op_pos_in_span(text: str, span: Optional[Tuple[int, int]]) -> Optional[int]:
    if not span:
        return None
    start, end = span
    sub = text[start:end]
    pos = _first_op_pos(sub)
    if pos is None:
        return None
    return start + pos


def _first_lparen_pos_in_span(text: str, span: Optional[Tuple[int, int]]) -> Optional[int]:
    if not span:
        return None
    start, end = span
    sub = text[start:end]
    pos = _first_lparen_pos(sub)
    if pos is None:
        return None
    return start + pos


def _first_balanced_paren_end_in_span(text: str, span: Optional[Tuple[int, int]]) -> Optional[int]:
    if not span:
        return None
    start, end = span
    sub = text[start:end]
    pos = _first_balanced_paren_end(sub)
    if pos is None:
        return None
    return start + pos


def extract_milestones(text: str, numbers: List[int]) -> Dict[str, Optional[int]]:
    """
    Extract character positions of structural milestones within a completion.
    Returns dict mapping milestone name -> char position (or None).
    """
    span = answer_span(text)
    return {
        "first_number": _first_number_pos(text, numbers),
        "first_op": _first_op_pos(text),
        "first_lparen": _first_lparen_pos(text),
        "first_balanced_paren": _first_balanced_paren_end(text),
        "feasible_tag_end": _tag_end_pos(text, r"</feasible>"),
        "answer_tag_start": _tag_start_content_pos(text, r"<answer>"),
        "answer_first_number": _first_number_pos_in_span(text, numbers, span),
        "answer_first_op": _first_op_pos_in_span(text, span),
        "answer_first_lparen": _first_lparen_pos_in_span(text, span),
        "answer_first_balanced_paren": _first_balanced_paren_end_in_span(text, span),
    }


def extract_answer_op_positions(text: str) -> List[int]:
    """
    Return character positions (pointing to one past the operator) of each
    operator inside the <answer> block.
    """
    span = answer_span(text)
    if not span:
        return []
    start, end = span
    answer_text = text[start:end]
    ops = [m.start() for m in re.finditer(r"[+\-*/]", answer_text)]
    return [start + pos + 1 for pos in ops]


def char_pos_to_token_len(text: str, pos: Optional[int], tokenizer) -> Optional[int]:
    """Convert a character position in text to the number of tokens up to that position."""
    if pos is None or pos < 0:
        return None
    prefix = text[:pos]
    return len(tokenizer.encode(prefix, add_special_tokens=False))


# ----------------------------
# Fast-tokenizer position alignment (merged from countdown_position_utils)
# ----------------------------

COUNTDOWN_OPERATOR_SYMBOLS = ("+", "-", "*", "/")


def _encode_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer with offset mapping is required for position alignment.")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [tuple(item) for item in encoded["offset_mapping"]]


def _find_span(text: str, marker: str, start: int = 0) -> Optional[Tuple[int, int]]:
    index = text.find(marker, start)
    if index < 0:
        return None
    return index, index + len(marker)


def _first_token_from_char(
    offsets: Sequence[Tuple[int, int]], char_pos: int, start_idx: int = 0
) -> Optional[int]:
    for index in range(start_idx, len(offsets)):
        _, end = offsets[index]
        if end > char_pos:
            return index
    return None


def _first_token_overlapping(
    offsets: Sequence[Tuple[int, int]],
    span_start: int,
    span_end: int,
    start_idx: int = 0,
) -> Optional[int]:
    for index in range(start_idx, len(offsets)):
        token_start, token_end = offsets[index]
        if token_start >= span_end:
            return None
        if token_end > span_start and token_start < span_end:
            return index
    return None


def _last_token_overlapping(
    offsets: Sequence[Tuple[int, int]],
    span_start: int,
    span_end: int,
    start_idx: int = 0,
) -> Optional[int]:
    last = None
    for index in range(start_idx, len(offsets)):
        token_start, token_end = offsets[index]
        if token_start >= span_end:
            break
        if token_end > span_start and token_start < span_end:
            last = index
    return last


def _build_operator_token_ids(tokenizer) -> set[int]:
    token_ids: set[int] = set()
    for operator in COUNTDOWN_OPERATOR_SYMBOLS:
        for text in (operator, f" {operator}"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
    return token_ids


def locate_positions_from_text(
    full_text: str,
    prompt_char_len: int,
    tokenizer,
    position_names: List[str],
    input_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Optional[int]]:
    """Map Countdown character milestones to token indices using offsets."""
    token_ids, offsets = _encode_with_offsets(tokenizer, full_text)
    if input_ids is not None and list(input_ids) != token_ids:
        raise ValueError("Tokenization mismatch between supplied input_ids and tokenizer offsets.")

    positions: Dict[str, Optional[int]] = {name: None for name in position_names}
    completion = full_text[prompt_char_len:]
    completion_start = _first_token_from_char(offsets, prompt_char_len, 0) or 0

    feasible_open = _find_span(completion, FEASIBLE_TAG)
    if feasible_open is not None:
        open_start = prompt_char_len + feasible_open[0]
        open_end = prompt_char_len + feasible_open[1]
        feasible_token = _first_token_overlapping(offsets, open_start, open_end, completion_start)
        if "think_end" in positions and feasible_token is not None and feasible_token > 0:
            positions["think_end"] = feasible_token - 1

    feasible_close = _find_span(completion, FEASIBLE_CLOSE_TAG)
    if "feasible_end" in positions and feasible_close is not None:
        close_start = prompt_char_len + feasible_close[0]
        close_end = prompt_char_len + feasible_close[1]
        positions["feasible_end"] = _last_token_overlapping(
            offsets, close_start, close_end, completion_start
        )

    answer_open = _find_span(completion, ANSWER_TAG)
    answer_content_token = None
    answer_close_char = len(full_text)
    if answer_open is not None:
        content_char = prompt_char_len + answer_open[1]
        answer_content_token = _first_token_from_char(offsets, content_char, completion_start)
        if "answer_start" in positions and answer_content_token is not None:
            positions["answer_start"] = answer_content_token
        answer_close = _find_span(completion, ANSWER_CLOSE_TAG, answer_open[1])
        if answer_close is not None:
            answer_close_char = prompt_char_len + answer_close[0]

    if "op1_before" in positions and answer_content_token is not None:
        operator_ids = _build_operator_token_ids(tokenizer)
        for index in range(answer_content_token, len(token_ids)):
            token_start, _ = offsets[index]
            if token_start >= answer_close_char:
                break
            if token_ids[index] in operator_ids:
                previous = index - 1
                if previous >= completion_start:
                    positions["op1_before"] = previous
                break

    return positions
