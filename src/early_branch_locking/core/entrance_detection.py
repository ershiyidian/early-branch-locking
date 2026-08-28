"""Shared semantic detection of the first Countdown entrance.

An entrance is the first operand and the first arithmetic operator in the
generated reasoning continuation.  The detector deliberately receives only
the generated continuation; callers must not pass the prompt or a scaffold
when measuring natural access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


FIRST_ARITH_RE = re.compile(
    r"(?<![\w.])(?P<first>-?\d+(?:\.\d+)?)\s*"
    r"(?P<op>[+\-*/×÷])"
)

_STOP_MARKERS = ("</think>", "<feasible>", "<answer>")


def normalize_op(value: str) -> str:
    """Normalize ASCII and Unicode multiplication/division symbols."""

    return {"×": "*", "÷": "/"}.get(value, value)


def normalize_first(value: str) -> int | None:
    """Return an integer operand, rejecting decimal Countdown families."""

    if "." in value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class EntranceMatch:
    found: bool
    first: int | None
    op: str | None
    char_start: int | None
    char_end: int | None
    family: str | None


NOT_FOUND = EntranceMatch(False, None, None, None, None, None)


def _search_text(generated_text: str, stop_markers: Iterable[str]) -> tuple[str, int]:
    text = str(generated_text or "")
    stop = len(text)
    for marker in stop_markers:
        index = text.lower().find(str(marker).lower())
        if index >= 0:
            stop = min(stop, index)
    return text[:stop], stop


def find_first_reasoning_entrance(
    generated_text: str,
    *,
    stop_markers: Iterable[str] = _STOP_MARKERS,
) -> EntranceMatch:
    """Find the first integer `(operand, operator)` entrance in reasoning.

    The caller supplies only assistant-generated text.  We stop scanning at
    format/answer markers so an arithmetic expression in the answer cannot be
    counted as a natural reasoning entrance.  Decimal matches are deliberately
    rejected because the Countdown family key is integer-valued.  A leading
    minus is accepted as part of the first operand, which handles negative
    intermediate states without turning ``-3 + 4`` into a spurious unary
    subtraction family.
    """

    text, _ = _search_text(generated_text, stop_markers)
    match = FIRST_ARITH_RE.search(text)
    if match is None:
        return NOT_FOUND
    first = normalize_first(match.group("first"))
    if first is None:
        return NOT_FOUND
    op = normalize_op(match.group("op"))
    return EntranceMatch(
        found=True,
        first=first,
        op=op,
        char_start=match.start(),
        char_end=match.end(),
        family=f"{first}{op}",
    )


__all__ = [
    "FIRST_ARITH_RE",
    "EntranceMatch",
    "find_first_reasoning_entrance",
    "normalize_first",
    "normalize_op",
]
