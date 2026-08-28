"""Native Countdown reward adapter used by VERL M12/M13 runs."""

from __future__ import annotations

from typing import Any

from early_branch_locking.core.external_countdown import evaluate_native_countdown


def compute_score(solution_str: str, ground_truth: dict[str, Any], **_: Any) -> float:
    """Score only the generated response, never the pre-rendered prompt."""

    if not isinstance(ground_truth, dict):
        return 0.0
    try:
        numbers = [int(value) for value in ground_truth["numbers"]]
        target = int(ground_truth["target"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    result = evaluate_native_countdown(str(solution_str or ""), numbers, target)
    return float(bool(result.overall_ok))


__all__ = ["compute_score"]
