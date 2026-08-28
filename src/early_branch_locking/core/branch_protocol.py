#!/usr/bin/env python3
"""Frozen, training-free protocol shared by the open-domain RLVR experiments.

Question: How can early mathematical strategy branches and semantic milestones
be measured without using the final answer or a model-specific parser?
Inputs: one merged math-evaluation JSONL row per problem and its 64 samples.
Procedure: reuse the repository answer/first-calculation parser, locate complete
semantic units, apply a deterministic strategy rubric, and cluster only the
pre-P1 text with a fixed token-shingle similarity rule.
Metrics: sample labels, P0/P1/P2 offsets, parse/format diagnostics, and branch
co-membership labels.
Outputs: in-memory records consumed by the branch_decomposition-math_transfer_analysis entry points.
Statistical unit: problem; samples are nested below problem.
Known limitations: strategy labels are a frozen conservative rule labeler, not
an LLM or human semantic gold standard; the proxy calibration experiment must
quantify this limitation before semantic claims are made.
Status: frozen protocol v1; no training is performed.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from early_branch_locking.core.math_trace_utils import (
    BRANCH_PLACEHOLDER,
    ANSWER_PLACEHOLDER,
    evaluate_completion,
    extract_first_calc_branch,
    normalize_answer,
)


PROTOCOL_VERSION = "branch_protocol_v1"
MILESTONE_VERSION = "semantic_milestones_v1"
STRATEGY_RUBRIC_VERSION = "strategy_rule_rubric_v1"
CLUSTER_VERSION = "early_trace_shingle_agglomeration_v1"
NO_VALID_FIRST_CALC = "NO_VALID_FIRST_CALC"
UNKNOWN_STRATEGY = "UNKNOWN"
STRATEGY_LABELS = (
    "DIRECT_EQUATION",
    "SUBSTITUTION_ELIMINATION",
    "CASES",
    "SYMMETRY_INVARIANT",
    "RECURSION_INDUCTION",
    "BACKWARD_CONSTRUCTION",
    "EXTREMAL_BOUND",
    "GEOMETRY_COORDINATE",
    "COUNTING_PROBABILITY",
    "ENUMERATION_TRIAL",
    "OTHER_EXPLICIT",
    UNKNOWN_STRATEGY,
)

_SENTENCE_END = re.compile(r"(?:\n\s*\n|[.!?。！？](?:\s|$)|\n)")
_CALC = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*"
    r"(?:[+\-*/=×÷]|\\times|\\cdot|\\div)\s*"
    r"[-+]?\d+(?:\.\d+)?"
)
_EQUATION = re.compile(r"(?:=|\\leq|\\geq|\\equiv|\\cong)")
_MATH_OPEN_CLOSE = (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$"))


def completion_list(row: Mapping) -> list[str]:
    """Return completions from the preserved evaluation schema."""
    for key in ("code", "responses", "completions", "completion"):
        value = row.get(key)
        if isinstance(value, list):
            return ["" if item is None else str(item) for item in value]
        if isinstance(value, str):
            return [value]
    value = row.get("pred")
    if isinstance(value, list):
        return ["" if item is None else str(item) for item in value]
    if value is not None:
        return [str(value)]
    return []


def ground_truth(row: Mapping) -> str:
    value = row.get("gt", row.get("ground_truth", row.get("answer", "")))
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def official_scores(row: Mapping, completions: Sequence[str], gt: str) -> list[bool]:
    value = row.get("score", row.get("scores"))
    if isinstance(value, list) and len(value) == len(completions):
        return [bool(item) for item in value]
    return [evaluate_completion(text, gt).is_correct for text in completions]


def source_problem_id(benchmark: str, row: Mapping) -> str:
    return f"{benchmark}:{row.get('idx', row.get('index', 'unknown'))}"


def _balanced_end(text: str, start: int, end: int) -> int | None:
    """Extend a candidate to a complete bracketed math unit when possible."""
    fragment = text[start:end]
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for char in fragment:
        if char in pairs:
            stack.append(pairs[char])
        elif char in ")]}":
            if not stack or stack.pop() != char:
                return None
    if stack:
        return None
    return end


def _unit_end(text: str, start: int, minimum: int) -> tuple[int, str]:
    """Find a conservative end after a complete calculation."""
    line_end = text.find("\n", minimum)
    if line_end < 0:
        line_end = len(text)
    sentence = re.search(r"[.!?。！？](?:\s|$)", text[minimum:line_end])
    if sentence:
        candidate = minimum + sentence.end()
    else:
        candidate = line_end
    balanced = _balanced_end(text, start, candidate)
    if balanced is not None:
        return balanced, "line_or_sentence_balanced"
    # A closing delimiter often appears before the prose sentence terminator.
    for end in range(minimum, min(line_end + 1, len(text) + 1)):
        if text[end - 1 : end] in ")]}":
            balanced = _balanced_end(text, start, end)
            if balanced is not None:
                return end, "first_balanced_delimiter"
    return candidate, "line_or_sentence_unverified"


def _closed_math_end(text: str, match_start: int) -> int | None:
    for opening, closing in _MATH_OPEN_CLOSE:
        if text.startswith(opening, match_start):
            close = text.find(closing, match_start + len(opening))
            if close >= 0:
                return close + len(closing)
    return None


def _first_calculation_end(text: str) -> tuple[int | None, str, float]:
    matches = list(_CALC.finditer(text))
    equation = _EQUATION.search(text)
    if equation and (not matches or equation.start() < matches[0].start()):
        start = max(0, text.rfind("\n", 0, equation.start()) + 1)
        end, source = _unit_end(text, start, equation.end())
        return end, f"equation:{source}", 0.86 if source.endswith("balanced") else 0.68
    if not matches:
        return None, "missing", 0.0
    match = matches[0]
    closed = _closed_math_end(text, match.start())
    if closed is not None:
        return closed, "closed_math", 0.95
    start = max(0, text.rfind("\n", 0, match.start()) + 1)
    end, source = _unit_end(text, start, match.end())
    return end, f"calculation:{source}", 0.86 if source.endswith("balanced") else 0.64


def _first_strategy_sentence(text: str, before: int | None) -> tuple[int | None, str, float]:
    limit = len(text) if before is None else before
    prefix = text[:limit]
    candidates = list(_SENTENCE_END.finditer(prefix))
    start = 0
    cues = re.compile(
        r"\b(?:plan|approach|strategy|method|let us|we need|we can|consider|"
        r"first|suppose|assume|define|set|denote|want to|use the)\b",
        re.IGNORECASE,
    )
    for match in candidates:
        end = match.end()
        sentence = prefix[start:end].strip()
        if len(sentence) >= 12 and cues.search(sentence):
            return end, "explicit_strategy_sentence", 0.73
        start = end
    if prefix.strip() and len(prefix.strip()) >= 12 and cues.search(prefix):
        return min(len(prefix), 512), "strategy_prefix_fallback", 0.48
    return None, "missing", 0.0


def _next_subresult_end(text: str, p1: int | None) -> tuple[int | None, str, float]:
    if p1 is None:
        return None, "missing_p1", 0.0
    tail = text[p1:]
    marker = re.search(
        r"(?:\btherefore\b|\bthus\b|\bso\b|\bhence\b|\bwe get\b|\bwhich gives\b|=)",
        tail,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return None, "missing", 0.0
    absolute = p1 + marker.end()
    end, source = _unit_end(text, p1, absolute)
    return end, f"subresult:{source}", 0.7 if source.endswith("balanced") else 0.52


def normalize_for_cluster(text: str) -> str:
    """Remove answer-bearing numeric detail while retaining early semantics."""
    text = re.sub(r"\\boxed\s*\{.*?\}", " <answer> ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"[-+]?\d+(?:[.,]\d+)?", " <num> ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[^A-Za-z_]+", " ", text).lower()
    words = [word for word in text.split() if len(word) > 1]
    return " ".join(words[:96])


def _shingles(text: str, width: int = 3) -> set[str]:
    tokens = text.split()
    if len(tokens) <= width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def strategy_label(text: str, p1_end: int | None) -> tuple[str, str, float]:
    """Apply the frozen rubric to the text available before the first calc."""
    window = text[: p1_end or min(len(text), 640)].lower()
    rules = (
        ("SUBSTITUTION_ELIMINATION", r"substitut|eliminat|plug in|replace|solve for"),
        ("CASES", r"case[s]?\b|consider whether|if .* then|depending on|split into"),
        ("SYMMETRY_INVARIANT", r"symmetr|invariant|conservation|same for|without loss"),
        ("RECURSION_INDUCTION", r"induct|recurr|previous step|assume .* holds|base case"),
        ("BACKWARD_CONSTRUCTION", r"backward|work backwards|reverse|from the end|construct"),
        ("EXTREMAL_BOUND", r"maximum|minimum|extrem|upper bound|lower bound|at most|at least"),
        ("GEOMETRY_COORDINATE", r"triangle|angle|circle|coordinate|area|geometry|slope|radius"),
        ("COUNTING_PROBABILITY", r"probab|count|permutation|combination|choose|ways|expectation"),
        ("ENUMERATION_TRIAL", r"enumerat|list all|try |test |check each|brute force"),
    )
    for label, pattern in rules:
        if re.search(pattern, window):
            return label, "rule", 0.66
    if _CALC.search(window) or _EQUATION.search(window):
        return "DIRECT_EQUATION", "rule", 0.58
    if len(window.strip()) >= 18:
        return "OTHER_EXPLICIT", "rule", 0.35
    return UNKNOWN_STRATEGY, "rule", 0.15


def milestone_record(text: str, gt: str) -> dict:
    p1_end, p1_source, p1_conf = _first_calculation_end(text)
    p0_end, p0_source, p0_conf = _first_strategy_sentence(text, p1_end)
    p2_end, p2_source, p2_conf = _next_subresult_end(text, p1_end)
    gt_norm = normalize_answer(gt).replace(" ", "")
    prefix_for_leakage = text[: p2_end or p1_end or 0]
    normalized_prefix = normalize_answer(prefix_for_leakage).replace(" ", "")
    leaked = bool(gt_norm and gt_norm != ANSWER_PLACEHOLDER and gt_norm in normalized_prefix)
    strategy, strategy_source, strategy_conf = strategy_label(text, p1_end)
    early_end = p1_end if p1_end is not None else min(len(text), 640)
    early = text[:early_end]
    return {
        "p0_char_end": p0_end,
        "p1_char_end": p1_end,
        "p2_char_end": p2_end,
        "p0_source": p0_source,
        "p1_source": p1_source,
        "p2_source": p2_source,
        "p0_confidence": p0_conf,
        "p1_confidence": p1_conf,
        "p2_confidence": p2_conf,
        "answer_leakage_before_p2": leaked,
        "strategy_branch": strategy,
        "strategy_source": strategy_source,
        "strategy_confidence": strategy_conf,
        "early_cluster_text": normalize_for_cluster(early),
        "early_text_sha1": hashlib.sha1(early.encode("utf-8", errors="replace")).hexdigest(),
    }


def label_completion(
    *,
    benchmark: str,
    model: str,
    source_file: str,
    source_row: int,
    problem_id: str,
    sample_index: int,
    completion: str,
    gt: str,
    official_correct: bool,
) -> dict:
    evaluated = evaluate_completion(completion, gt)
    milestone = milestone_record(completion, gt)
    first_calc = extract_first_calc_branch(completion)
    if first_calc == BRANCH_PLACEHOLDER:
        first_calc = NO_VALID_FIRST_CALC
    text_clean = completion.strip()
    format_valid = evaluated.parsed and bool(
        re.search(r"(?:\\boxed|####|final answer|answer)", completion, re.IGNORECASE)
    )
    no_calc = first_calc == NO_VALID_FIRST_CALC
    malformed = (not text_clean) or (len(re.findall(r"[A-Za-z0-9]", completion)) < 4)
    record = {
        "benchmark": benchmark,
        "model": model,
        "problem_id": problem_id,
        "sample_index": sample_index,
        "source_file": source_file,
        "source_row": source_row,
        "completion_sha1": hashlib.sha1(completion.encode("utf-8", errors="replace")).hexdigest(),
        "completion_chars": len(completion),
        "token_count_proxy": len(completion.split()),
        "ground_truth": gt,
        "official_correct": bool(official_correct),
        "parser_correct": bool(evaluated.is_correct),
        "parsed_answer": bool(evaluated.parsed),
        "parsed_answer_value": evaluated.answer,
        "numeric_trace": evaluated.numeric_trace,
        "first_calc_branch": first_calc,
        "format_valid": bool(format_valid),
        "no_valid_first_calc": no_calc,
        "malformed_or_empty": malformed,
    }
    record.update(milestone)
    record["early_cluster_id"] = None
    record["early_cluster_similarity"] = None
    return record


def assign_early_clusters(records: Sequence[dict], threshold: float = 0.30) -> None:
    """Cluster each problem jointly across models using only early text."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["problem_id"]].append(index)
    for indices in grouped.values():
        parent = list(range(len(indices)))
        signatures = [_shingles(records[index].get("early_cluster_text", "")) for index in indices]

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(indices)):
            if not signatures[left]:
                continue
            for right in range(left):
                if signatures[right] and _jaccard(signatures[left], signatures[right]) >= threshold:
                    union(left, right)
        roots: dict[int, int] = {}
        for local, global_index in enumerate(indices):
            root = find(local)
            roots.setdefault(root, len(roots))
            record = records[global_index]
            if record["first_calc_branch"] == NO_VALID_FIRST_CALC or not signatures[local]:
                record["early_cluster_id"] = "NO_VALID_EARLY_TRACE"
                record["early_cluster_similarity"] = 0.0
            else:
                record["early_cluster_id"] = f"cluster_{roots[root]:03d}"
                record["early_cluster_similarity"] = 1.0


def entropy(labels: Iterable[str]) -> float:
    counts = Counter(label for label in labels if label)
    total = sum(counts.values())
    return float(-sum((count / total) * math.log(count / total) for count in counts.values())) if total else 0.0


def herfindahl(labels: Iterable[str]) -> float:
    counts = Counter(label for label in labels if label)
    total = sum(counts.values())
    return float(sum((count / total) ** 2 for count in counts.values())) if total else 0.0


def gini_counts(labels: Iterable[str]) -> float:
    values = sorted(Counter(label for label in labels if label).values())
    total = sum(values)
    if not values or total == 0:
        return 0.0
    n = len(values)
    return float(sum((2 * index - n - 1) * value for index, value in enumerate(values, 1)) / (n * total))


def jensen_shannon(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    labels = set(left_counts) | set(right_counts)
    n_left, n_right = max(1, len(left)), max(1, len(right))
    divergence = 0.0
    for label in labels:
        p = left_counts[label] / n_left
        q = right_counts[label] / n_right
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log(q / midpoint)
    return float(divergence)


def pairwise_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    matches = 0
    total = 0
    for i in range(len(left)):
        for j in range(i):
            matches += int((left[i] == left[j]) == (right[i] == right[j]))
            total += 1
    return matches / total if total else 0.0


def pairwise_precision_recall_f1(left: Sequence[str], right: Sequence[str]) -> dict[str, float | int]:
    """Compare positive co-membership pairs between two clusterings.

    This is deliberately different from :func:`pairwise_agreement`, which
    counts both co-clustered and separated pairs and can look high when most
    pairs are negative.  The returned precision/recall/F1 fields describe
    only the positive co-membership relation.
    """
    if len(left) != len(right) or len(left) < 2:
        return {"pairwise_tp": 0, "pairwise_fp": 0, "pairwise_fn": 0, "pairwise_precision": 0.0, "pairwise_recall": 0.0, "pairwise_f1": 0.0}
    true_positive = false_positive = false_negative = 0
    for index in range(len(left)):
        for other in range(index):
            predicted = left[index] == left[other]
            target = right[index] == right[other]
            if predicted and target:
                true_positive += 1
            elif predicted:
                false_positive += 1
            elif target:
                false_negative += 1
    predicted_positive = true_positive + false_positive
    target_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else float(target_positive == 0)
    recall = true_positive / target_positive if target_positive else float(predicted_positive == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pairwise_tp": true_positive,
        "pairwise_fp": false_positive,
        "pairwise_fn": false_negative,
        "pairwise_precision": float(precision),
        "pairwise_recall": float(recall),
        "pairwise_f1": float(f1),
    }


def contingency(left: Sequence[str], right: Sequence[str]) -> dict[tuple[str, str], int]:
    return Counter(zip(left, right))


def adjusted_rand_index(left: Sequence[str], right: Sequence[str]) -> float:
    n = len(left)
    if n < 2:
        return 0.0
    table = contingency(left, right)
    left_counts, right_counts = Counter(left), Counter(right)
    choose2 = lambda value: value * (value - 1) / 2
    observed = sum(choose2(value) for value in table.values())
    expected = sum(choose2(value) for value in left_counts.values()) * sum(choose2(value) for value in right_counts.values())
    total = choose2(n)
    denominator = 0.5 * (sum(choose2(value) for value in left_counts.values()) + sum(choose2(value) for value in right_counts.values())) - expected / total
    numerator = observed - expected / total
    return float(numerator / denominator) if denominator else 1.0


def normalized_mutual_information(left: Sequence[str], right: Sequence[str]) -> float:
    n = len(left)
    if n == 0:
        return 0.0
    table = contingency(left, right)
    left_counts, right_counts = Counter(left), Counter(right)
    mutual = 0.0
    for (left_label, right_label), count in table.items():
        mutual += count / n * math.log((count * n) / (left_counts[left_label] * right_counts[right_label]))
    h_left, h_right = entropy(left), entropy(right)
    denominator = math.sqrt(h_left * h_right)
    return float(mutual / denominator) if denominator else 0.0


def variation_of_information(left: Sequence[str], right: Sequence[str]) -> float:
    nmi = normalized_mutual_information(left, right)
    return float(entropy(left) + entropy(right) - 2 * nmi * math.sqrt(entropy(left) * entropy(right)))


def bootstrap_mean(values: Sequence[float], seed: int = 1729, draws: int = 1000) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    import numpy as np

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def protocol_metadata() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "milestone_version": MILESTONE_VERSION,
        "strategy_rubric_version": STRATEGY_RUBRIC_VERSION,
        "cluster_version": CLUSTER_VERSION,
        "cluster_similarity_threshold": 0.30,
        "cluster_input": "normalized text through P1, numeric values and boxed answer markers removed",
        "p0": "explicit plan/strategy sentence before P1; missing when absent",
        "p1": "first complete numeric calculation/equation with balanced line/sentence unit",
        "p2": "first complete post-P1 subresult marker and balanced unit",
        "answer_leakage": "normalized ground truth substring before P2; diagnostic only",
        "primary_correctness": "official per-sample score when present; parser correctness retained as diagnostic",
        "raw_reuse": "source raw JSONL is retained; branch_decomposition labels reference source_file/source_row and SHA1",
        "strategy_labels": list(STRATEGY_LABELS),
    }
