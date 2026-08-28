#!/usr/bin/env python3
"""Build leak-free, solver-derived supervision for the GRPO-line SFT control."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from early_branch_locking._repo import METRICS_DIR, RLVR_DATA_ROOT  # noqa: E402
from early_branch_locking.core.countdown_shared import canonicalize_expression, enumerate_solution_set  # noqa: E402
from early_branch_locking.core.external_countdown import evaluate_native_countdown  # noqa: E402
from early_branch_locking.countdown.prefix_splice_recovery import entrance_of, minimal_render  # noqa: E402
from early_branch_locking.countdown.public_grpo_replication import semantic_key, semantic_key_text  # noqa: E402

TEMPLATES = (
    "We need {target} from {numbers}. I will calculate it in stages: {steps}.",
    "Let me work toward {target} using {numbers}. The arithmetic is: {steps}.",
    "A direct construction for target {target} is: {steps}.",
    "Using each of {numbers} once, compute as follows: {steps}.",
    "I will combine the numbers carefully: {steps}; this reaches {target}.",
    "Here is a step-by-step route to {target}: {steps}.",
    "The useful intermediate calculations are {steps}.",
    "Start from {numbers}; evaluate {steps} to obtain {target}.",
    "One valid calculation chain is {steps}.",
    "I can form {target} by these operations: {steps}.",
    "Compute sequentially: {steps}. This gives the requested target {target}.",
    "A valid way to use {numbers} is: {steps}.",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("generate-problems", "build-supervision", "audit"), default="generate-problems")
    p.add_argument("--tag", default="v1")
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--num-low", type=int, default=1)
    p.add_argument("--num-high", type=int, default=100)
    p.add_argument("--target-low", type=int, default=1)
    p.add_argument("--target-high", type=int, default=100)
    p.add_argument("--p-three-numbers", type=float, default=0.5)
    p.add_argument("--k", type=int, choices=(1, 2, 4, 8), default=4)
    p.add_argument("--sampling", choices=("entrance-diverse", "uniform"), default="entrance-diverse")
    p.add_argument("--out-dir", type=Path, default=RLVR_DATA_ROOT / "outputs" / "grpo_sft")
    p.add_argument("--safe-eval", type=Path, default=METRICS_DIR / "s2_philschmid_safe_eval_rows_v1.csv")
    p.add_argument("--problems", type=Path, default=None)
    return p.parse_args(argv)


def _commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or "unavailable"


def _sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _paths(a):
    root = a.out_dir
    return {
        "problems": a.problems or root / f"grpo_line_sft_problems_{a.tag}.jsonl",
        "dedup": root / f"grpo_line_sft_dedup_manifest_{a.tag}.json",
        "supervision": root / f"grpo_line_sft_supervision_k{a.k}_{a.sampling}_{a.tag}.jsonl",
    }


def _write_jsonl(path: Path, rows):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite maintained artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _eval_keys(path: Path):
    frame = pd.read_csv(path)
    kept = frame[frame["keep_for_external_eval"].astype(str).str.lower().eq("true")]
    if len(kept) != 135:
        raise ValueError(f"expected 135 safe external-eval rows, got {len(kept)}")
    keys = set()
    for row in kept.itertuples():
        numbers = json.loads(row.numbers) if isinstance(row.numbers, str) else row.numbers
        keys.add(semantic_key(numbers, int(row.target)))
    if len(keys) != 135:
        raise ValueError("safe-eval semantic keys are not unique")
    return keys


def generate(a):
    paths = _paths(a)
    eval_keys = _eval_keys(a.safe_eval)
    rng = np.random.default_rng(a.seed)
    seen, rows = set(eval_keys), []
    needed = a.n_train + a.n_val
    attempts = 0
    while len(rows) < needed:
        attempts += 1
        n = 3 if rng.random() < a.p_three_numbers else 4
        numbers = rng.integers(a.num_low, a.num_high + 1, size=n).astype(int).tolist()
        target = int(rng.integers(a.target_low, a.target_high + 1))
        key = semantic_key(numbers, target)
        if key in seen:
            continue
        leaves = enumerate_solution_set(numbers, target)
        if not leaves:
            continue
        seen.add(key)
        idx = len(rows)
        rows.append({"problem_uid": f"grpo_sft_{idx:05d}", "numbers": numbers, "target": target,
                     "split": "train" if idx < a.n_train else "val", "semantic_key": semantic_key_text(numbers, target),
                     "n_canonical_leaves": len(leaves)})
    _write_jsonl(paths["problems"], rows)
    payload = {"artifact": "GRPO-line SFT problem generation", "status": "complete", "tag": a.tag, "git_commit": _commit(),
               "safe_eval_sha256": _sha(a.safe_eval), "n_train": a.n_train, "n_val": a.n_val, "n_eval": len(eval_keys),
               "semantic_key_counts": {"train": a.n_train, "val": a.n_val, "eval": len(eval_keys), "union": len(seen)},
               "intersections": {"train_val": 0, "train_eval": 0, "val_eval": 0},
               "generator": {"number_range": [a.num_low, a.num_high], "target_range": [a.target_low, a.target_high], "p_three_numbers": a.p_three_numbers, "seed": a.seed, "attempts": attempts},
               "outputs": {"problems": str(paths["problems"])}}
    if paths["dedup"].exists():
        raise FileExistsError(paths["dedup"])
    tmp = paths["dedup"].with_suffix(".partial"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)); tmp.replace(paths["dedup"])
    print(json.dumps({"problems": str(paths["problems"]), "rows": len(rows), "attempts": attempts}, sort_keys=True))


def _family(expr):
    found = entrance_of(minimal_render(expr))
    return (found[0], found[1]) if found else ("?", "?")


def _pick(leaves, k, sampling, rng):
    leaves = sorted(leaves)
    if sampling == "uniform":
        return rng.sample(leaves, k=min(k, len(leaves)))
    groups = defaultdict(list)
    for leaf in leaves:
        groups[_family(leaf)].append(leaf)
    for values in groups.values():
        values.sort()
    order = sorted(groups, key=lambda x: (-len(groups[x]), x))
    picked = []
    while len(picked) < min(k, len(leaves)):
        for family in order:
            if groups[family]:
                picked.append(groups[family].pop(0))
            if len(picked) == min(k, len(leaves)):
                break
    return picked


def _render(node, rng):
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.UnaryOp):
        return "-" + _render(node.operand, rng)
    if not isinstance(node, ast.BinOp):
        raise ValueError(type(node).__name__)
    op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
    left, right = _render(node.left, rng), _render(node.right, rng)
    if op in "+*" and rng.random() < .5:
        left, right = right, left
    return f"({left} {op} {right})"


def _steps(expr):
    tree = ast.parse(expr, mode="eval")
    output = []
    def walk(node):
        if isinstance(node, ast.BinOp):
            walk(node.left); walk(node.right)
            text = ast.unparse(node)
            try:
                value = eval(compile(ast.Expression(node), "<expr>", "eval"), {"__builtins__": {}}, {})
                output.append(f"{text} = {value}")
            except Exception:
                pass
    walk(tree.body)
    return "; then ".join(output[-3:]) or minimal_render(expr)


def build(a):
    paths = _paths(a)
    rows = [json.loads(line) for line in paths["problems"].open(encoding="utf-8") if line.strip()]
    rng = random.Random(a.seed + 10_000 * a.k + (1 if a.sampling == "uniform" else 0))
    output, failures = [], defaultdict(int)
    for problem in rows:
        leaves = enumerate_solution_set(list(map(int, problem["numbers"])), int(problem["target"]))
        picked = _pick(leaves, a.k, a.sampling, rng)
        seen = set()
        for ordinal, leaf in enumerate(picked):
            raw = _render(ast.parse(leaf, mode="eval").body, rng)
            canonical, _ = canonicalize_expression(raw)
            if canonical != leaf or canonical in seen:
                failures["canonical_mismatch_or_duplicate"] += 1; continue
            think = TEMPLATES[rng.randrange(len(TEMPLATES))].format(numbers=problem["numbers"], target=problem["target"], steps=_steps(raw))
            answer = f"{raw} = {int(problem['target'])}"
            completion = think + "\n</think>\n\n<answer>" + answer + "</answer>"
            check = evaluate_native_countdown(completion, list(map(int, problem["numbers"])), int(problem["target"]))
            if not (check.overall_ok and check.native_format_ok and check.canonical_expr == leaf):
                failures["native_evaluator_failed"] += 1; continue
            seen.add(canonical)
            output.append({**problem, "canonical_leaf": leaf, "raw_expression": raw, "template_id": ordinal % len(TEMPLATES),
                           "think_body": think, "answer_text": answer, "completion": completion, "sampling": a.sampling, "k": a.k,
                           "entrance_family": _family(leaf)})
    if not output:
        raise RuntimeError("all supervision rows failed validation")
    _write_jsonl(paths["supervision"], output)
    print(json.dumps({"supervision": str(paths["supervision"]), "rows": len(output), "failures": dict(failures)}, sort_keys=True))


def audit(a):
    paths = _paths(a)
    problems = [json.loads(line) for line in paths["problems"].open() if line.strip()]
    keys = defaultdict(set)
    for row in problems:
        keys[row["split"]].add(row["semantic_key"])
    eval_keys = {semantic_key_text(list(key[0]), key[1]) for key in _eval_keys(a.safe_eval)}
    intersections = {"train_val": keys["train"] & keys["val"], "train_eval": keys["train"] & eval_keys, "val_eval": keys["val"] & eval_keys}
    if any(intersections.values()):
        raise AssertionError({k: len(v) for k, v in intersections.items()})
    print(json.dumps({"status": "complete", "train": len(keys["train"]), "val": len(keys["val"]), "eval": len(eval_keys)}, sort_keys=True))


def main(argv=None):
    a = parse_args(argv)
    {"generate-problems": generate, "build-supervision": build, "audit": audit}[a.mode](a)

if __name__ == "__main__":
    main()
