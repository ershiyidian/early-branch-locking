#!/usr/bin/env python3
"""Build auditable native Countdown parquet data for M12/M13 GRPO.

The local VERL dataset reader consumes a pre-rendered prompt string from the
first item of ``prompt``.  This module therefore keeps prompt rendering,
semantic identity, reward provenance, and source completions in one explicit
conversion boundary.  It also provides the fresh on-policy source required by
the M12 data factor; historical rollout files are never accepted as an
implicit substitute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import REPO_ROOT, RLVR_DATA_ROOT  # noqa: E402
from early_branch_locking.core.external_countdown import evaluate_native_countdown  # noqa: E402
from early_branch_locking.countdown.public_grpo_replication import (  # noqa: E402
    build_philschmid_messages,
    semantic_key_text,
)


DATA_SOURCE = "countdown-native"
DATA_VERSION = "factorial_intervention-native-countdown-v1"
REQUIRED_PARQUET_COLUMNS = {
    "prompt",
    "data_source",
    "reward_model",
    "extra_info",
    "problem_uid",
    "semantic_key",
    "split",
    "source_kind",
}
HEALTH_RE = re.compile(
    r"""
    (?:
        \b(?:float|math|np|numpy)\s*
        (?:
            \(\s*['\"]?[-+]?(?:nan|inf(?:inity)?)['\"]?\s*\)
            |\.\s*[-+]?(?:nan|inf(?:inity)?)\b
        )
        |
        (?<![A-Za-z0-9_])(?:nan|inf)(?![A-Za-z0-9_])
        |
        (?:/|÷)\s*0\b[^\n]{0,120}?\b(?:nan|inf(?:inity)?)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
HEALTH_STANDALONE = {
    "nan",
    "inf",
    "infinity",
    "+nan",
    "+inf",
    "+infinity",
    "-nan",
    "-inf",
    "-infinity",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("build-solver", "generate-on-policy", "build-on-policy", "audit"),
        default="audit",
    )
    parser.add_argument("--tag", default="factorial_intervention_v1")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=RLVR_DATA_ROOT / "outputs" / "grpo_sft" / "grpo_line_sft_problems_v1.jsonl",
    )
    parser.add_argument(
        "--supervision",
        type=Path,
        default=RLVR_DATA_ROOT
        / "outputs"
        / "grpo_sft"
        / "grpo_line_sft_supervision_k4_entrance-diverse_v1.jsonl",
    )
    parser.add_argument("--raw-on-policy", type=Path, default=None)
    parser.add_argument("--raw-on-policy-val", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=RLVR_DATA_ROOT / "outputs" / "factorial_intervention_native")
    parser.add_argument("--train-parquet", type=Path, default=None)
    parser.add_argument("--val-parquet", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "model" / "qwen253B")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--prompt-batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def source_row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def has_health_failure(text: str) -> bool:
    """Reject empty or numeric/code NaN/Inf completions before aggregation."""

    stripped = str(text or "").strip()
    return not stripped or stripped.casefold() in HEALTH_STANDALONE or bool(HEALTH_RE.search(stripped))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def validate_ledger(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    by_key: dict[str, str] = {}
    for row in rows:
        uid = str(row.get("problem_uid", ""))
        if not uid or uid in by_uid:
            raise ValueError(f"duplicate or missing problem_uid: {uid!r}")
        numbers = [int(value) for value in row.get("numbers", [])]
        target = int(row["target"])
        key = str(row.get("semantic_key", ""))
        expected_key = semantic_key_text(numbers, target)
        if key != expected_key:
            raise ValueError(f"semantic key mismatch for {uid}: {key!r} != {expected_key!r}")
        if str(row.get("split")) not in {"train", "val"}:
            raise ValueError(f"unknown split for {uid}: {row.get('split')!r}")
        if key in by_key:
            raise ValueError(f"duplicate semantic key {key}: {uid} and {by_key[key]}")
        by_uid[uid] = {**row, "numbers": numbers, "target": target, "semantic_key": key}
        by_key[key] = uid
    counts = {split: sum(row["split"] == split for row in by_uid.values()) for split in ("train", "val")}
    if counts != {"train": 8000, "val": 500}:
        raise ValueError(f"M12 ledger must be 8000 train + 500 val rows, got {counts}")
    return by_uid


def render_native_prompt(tokenizer: Any, numbers: Iterable[int], target: int) -> str:
    messages = build_philschmid_messages(list(map(int, numbers)), int(target))
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


def _parquet_paths(args: argparse.Namespace, source_kind: str, objective: str = "mle") -> tuple[Path, Path]:
    suffix = "" if objective == "mle" else f"_{objective}"
    train = args.train_parquet or args.out_dir / f"factorial_intervention_{source_kind}_train{suffix}_{args.tag}.parquet"
    val = args.val_parquet or args.out_dir / f"factorial_intervention_{source_kind}_val{suffix}_{args.tag}.parquet"
    return train, val


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, engine="pyarrow")
    os.replace(partial, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def write_mle_jsonl(
    path: Path,
    source_rows: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    source_kind: str,
) -> dict[str, Any]:
    """Export validated source rows in the JSONL shape used by the LoRA MLE trainer."""

    if path.exists():
        raise FileExistsError(path)
    _validate_source_rows(source_rows, ledger, source_kind=source_kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    written = 0
    with partial.open("w", encoding="utf-8") as handle:
        for row in source_rows:
            problem = ledger[str(row["problem_uid"])]
            record = {
                "problem_uid": problem["problem_uid"],
                "semantic_key": problem["semantic_key"],
                "numbers": problem["numbers"],
                "target": problem["target"],
                "split": problem["split"],
                "completion": str(row["completion"]),
                "source_kind": source_kind,
                "source_row_hash": source_row_hash(row),
                "sample_id": int(row.get("sample_id", row.get("sample_index", 0))),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    os.replace(partial, path)
    return {"path": str(path), "rows": written, "sha256": sha256(path), "source_kind": source_kind}


def _make_parquet_row(
    problem: dict[str, Any],
    prompt_text: str,
    source_kind: str,
    source_row: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    completion = str(source_row.get("completion", ""))
    source_hash = source_row_hash(source_row)
    extra_info = {
        "index": ordinal,
        "problem_uid": problem["problem_uid"],
        "semantic_key": problem["semantic_key"],
        "split": problem["split"],
        "source_kind": source_kind,
        "source_row_hash": source_hash,
        "sample_id": int(
            source_row.get(
                "sample_id",
                source_row.get("sample_index", source_row.get("_derived_sample_id", ordinal)),
            )
        ),
        "data_version": DATA_VERSION,
    }
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "data_source": DATA_SOURCE,
        "ability": "countdown",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "numbers": problem["numbers"],
                "target": int(problem["target"]),
                "semantic_key": problem["semantic_key"],
                "problem_uid": problem["problem_uid"],
            },
        },
        "extra_info": extra_info,
        "problem_uid": problem["problem_uid"],
        "semantic_key": problem["semantic_key"],
        "split": problem["split"],
        "source_kind": source_kind,
        "source_row_hash": source_hash,
        "sample_id": extra_info["sample_id"],
        "completion": completion,
        "completion_health_ok": not has_health_failure(completion) if completion else None,
        "native_eval_ok": bool(source_row.get("overall_ok")) if "overall_ok" in source_row else None,
    }


def _validate_source_rows(
    source_rows: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    *,
    source_kind: str,
) -> None:
    seen: set[tuple[str, int]] = set()
    fallback_counts: dict[str, int] = {}
    for row in source_rows:
        uid = str(row.get("problem_uid", ""))
        if uid not in ledger:
            raise ValueError(f"source row references unknown problem_uid: {uid}")
        problem = ledger[uid]
        completion = str(row.get("completion", ""))
        if has_health_failure(completion):
            raise ValueError(f"empty/NaN/Inf completion in {source_kind} source: {uid}")
        result = evaluate_native_countdown(completion, problem["numbers"], problem["target"])
        if source_kind == "solver-rich" and not result.overall_ok:
            raise ValueError(f"solver-rich source is not natively valid: {uid}")
        if "sample_id" in row or "sample_index" in row:
            sample_id = int(row.get("sample_id", row.get("sample_index")))
        else:
            sample_id = fallback_counts.get(uid, 0)
            fallback_counts[uid] = sample_id + 1
        key = (uid, sample_id)
        if key in seen:
            raise ValueError(f"duplicate source identity: {(uid, sample_id)}")
        seen.add(key)


def _build_rows(
    ledger: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    tokenizer: Any,
    source_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_source_rows(source_rows, ledger, source_kind=source_kind)
    prompt_cache = {
        uid: render_native_prompt(tokenizer, problem["numbers"], problem["target"])
        for uid, problem in ledger.items()
    }
    train_rows = []
    val_source_rows = []
    fallback_counts: dict[str, int] = {}
    for ordinal, source in enumerate(source_rows):
        uid = str(source["problem_uid"])
        source = dict(source)
        if "sample_id" not in source and "sample_index" not in source:
            source["_derived_sample_id"] = fallback_counts.get(uid, 0)
            fallback_counts[uid] = source["_derived_sample_id"] + 1
        target_rows = train_rows if ledger[uid]["split"] == "train" else val_source_rows
        target_rows.append(_make_parquet_row(ledger[uid], prompt_cache[uid], source_kind, source, ordinal))
    if not train_rows:
        raise ValueError("no train source rows")
    # Validation parquet for MLE retains the validated completions available
    # in the source supervision.  If a source lacks a validation completion,
    # retain one identity row so VERL still has the exact 500-key validation
    # ledger; MLE launchers can choose the MLE validation file explicitly.
    val_by_uid = {row["problem_uid"]: row for row in val_source_rows}
    val_rows = []
    for ordinal, problem in enumerate(ledger.values()):
        if problem["split"] == "val":
            if problem["problem_uid"] in val_by_uid:
                val_rows.append(val_by_uid[problem["problem_uid"]])
            else:
                val_rows.append(_make_parquet_row(problem, prompt_cache[problem["problem_uid"]], source_kind,
                                                  {"completion": "", "sample_id": 0, "validation_only": True}, ordinal))
    if not train_rows or len(val_rows) != 500:
        raise ValueError(f"bad native conversion sizes: train={len(train_rows)} val={len(val_rows)}")
    return train_rows, val_rows


def _write_factor_parquet(
    args: argparse.Namespace,
    ledger: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    tokenizer: Any,
    source_kind: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    train_rows, val_rows = _build_rows(ledger, source_rows, tokenizer, source_kind)
    train_path, val_path = _parquet_paths(args, source_kind, "mle")
    grpo_train_path, grpo_val_path = _parquet_paths(args, source_kind, "grpo")
    grpo_train_rows, grpo_val_rows = _build_grpo_rows(ledger, source_rows, tokenizer, source_kind)
    if len(grpo_train_rows) != 8000 or len(grpo_val_rows) != 500:
        raise ValueError(f"bad GRPO prompt conversion sizes: train={len(grpo_train_rows)} val={len(grpo_val_rows)}")
    _atomic_parquet(train_path, pd.DataFrame(train_rows))
    _atomic_parquet(val_path, pd.DataFrame(val_rows))
    _atomic_parquet(grpo_train_path, pd.DataFrame(grpo_train_rows))
    _atomic_parquet(grpo_val_path, pd.DataFrame(grpo_val_rows))
    manifest = {
        "artifact": DATA_VERSION,
        "status": "complete",
        "source_kind": source_kind,
        "data_source": DATA_SOURCE,
        "ledger": str(args.ledger),
        "ledger_sha256": sha256(args.ledger),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "grpo_train_rows": len(grpo_train_rows),
        "grpo_val_rows": len(grpo_val_rows),
        "train_problems": len({row["problem_uid"] for row in train_rows}),
        "val_problems": len({row["problem_uid"] for row in val_rows}),
        "train_parquet": str(train_path),
        "val_parquet": str(val_path),
        "grpo_train_parquet": str(grpo_train_path),
        "grpo_val_parquet": str(grpo_val_path),
        "train_parquet_sha256": sha256(train_path),
        "val_parquet_sha256": sha256(val_path),
        "grpo_train_parquet_sha256": sha256(grpo_train_path),
        "grpo_val_parquet_sha256": sha256(grpo_val_path),
        "provenance": provenance,
    }
    _atomic_json(args.out_dir / f"factorial_intervention_{source_kind}_manifest_{args.tag}.json", manifest)
    return manifest


def _build_grpo_rows(
    ledger: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    tokenizer: Any,
    source_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make one prompt row per ledger problem for objective-orthogonal GRPO.

    Completion rows belong to MLE.  Native GRPO samples responses itself, so
    retaining one row per problem prevents solver-rich/on-policy source row
    counts from changing the prompt weighting of the GRPO cells.
    """

    source_by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        source_by_uid.setdefault(str(row.get("problem_uid", "")), []).append(row)
    prompts = {
        uid: render_native_prompt(tokenizer, problem["numbers"], problem["target"])
        for uid, problem in ledger.items()
    }
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for ordinal, problem in enumerate(ledger.values()):
        uid = problem["problem_uid"]
        source_fingerprint = hashlib.sha256(
            _canonical_json(source_by_uid.get(uid, [{"problem_uid": uid, "source_kind": source_kind}])).encode("utf-8")
        ).hexdigest()
        source = {
            "completion": "",
            "sample_id": 0,
            "source_fingerprint": source_fingerprint,
            "objective": "grpo",
        }
        row = _make_parquet_row(problem, prompts[uid], source_kind, source, ordinal)
        row["completion"] = None
        row["completion_health_ok"] = None
        row["native_eval_ok"] = None
        row["extra_info"]["objective"] = "grpo"
        row["extra_info"]["source_fingerprint"] = source_fingerprint
        (train_rows if problem["split"] == "train" else val_rows).append(row)
    return train_rows, val_rows


def build_solver(args: argparse.Namespace) -> dict[str, Any]:
    ledger = validate_ledger(read_jsonl(args.ledger))
    source_rows = read_jsonl(args.supervision)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=False)
    return _write_factor_parquet(
        args,
        ledger,
        source_rows,
        tokenizer,
        "solver-rich",
        {
            "supervision": str(args.supervision),
            "supervision_sha256": sha256(args.supervision),
            "supervision_rows": len(source_rows),
            "generation": "deterministic solver enumeration and native evaluator",
        },
    )


def _raw_output(args: argparse.Namespace) -> Path:
    return args.raw_on_policy or args.out_dir / f"factorial_intervention_on_policy_raw_{args.tag}.jsonl"


def generate_on_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples_per_problem < 1 or args.prompt_batch_size < 1:
        raise ValueError("samples-per-problem and prompt-batch-size must be positive")
    ledger = validate_ledger(read_jsonl(args.ledger))
    problems = [row for row in ledger.values() if row["split"] == args.split]
    output = _raw_output(args)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=False)
    prompts = [render_native_prompt(tokenizer, row["numbers"], row["target"]) for row in problems]
    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        seed=args.seed,
    )
    written = 0
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for start in range(0, len(problems), args.prompt_batch_size):
                batch_problems = problems[start : start + args.prompt_batch_size]
                batch_prompts = prompts[start : start + args.prompt_batch_size]
                sampling = SamplingParams(
                    n=args.samples_per_problem,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    seed=args.seed + start,
                )
                outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
                if len(outputs) != len(batch_problems):
                    raise RuntimeError("vLLM request count mismatch")
                for problem, request in zip(batch_problems, outputs):
                    if len(request.outputs) != args.samples_per_problem:
                        raise RuntimeError("vLLM sample count mismatch")
                    for sample_id, sample in enumerate(request.outputs):
                        completion = str(sample.text or "")
                        if has_health_failure(completion):
                            raise RuntimeError(
                                f"on-policy generation emitted empty/NaN/Inf completion at "
                                f"{problem['problem_uid']} sample {sample_id}"
                            )
                        result = evaluate_native_countdown(completion, problem["numbers"], problem["target"])
                        row = {
                            "problem_uid": problem["problem_uid"],
                            "semantic_key": problem["semantic_key"],
                            "split": problem["split"],
                            "numbers": problem["numbers"],
                            "target": problem["target"],
                            "sample_id": sample_id,
                            "completion": completion,
                            "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                            "overall_ok": bool(result.overall_ok),
                            "native_format_ok": bool(result.native_format_ok),
                            "sampling": {
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "max_new_tokens": args.max_new_tokens,
                                "seed": args.seed + start,
                            },
                            "source_model": str(args.model_path),
                        }
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        written += 1
                handle.flush()
                print(f"[on-policy] {start + len(batch_problems)}/{len(problems)} prompts", flush=True)
        expected = len(problems) * args.samples_per_problem
        if written != expected:
            raise RuntimeError(f"raw row count mismatch: {written} != {expected}")
        os.replace(partial, output)
    finally:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
    manifest = {
        "artifact": DATA_VERSION,
        "status": "complete",
        "source_kind": "on-policy",
        "raw_path": str(output),
        "raw_sha256": sha256(output),
        "ledger": str(args.ledger),
        "ledger_sha256": sha256(args.ledger),
        "split": args.split,
        "problems": len(problems),
        "samples_per_problem": args.samples_per_problem,
        "rows": written,
        "model_path": str(args.model_path),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
    }
    _atomic_json(args.out_dir / f"factorial_intervention_on_policy_raw_manifest_{args.tag}.json", manifest)
    return manifest


def build_on_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.raw_on_policy is None:
        raise ValueError("--raw-on-policy is required for build-on-policy")
    ledger = validate_ledger(read_jsonl(args.ledger))
    source_rows = read_jsonl(args.raw_on_policy)
    if args.raw_on_policy_val is None:
        raise ValueError("--raw-on-policy-val is required for the complete train/validation factor")
    source_rows.extend(read_jsonl(args.raw_on_policy_val))
    expected_problems = {
        uid for uid, row in ledger.items() if row["split"] in {"train", "val"}
    }
    counts = {uid: [] for uid in expected_problems}
    for row in source_rows:
        uid = str(row.get("problem_uid", ""))
        if uid in counts:
            counts[uid].append(int(row.get("sample_id", row.get("sample_index", -1))))
    if set(counts) != expected_problems or any(
        set(value) != set(range(args.samples_per_problem)) for value in counts.values()
    ):
        bad = {
            uid: sorted(sample_ids)
            for uid, sample_ids in counts.items()
            if set(sample_ids) != set(range(args.samples_per_problem))
        }
        raise ValueError(f"on-policy raw does not have exact per-problem sample IDs: {list(bad.items())[:5]}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=False)
    result = _write_factor_parquet(
        args,
        ledger,
        source_rows,
        tokenizer,
        "on-policy",
        {
            "raw_on_policy": str(args.raw_on_policy),
            "raw_on_policy_sha256": sha256(args.raw_on_policy),
            "raw_on_policy_val": str(args.raw_on_policy_val),
            "raw_on_policy_val_sha256": sha256(args.raw_on_policy_val),
            "raw_rows": len(source_rows),
            "source_policy": "fresh base-model native rollouts on both frozen train and validation splits; invalid but healthy completions retained for MLE provenance",
            "samples_per_problem": args.samples_per_problem,
        },
    )
    mle_jsonl = args.out_dir / f"factorial_intervention_on_policy_supervision_{args.tag}.jsonl"
    result["mle_jsonl"] = write_mle_jsonl(mle_jsonl, source_rows, ledger, "on-policy")
    manifest_path = args.out_dir / f"factorial_intervention_on-policy_mle_manifest_{args.tag}.json"
    _atomic_json(manifest_path, {**result, "mle_jsonl": result["mle_jsonl"]})
    return result


def audit_parquet(path: Path, ledger: dict[str, dict[str, Any]], expected_split: str, expected_source: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = REQUIRED_PARQUET_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(frame["data_source"].astype(str)) != {DATA_SOURCE}:
        raise ValueError(f"unexpected data_source in {path}")
    if set(frame["source_kind"].astype(str)) != {expected_source}:
        raise ValueError(f"unexpected source_kind in {path}")
    if set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError(f"unexpected split in {path}")
    seen: set[tuple[str, int]] = set()
    for row in frame.to_dict("records"):
        uid = str(row["problem_uid"])
        if uid not in ledger or ledger[uid]["split"] != expected_split:
            raise ValueError(f"parquet row has invalid problem identity: {uid}")
        if str(row["semantic_key"]) != ledger[uid]["semantic_key"]:
            raise ValueError(f"parquet semantic-key mismatch: {uid}")
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if not isinstance(prompt, list) or len(prompt) != 1 or not str(prompt[0].get("content", "")).strip():
            raise ValueError(f"invalid pre-rendered prompt for {uid}")
        info = row["extra_info"]
        if str(info.get("data_version")) != DATA_VERSION or str(info.get("source_kind")) != expected_source:
            raise ValueError(f"missing provenance for {uid}")
        identity = (uid, int(row["sample_id"]))
        if identity in seen:
            raise ValueError(f"duplicate parquet identity: {identity}")
        seen.add(identity)
    return {"path": str(path), "rows": len(frame), "problems": frame["problem_uid"].nunique(), "sha256": sha256(path)}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    ledger = validate_ledger(read_jsonl(args.ledger))
    source_kinds = ("solver-rich", "on-policy")
    reports = []
    for source_kind in source_kinds:
        train, val = _parquet_paths(args, source_kind, "mle")
        grpo_train, grpo_val = _parquet_paths(args, source_kind, "grpo")
        reports.append(audit_parquet(train, ledger, "train", source_kind))
        reports.append(audit_parquet(val, ledger, "val", source_kind))
        reports.append(audit_parquet(grpo_train, ledger, "train", source_kind))
        reports.append(audit_parquet(grpo_val, ledger, "val", source_kind))
    result = {"status": "complete", "data_source": DATA_SOURCE, "ledger_sha256": sha256(args.ledger), "parquet": reports}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "build-solver":
        result = build_solver(args)
    elif args.mode == "generate-on-policy":
        result = generate_on_policy(args)
    elif args.mode == "build-on-policy":
        result = build_on_policy(args)
    else:
        result = audit(args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
