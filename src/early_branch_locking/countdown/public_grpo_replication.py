#!/usr/bin/env python
"""Replicate the public Philschmid Countdown trajectory under a native prompt.

The runner keeps the external protocol independent from the local
``<feasible>`` task.  It provides CPU-only inventory/safe-set preparation and
vLLM collection/aggregation entry points.  Full n=320 runs must be launched
only after ``--mode select`` has written the immutable peak/late selection.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    enumerate_solution_set,
    extract_ground_truth,
    load_parquet_sorted,
    pass_at_k,
)
from early_branch_locking.core.external_countdown import (  # noqa: E402
    evaluate_native_countdown,
)
from early_branch_locking.core.entrance_detection import (  # noqa: E402
    find_first_reasoning_entrance,
)


REPO_ID = "philschmid/qwen-2.5-3b-r1-countdown"
REPO_REVISION = "8da99c421d1fc90f2d6d28dce8ac55cd59b05982"
DEFAULT_LOCAL_ROOT = ROOT / "checkpoints" / "TinyZero" / "external-countdown" / "philschmid_qwen-2.5-3b-r1-countdown"
DEFAULT_PUBLIC_CACHE = Path.home() / ".cache" / "huggingface" / "datasets" / "Jiayi-Pan___countdown-tasks-3to4"
PUBLIC_DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
PUBLIC_TRAIN_SEED = 42
SAFE_SET_SEED = 1729
BOOTSTRAP_DRAWS = 10_000
CHECKPOINT_RE = re.compile(r"(?:^|/)checkpoint-(\d+)(?:/|$)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "inventory", "download", "prepare-safe-set", "collect", "aggregate",
            "select", "tf-profile", "staircase-prepare", "staircase-run",
            "staircase-aggregate", "summarize",
        ),
        default="inventory",
    )
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--public-cache-path", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--raw-path", action="append", type=Path, default=[])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--steps", default="auto")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--sample-chunk-size", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-training-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--peak-step", type=int, default=None)
    parser.add_argument("--late-step", type=int, default=None)
    parser.add_argument("--n-continuations", type=int, default=16)
    parser.add_argument("--rungs", default="L0,L3,L4")
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--selection-path", type=Path, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    return parser.parse_args(argv)


def _tagged(stem: str, tag: str, suffix: str) -> Path:
    return METRICS_DIR / f"{stem}_{tag}.{suffix}"


def _raw_tagged(label: str, n: int, tag: str) -> Path:
    return RAW_DIR / f"s2_philschmid_{label}_n{n}_{tag}.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def semantic_key(numbers: Iterable[int], target: int) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(int(value) for value in numbers)), int(target)


def semantic_key_text(numbers: Iterable[int], target: int) -> str:
    numbers_key, target_key = semantic_key(numbers, target)
    return json.dumps([list(numbers_key), target_key], separators=(",", ":"))


def build_philschmid_messages(numbers: Sequence[int], target: int) -> list[dict[str, str]]:
    """Exact public system/user/assistant prefill contract."""

    number_text = str(list(map(int, numbers)))
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You first thinks about the reasoning "
                "process in the mind and then provides the user with the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Using the numbers {number_text}, create an equation that equals {int(target)}. "
                "You can use basic arithmetic operations (+, -, *, /) and each number can "
                "only be used once. Show your work in <think> </think> tags. And return the "
                "final equation and answer in <answer> </answer> tags, for example "
                "<answer> (1 + 2) / 3 = 1 </answer>."
            ),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>",
        },
    ]


def build_philschmid_prompt(tokenizer, numbers: Sequence[int], target: int) -> str:
    messages = build_philschmid_messages(numbers, target)
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


def _find_public_arrow(cache_root: Path | None = None) -> Path:
    roots = [cache_root] if cache_root is not None else [DEFAULT_PUBLIC_CACHE]
    candidates: list[Path] = []
    for root in roots:
        if root is None:
            continue
        if root.is_file():
            candidates.append(root)
        elif root.exists():
            candidates.extend(root.rglob("countdown-tasks-3to4-train.arrow"))
    if not candidates:
        raise FileNotFoundError(
            "public Countdown cache not found; pass --public-cache-path or populate "
            f"the cache for {PUBLIC_DATASET_ID}"
        )
    candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    return candidates[0]


def _load_public_dataset(cache_path: Path | None = None):
    path = _find_public_arrow(cache_path)
    try:
        from datasets import Dataset

        dataset = Dataset.from_file(str(path))
        if len(dataset) < 50_000:
            raise ValueError(f"public cache has only {len(dataset)} rows")
        return dataset, path, "datasets.Dataset.from_file"
    except Exception as exc:
        # The fallback preserves row order but cannot guarantee the exact
        # datasets-library shuffle implementation; the manifest records it.
        import pyarrow as pa
        import pyarrow.ipc as ipc

        rows: list[dict[str, Any]] = []
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                rows.extend(batch.to_pylist())
        if len(rows) < 50_000:
            raise ValueError(f"public cache has only {len(rows)} rows") from exc
        return rows, path, f"pyarrow fallback after {type(exc).__name__}"


def _dataset_rows(dataset: Any, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    if indices is None:
        if hasattr(dataset, "to_list"):
            return dataset.to_list()
        return list(dataset)
    if hasattr(dataset, "select"):
        return dataset.select(list(indices)).to_list()
    return [dataset[int(index)] for index in indices]


def _training_superset(dataset: Any) -> tuple[list[dict[str, Any]], str]:
    if hasattr(dataset, "shuffle"):
        selected = dataset.shuffle(seed=PUBLIC_TRAIN_SEED).select(range(50_000))
        return selected.to_list(), "datasets.Dataset.shuffle(seed=42).select(range(50000))"
    rng = np.random.default_rng(PUBLIC_TRAIN_SEED)
    indices = rng.permutation(len(dataset))[:50_000]
    return _dataset_rows(dataset, indices), "numpy permutation fallback (datasets unavailable)"


def _safe_manifest_path(args: argparse.Namespace) -> Path:
    return args.eval_manifest or _tagged("s2_philschmid_safe_eval_manifest", args.tag, "json")


def prepare_safe_set(args: argparse.Namespace) -> Path:
    dataset, cache_path, loader = _load_public_dataset(args.public_cache_path)
    train_rows, shuffle_rule = _training_superset(dataset)
    training_keys = {semantic_key(row.get("nums", []), row["target"]) for row in train_rows}
    current_records = load_parquet_sorted(TEST_PARQUET, n=150, sort_key="sample_id")
    rows: list[dict[str, Any]] = []
    for problem_index, record in enumerate(current_records):
        numbers, target, feasible = extract_ground_truth(record)
        key = semantic_key(numbers, target)
        rows.append(
            {
                "our_problem_index": problem_index,
                "numbers": json.dumps(list(map(int, numbers))),
                "target": int(target),
                "semantic_key": semantic_key_text(numbers, target),
                "in_philschmid_50k_superset": key in training_keys,
                "keep_for_external_eval": key not in training_keys,
                "source": "our_test",
                "feasible_label": str(feasible),
            }
        )
    primary_safe = [row for row in rows if row["keep_for_external_eval"]]
    fallback_used = False
    fallback_rows: list[dict[str, Any]] = []
    if len(primary_safe) < 100:
        fallback_used = True
        complement: list[tuple[int, dict[str, Any]]] = []
        seen: set[tuple[tuple[int, ...], int]] = set(training_keys)
        for index, row in enumerate(_dataset_rows(dataset)):
            key = semantic_key(row.get("nums", []), row["target"])
            if key in seen:
                continue
            seen.add(key)
            complement.append((index, row))
        rng = np.random.default_rng(SAFE_SET_SEED)
        order = rng.permutation(len(complement))
        for selected_index in order[:150]:
            source_index, row = complement[int(selected_index)]
            numbers = list(map(int, row["nums"]))
            target = int(row["target"])
            fallback_rows.append(
                {
                    "our_problem_index": f"external:{source_index}",
                    "numbers": json.dumps(numbers),
                    "target": target,
                    "semantic_key": semantic_key_text(numbers, target),
                    "in_philschmid_50k_superset": False,
                    "keep_for_external_eval": True,
                    "source": "public_complement_seed1729",
                    "feasible_label": "yes",
                }
            )
        selected_rows = fallback_rows
    else:
        selected_rows = primary_safe
    output_csv = _tagged("s2_philschmid_safe_eval_rows", args.tag, "csv")
    output_manifest = _safe_manifest_path(args)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows + fallback_rows).to_csv(output_csv, index=False)
    manifest = {
        "experiment_id": "public_grpo_replication",
        "artifact": "safe_eval_set",
        "version": args.tag,
        "public_dataset": PUBLIC_DATASET_ID,
        "public_cache_path": str(cache_path),
        "public_cache_sha256": _sha256(cache_path),
        "public_loader": loader,
        "public_dataset_rows": len(dataset),
        "train_superset_size": len(train_rows),
        "train_superset_shuffle_rule": shuffle_rule,
        "train_superset_seed": PUBLIC_TRAIN_SEED,
        "semantic_key_definition": "(tuple(sorted(int(numbers))), int(target))",
        "our_test_rows": len(rows),
        "primary_safe_rows": len(primary_safe),
        "overlap_excluded": len(rows) - len(primary_safe),
        "fallback_used": fallback_used,
        "fallback_seed": SAFE_SET_SEED if fallback_used else None,
        "selected_eval_rows": len(selected_rows),
        "selected_source": "public complement" if fallback_used else "our 150 held-out rows",
        "own_step50_step275_required_on_fallback": fallback_used,
        "rows_path": str(output_csv),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))
    return output_manifest


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _known_specs() -> list[tuple[str, int]]:
    import importlib.util

    helper_path = ROOT / "scripts" / "download_hf_mirror_models.py"
    spec = importlib.util.spec_from_file_location("tinyzero_hf_mirror_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load download helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    known_files = module.known_files

    specs = known_files(REPO_ID)
    if specs is None:
        raise RuntimeError(f"no fallback file manifest for {REPO_ID}")
    return specs


def _step_specs(step: int) -> list[tuple[str, int]]:
    prefix = f"checkpoint-{int(step)}/"
    return [(name[len(prefix):], size) for name, size in _known_specs() if name.startswith(prefix)]


def _local_file_status(step: int, root: Path) -> dict[str, Any]:
    specs = _step_specs(step)
    directory = root / f"checkpoint-{step}"
    inference_names = {
        "config.json", "generation_config.json", "merges.txt", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors", "model.safetensors.index.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    }
    inference_missing: list[str] = []
    inference_bad: list[str] = []
    training_missing: list[str] = []
    partial_files: list[str] = []
    for name, expected in specs:
        path = directory / name
        part_path = path.with_name(path.name + ".part")
        if part_path.is_file():
            partial_files.append(str(part_path))
        if not path.is_file():
            if name in inference_names:
                inference_missing.append(name)
            training_missing.append(name)
        elif path.stat().st_size != expected:
            if name in inference_names:
                inference_bad.append(name)
            training_missing.append(name)
    return {
        "step": int(step),
        "path": str(directory),
        "inference_complete": not inference_missing and not inference_bad,
        "inference_missing": inference_missing,
        "inference_bad": inference_bad,
        "training_directory_complete": not training_missing and not partial_files,
        "training_missing_or_bad": training_missing,
        "partial_files": partial_files,
        "file_count_expected": len(specs),
        "file_count_present": sum((directory / name).is_file() and (directory / name).stat().st_size == expected for name, expected in specs),
    }


def inventory(args: argparse.Namespace) -> Path:
    remote_files: list[str] = []
    remote_revision = REPO_REVISION
    remote_error: str | None = None
    try:
        from huggingface_hub import HfApi

        api = HfApi(endpoint="https://hf-mirror.com")
        info = api.model_info(REPO_ID, revision="main", files_metadata=False)
        remote_revision = str(getattr(info, "sha", None) or REPO_REVISION)
        remote_files = [str(item.rfilename) for item in info.siblings]
    except Exception as exc:
        remote_error = f"{type(exc).__name__}: {exc}"
    if not remote_files:
        remote_files = [name for name, _ in _known_specs()]
    remote_steps = sorted({int(match.group(1)) for name in remote_files if (match := CHECKPOINT_RE.search(name))})
    if not remote_steps:
        remote_steps = sorted({int(name.split("/")[0].split("-")[-1]) for name, _ in _known_specs() if name.startswith("checkpoint-")})
    local_steps = {int(path.name.split("-")[-1]) for path in args.local_root.glob("checkpoint-*") if path.is_dir() and path.name.split("-")[-1].isdigit()}
    statuses = [_local_file_status(step, args.local_root) for step in sorted(local_steps | set(remote_steps))]
    output = _tagged("s2_remote_inventory", args.tag, "json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "public_grpo_replication",
        "repo_id": REPO_ID,
        "remote_revision": remote_revision,
        "known_revision_fallback": REPO_REVISION,
        "remote_inventory_source": "HfApi mirror" if remote_error is None else "known local manifest fallback",
        "remote_error": remote_error,
        "remote_file_count": len(remote_files),
        "discovered_checkpoint_steps": remote_steps,
        "local_root": str(args.local_root),
        "checkpoints": statuses,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=_json_default))
    return output


def _selected_steps(text: str, available: Sequence[int]) -> list[int]:
    if str(text).strip().lower() == "auto":
        return list(available)
    requested = sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"requested checkpoint steps are unavailable: {unknown}")
    return requested


def download(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import importlib.util

    helper_path = ROOT / "scripts" / "download_hf_mirror_models.py"
    spec = importlib.util.spec_from_file_location("tinyzero_hf_mirror_download_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load download helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    download_file = module.download_file

    remote_steps = sorted({int(path.name.split("-")[-1]) for path in args.local_root.glob("checkpoint-*") if path.is_dir() and path.name.split("-")[-1].isdigit()})
    if not remote_steps:
        remote_steps = [step for step, _ in [(int(name.split("/")[0].split("-")[-1]), size) for name, size in _known_specs() if name.startswith("checkpoint-")]]
    steps = _selected_steps(args.steps, remote_steps)
    files = {"config.json", "generation_config.json", "merges.txt", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    jobs: list[tuple[str, str, str, int, Path]] = []
    for step in steps:
        for name, size in _step_specs(step):
            if args.include_training_files or name in files:
                jobs.append((REPO_ID, REPO_REVISION, f"checkpoint-{step}/{name}", size, args.local_root))
    total = sum(job[3] for job in jobs)
    payload = {"repo_id": REPO_ID, "steps": steps, "files": len(jobs), "bytes": total, "dry_run": args.dry_run, "include_training_files": args.include_training_files}
    print(json.dumps(payload, indent=2, default=_json_default))
    if args.dry_run:
        return
    args.local_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_file, *job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), default=_json_default), flush=True)


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    selected = frame[frame["keep_for_external_eval"].astype(str).str.lower().isin({"true", "1"})]
    rows = []
    for _, row in selected.iterrows():
        rows.append({
            "problem_id": str(row["our_problem_index"]),
            "numbers": list(map(int, json.loads(row["numbers"]))),
            "target": int(row["target"]),
            "semantic_key": str(row["semantic_key"]),
            "source": str(row.get("source", "our_test")),
        })
    return rows


def _manifest_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = _safe_manifest_path(args)
    if not manifest.exists():
        raise FileNotFoundError(f"safe-set manifest not found: {manifest}; run prepare-safe-set first")
    payload = json.loads(manifest.read_text())
    rows_path = Path(payload["rows_path"])
    rows = _load_eval_rows(rows_path)
    if args.num_problems is not None:
        rows = rows[: int(args.num_problems)]
    return rows


def _model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return str(args.model_label)
    if args.checkpoint_step is not None:
        return f"step{int(args.checkpoint_step)}"
    if args.model_path is not None:
        return args.model_path.name
    raise ValueError("collect requires --model-path or --checkpoint-step")


def _model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path
    if args.checkpoint_step is None:
        raise ValueError("collect requires --model-path or --checkpoint-step")
    return args.local_root / f"checkpoint-{int(args.checkpoint_step)}"


def collect(args: argparse.Namespace) -> Path:
    model_path = _model_path(args)
    label = _model_label(args)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    required_inference_files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
    )
    incomplete = [name for name in required_inference_files if not (model_path / name).is_file()]
    if incomplete:
        raise RuntimeError(f"inference checkpoint is incomplete: {model_path}; missing={incomplete}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    eval_rows = _manifest_rows(args)
    prompts = [build_philschmid_prompt(tokenizer, row["numbers"], row["target"]) for row in eval_rows]
    n = int(args.n_samples)
    chunk = n if int(args.sample_chunk_size) <= 0 else min(n, int(args.sample_chunk_size))
    if chunk <= 0:
        raise ValueError("--sample-chunk-size must be positive")
    output = _raw_tagged(label, n, args.tag)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    temporary = output.with_suffix(output.suffix + ".partial")
    llm = LLM(
        model=str(model_path), tensor_parallel_size=1, trust_remote_code=True,
        dtype=args.dtype, max_model_len=args.max_model_len, seed=args.seed,
        enforce_eager=False,
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for start in range(0, n, chunk):
                current = min(chunk, n - start)
                params = SamplingParams(
                    n=current, temperature=args.temperature, top_p=args.top_p,
                    max_tokens=args.max_new_tokens, seed=args.seed + start,
                )
                outputs = llm.generate(prompts, params)
                if len(outputs) != len(eval_rows):
                    raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(eval_rows)} prompts")
                for problem, request in zip(eval_rows, outputs):
                    if len(request.outputs) != current:
                        raise RuntimeError("vLLM returned an unexpected sample count")
                    for offset, sequence in enumerate(request.outputs):
                        completion = str(sequence.text or "")
                        result = evaluate_native_countdown(completion, problem["numbers"], problem["target"])
                        row = {
                            "experiment_id": "public_grpo_replication",
                            "model_label": label,
                            "model_path": str(model_path),
                            "checkpoint_step": args.checkpoint_step,
                            "problem_id": problem["problem_id"],
                            "numbers": problem["numbers"],
                            "target": problem["target"],
                            "semantic_key": problem["semantic_key"],
                            "problem_source": problem["source"],
                            "sample_id": start + offset,
                            "completion": completion,
                            "n_generated_tokens": len(getattr(sequence, "token_ids", ()) or ()),
                            "sampling_seed": args.seed + start,
                            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
                            **result.as_dict(),
                        }
                        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
                handle.flush()
                print(f"[public_grpo_replication] {label}: samples {start + current}/{n}", flush=True)
        os.replace(temporary, output)
    finally:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        temporary.unlink(missing_ok=True)
    manifest = _tagged(f"s2_philschmid_collect_{label}_n{n}", args.tag, "json")
    manifest.write_text(json.dumps({
        "raw_path": str(output), "model_label": label, "model_path": str(model_path),
        "checkpoint_step": args.checkpoint_step, "n_problems": len(eval_rows),
        "n_samples": n, "expected_raw_rows": len(eval_rows) * n,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "safe_eval_manifest": str(_safe_manifest_path(args)), "git_commit": _git_commit(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return output


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selection_steps(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve the immutable pilot selection before any follow-up run."""

    selection_path = args.selection_path or _tagged("s2_philschmid_selection", args.tag, "json")
    if selection_path.exists():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        status = str(payload.get("selection_status", "selected"))
        if status != "selected" or payload.get("breadth_peak_step") is None or payload.get("late_step") is None:
            raise RuntimeError(
                f"immutable public_grpo_replication selection is unavailable: status={status}; "
                f"selection_manifest={selection_path}"
            )
        return int(payload["breadth_peak_step"]), int(payload["late_step"])
    if args.peak_step is None or args.late_step is None:
        raise FileNotFoundError(
            f"selection manifest not found: {selection_path}; run --mode select first "
            "or pass --peak-step and --late-step"
        )
    return int(args.peak_step), int(args.late_step)


def _external_family_ledger(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the deterministic two-family witness ledger shared by public_grpo_replication follow-ups."""

    rows = _manifest_rows(args)
    ledger: list[dict[str, Any]] = []
    for problem in rows:
        solutions = sorted(enumerate_solution_set(problem["numbers"], problem["target"]))
        if not solutions:
            continue
        family_counts: Counter[str] = Counter()
        family_expr: dict[str, str] = {}
        for expression in solutions:
            result = evaluate_native_countdown(
                f"</think><answer>{expression}</answer>",
                problem["numbers"],
                problem["target"],
            )
            family = result.answer_entrance_family
            if family:
                family_counts[family] += 1
                family_expr.setdefault(family, expression)
        designated = [family for family, _ in family_counts.most_common(2)]
        for family in designated:
            expression = family_expr[family]
            entrance = find_first_reasoning_entrance(expression)
            if not entrance.found:
                continue
            ledger.append({
                **problem,
                "designated_family": family,
                "designated_families": designated,
                "witness_expression": expression,
                "witness_entrance_char_start": entrance.char_start,
                "witness_entrance_char_end": entrance.char_end,
                "family_solution_count": int(family_counts[family]),
            })
    return ledger


def _first_calc_parts(expression: str) -> tuple[int, str, int, str] | None:
    """Return first operand/operator/second operand and a stable value string."""

    match = re.search(r"(?<![\w.])(-?\d+)\s*([+\-*/])\s*(-?\d+)", str(expression))
    if match is None:
        return None
    first, op, second = int(match.group(1)), match.group(2), int(match.group(3))
    left, right = Fraction(first), Fraction(second)
    if op == "+":
        value = left + right
    elif op == "-":
        value = left - right
    elif op == "*":
        value = left * right
    else:
        if right == 0:
            return None
        value = left / right
    value_text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return first, op, second, value_text


def _staircase_scaffold(rung: str, expression: str) -> str:
    if rung == "L0":
        return ""
    parts = _first_calc_parts(expression)
    if parts is None:
        raise ValueError(f"cannot render staircase scaffold for witness: {expression}")
    first, op, second, value = parts
    if rung == "L3":
        return f"Let me try: {first} {op}"
    if rung == "L4":
        return f"Let me try: {first} {op} {second} = {value}."
    raise ValueError(f"unsupported public_grpo_replication staircase rung: {rung}")


def _load_model_selection(args: argparse.Namespace) -> tuple[int, int, Path, Path]:
    peak, late = _selection_steps(args)
    peak_path = args.local_root / f"checkpoint-{peak}"
    late_path = args.local_root / f"checkpoint-{late}"
    return peak, late, peak_path, late_path


def _token_nll_rows(model, tokenizer, prompt: str, continuation: str, boundary_char_end: int | None, *, device, max_input_tokens: int) -> dict[str, Any]:
    """Score a native continuation using separately encoded prompt/target ids."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(continuation, add_special_tokens=False)
    full_ids = prompt_ids + target_ids
    truncated = len(full_ids) > max_input_tokens
    if truncated:
        full_ids = full_ids[:max_input_tokens]
        target_ids = full_ids[len(prompt_ids):]
    if len(full_ids) <= len(prompt_ids) or not target_ids:
        return {
            "nll_values": [],
            "target_tokens": 0,
            "entrance_token_end": 0,
            "truncated": bool(truncated),
            "score_status": "empty_target",
        }
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[0].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        labels = input_ids[0, 1:]
        values = -logp.gather(1, labels[:, None]).squeeze(1)
    values = values[len(prompt_ids) - 1 :].detach().cpu().numpy().astype(float)
    boundary = None if boundary_char_end is None else len(tokenizer.encode(continuation[:boundary_char_end], add_special_tokens=False))
    entrance_token_end = int(min(max(boundary if boundary is not None else len(values), 0), len(values)))
    return {
        "nll_values": values.tolist(),
        "target_tokens": int(len(values)),
        "entrance_token_end": entrance_token_end,
        "truncated": bool(truncated),
        "score_status": "ok",
    }


def _load_transformer_model(model_path: Path, device: Any):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
        **({"attn_implementation": "sdpa"} if device.type == "cuda" else {}),
    ).to(device).eval()
    return tokenizer, model


def tf_profile(args: argparse.Namespace) -> Path:
    """Score exact native witness continuations at selected peak and late steps."""

    peak, late, peak_path, late_path = _load_model_selection(args)
    for path in (peak_path, late_path):
        if not path.is_dir():
            raise FileNotFoundError(path)
    ledger = _external_family_ledger(args)
    if not ledger:
        raise ValueError("external witness ledger is empty")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("tf-profile requires PyTorch") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(str(args.gpu_id).split(",")[0]))
    prompt_cache: dict[tuple[str, tuple[int, ...], int], str] = {}
    raw_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    for step, model_path in ((peak, peak_path), (late, late_path)):
        tokenizer, model = _load_transformer_model(model_path, device)
        for index, item in enumerate(ledger):
            key = (str(model_path), tuple(item["numbers"]), int(item["target"]))
            prompt = prompt_cache.setdefault(key, build_philschmid_prompt(tokenizer, item["numbers"], item["target"]))
            continuation = f"Let me try: {item['witness_expression']}"
            scored = _token_nll_rows(
                model, tokenizer, prompt, continuation,
                int(continuation.find(str(item["witness_expression"]))) + int(item["witness_entrance_char_end"] or 0)
                if item.get("witness_entrance_char_end") is not None else None,
                device=device, max_input_tokens=args.max_input_tokens,
            )
            values = np.asarray(scored.pop("nll_values"), dtype=float)
            split = int(scored["entrance_token_end"])
            entrance_values = values[:split]
            execution_values = values[split:]
            witness_rows.append({
                "model_label": f"step{step}", "checkpoint_step": step,
                "problem_id": str(item["problem_id"]), "designated_family": item["designated_family"],
                "entrance_nll": float(entrance_values.sum()) if len(entrance_values) else np.nan,
                "execution_nll": float(execution_values.sum()) if len(execution_values) else np.nan,
                "entrance_tokens": int(len(entrance_values)), "execution_tokens": int(len(execution_values)),
                "entrance_nll_per_token": float(entrance_values.mean()) if len(entrance_values) else np.nan,
                "execution_nll_per_token": float(execution_values.mean()) if len(execution_values) else np.nan,
                "score_status": scored["score_status"], "truncated": scored["truncated"],
            })
            for token_index, value in enumerate(values):
                raw_rows.append({
                    "model_label": f"step{step}", "checkpoint_step": step,
                    "problem_id": str(item["problem_id"]), "designated_family": item["designated_family"],
                    "witness_expression": item["witness_expression"], "token_index": int(token_index),
                    "relative_pos": int(token_index - split), "nll": float(value),
                    "entrance_token_end": split, "target_tokens": int(len(values)),
                    "score_status": scored["score_status"], "truncated": scored["truncated"],
                })
            if (index + 1) % 100 == 0:
                print(f"[public_grpo_replication TF] step={step} witnesses={index + 1}/{len(ledger)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    raw_path = _tagged("s2_philschmid_tf_profile_raw", args.tag, "jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    witness_frame = pd.DataFrame(witness_rows)
    summary_rows: list[dict[str, Any]] = []
    for label, frame in witness_frame.groupby("model_label", sort=True):
        per_problem = frame.groupby("problem_id", as_index=False)[
            ["entrance_nll", "execution_nll", "entrance_nll_per_token", "execution_nll_per_token"]
        ].mean()
        row: dict[str, Any] = {"row_type": "model", "model_label": label, "checkpoint_step": int(frame["checkpoint_step"].iloc[0]), "n_problems": int(len(per_problem)), "n_witnesses": int(len(frame))}
        for metric in ("entrance_nll", "execution_nll", "entrance_nll_per_token", "execution_nll_per_token"):
            mean, lo, hi = _bootstrap(per_problem[metric].to_numpy(float), args.bootstrap_draws, args.seed + len(metric) + int(frame["checkpoint_step"].iloc[0]))
            row[metric], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = mean, lo, hi
        summary_rows.append(row)
    if {peak, late}.issubset(set(witness_frame["checkpoint_step"].dropna().astype(int))):
        by_step = witness_frame.groupby(["checkpoint_step", "problem_id"], as_index=False)[
            ["entrance_nll", "execution_nll", "entrance_nll_per_token", "execution_nll_per_token"]
        ].mean()
        pivot = by_step.pivot(index="problem_id", columns="checkpoint_step")
        for metric in ("entrance_nll", "execution_nll", "entrance_nll_per_token", "execution_nll_per_token"):
            delta = (pivot[metric][late] - pivot[metric][peak]).to_numpy(float)
            mean, lo, hi = _bootstrap(delta, args.bootstrap_draws, args.seed + 7000 + len(metric))
            summary_rows.append({
                "row_type": "delta_late_minus_peak", "model_label": f"step{late}-step{peak}",
                "checkpoint_step": late, "reference_step": peak, "n_problems": int(len(delta)), "n_witnesses": int(len(witness_frame)),
                metric: mean, f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi,
            })
    summary = pd.DataFrame(summary_rows)
    output = _tagged("s2_philschmid_tf_profile", args.tag, "csv")
    summary.to_csv(output, index=False)
    manifest = _tagged("s2_philschmid_tf_profile_manifest", args.tag, "json")
    manifest.write_text(json.dumps({
        "experiment_id": "public_grpo_replication", "artifact": "teacher_forced_native_profile", "protocol_version": "v1",
        "peak_step": peak, "late_step": late, "ledger_rows": len(ledger), "raw_token_rows": len(raw_rows),
        "statistical_unit": "problem", "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed},
        "sampling": {"prompt": "public Philschmid native prompt", "witness": "Let me try: <solver witness>", "boundary": "first operand+operator in witness"},
        "raw_path": str(raw_path), "summary_path": str(output), "git_commit": _git_commit(),
    }, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"raw": str(raw_path), "summary": str(output), "witness_rows": len(witness_rows)}, sort_keys=True))
    return output


def staircase_prepare(args: argparse.Namespace) -> Path:
    ledger = _external_family_ledger(args)
    rungs = [value.strip() for value in args.rungs.split(",") if value.strip()]
    if set(rungs) - {"L0", "L3", "L4"}:
        raise ValueError("staircase rungs must be from L0,L3,L4")
    rows: list[dict[str, Any]] = []
    for item in ledger:
        for rung in rungs:
            rows.append({
                **item, "rung": rung, "scaffold": _staircase_scaffold(rung, item["witness_expression"]),
            })
    output = args.ledger_path or _tagged("s2_philschmid_staircase_ledger", args.tag, "jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    manifest = _tagged("s2_philschmid_staircase_prepare", args.tag, "json")
    manifest.write_text(json.dumps({
        "experiment_id": "public_grpo_replication", "artifact": "native_staircase_ledger", "rungs": rungs,
        "rows": len(rows), "problems": len({str(row["problem_id"]) for row in rows}),
        "family_selection": "top two solver-enumerated answer entrance families per problem",
        "ledger_path": str(output), "git_commit": _git_commit(),
    }, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(output), "rows": len(rows)}, sort_keys=True))
    return output


def staircase_run(args: argparse.Namespace) -> Path:
    if args.model_path is None or args.model_label is None:
        raise ValueError("staircase-run requires --model-path and --model-label")
    ledger_path = args.ledger_path or _tagged("s2_philschmid_staircase_ledger", args.tag, "jsonl")
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)
    model_path = args.model_path
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"inference checkpoint is incomplete: {model_path}; missing={missing}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ledger = _load_jsonl(ledger_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    prompts = [
        build_philschmid_prompt(tokenizer, row["numbers"], row["target"]) + str(row["scaffold"])
        for row in ledger
    ]
    llm = LLM(model=str(model_path), tensor_parallel_size=1, trust_remote_code=True,
              dtype=args.dtype, max_model_len=args.max_model_len, seed=args.seed, enforce_eager=False)
    output = RAW_DIR / f"s2_philschmid_staircase_raw_{args.model_label}_{args.tag}.jsonl"
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        requests = llm.generate(prompts, SamplingParams(
            n=args.n_continuations, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_new_tokens, seed=args.seed,
        ))
        with temporary.open("w", encoding="utf-8") as handle:
            for row, request in zip(ledger, requests):
                for sample_id, sequence in enumerate(request.outputs):
                    continuation = str(sequence.text or "")
                    full_generated = f"{row['scaffold']}{continuation}"
                    result = evaluate_native_countdown(full_generated, row["numbers"], row["target"])
                    handle.write(json.dumps({
                        "experiment_id": "public_grpo_replication", "model_label": args.model_label,
                        "model_path": str(model_path), "problem_id": row["problem_id"],
                        "numbers": row["numbers"], "target": row["target"],
                        "designated_family": row["designated_family"], "rung": row["rung"],
                        "scaffold": row["scaffold"], "sample_id": sample_id,
                        "continuation": continuation,
                        "n_generated_tokens": len(getattr(sequence, "token_ids", ()) or ()),
                        "designated_family_ok": bool(result.overall_ok and result.answer_entrance_family == row["designated_family"]),
                        "any_valid_ok": bool(result.overall_ok),
                        **result.as_dict(),
                    }, ensure_ascii=False, default=_json_default) + "\n")
        os.replace(temporary, output)
    finally:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        temporary.unlink(missing_ok=True)
    manifest = _tagged(f"s2_philschmid_staircase_{args.model_label}", args.tag, "json")
    manifest.write_text(json.dumps({
        "experiment_id": "public_grpo_replication", "artifact": "native_staircase_raw", "model_label": args.model_label,
        "model_path": str(model_path), "ledger_path": str(ledger_path), "n_ledger_rows": len(ledger),
        "n_continuations": args.n_continuations, "expected_raw_rows": len(ledger) * args.n_continuations,
        "raw_path": str(output), "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "git_commit": _git_commit(),
    }, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"raw": str(output), "rows": len(ledger) * args.n_continuations}, sort_keys=True))
    return output


def staircase_aggregate(args: argparse.Namespace) -> Path:
    paths = args.raw_path or sorted(RAW_DIR.glob(f"s2_philschmid_staircase_raw_*_{args.tag}.jsonl"))
    if not paths:
        raise ValueError("staircase-aggregate found no raw paths")
    rows = [row for path in paths for row in _load_jsonl(path)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("staircase raw files are empty")
    per_problem = frame.groupby(["model_label", "rung", "problem_id"], as_index=False).agg(
        designated_family_completion=("designated_family_ok", "mean"),
        any_valid_completion=("any_valid_ok", "mean"),
        native_format_rate=("native_format_ok", "mean"),
        answer_family_parse_rate=("answer_entrance_family", lambda values: float(pd.Series(values).notna().mean())),
    )
    summary_rows: list[dict[str, Any]] = []
    for (label, rung), group in per_problem.groupby(["model_label", "rung"], sort=True):
        row: dict[str, Any] = {"model_label": label, "rung": rung, "n_problems": len(group), "n_samples_per_problem": int(frame[(frame.model_label == label) & (frame.rung == rung)].groupby("problem_id").size().mode().iloc[0])}
        for metric in ("designated_family_completion", "any_valid_completion", "native_format_rate", "answer_family_parse_rate"):
            mean, lo, hi = _bootstrap(group[metric].to_numpy(float), args.bootstrap_draws, args.seed + len(metric) + len(summary_rows) * 31)
            row[metric], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = mean, lo, hi
        summary_rows.append(row)
    output = _tagged("s2_philschmid_staircase", args.tag, "csv")
    pd.DataFrame(summary_rows).to_csv(output, index=False)
    per_output = _tagged("s2_philschmid_staircase_per_problem", args.tag, "parquet")
    per_problem.to_parquet(per_output, index=False)
    manifest = _tagged("s2_philschmid_staircase_manifest", args.tag, "json")
    manifest.write_text(json.dumps({
        "experiment_id": "public_grpo_replication", "artifact": "native_staircase_summary", "input_paths": [str(path) for path in paths],
        "raw_rows": len(rows), "statistical_unit": "problem", "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed},
        "summary_path": str(output), "per_problem_path": str(per_output), "git_commit": _git_commit(),
    }, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return output


def summarize(args: argparse.Namespace) -> Path:
    peak = late = None
    selection_path = args.selection_path or _tagged("s2_philschmid_selection", args.tag, "json")
    if selection_path.exists():
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        peak, late = selected.get("breadth_peak_step"), selected.get("late_step")
    paths = {
        "selection": str(selection_path) if selection_path.exists() else None,
        "pilot_curve": str(_tagged("s2_philschmid_pilot_curve", args.tag, "csv")),
        "full_summary": str(_tagged("s2_philschmid_full_summary", args.tag, "csv")),
        "tf_profile": str(_tagged("s2_philschmid_tf_profile", args.tag, "csv")),
        "staircase": str(_tagged("s2_philschmid_staircase", args.tag, "csv")),
    }
    payload = {
        "experiment_id": "public_grpo_replication", "artifact": "summary_index", "tag": args.tag,
        "selected_breadth_peak_step": peak, "selected_late_step": late,
        "artifacts": paths, "artifact_status": {name: Path(path).exists() if path else False for name, path in paths.items()},
        "git_commit": _git_commit(),
    }
    output = _tagged("s2_philschmid_summary", args.tag, "json")
    output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return output


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _problem_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_model_problem: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_model_problem.setdefault((str(row["model_label"]), str(row["problem_id"])), []).append(row)
    for (model, pid), group in by_model_problem.items():
        first = group[0]
        numbers, target = list(map(int, first["numbers"])), int(first["target"])
        solution_set = enumerate_solution_set(numbers, target)
        correct = [bool(row.get("overall_ok")) for row in group]
        answer_families = Counter(str(row.get("answer_entrance_family")) for row in group if row.get("answer_entrance_family"))
        think_families = Counter(str(row.get("think_entrance_family")) for row in group if row.get("think_entrance_family"))
        feasible_answer_families = {
            str(evaluate_native_countdown(f"</think><answer>{expr}</answer>", numbers, target).answer_entrance_family)
            for expr in solution_set
        }
        observed_solutions = {str(row.get("canonical_expr")) for row in group if row.get("overall_ok") and row.get("canonical_expr") in solution_set}
        answer_observed = set(answer_families) & feasible_answer_families
        feasible_operator_classes = {family[-1] for family in feasible_answer_families if family}
        observed_operator_classes = {family[-1] for family in answer_families if family}
        family_total = sum(answer_families.values())
        think_total = sum(think_families.values())
        records.append({
            "model_label": model,
            "checkpoint_step": first.get("checkpoint_step"),
            "problem_id": pid,
            "n": len(group),
            "correct_count": int(sum(correct)),
            "pass_at_1": pass_at_k(len(group), sum(correct), 1),
            "pass_at_64": pass_at_k(len(group), sum(correct), min(64, len(group))),
            "pass_at_256": pass_at_k(len(group), sum(correct), min(256, len(group))),
            "native_format_rate": float(np.mean([bool(row.get("native_format_ok")) for row in group])),
            "exact_coverage": len(observed_solutions) / len(solution_set) if solution_set else 0.0,
            "answer_entrance_family_coverage": len(answer_observed) / len(feasible_answer_families) if feasible_answer_families else 0.0,
            "operator_class_coverage": len(observed_operator_classes & feasible_operator_classes) / len(feasible_operator_classes) if feasible_operator_classes else 0.0,
            "answer_entrance_family_entropy": _entropy(answer_families),
            "think_entrance_family_entropy": _entropy(think_families),
            "answer_entrance_access_rate": len(answer_observed) / family_total if family_total else 0.0,
            "think_entrance_access_rate": sum(value for key, value in think_families.items() if key in feasible_answer_families) / think_total if think_total else 0.0,
            "unique_solution_count": len(observed_solutions),
            "zero_access_answer_family_count": len(feasible_answer_families - set(answer_families)),
        })
    return pd.DataFrame(records)


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * np.log(count / total) for count in counts.values() if count))


def aggregate(args: argparse.Namespace) -> Path:
    paths = args.raw_path
    if not paths:
        raise ValueError("aggregate requires one or more --raw-path arguments")
    rows = [row for path in paths for row in _load_jsonl(path)]
    if not rows:
        raise ValueError("raw files are empty")
    per_problem = _problem_frame(rows)
    labels = []
    for label, frame in per_problem.groupby("model_label", sort=False):
        row: dict[str, Any] = {"model_label": label, "checkpoint_step": frame["checkpoint_step"].iloc[0], "n_problems": len(frame), "n_samples_per_problem": int(frame["n"].mode().iloc[0])}
        for metric in ("pass_at_1", "pass_at_64", "pass_at_256", "native_format_rate", "exact_coverage", "answer_entrance_family_coverage", "operator_class_coverage", "answer_entrance_family_entropy", "think_entrance_family_entropy", "zero_access_answer_family_count"):
            mean, lo, hi = _bootstrap(frame[metric].to_numpy(float), args.bootstrap_draws, args.seed + len(labels) * 100 + len(metric))
            row[metric], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = mean, lo, hi
        labels.append(row)
    summary = pd.DataFrame(labels)
    kind = "full_summary" if args.full else "pilot_curve"
    output = _tagged(f"s2_philschmid_{kind}", args.tag, "csv")
    per_stem = "s2_philschmid_full_per_problem" if args.full else "s2_philschmid_pilot_per_problem"
    per_output = _tagged(per_stem, args.tag, "parquet")
    summary.to_csv(output, index=False)
    per_problem.to_parquet(per_output, index=False)
    manifest = _tagged(f"s2_philschmid_{kind}_manifest", args.tag, "json")
    manifest.write_text(json.dumps({
        "kind": kind, "input_paths": [str(path) for path in paths],
        "raw_rows": len(rows), "expected_rows_by_label": {label: int(len(frame)) for label, frame in pd.DataFrame(rows).groupby("model_label")},
        "statistical_unit": "problem", "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed},
        "summary_path": str(output), "per_problem_path": str(per_output), "git_commit": _git_commit(),
    }, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return output


def select(args: argparse.Namespace) -> Path:
    pilot_path = _tagged("s2_philschmid_pilot_curve", args.tag, "csv")
    if not pilot_path.exists():
        raise FileNotFoundError(pilot_path)
    frame = pd.read_csv(pilot_path)
    frame["checkpoint_step"] = pd.to_numeric(frame["checkpoint_step"], errors="coerce")
    checkpoints = frame[frame["checkpoint_step"].notna() & (frame["checkpoint_step"] > 0)].copy()
    if checkpoints.empty:
        raise ValueError("pilot has no numeric checkpoint rows")
    output = _tagged("s2_philschmid_selection", args.tag, "json")
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    observed = [
        {
            "step": int(row["checkpoint_step"]),
            "native_format_rate": float(row["native_format_rate"]),
            "exact_coverage": float(row["exact_coverage"]),
        }
        for _, row in checkpoints.sort_values("checkpoint_step").iterrows()
    ]
    threshold = 0.90
    eligible = checkpoints[checkpoints["native_format_rate"] >= threshold]
    if eligible.empty:
        threshold = 0.80
        eligible = checkpoints[checkpoints["native_format_rate"] >= threshold]
    if eligible.empty:
        payload = {
            "selection_status": "failed_no_eligible_checkpoint",
            "rule": "max exact_cov among format competent checkpoints; ties choose earliest step",
            "format_threshold_requested": 0.90,
            "format_threshold_fallback": 0.80,
            "format_threshold_used": None,
            "breadth_peak_step": None,
            "late_step": int(checkpoints["checkpoint_step"].max()),
            "selected_before_full_n320": False,
            "observed_checkpoints": observed,
            "pilot_path": str(pilot_path),
            "created_at_utc": (existing or {}).get("created_at_utc", datetime.now(timezone.utc).isoformat()),
            "git_commit": _git_commit(),
        }
        if output.exists() and not args.overwrite:
            old = dict(existing or {})
            old.pop("created_at_utc", None)
            comparable = dict(payload)
            comparable.pop("created_at_utc", None)
            if old != comparable:
                raise FileExistsError(f"immutable selection already exists: {output}")
        else:
            output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return output
    max_cov = float(eligible["exact_coverage"].max())
    peak_step = int(eligible[eligible["exact_coverage"] == max_cov]["checkpoint_step"].min())
    late_step = int(checkpoints["checkpoint_step"].max())
    payload = {
        "selection_status": "selected",
        "rule": "max exact_cov among format competent checkpoints; ties choose earliest step",
        "format_threshold_requested": 0.90,
        "format_threshold_used": threshold,
        "breadth_peak_step": peak_step,
        "late_step": late_step,
        "selected_before_full_n320": True,
        "observed_checkpoints": observed,
        "pilot_path": str(pilot_path),
        "created_at_utc": (existing or {}).get("created_at_utc", datetime.now(timezone.utc).isoformat()),
        "git_commit": _git_commit(),
    }
    if output.exists() and not args.overwrite:
        old = dict(existing or {})
        old.pop("created_at_utc", None)
        comparable = dict(payload)
        comparable.pop("created_at_utc", None)
        if old != comparable:
            raise FileExistsError(f"immutable selection already exists: {output}")
    else:
        output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "inventory":
        inventory(args)
    elif args.mode == "download":
        download(args)
    elif args.mode == "prepare-safe-set":
        prepare_safe_set(args)
    elif args.mode == "collect":
        print(collect(args))
    elif args.mode == "aggregate":
        print(aggregate(args))
    elif args.mode == "select":
        print(select(args))
    elif args.mode == "tf-profile":
        print(tf_profile(args))
    elif args.mode == "staircase-prepare":
        print(staircase_prepare(args))
    elif args.mode == "staircase-run":
        print(staircase_run(args))
    elif args.mode == "staircase-aggregate":
        print(staircase_aggregate(args))
    elif args.mode == "summarize":
        print(summarize(args))


if __name__ == "__main__":
    main()
