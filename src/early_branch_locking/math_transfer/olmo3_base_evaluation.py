#!/usr/bin/env python3
"""Resumable OLMo-3 base evaluation with per-sequence stopping.

The external Limit-of-RLVR evaluator uses a batch-global stopping criterion.
That is unsafe for the OLMo base because one long continuation keeps every
sequence in the batch alive.  This runner keeps the fixed plain ``cot``
prompt and sampling contract, but tracks stop-token suffixes per sequence and
forces EOS only for rows that have reached a stop marker.

Generation output is deliberately kept separate from scoring.  Each problem
is written as one JSON object with its 64 completions, source example, and
generation telemetry.  A completed shard is renamed from ``.partial`` only
after all requested problems have been flushed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "dataset" / "math_eval"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "rlvr" / "outputs" / "experiments" / "olmo3_full_trajectory_v2"
MODEL_DEFAULT = ROOT / "model" / "Olmo-3-1025-7B"
BENCHMARK_COUNTS = {
    "gsm8k": 500,
    "math500": 500,
    "minerva_math": 272,
    "olympiadbench": 500,
    "amc23": 40,
    "aime24": 30,
}
STOP_STRINGS = ("</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:")
MODEL_ALIAS = "math_olmo3_base_7b"
EXPERIMENT_ID = "olmo3_benchmark-olmo3-base-full-trajectory-v2"

# Reject executable/numeric non-finite artifacts without rejecting ordinary
# mathematical prose such as "negative infinity" in a derivation.
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
HEALTH_STANDALONE = {"nan", "inf", "infinity", "+nan", "+inf", "+infinity", "-nan", "-inf", "-infinity"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate", "finalize"), default="generate")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_COUNTS))
    parser.add_argument("--model-path", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw", type=Path, help="Completed or partial JSONL for --mode validate/finalize")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--n-sampling", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected a JSON list in {path}")
        rows = payload
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("idx", index)
        normalized.append(item)
    return sorted(normalized, key=lambda item: int(item["idx"]))


def dataset_path(data_root: Path, benchmark: str) -> Path:
    for name in ("test.jsonl", "test.json"):
        candidate = data_root / benchmark / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No test file for benchmark={benchmark} under {data_root}")


def question_for(row: dict, benchmark: str) -> str:
    if benchmark in {"gsm8k", "olympiadbench"}:
        value = row.get("question")
    else:
        value = row.get("problem", row.get("question"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing question for {benchmark} idx={row.get('idx')}")
    return value.strip()


def boxed_answer(text: str) -> str | None:
    marker = "\\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    open_index = text.find("{", start)
    if open_index < 0:
        return None
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index].strip()
    return None


def ground_truth_for(row: dict, benchmark: str) -> str:
    if benchmark == "gsm8k":
        return str(row["answer"].split("####", 1)[-1]).strip()
    if benchmark in {"math500", "minerva_math"}:
        answer = row.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        solution = str(row.get("solution", ""))
        return boxed_answer(solution) or solution.strip()
    if benchmark == "olympiadbench":
        answer = row.get("final_answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return str(answer).strip().strip("$")
    return str(row["answer"]).strip()


def build_prompt(question: str) -> str:
    # This is the external evaluator's cot template after construct_prompt's
    # `.strip(" ")`: the final prompt has no trailing space after Answer:.
    return f"Question: {question}\nAnswer:"


def load_problem_slice(args: argparse.Namespace) -> tuple[list[dict], Path, int, int]:
    if not args.benchmark:
        raise ValueError("--benchmark is required for --mode generate")
    source = dataset_path(args.data_root, args.benchmark)
    rows = read_rows(source)
    expected = BENCHMARK_COUNTS[args.benchmark]
    if len(rows) < expected:
        raise ValueError(f"{args.benchmark} has {len(rows)} rows, expected at least {expected}")
    start = max(0, args.start)
    end = expected if args.end < 0 else min(args.end, expected)
    if end <= start:
        raise ValueError(f"Invalid slice [{start}, {end})")
    return rows[start:end], source, start, end


def shard_paths(args: argparse.Namespace, start: int, end: int) -> tuple[Path, Path, Path]:
    shard_dir = args.output_root / args.benchmark
    stem = f"records_s{start}_e{end}"
    return shard_dir / f"{stem}.jsonl", shard_dir / f"{stem}.jsonl.partial", shard_dir / f"{stem}.manifest.json"


def progress_path_for(partial_path: Path) -> Path:
    """Return the non-aggregate live heartbeat sidecar for a shard."""
    return partial_path.with_suffix(partial_path.suffix + ".progress.json")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_completed_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    indices: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line in {path} at {line_number}")
            row = json.loads(line)
            if "idx" not in row or not isinstance(row.get("code"), list):
                raise ValueError(f"Malformed completed row in {path} at {line_number}")
            if int(row["idx"]) in indices:
                raise ValueError(f"Duplicate idx={row['idx']} in {path}")
            indices.add(int(row["idx"]))
    return indices


def suffix_match(values: list[int], stop_sequences: list[tuple[int, ...]]) -> str | None:
    for label, sequence in stop_sequences:
        if sequence and len(values) >= len(sequence) and tuple(values[-len(sequence) :]) == sequence:
            return label
    return None


def truncate_stops(text: str) -> tuple[str, str | None]:
    hits = [(text.find(marker), marker) for marker in STOP_STRINGS if marker in text]
    if not hits:
        return text, None
    index, marker = min(hits, key=lambda item: item[0])
    return text[:index], marker


def has_health_failure(completion: str) -> bool:
    return completion.strip().casefold() in HEALTH_STANDALONE or HEALTH_RE.search(completion) is not None


def generation_status(
    token_ids: list[int],
    eos_id: int | None,
    stop_sequences: list[tuple[str, tuple[int, ...]]],
    max_new_tokens: int,
) -> str:
    stop = suffix_match(token_ids, stop_sequences)
    if stop is not None:
        return f"stop:{stop}"
    if eos_id is not None and eos_id in token_ids:
        return "eos"
    if len(token_ids) >= max_new_tokens:
        return "length_cap"
    return "unknown"


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[dict]:
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    stop_sequences: list[tuple[str, tuple[int, ...]]] = []
    for marker in STOP_STRINGS:
        ids = tuple(tokenizer.encode(marker, add_special_tokens=False))
        if ids:
            stop_sequences.append((marker, ids))

    class PerSequenceStop(LogitsProcessor):
        def __init__(self):
            self.finished = [False] * len(prompts)
            self.stop_reasons: list[str | None] = [None] * len(prompts)
            self.max_stop_length = max((len(sequence) for _, sequence in stop_sequences), default=1)

        def __call__(self, input_ids, scores):
            for index in range(len(prompts)):
                # Copy only the small suffix needed for stop matching. Copying
                # the whole growing continuation to CPU here causes an O(n^2)
                # synchronization cost on 16k-token base-model traces.
                generated_length = input_ids.shape[1] - prompt_width
                if generated_length <= 0:
                    continue
                tail_start = max(prompt_width, input_ids.shape[1] - self.max_stop_length)
                continuation = input_ids[index, tail_start:].tolist()
                reason = suffix_match(continuation, stop_sequences)
                if reason is not None and not self.finished[index]:
                    self.finished[index] = True
                    self.stop_reasons[index] = f"stop:{reason}"
                if self.finished[index]:
                    scores[index, :] = -torch.inf
                    scores[index, tokenizer.eos_token_id] = 0
            return scores

    class MinNewTokens(LogitsProcessor):
        def __init__(self, min_new_tokens: int):
            self.min_new_tokens = min_new_tokens

        def __call__(self, input_ids, scores):
            if input_ids.shape[1] - prompt_width < self.min_new_tokens:
                scores[:, tokenizer.eos_token_id] = -torch.inf
            return scores

    stopper = PerSequenceStop()
    processors = [stopper]
    if args.min_new_tokens:
        processors.insert(0, MinNewTokens(args.min_new_tokens))
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList(processors),
        )
    rows = []
    for index, token_row in enumerate(generated[:, prompt_width:]):
        ids = token_row.detach().cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        text, text_stop = truncate_stops(text)
        reason = stopper.stop_reasons[index] or text_stop
        if reason is None:
            reason = generation_status(ids, tokenizer.eos_token_id, stop_sequences, args.max_new_tokens)
        if reason == "unknown":
            reason = "eos" if tokenizer.eos_token_id in ids else "length_cap"
        completion = text.strip()
        if not completion:
            raise RuntimeError("olmo3_benchmark empty completion after decoding")
        if has_health_failure(completion):
            raise RuntimeError("olmo3_benchmark standalone numeric NaN/Inf completion after decoding")
        completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        if completion_tokens <= 0:
            raise RuntimeError("olmo3_benchmark zero-token completion after decoding")
        rows.append(
            {
                "completion": completion,
                "completion_tokens": completion_tokens,
                "finish_reason": reason,
            }
        )
    return rows


def manifest_base(args: argparse.Namespace, source: Path, start: int, end: int, raw_path: Path, partial_path: Path) -> dict:
    return {
        "git_commit": git_commit(),
        "status": "running",
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL_ALIAS,
        "model_path": str(args.model_path),
        "benchmark": args.benchmark,
        "problem_slice": {"start": start, "end": end, "n_problems": end - start},
        "requested": {
            "n_sampling": args.n_sampling,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "seed": args.seed,
            "prompt_type": "cot",
            "apply_chat_template": False,
        },
        "backend": "transformers_hf_per_sequence_stop",
        "cuda_visible_devices": args.cuda_visible_devices,
        "batch_size": args.batch_size,
        "dataset_path": str(source),
        "dataset_sha256": sha256_file(source),
        "raw_path": str(raw_path),
        "partial_path": str(partial_path),
        "stop_strings": list(STOP_STRINGS),
        "notes": [
            "Per-sequence stop tracking removes the external evaluator's batch-global stopping hazard.",
            "Raw completion generation is separate from official math scoring.",
        ],
    }


def run_generate(args: argparse.Namespace) -> None:
    import random
    import numpy as np
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    problems, source, start, end = load_problem_slice(args)
    raw_path, partial_path, manifest_path = shard_paths(args, start, end)
    progress_path = progress_path_for(partial_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"Completed shard exists: {raw_path}; choose a new tag or pass --overwrite")
    if args.overwrite:
        for path in (raw_path, partial_path, manifest_path, progress_path):
            if path.exists():
                path.unlink()

    completed = load_completed_indices(partial_path) if args.resume else set()
    manifest = manifest_base(args, source, start, end, raw_path, partial_path)
    manifest["resumed_completed_indices"] = sorted(completed)
    atomic_json(manifest_path, manifest)

    model, tokenizer = load_model(args)
    print(
        json.dumps(
            {
                "event": "olmo3_generation_started",
                "benchmark": args.benchmark,
                "start": start,
                "end": end,
                "completed": len(completed),
                "gpu": args.cuda_visible_devices,
                "batch_size": args.batch_size,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    total_started = time.time()
    mode = "a" if partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8") as handle:
        for offset, problem in enumerate(problems, start=start):
            idx = int(problem["idx"])
            if idx in completed:
                continue
            question = question_for(problem, args.benchmark)
            prompt = build_prompt(question)
            completions: list[dict] = []
            problem_started = time.time()
            for sample_start in range(0, args.n_sampling, args.batch_size):
                sample_end = min(args.n_sampling, sample_start + args.batch_size)
                completions.extend(generate_batch(model, tokenizer, [prompt] * (sample_end - sample_start), args))
                token_values = [item["completion_tokens"] for item in completions]
                reason_counts = Counter(item["finish_reason"] for item in completions)
                heartbeat = {
                    "event": "olmo3_problem_batch_complete",
                    "benchmark": args.benchmark,
                    "idx": idx,
                    "problem_position": offset,
                    "sample_range": [sample_start, sample_end - 1],
                    "completed_samples_for_problem": len(completions),
                    "requested_samples_for_problem": args.n_sampling,
                    "min_completion_tokens": min(token_values),
                    "mean_completion_tokens": sum(token_values) / len(token_values),
                    "max_completion_tokens": max(token_values),
                    "finish_reason_counts": dict(sorted(reason_counts.items())),
                    "elapsed_problem_s": round(time.time() - problem_started, 2),
                    "elapsed_total_s": round(time.time() - total_started, 2),
                }
                atomic_json(progress_path, heartbeat)
                print(json.dumps(heartbeat, ensure_ascii=False), flush=True)
            record = {
                "idx": idx,
                "benchmark": args.benchmark,
                "question": question,
                "gt": ground_truth_for(problem, args.benchmark),
                "source_example": problem,
                "prompt": prompt,
                "code": [item["completion"] for item in completions],
                "completion_tokens": [item["completion_tokens"] for item in completions],
                "finish_reason": [item["finish_reason"] for item in completions],
                "n_sampling": len(completions),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            progress_path.unlink(missing_ok=True)
            completed.add(idx)
            token_values = record["completion_tokens"]
            print(
                json.dumps(
                    {
                        "event": "olmo3_problem_complete",
                        "benchmark": args.benchmark,
                        "idx": idx,
                        "problem_position": offset,
                        "completed": len(completed),
                        "requested_problems": end - start,
                        "samples": len(completions),
                        "mean_completion_tokens": sum(token_values) / len(token_values),
                        "max_completion_tokens": max(token_values),
                        "elapsed_problem_s": round(time.time() - problem_started, 2),
                        "elapsed_total_s": round(time.time() - total_started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if len(completed) != end - start:
        raise RuntimeError(f"Completed {len(completed)} rows, expected {end - start}")
    os.replace(partial_path, raw_path)
    manifest.update(
        {
            "status": "complete",
            "completed_problem_count": len(completed),
            "raw_sha256": sha256_file(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "completed_at_epoch": time.time(),
        }
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps({"event": "olmo3_generation_complete", "raw": str(raw_path), "manifest": str(manifest_path)}), flush=True)


def validate_raw(path: Path, expected_problems: int | None = None, expected_samples: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line at {line_number} in {path}")
            row = json.loads(line)
            rows.append(row)
    indices = [int(row["idx"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate problem idx in {path}")
    sample_counts = []
    empty_completion_count = 0
    zero_token_completion_count = 0
    health_failure_count = 0
    for row_number, row in enumerate(rows, start=1):
        completions = row.get("code")
        token_counts = row.get("completion_tokens")
        finish_reasons = row.get("finish_reason")
        if not isinstance(completions, list) or not isinstance(token_counts, list) or not isinstance(finish_reasons, list):
            raise ValueError(f"Missing completion telemetry in {path} row {row_number}")
        if len(completions) != len(token_counts) or len(completions) != len(finish_reasons):
            raise ValueError(f"Mismatched completion telemetry lengths in {path} row {row_number}")
        sample_counts.append(len(completions))
        for sample_number, (completion, token_count, reason) in enumerate(zip(completions, token_counts, finish_reasons)):
            if not isinstance(completion, str) or not completion.strip():
                empty_completion_count += 1
            if not isinstance(token_count, int) or token_count <= 0:
                zero_token_completion_count += 1
            if isinstance(completion, str) and has_health_failure(completion):
                health_failure_count += 1
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"Missing finish reason in {path} row {row_number} sample {sample_number}")
    if empty_completion_count or zero_token_completion_count or health_failure_count:
        raise ValueError(
            f"Unhealthy completions in {path}: empty={empty_completion_count} zero_token={zero_token_completion_count} health={health_failure_count}"
        )
    if expected_samples is not None and any(value != expected_samples for value in sample_counts):
        raise ValueError(f"Sample counts {sorted(set(sample_counts))} != {expected_samples}")
    result = {
        "status": "valid",
        "path": str(path),
        "problem_count": len(rows),
        "sample_counts": sorted(set(sample_counts)),
        "raw_rows": sum(sample_counts),
        "empty_completion_count": empty_completion_count,
        "zero_token_completion_count": zero_token_completion_count,
        "health_failure_count": health_failure_count,
        "finish_reason_counts": {},
        "sha256": sha256_file(path),
    }
    for row in rows:
        for reason in row.get("finish_reason", []):
            result["finish_reason_counts"][reason] = result["finish_reason_counts"].get(reason, 0) + 1
    if expected_problems is not None and len(rows) != expected_problems:
        raise ValueError(f"Problem count {len(rows)} != {expected_problems}")
    return result


def finalize_completed_manifest(
    raw_path: Path,
    manifest_path: Path,
    expected_problems: int,
    expected_samples: int,
    partial_path: Path | None = None,
) -> dict:
    """Repair completion-time provenance after an older runner exits.

    The active v2 shards were launched before the experiment-id metadata fix,
    so this operation is deliberately separate from generation. It refuses
    to touch a still-growing partial file and validates the complete raw
    shard before changing its manifest.
    """
    validation = validate_raw(
        raw_path,
        expected_problems=expected_problems,
        expected_samples=expected_samples,
    )
    partial_path = partial_path or raw_path.with_suffix(raw_path.suffix + ".partial")
    if partial_path.is_file():
        raise RuntimeError(f"Refusing to finalize while partial shard exists: {partial_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected manifest object in {manifest_path}")
    previous_experiment_id = manifest.get("experiment_id")
    manifest.update(
        {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "completed_problem_count": validation["problem_count"],
            "raw_sha256": validation["sha256"],
            "raw_bytes": raw_path.stat().st_size,
            "completed_at_epoch": manifest.get("completed_at_epoch", time.time()),
            "finalized_by": "olmo3_olmo_base_eval.finalize_completed_manifest",
        }
    )
    if previous_experiment_id != EXPERIMENT_ID:
        manifest["provenance_repaired_from_experiment_id"] = previous_experiment_id
    atomic_json(manifest_path, manifest)
    return {
        "status": "complete",
        "raw": str(raw_path),
        "manifest": str(manifest_path),
        "validation": validation,
    }


def run_validate(args: argparse.Namespace) -> None:
    if args.raw is None:
        raise ValueError("--raw is required for --mode validate")
    result = validate_raw(args.raw, expected_samples=args.n_sampling)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_finalize(args: argparse.Namespace) -> None:
    if args.raw is None:
        raise ValueError("--raw is required for --mode finalize")
    if args.benchmark is None:
        raise ValueError("--benchmark is required for --mode finalize")
    manifest_path = args.raw.with_suffix(".manifest.json")
    result = finalize_completed_manifest(
        args.raw,
        manifest_path,
        expected_problems=BENCHMARK_COUNTS[args.benchmark],
        expected_samples=args.n_sampling,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "validate":
        run_validate(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
