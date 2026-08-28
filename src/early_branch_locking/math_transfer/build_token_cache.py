#!/usr/bin/env python3
"""Cache teacher-forced completion token NLLs for semantic_boundary_analysis boundary resegmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from early_branch_locking.math_transfer.analyze_segment_suppression import MODEL_DIRS, build_sequence, prompt_for  # noqa

REQUIRED_CACHE_FIELDS = frozenset(("nll", "prompt_len", "completion_tokens", "truncated", "max_input_tokens"))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=ROOT / "data/rlvr/outputs/experiments/e4_semantic_boundary_v1/packet_blind.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/rlvr/outputs/experiments/e4_semantic_boundary_v1/e4_token_cache")
    parser.add_argument("--model-alias", choices=("math_base_7b", "math_simple_rl_7b"), required=True)
    parser.add_argument("--models-root", type=Path, default=ROOT / "model")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args(argv)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def cache_path(cache_dir: Path, item_id: str, model_alias: str) -> Path:
    return cache_dir / f"{item_id}__{model_alias}.npz"


def validate_cache(path: Path, max_input_tokens: int) -> tuple[bool, str]:
    """Return whether a cache can safely stand in for a new scoring result."""
    if not path.is_file():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = REQUIRED_CACHE_FIELDS.difference(payload.files)
            if missing:
                return False, f"missing_fields:{','.join(sorted(missing))}"
            nll = np.asarray(payload["nll"])
            completion_tokens = int(np.asarray(payload["completion_tokens"]).item())
            prompt_len = int(np.asarray(payload["prompt_len"]).item())
            recorded_limit = int(np.asarray(payload["max_input_tokens"]).item())
            truncated = np.asarray(payload["truncated"]).item()
    except (OSError, ValueError, KeyError, EOFError, TypeError, OverflowError, zipfile.BadZipFile) as exc:
        return False, f"unreadable:{type(exc).__name__}"
    if nll.ndim != 1 or not np.issubdtype(nll.dtype, np.number):
        return False, "nll_not_numeric_vector"
    if len(nll) != completion_tokens or completion_tokens <= 0:
        return False, "completion_length_mismatch"
    if prompt_len <= 0 or recorded_limit != max_input_tokens:
        return False, "metadata_mismatch"
    if not isinstance(truncated, (bool, np.bool_)):
        return False, "truncated_not_boolean"
    if not np.isfinite(nll).all():
        return False, "nonfinite_nll"
    return True, "valid"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_packet_items(packet: Path) -> list[dict]:
    items = [json.loads(line) for line in packet.open(encoding="utf-8") if line.strip() and int(json.loads(line).get("dup_index", 0)) == 0]
    item_ids = [str(item["item_id"]) for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("packet has duplicate unique-item IDs for dup_index=0")
    return items


def manifest_payload(args, model_path: Path, packet_sha256: str, n_unique: int, n_selected: int, written: int, existing: int, invalid: int, empty_ids: list[str], ready: int, status: str) -> dict:
    return {
        "artifact": "semantic_boundary_analysis token cache",
        "status": status,
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "packet_sha256": packet_sha256,
        "git_commit": git_commit(),
        "n_unique_packet_items": n_unique,
        "n_selected_items": n_selected,
        "n_cache_written_this_run": written,
        "n_valid_existing": existing,
        "n_invalid_existing_recomputed": invalid,
        "n_empty_completion": len(empty_ids),
        "empty_completion_item_ids": empty_ids,
        "n_valid_cache_after_run": ready,
        "max_input_tokens": args.max_input_tokens,
        "dtype": "float16_nll",
        "tokenizer_use_fast": False,
        "max_items": args.max_items,
        "formal_complete": status == "complete",
    }


def main(argv=None):
    args = parse(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    packet_items = load_packet_items(args.packet)
    items = packet_items[:args.max_items] if args.max_items else packet_items
    model_path = args.models_root / MODEL_DIRS[args.model_alias]
    packet_sha256 = sha(args.packet)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True).cuda().eval()
    written = existing = invalid = 0
    empty_ids: list[str] = []
    for index, item in enumerate(items, 1):
        out = cache_path(args.cache_dir, str(item["item_id"]), args.model_alias)
        valid, reason = validate_cache(out, args.max_input_tokens)
        if valid:
            existing += 1
            continue
        if reason != "missing":
            invalid += 1
        ids, prompt_len, completion_ids, truncated = build_sequence(tokenizer, prompt_for(item["question"]), item["response"], args.max_input_tokens)
        if not completion_ids:
            empty_ids.append(str(item["item_id"]))
            continue
        input_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids, use_cache=False).logits[0, :-1].float()
            target = input_ids[0, 1:]
            nll = (-logits.log_softmax(-1).gather(1, target[:, None]).squeeze(1)[prompt_len - 1 :]).cpu().numpy().astype(np.float16)
        temporary = out.with_suffix(".partial.npz")
        np.savez_compressed(temporary, nll=nll, prompt_len=prompt_len, completion_tokens=len(completion_ids), truncated=truncated, max_input_tokens=args.max_input_tokens)
        os.replace(temporary, out)
        valid, reason = validate_cache(out, args.max_input_tokens)
        if not valid:
            raise RuntimeError(f"atomic cache validation failed for {out}: {reason}")
        written += 1
        if index % 100 == 0:
            print(json.dumps({"processed": index, "written": written}, sort_keys=True), flush=True)
    ready = sum(validate_cache(cache_path(args.cache_dir, str(item["item_id"]), args.model_alias), args.max_input_tokens)[0] for item in items)
    status = "diagnostic" if args.max_items else ("complete" if ready == len(packet_items) and not empty_ids else "incomplete")
    manifest = args.cache_dir / f"manifest_{args.model_alias}.json"
    payload = manifest_payload(args, model_path, packet_sha256, len(packet_items), len(items), written, existing, invalid, empty_ids, ready, status)
    atomic_json(manifest, payload)
    if status == "incomplete":
        raise RuntimeError(f"semantic_boundary_analysis token cache incomplete: ready={ready}/{len(packet_items)}, empty_completion={len(empty_ids)}")
    print(json.dumps({"status": status, "written": written, "cache_dir": str(args.cache_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
