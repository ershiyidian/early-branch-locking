#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""entrance_entropy_training entrance-targeted entropy training protocol and calibration driver.

The script owns the entrance_entropy_training protocol boundary without duplicating the PPO trainer:
``plan`` freezes the three controlled-fork arms, ``smoke`` exercises the
semantic mask locally, ``calibrate`` derives the global-extra coefficient from
a response ledger, and ``launch`` delegates to the shared Countdown launcher.
Formal training is intentionally gated on a calibration manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR, METRICS_DIR, REPO_ROOT  # noqa: E402
try:
    from verl.trainer.ppo.regularization import entrance_entropy_mask  # noqa: E402
except ModuleNotFoundError:
    def entrance_entropy_mask(response_text, response_input_ids, tokenizer):
        """Dependency-free calibration mask used when veRL is not installed."""
        values = response_input_ids.detach().cpu().tolist() if isinstance(response_input_ids, torch.Tensor) else list(response_input_ids)
        mask = torch.zeros(len(values), dtype=torch.float32)
        if not getattr(tokenizer, "is_fast", False):
            return mask
        from early_branch_locking.core.entrance_detection import find_first_reasoning_entrance
        match = find_first_reasoning_entrance(str(response_text))
        if not match.found or match.char_start is None or match.char_end is None:
            return mask
        encoded = tokenizer(str(response_text), add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        if list(encoded["input_ids"]) != values:
            return mask
        for index, (left, right) in enumerate(encoded["offset_mapping"]):
            if right > match.char_start and left < match.char_end:
                mask[index] = 1.0
        return mask


PROTOCOL_VERSION = "s3-entrance-entropy-v1"
SEED = 1729
BASE_GLOBAL_ENTROPY = 0.001
DEFAULT_TARGETED_COEFF = 0.02
DEFAULT_EXTRA_STEPS = 225
DEFAULT_TAG = "entrance_entropy_training_v1"
LAUNCHER = None


@dataclass(frozen=True)
class Arm:
    name: str
    note: str
    extra_global_coeff: float = 0.0
    entrance_coeff: float = 0.0


class _CharacterTokenizer:
    """Small deterministic tokenizer used by the non-paper smoke gate."""

    is_fast = True

    def decode(self, ids, **_kwargs):
        return "".join(chr(token) for token in ids)

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "calibrate", "launch"), default="plan")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--resume-step", type=int, default=50)
    parser.add_argument("--extra-steps", type=int, default=DEFAULT_EXTRA_STEPS)
    parser.add_argument("--targeted-coeff", type=float, default=DEFAULT_TARGETED_COEFF)
    parser.add_argument("--global-extra-coeff", type=float, default=None)
    parser.add_argument("--calibration-input", type=Path, default=None)
    parser.add_argument("--calibration-output", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--arms", default="control,global_extra_entropy,entrance_extra_entropy")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--ray-tmpdir", default="/tmp/tinyzero-s3-ray")
    parser.add_argument("--env-name", default="tinyzero")
    parser.add_argument("--run-prefix", default="countdown-qwen2.5-3b-s3")
    parser.add_argument("--actor-dir", type=Path, default=COUNTDOWN_ACTOR_DIR)
    parser.add_argument("--critic-dir", type=Path, default=None)
    parser.add_argument("--train-files", type=Path, default=ROOT / "dataset" / "train.parquet")
    parser.add_argument("--val-files", type=Path, default=ROOT / "dataset" / "test.parquet")
    parser.add_argument("--train-batch-size", type=int, default=160)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--save-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint is missing: {path}")
    return path


def _source_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    actor = _checkpoint(args.actor_dir / f"global_step_{args.resume_step}", "actor")
    critic_root = args.critic_dir or args.actor_dir.parent / "critic"
    critic = _checkpoint(critic_root / f"global_step_{args.resume_step}", "critic")
    return actor, critic


def _arm_names(args: argparse.Namespace) -> list[str]:
    names = [item.strip() for item in args.arms.split(",") if item.strip()]
    expected = {"control", "global_extra_entropy", "entrance_extra_entropy"}
    unknown = sorted(set(names) - expected)
    if unknown or not names:
        raise ValueError(f"invalid entrance_entropy_training arms {unknown or names}; expected subset of {sorted(expected)}")
    return names


def _calibration_path(args: argparse.Namespace) -> Path:
    return args.calibration_output or METRICS_DIR / f"entrance_entropy_training_calibration_manifest_{args.tag}.json"


def _manifest_path(args: argparse.Namespace) -> Path:
    return METRICS_DIR / f"entrance_entropy_training_manifest_{args.tag}.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _response_text(row: dict[str, Any]) -> str:
    for key in ("response_text", "completion", "response", "continuation", "generated_text", "text"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _load_tokenizer(args: argparse.Namespace):
    path = args.tokenizer_path or args.actor_dir / f"global_step_{args.resume_step}"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("entrance_entropy_training calibration requires a fast tokenizer for offset mapping")
    return tokenizer, path


def _mask_stats(rows: Iterable[dict[str, Any]], tokenizer) -> tuple[list[dict[str, Any]], dict[str, float]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        response = _response_text(row)
        ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = torch.tensor(ids, dtype=torch.long)
        mask = entrance_entropy_mask(response, response_ids, tokenizer)
        response_tokens = len(ids)
        mask_tokens = int(mask.sum().item())
        target_entropy = row.get("mean_entropy_on_entrance", row.get("entrance_entropy"))
        global_entropy = row.get("mean_entropy_elsewhere", row.get("global_entropy", row.get("entropy")))
        records.append({
            "row_index": index,
            "problem_id": row.get("problem_id"),
            "response_tokens": response_tokens,
            "entrance_mask_tokens": mask_tokens,
            "entrance_detected": bool(mask_tokens),
            "target_entropy": float(target_entropy) if target_entropy is not None else None,
            "global_entropy": float(global_entropy) if global_entropy is not None else None,
        })
    if not records:
        raise ValueError("calibration input contains no response rows")
    response_counts = np.asarray([r["response_tokens"] for r in records], dtype=float)
    mask_counts = np.asarray([r["entrance_mask_tokens"] for r in records], dtype=float)
    target_entropy = np.asarray([r["target_entropy"] for r in records if r["target_entropy"] is not None], dtype=float)
    global_entropy = np.asarray([r["global_entropy"] for r in records if r["global_entropy"] is not None], dtype=float)
    summary = {
        "n_rows": float(len(records)),
        "parse_rate": float(np.mean(mask_counts > 0)),
        "mean_response_tokens": float(response_counts.mean()),
        "mean_entrance_mask_tokens": float(mask_counts.mean()),
        "median_entrance_mask_tokens": float(np.median(mask_counts)),
        "p95_entrance_mask_tokens": float(np.quantile(mask_counts, 0.95)),
        "target_token_fraction": float(np.mean(mask_counts / np.maximum(response_counts, 1.0))),
        "target_entropy": float(target_entropy.mean()) if len(target_entropy) else 1.0,
        "global_entropy": float(global_entropy.mean()) if len(global_entropy) else 1.0,
        "entropy_source": "telemetry" if len(target_entropy) and len(global_entropy) else "unit_fallback",
    }
    return records, summary


def calibrate(args: argparse.Namespace) -> Path:
    if args.calibration_input is None:
        raise ValueError("--mode calibrate requires --calibration-input JSONL")
    if not args.calibration_input.is_file():
        raise FileNotFoundError(args.calibration_input)
    tokenizer, tokenizer_path = _load_tokenizer(args)
    rows = _load_rows(args.calibration_input)
    records, summary = _mask_stats(rows, tokenizer)
    if summary["parse_rate"] < 0.8:
        raise RuntimeError(
            f"entrance parse rate {summary['parse_rate']:.3f} is below the entrance_entropy_training gate 0.8"
        )
    global_extra = args.global_extra_coeff
    if global_extra is None:
        global_extra = args.targeted_coeff * summary["target_token_fraction"] * (
            summary["target_entropy"] / max(summary["global_entropy"], 1e-12)
        )
    payload = {
        "experiment_id": "entrance_entropy_training",
        "protocol_version": PROTOCOL_VERSION,
        "tag": args.tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "source": {
            "calibration_input": str(args.calibration_input),
            "calibration_input_sha256": _sha256(args.calibration_input),
            "tokenizer_path": str(tokenizer_path),
        },
        "fixed_rule": "global_extra = targeted_coeff * target_mask_token_fraction * target_entropy / global_entropy",
        "targeted_coeff": float(args.targeted_coeff),
        "global_extra_coeff": float(global_extra),
        "base_global_entropy_coeff": BASE_GLOBAL_ENTROPY,
        "summary": summary,
        "row_audit": records,
        "paper_status": "calibration_only_non_paper",
    }
    output = _calibration_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"calibration": str(output), "parse_rate": summary["parse_rate"], "global_extra_coeff": global_extra}, sort_keys=True))
    return output


def smoke(args: argparse.Namespace) -> Path:
    tokenizer = _CharacterTokenizer()
    fixtures = [
        "<think>Let me try: 88 - 87 = 1.</think><answer>88 - 87</answer>",
        "<think>Try 83 + 87</think><answer>83 + 87</answer>",
        "<think>Numbers are [83, 87, 88]</think><answer>NO_SOLUTION</answer>",
    ]
    rows = []
    for text in fixtures:
        ids = torch.tensor([ord(character) for character in text], dtype=torch.long)
        mask = entrance_entropy_mask(text, ids, tokenizer)
        rows.append({"text": text, "mask_tokens": int(mask.sum().item()), "parse": bool(mask.sum().item())})
    if sum(row["parse"] for row in rows) != 2:
        raise AssertionError(f"unexpected entrance_entropy_training smoke mask results: {rows}")
    output = METRICS_DIR / f"entrance_entropy_training_smoke_manifest_{args.tag}.json"
    output.write_text(json.dumps({
        "experiment_id": "entrance_entropy_training", "protocol_version": PROTOCOL_VERSION,
        "paper_status": "SMOKE_NON_PAPER", "rows": rows,
        "expected": "operand-plus-operator span only; prompt-free response detector",
        "git_commit": _git_commit(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"smoke": str(output), "rows": rows}, sort_keys=True))
    return output


def _arm_profiles(args: argparse.Namespace, calibration: dict[str, Any] | None) -> list[dict[str, Any]]:
    actor, critic = _source_paths(args)
    global_extra = float((calibration or {}).get("global_extra_coeff", 0.0))
    targeted = float((calibration or {}).get("targeted_coeff", args.targeted_coeff))
    arms = {
        "control": Arm("control", "paper recipe; no additional entropy"),
        "global_extra_entropy": Arm("global_extra_entropy", "matched global extra entropy", extra_global_coeff=global_extra),
        "entrance_extra_entropy": Arm("entrance_extra_entropy", "matched entrance-span extra entropy", entrance_coeff=targeted),
    }
    records = []
    for name in _arm_names(args):
        arm = arms[name]
        run_name = f"{args.run_prefix}-{args.tag}-{name}"
        environment = {
            "ENV_NAME": args.env_name,
            "GPU_IDS": args.gpu_ids,
            "N_GPUS_PER_NODE": str(len([x for x in args.gpu_ids.split(",") if x.strip()])),
            "RAY_TMPDIR": args.ray_tmpdir,
            "ACTOR_MODEL_DTYPE": "bfloat16",
            "CRITIC_MODEL_DTYPE": "bfloat16",
            "REF_MODEL_DTYPE": "bfloat16",
            "ACTOR_ENABLE_GRADIENT_CHECKPOINTING": "True",
            "CRITIC_ENABLE_GRADIENT_CHECKPOINTING": "True",
            "ACTOR_USE_REMOVE_PADDING": "True",
            "CRITIC_USE_REMOVE_PADDING": "True",
            "ACTOR_PARAM_OFFLOAD": str(args.cpu_offload),
            "ACTOR_GRAD_OFFLOAD": str(args.cpu_offload),
            "ACTOR_OPTIMIZER_OFFLOAD": str(args.cpu_offload),
            "CRITIC_PARAM_OFFLOAD": str(args.cpu_offload),
            "CRITIC_GRAD_OFFLOAD": str(args.cpu_offload),
            "CRITIC_OPTIMIZER_OFFLOAD": str(args.cpu_offload),
            "REF_PARAM_OFFLOAD": str(args.cpu_offload),
            # The FSDP workers divide these values by the world size during
            # initialization. Two is therefore the smallest valid value for
            # a two-GPU run while preserving one sample per rank.
            "ACTOR_PPO_MICRO_BATCH_SIZE": "2",
            "CRITIC_PPO_MICRO_BATCH_SIZE": "2",
            "CRITIC_FORWARD_MICRO_BATCH_SIZE": "2",
            "REF_LOG_PROB_MICRO_BATCH_SIZE": "2",
            "ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE": "2",
            "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "16384",
            "CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU": "16384",
            "CRITIC_FORWARD_MAX_TOKEN_LEN_PER_GPU": "16384",
            "REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "16384",
            "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "16384",
            "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.20",
            "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "4096",
            "ROLLOUT_MAX_NUM_SEQS": "64",
            "ROLLOUT_FREE_CACHE_ENGINE": "True",
            "BASE_MODEL": str(actor),
            "CRITIC_MODEL": str(critic),
            "TRAIN_FILES": str(args.train_files),
            "VAL_FILES": str(args.val_files),
            "TRAIN_BATCH_SIZE": str(args.train_batch_size),
            "MAX_RESPONSE_LENGTH": str(args.max_response_length),
            "ROLLOUT_RESPONSE_LENGTH": str(args.max_response_length),
            "RUN_NAME": run_name,
            "OUTPUT_DIR": str(ROOT / "checkpoints" / "TinyZero" / run_name),
            "LOG_DIR": str(ROOT / "logs" / "s3"),
            "LOG_FILE": str(ROOT / "logs" / "s3" / f"{run_name}.log"),
            "TOTAL_TRAINING_STEPS": str(args.extra_steps + 1),
            "SAVE_FREQ": str(args.save_freq),
            "SEED": str(args.seed),
            "ACTOR_ENTROPY_COEFF": str(BASE_GLOBAL_ENTROPY + arm.extra_global_coeff),
            "ACTOR_ENTRANCE_ENTROPY_COEFF": str(arm.entrance_coeff),
            "ACTOR_OP1_ENTROPY_COEFF": "0.0",
            "ACTOR_USE_KL_LOSS": "False",
            "ALGORITHM_USE_KL_IN_REWARD": "True",
            "VALIDATE_AFTER_TRAIN": "False",
            "SAVE_CRITIC": "True",
        }
        records.append({"name": name, "note": arm.note, "environment": environment})
    return records


def plan(args: argparse.Namespace) -> Path:
    actor, critic = _source_paths(args)
    calibration_path = _calibration_path(args)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None
    payload = {
        "experiment_id": "entrance_entropy_training",
        "protocol_version": PROTOCOL_VERSION,
        "tag": args.tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "fork": {"resume_step": args.resume_step, "actor": str(actor), "critic": str(critic), "optimizer_state": "reset"},
        "fixed_training": {"extra_steps": args.extra_steps, "save_steps": [25, 50, 100, 150, 225], "base_global_entropy_coeff": BASE_GLOBAL_ENTROPY},
        "calibration_manifest": str(calibration_path) if calibration else None,
        "arms": _arm_profiles(args, calibration),
        "launcher": None,
        "paper_status": "planned_controlled_fork",
    }
    output = _manifest_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "arms": [row["name"] for row in payload["arms"]]}, sort_keys=True))
    return output


def launch(args: argparse.Namespace) -> None:
    raise RuntimeError("Training launch requires the host veRL/TinyZero launcher; use --mode plan or --mode smoke in this repository.")
    calibration_path = _calibration_path(args)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"formal entrance_entropy_training launch requires calibration manifest: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("paper_status") != "calibration_only_non_paper":
        raise ValueError("calibration manifest has an unexpected status")
    manifest_path = plan(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for arm in manifest["arms"]:
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in arm["environment"].items()})
        environment["ALLOW_EXISTING_OUTPUT"] = "0"
        log_path = Path(arm["environment"]["LOG_FILE"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            subprocess.run(["bash", str(LAUNCHER), "--launch"], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "calibrate":
        calibrate(args)
    elif args.mode == "plan":
        plan(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
