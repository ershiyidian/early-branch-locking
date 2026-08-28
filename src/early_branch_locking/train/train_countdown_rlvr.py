#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C-1: the three-arm Countdown PPO regularization control.

The arms fork from the exported actor and critic weights at step 100.  They
are intentionally *weight* forks: no optimizer, scheduler, dataloader, or
trainer state is resumed.  The single-GPU profile is accepted as a documented
bf16/offload feasibility compromise, so downstream results are interpreted as
a controlled fork rather than an exact replay of the original fp32 run.

Modes:
  plan       resolve the three arms and write provenance without launching;
  dry-run    run launcher preflight and Hydra composition for every arm;
  smoke      run one formal-shaped update (use --arms paper_recipe first);
  launch     run selected arms sequentially, preserving each arm's log.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (  # noqa: E402
    COUNTDOWN_ACTOR_DIR,
    DATASET_DIR,
    METRICS_DIR,
    REPO_ROOT,
)


LAUNCHER = REPO_ROOT / "analysis" / "rlvr" / "tools" / "run_countdown_rlvr_2gpu.sh"
LOG_ROOT = REPO_ROOT / "logs" / "countdown_rlvr_arms"
COUNTDOWN_ROOT = REPO_ROOT / "checkpoints" / "TinyZero" / "countdown-qwen2.5-3b"
COUNTDOWN_CRITIC_DIR = COUNTDOWN_ROOT / "critic"
SUPPORTED_ARMS = ("paper_recipe", "actor_kl_0p01", "op1_entropy_0p02", "retry_entropy_0p02")


@dataclass(frozen=True)
class Arm:
    name: str
    overrides: dict[str, str] = field(default_factory=dict)
    note: str = ""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "dry-run", "smoke", "launch"), default="plan")
    parser.add_argument("--arms", default=", ".join(SUPPORTED_ARMS).replace(", ", ","))
    parser.add_argument("--resume-step", type=int, default=100)
    parser.add_argument("--extra-steps", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=160)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--actor-dir", type=Path, default=COUNTDOWN_ACTOR_DIR)
    parser.add_argument("--critic-dir", type=Path, default=COUNTDOWN_CRITIC_DIR)
    parser.add_argument("--train-files", type=Path, default=DATASET_DIR / "train.parquet")
    parser.add_argument("--val-files", type=Path, default=DATASET_DIR / "test.parquet")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--env-name", default="tinyzero")
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="c1_v1")
    parser.add_argument("--run-prefix", default="countdown-qwen2.5-3b-c1")
    parser.add_argument("--ray-tmpdir", type=Path, default=Path(os.environ.get("RAY_TMPDIR", "/tmp/early-branch-locking-ray")))
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="offload FSDP parameters, gradients, and optimizer state to CPU (slower but lower VRAM)",
    )
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args(argv)


def _checkpoint(directory: Path, step: int, label: str) -> Path:
    path = directory / f"global_step_{step}"
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint is missing: {path}")
    return path


def resume_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        _checkpoint(args.actor_dir, args.resume_step, "actor"),
        _checkpoint(args.critic_dir, args.resume_step, "critic"),
    )


def build_arms(args: argparse.Namespace) -> list[Arm]:
    names = [item.strip() for item in args.arms.split(",") if item.strip()]
    unknown = sorted(set(names) - set(SUPPORTED_ARMS))
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; supported arms are {SUPPORTED_ARMS}")
    if not names:
        raise SystemExit("--arms selected no arms")

    common = {
        "actor_rollout_ref.actor.entropy_coeff": "0.001",
        "actor_rollout_ref.actor.op1_entropy_coeff": "0.0",
        "actor_rollout_ref.actor.retry_entropy_coeff": "0.0",
        "actor_rollout_ref.actor.use_kl_loss": "False",
        "actor_rollout_ref.actor.kl_loss_coef": "0.001",
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "algorithm.use_kl_in_reward": "True",
    }
    arms = {
        "paper_recipe": Arm(
            "paper_recipe",
            dict(common),
            "A0: paper PPO/GAE recipe with fixed reward-KL beta=0.001 and global entropy=0.001",
        ),
        "actor_kl_0p01": Arm(
            "actor_kl_0p01",
            {
                **common,
                "actor_rollout_ref.actor.use_kl_loss": "True",
                "actor_rollout_ref.actor.kl_loss_coef": "0.01",
            },
            "A1: A0 plus low-variance actor-KL coefficient=0.01; reward-KL remains enabled",
        ),
        "op1_entropy_0p02": Arm(
            "op1_entropy_0p02",
            {**common, "actor_rollout_ref.actor.op1_entropy_coeff": "0.02"},
            "A2: A0 plus entropy coefficient=0.02 only at the tokenizer-mapped first binary answer operator",
        ),
        "retry_entropy_0p02": Arm(
            "retry_entropy_0p02",
            {**common, "actor_rollout_ref.actor.retry_entropy_coeff": "0.02"},
            "olmo3_benchmark: A0 plus entropy coefficient=0.02 on the first tokens of a new trial after a falsified equation",
        ),
    }
    return [arms[name] for name in names]


def _single_gpu_profile(gpu_count: int = 1, cpu_offload: bool = True) -> dict[str, str]:
    """Build the bf16/offload profile for one or more local 4090s."""

    micro_batch = str(max(1, gpu_count))

    return {
        "N_GPUS_PER_NODE": str(gpu_count),
        "ROLLOUT_TP_SIZE": "1",
        "ACTOR_MODEL_DTYPE": "bfloat16",
        "CRITIC_MODEL_DTYPE": "bfloat16",
        "REF_MODEL_DTYPE": "bfloat16",
        "ACTOR_ENABLE_GRADIENT_CHECKPOINTING": "True",
        "CRITIC_ENABLE_GRADIENT_CHECKPOINTING": "True",
        "ACTOR_USE_REMOVE_PADDING": "True",
        "CRITIC_USE_REMOVE_PADDING": "True",
        "ACTOR_PARAM_OFFLOAD": str(cpu_offload),
        "ACTOR_GRAD_OFFLOAD": str(cpu_offload),
        "ACTOR_OPTIMIZER_OFFLOAD": str(cpu_offload),
        "CRITIC_PARAM_OFFLOAD": str(cpu_offload),
        "CRITIC_GRAD_OFFLOAD": str(cpu_offload),
        "CRITIC_OPTIMIZER_OFFLOAD": str(cpu_offload),
        "REF_PARAM_OFFLOAD": str(cpu_offload),
        "ACTOR_PPO_MICRO_BATCH_SIZE": micro_batch,
        "CRITIC_PPO_MICRO_BATCH_SIZE": micro_batch,
        "CRITIC_FORWARD_MICRO_BATCH_SIZE": micro_batch,
        "REF_LOG_PROB_MICRO_BATCH_SIZE": micro_batch,
        "ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE": micro_batch,
        "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "16384",
        "CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU": "16384",
        "CRITIC_FORWARD_MAX_TOKEN_LEN_PER_GPU": "16384",
        "REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "16384",
        "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "16384",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.20",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "4096",
        "ROLLOUT_MAX_NUM_SEQS": "64",
        "ROLLOUT_FREE_CACHE_ENGINE": "True",
        "SAVE_CRITIC": "False",
        "TEST_FREQ": "0",
        "VAL_BEFORE_TRAIN": "False",
        "VALIDATE_AFTER_TRAIN": "False",
        "RAY_TMPDIR": str(args.ray_tmpdir),
    }


def _profile_hash(profile: dict) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def arm_environment(
    args: argparse.Namespace,
    arm: Arm,
    actor_source: Path,
    critic_source: Path,
    mode: str,
) -> dict[str, str]:
    run_prefix = args.run_prefix if mode != "smoke" else f"{args.run_prefix}-smoke"
    run_name = f"{run_prefix}-{arm.name}"
    gpu_count = len([item for item in args.gpu_ids.split(",") if item.strip()])
    profile = _single_gpu_profile(gpu_count=max(1, gpu_count), cpu_offload=args.cpu_offload)
    profile["GPU_IDS"] = args.gpu_ids
    profile["ENV_NAME"] = args.env_name
    profile["BASE_MODEL"] = str(actor_source)
    profile["CRITIC_MODEL"] = str(critic_source)
    profile["TRAIN_FILES"] = str(args.train_files)
    profile["VAL_FILES"] = str(args.val_files)
    profile["TRAIN_BATCH_SIZE"] = str(args.train_batch_size)
    profile["VAL_BATCH_SIZE"] = "640"
    profile["MAX_RESPONSE_LENGTH"] = str(args.max_response_length)
    profile["MAX_PROMPT_LENGTH"] = "384"
    # Keep the rollout engine's generation cap aligned with the data cap.  The
    # launcher has a separate ROLLOUT_RESPONSE_LENGTH default, so omitting this
    # would silently generate 1024 tokens while the trainer is configured for
    # the requested shorter response length.
    profile["ROLLOUT_RESPONSE_LENGTH"] = str(args.max_response_length)
    profile["ROLLOUT_PROMPT_LENGTH"] = "384"
    profile["RUN_NAME"] = run_name
    profile["OUTPUT_DIR"] = str(REPO_ROOT / "checkpoints" / "TinyZero" / run_name)
    profile["LOG_DIR"] = str(LOG_ROOT)
    profile["LOG_FILE"] = str(LOG_ROOT / f"{run_name}.log")
    profile["RAY_TMPDIR"] = str(args.ray_tmpdir)
    profile["SAVE_FREQ"] = str(0 if mode == "smoke" else args.save_freq)
    profile["SEED"] = str(args.seed)
    profile["TOTAL_TRAINING_STEPS"] = str(2 if mode == "smoke" else args.extra_steps + 1)
    profile["ROLLOUT_GPU_MEMORY_UTILIZATION"] = str(args.rollout_gpu_memory_utilization)
    profile["ROLLOUT_MAX_NUM_BATCHED_TOKENS"] = str(args.rollout_max_num_batched_tokens)
    profile["EXTRA_OVERRIDES"] = " ".join(f"{key}={value}" for key, value in arm.overrides.items())
    return profile


def launcher_command(mode: str) -> list[str]:
    return ["bash", str(LAUNCHER), "--dry-run" if mode == "dry-run" else "--launch"]


def _manifest_path(args: argparse.Namespace) -> Path:
    return args.out_dir / f"solution_coverage1_arm_manifest_{args.tag}.json"


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    if not LAUNCHER.exists():
        raise FileNotFoundError(f"launcher is missing: {LAUNCHER}")
    actor_source, critic_source = resume_paths(args)
    arms = build_arms(args)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    arm_records = []
    for arm in arms:
        profile = arm_environment(args, arm, actor_source, critic_source, args.mode)
        arm_records.append(
            {
                "name": arm.name,
                "note": arm.note,
                "overrides": arm.overrides,
                "profile": profile,
                "profile_hash": _profile_hash(profile),
                "run_name": profile["RUN_NAME"],
                "output_dir": profile["OUTPUT_DIR"],
                "log_file": profile["LOG_FILE"],
                "status": "planned",
            }
        )

    manifest = {
        "experiment_id": "countdown_rlvr",
        "tag": args.tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "seed": args.seed,
        "requested_extra_steps": args.extra_steps,
        "resume_step": args.resume_step,
        "actor_source_checkpoint": str(actor_source),
        "critic_source_checkpoint": str(critic_source),
        "reference_source_checkpoint": str(actor_source),
        "state_provenance": (
            "forked from exported actor/critic weights; optimizer, scheduler, "
            "dataloader, and trainer state reset"
        ),
        "single_gpu_compromise": (
            "GPU0 only; bf16 actor/critic/reference parameters, gradient checkpointing, "
            "remove-padding, and CPU parameter/gradient/optimizer offload"
        ),
        "shared_launcher": str(LAUNCHER.relative_to(REPO_ROOT)),
        "ray_tmpdir": str(args.ray_tmpdir),
        "arms": arm_records,
        "downstream_evaluation": {
            "sampling": "150 problems x 320 samples, temperature=0.7, top_p=0.9, max_new_tokens=256, seed=42",
            "labeling": "explicit --model_label per C-1 checkpoint; never basename-only global_step_100",
            "coverage": "countdown/03_solution_coverage.py --save_per_problem",
            "op1_family": "countdown/public_grpo_replication.py --raw-glob",
            "reachability": "countdown/problem_sensitivity.py score/bound with explicit model paths",
        },
        "baseline_gate": {
            "step100_pass_at_1": 0.177687,
            "step100_exact_coverage": 0.229056,
            "step100_op1_family_coverage": 0.447778,
            "required": "A0 endpoint pass@1 rises, both breadth metrics fall, and exact<=0.176 or op1<=0.374",
            "failure_action": "stop before spending GPU budget on A1/A2 and report the fork as non-reproducing",
        },
    }
    manifest_path = _manifest_path(args)
    _write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "arms": [arm.name for arm in arms], "mode": args.mode}, sort_keys=True))
    if args.mode == "plan":
        for record in arm_records:
            print(json.dumps(record, sort_keys=True))
        return

    for index, record in enumerate(manifest["arms"]):
        environment = os.environ.copy()
        environment.update(record["profile"])
        record["status"] = "running"
        _write_manifest(manifest_path, manifest)
        print(f"[c1] {args.mode} arm={record['name']} profile_hash={record['profile_hash']}", flush=True)
        try:
            subprocess.run(
                launcher_command(args.mode),
                cwd=REPO_ROOT,
                env=environment,
                check=True,
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            _write_manifest(manifest_path, manifest)
            raise
        except KeyboardInterrupt as exc:
            record["status"] = "interrupted"
            record["error"] = repr(exc)
            _write_manifest(manifest_path, manifest)
            raise
        record["status"] = "complete"
        _write_manifest(manifest_path, manifest)
        print(f"[c1] complete arm={record['name']} ({index + 1}/{len(arm_records)})", flush=True)


if __name__ == "__main__":
    main()
