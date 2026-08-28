#!/usr/bin/env python3
"""Prepare and launch the protocol-complete M13 Base/SFT -> native GRPO pair.

The original Todo snippet only merged a LoRA adapter.  This wrapper performs
the merge atomically, records an immutable run contract, and launches the
repository's native VERL GRPO trainer with identical data and optimization
settings for Base and merged-SFT initializations.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import REPO_ROOT, RLVR_DATA_ROOT  # noqa: E402


PROTOCOL_VERSION = "sft_grpo_continuation-native-grpo-continuation-v1"
DEFAULT_LEDGER = RLVR_DATA_ROOT / "outputs" / "grpo_sft" / "grpo_line_sft_problems_v1.jsonl"
DEFAULT_TRAIN = RLVR_DATA_ROOT / "outputs" / "factorial_intervention_native" / "factorial_intervention_solver-rich_train_grpo_factorial_intervention_v1.parquet"
DEFAULT_VAL = RLVR_DATA_ROOT / "outputs" / "factorial_intervention_native" / "factorial_intervention_solver-rich_val_grpo_factorial_intervention_v1.parquet"
DEFAULT_BASE = REPO_ROOT / "model" / "qwen253B"
DEFAULT_ADAPTER = REPO_ROOT / "checkpoints" / "grpo_line_sft" / "k4_v1" / "step_02750"
DEFAULT_ROOT = REPO_ROOT / "checkpoints" / "grpo_continuation_v1"
DEFAULT_RAY_ROOT = Path("/tmp") / "sft_grpo_continuation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "merge", "launch"), default="dry-run")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--merged-sft", type=Path, default=None)
    parser.add_argument("--gpu-id", default="0,1", help="physical GPU IDs for Base,SFT in launch mode")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-response-length", type=int, default=256)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--rollout-memory", type=float, default=0.35)
    parser.add_argument("--rollout-max-batched-tokens", type=int, default=8192)
    parser.add_argument("--rollout-max-seqs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--env-name", default="tinyzero")
    parser.add_argument("--ray-tmpdir", type=Path, default=DEFAULT_RAY_ROOT)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs" / "sft_grpo_continuation_grpo")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.base_model, args.sft_adapter, args.train_data, args.val_data, args.ledger):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (args.sft_adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(args.sft_adapter / "adapter_model.safetensors")
    if args.total_steps < 1 or args.save_every < 1 or args.rollout_n < 2:
        raise ValueError("M13 requires positive steps/save cadence and rollout n > 1")
    if args.total_steps % args.save_every:
        raise ValueError("M13 total_steps must be divisible by save_every for exact checkpoints")
    gpu_ids = [item.strip() for item in str(args.gpu_id).split(",") if item.strip()]
    if len(gpu_ids) != 2:
        raise ValueError("M13 launch requires exactly two physical GPU IDs, one per arm")


def merge_sft_adapter_atomic(base_model_path: Path, adapter_path: Path, output_path: Path) -> dict[str, Any]:
    """Merge LoRA weights into a plain HF directory using a directory rename."""

    _validate_merge_paths(base_model_path, adapter_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
        merged.save_pretrained(str(temp_path), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True, use_fast=False)
        tokenizer.save_pretrained(str(temp_path))
        manifest = {
            "artifact": PROTOCOL_VERSION,
            "artifact_kind": "merged_sft_initialization",
            "status": "complete",
            "base_model": str(base_model_path),
            "sft_adapter": str(adapter_path),
            "adapter_config_sha256": sha256(adapter_path / "adapter_config.json"),
            "adapter_weights_sha256": sha256(adapter_path / "adapter_model.safetensors"),
            "output": str(output_path),
            "merge": "PeftModel.merge_and_unload; output directory atomically renamed from .partial",
        }
        (temp_path / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_path, output_path)
        return manifest
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def _validate_merge_paths(base_model_path: Path, adapter_path: Path, output_path: Path) -> None:
    if not base_model_path.is_dir():
        raise FileNotFoundError(base_model_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if output_path.parent.exists():
        collisions = list(output_path.parent.glob(f".{output_path.name}.partial-*"))
        if collisions:
            raise FileExistsError(f"stale atomic merge directories: {collisions}")


def merged_sft_path(args: argparse.Namespace) -> Path:
    return args.merged_sft or args.output_root / "merged_sft_k4_v1"


def _arm_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "base": args.output_root / f"base_grpo_seed{args.seed}",
        "sft": args.output_root / f"sft_grpo_seed{args.seed}",
    }


def _gpu_ids(args: argparse.Namespace) -> list[str]:
    return [item.strip() for item in str(args.gpu_id).split(",") if item.strip()]


def _ray_tmpdir(args: argparse.Namespace, arm: str) -> Path:
    """Use a short, arm-specific Ray root to stay under AF_UNIX's path limit."""
    code = {"base": "b", "sft": "s"}[arm]
    path = args.ray_tmpdir / f"{code}{int(args.seed)}"
    if len(str(path.resolve())) + 82 > 107:
        raise ValueError(f"Ray root is too long for AF_UNIX sockets: {path}")
    return path


def grpo_command(args: argparse.Namespace, arm: str, gpu_id: str) -> list[str]:
    model = args.base_model if arm == "base" else merged_sft_path(args)
    root = _arm_roots(args)[arm]
    experiment = f"sft_grpo_continuation-{arm}-grpo-seed-{args.seed}"
    return [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={args.train_data}",
        f"data.val_files={args.val_data}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size=500",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.optim.type=adamw",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.grad_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_memory}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.rollout_max_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.rollout_max_seqs}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"trainer.seed={args.seed}",
        "trainer.critic_warmup=0",
        "trainer.logger=[console]",
        "+trainer.val_before_train=False",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        f"trainer.save_freq={args.save_every}",
        "trainer.test_freq=-1",
        f"trainer.total_training_steps={args.total_steps}",
        "trainer.default_hdfs_dir=null",
        f"trainer.default_local_dir={root}",
        "trainer.project_name=TinyZero-M13",
        f"trainer.experiment_name={experiment}",
    ]


def contract(args: argparse.Namespace) -> dict[str, Any]:
    _validate_inputs(args)
    merged = merged_sft_path(args)
    roots = _arm_roots(args)
    if args.mode in {"dry-run", "launch"} and not merged.is_dir():
        # Dry-run may prepare the merge command, but launch must have a plain
        # HF checkpoint because VERL cannot consume a LoRA adapter path here.
        if args.mode == "launch":
            raise FileNotFoundError(f"merged SFT checkpoint required before launch: {merged}")
    for root in roots.values():
        if root.exists():
            raise FileExistsError(f"M13 output root already exists: {root}")
    gpu_ids = _gpu_ids(args)
    return {
        "artifact": PROTOCOL_VERSION,
        "status": "planned",
        "base_model": str(args.base_model),
        "sft_adapter": str(args.sft_adapter),
        "merged_sft": str(merged),
        "ledger": str(args.ledger),
        "ledger_sha256": sha256(args.ledger),
        "train_data": str(args.train_data),
        "train_data_sha256": sha256(args.train_data),
        "val_data": str(args.val_data),
        "val_data_sha256": sha256(args.val_data),
        "seed": args.seed,
        "arms": {
            arm: {
                "gpu_id": gpu_ids[index],
                "model": str(args.base_model if arm == "base" else merged),
                "output_root": str(roots[arm]),
                "command": grpo_command(args, arm, gpu_ids[index]),
            }
            for index, arm in enumerate(("base", "sft"))
        },
        "contract": {
            "algorithm": "grpo",
            "rollout_n": args.rollout_n,
            "total_training_steps": args.total_steps,
            "save_freq": args.save_every,
            "same_data": True,
            "same_seed": True,
            "response_only_native_reward": True,
        },
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    payload = contract(args)
    for arm, item in payload["arms"].items():
        print(f"{arm}: CUDA_VISIBLE_DEVICES={item['gpu_id']} {shlex.join(item['command'])}")
    return payload


def launch(args: argparse.Namespace) -> dict[str, Any]:
    payload = contract(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.ray_tmpdir.mkdir(parents=True, exist_ok=True)
    payload["status"] = "running"
    payload["started"] = True
    _atomic_json(args.output_root / f"sft_grpo_continuation_run_manifest_seed{args.seed}.json", payload)
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    try:
        for arm in ("base", "sft"):
            item = payload["arms"][arm]
            log_path = args.log_dir / f"sft_grpo_continuation_{arm}_seed{args.seed}.log"
            env = os.environ.copy()
            ray_tmpdir = _ray_tmpdir(args, arm)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(item["gpu_id"]),
                    "VLLM_ATTENTION_BACKEND": "XFORMERS",
                    "RAY_TMPDIR": str(ray_tmpdir),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            )
            ray_tmpdir.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                ["conda", "run", "-n", args.env_name, *item["command"]],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((arm, process, handle))
        statuses = {arm: process.wait() for arm, process, _ in processes}
        for _, _, handle in processes:
            handle.close()
        payload["statuses"] = statuses
        payload["status"] = "complete" if all(code == 0 for code in statuses.values()) else "failed"
        _atomic_json(args.output_root / f"sft_grpo_continuation_run_manifest_seed{args.seed}.json", payload)
        if payload["status"] != "complete":
            raise subprocess.CalledProcessError(1, "M13 paired launch", output=statuses)
        return payload
    except Exception:
        payload["status"] = "failed"
        _atomic_json(args.output_root / f"sft_grpo_continuation_run_manifest_seed{args.seed}.json", payload)
        raise


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "merge":
        _validate_inputs(args)
        result = merge_sft_adapter_atomic(args.base_model, args.sft_adapter, merged_sft_path(args))
    elif args.mode == "dry-run":
        result = dry_run(args)
    else:
        result = launch(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
