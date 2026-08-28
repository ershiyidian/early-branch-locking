#!/usr/bin/env python3
"""Freeze, audit, and launch the M12 2x2 objective/data factorial.

The source factor is solver-rich versus fresh base-model on-policy
supervision.  MLE consumes the source completions; native VERL GRPO samples
its own completions, so its one-row-per-problem parquet keeps the source
factor as explicit provenance without changing prompt multiplicity.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking._repo import REPO_ROOT, RLVR_DATA_ROOT  # noqa: E402
from early_branch_locking.train import prepare_native_grpo_data as native  # noqa: E402


PROTOCOL_VERSION = "factorial_intervention-native-factorial-v1"
SEEDS = (1729, 1730)
SOURCE_KINDS = ("solver-rich", "on-policy")
OBJECTIVES = ("mle", "grpo")
DEFAULT_TAG = "factorial_intervention_v1"
DEFAULT_OUT_DIR = RLVR_DATA_ROOT / "outputs" / "factorial_intervention_native"
DEFAULT_LEDGER = RLVR_DATA_ROOT / "outputs" / "grpo_sft" / "grpo_line_sft_problems_v1.jsonl"
DEFAULT_SOLVER_SOURCE = (
    RLVR_DATA_ROOT / "outputs" / "grpo_sft" / "grpo_line_sft_supervision_k4_entrance-diverse_v1.jsonl"
)
DEFAULT_ON_POLICY_SOURCE = DEFAULT_OUT_DIR / "factorial_intervention_on_policy_supervision_factorial_intervention_v1.jsonl"
DEFAULT_BASE = REPO_ROOT / "model" / "qwen253B"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "factorial_intervention_factorial_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "factorial_intervention_factorial_v1"
# Ray appends a timestamped session directory and Unix socket names below
# this root. Keep the default root short enough for AF_UNIX's 107-byte limit.
DEFAULT_RAY_ROOT = Path("/tmp") / "factorial_intervention"
FORMAL_ROLLOUT_MEMORY = 0.35
FORMAL_ROLLOUT_MAX_BATCHED_TOKENS = 8192
FORMAL_ROLLOUT_MAX_SEQS = 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "audit", "dry-run", "launch"), default="audit")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--solver-source", type=Path, default=DEFAULT_SOLVER_SOURCE)
    parser.add_argument("--on-policy-source", type=Path, default=DEFAULT_ON_POLICY_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--ray-root", type=Path, default=DEFAULT_RAY_ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--env-name", default="tinyzero")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-response-length", type=int, default=256)
    parser.add_argument("--sft-epochs", type=float, default=2.0)
    parser.add_argument("--sft-batch-size", type=int, default=2)
    parser.add_argument("--sft-grad-accum", type=int, default=8)
    parser.add_argument("--sft-save-every", type=int, default=250)
    parser.add_argument("--grpo-save-every", type=int, default=250)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--rollout-memory", type=float, default=FORMAL_ROLLOUT_MEMORY)
    parser.add_argument("--rollout-max-batched-tokens", type=int, default=FORMAL_ROLLOUT_MAX_BATCHED_TOKENS)
    parser.add_argument("--rollout-max-seqs", type=int, default=FORMAL_ROLLOUT_MAX_SEQS)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--actor-model-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Actor parameter dtype for GRPO; BF16 keeps AdamW state within one 48-GiB GPU.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically update a launcher status file, including on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _gpu_ids(args: argparse.Namespace) -> list[str]:
    ids = [item.strip() for item in str(args.gpu_ids).split(",") if item.strip()]
    if len(ids) != 2:
        raise ValueError("M12 requires exactly two physical GPU IDs")
    return ids


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {"solver-rich": args.solver_source, "on-policy": args.on_policy_source}


def _mle_supervision_path(args: argparse.Namespace, source_kind: str) -> Path:
    return args.data_dir / f"factorial_intervention_{source_kind}_mle_supervision_{args.tag}.jsonl"


def _parquet_paths(args: argparse.Namespace, source_kind: str) -> dict[str, Path]:
    return {
        "mle_train": args.data_dir / f"factorial_intervention_{source_kind}_train_{args.tag}.parquet",
        "mle_val": args.data_dir / f"factorial_intervention_{source_kind}_val_{args.tag}.parquet",
        "grpo_train": args.data_dir / f"factorial_intervention_{source_kind}_train_grpo_{args.tag}.parquet",
        "grpo_val": args.data_dir / f"factorial_intervention_{source_kind}_val_grpo_{args.tag}.parquet",
    }


def _manifest_path(args: argparse.Namespace) -> Path:
    return args.data_dir / f"factorial_intervention_factorial_manifest_{args.tag}.json"


def _run_manifest_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_root / f"factorial_intervention_run_manifest_{args.tag}.json"


def _run_final_manifest_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_root / f"factorial_intervention_run_manifest_{args.tag}_final.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return native.read_jsonl(path)


def _build_train_supervision(
    args: argparse.Namespace,
    ledger: dict[str, dict[str, Any]],
    source_kind: str,
) -> dict[str, Any]:
    source_path = _source_paths(args)[source_kind]
    output = _mle_supervision_path(args, source_kind)
    source_rows = _read_jsonl(source_path)
    train_rows = [row for row in source_rows if ledger.get(str(row.get("problem_uid")), {}).get("split") == "train"]
    if not train_rows:
        raise ValueError(f"no train rows in {source_kind} source: {source_path}")
    native._validate_source_rows(train_rows, ledger, source_kind=source_kind)
    if output.exists():
        existing = _read_jsonl(output)
        _validate_train_supervision(existing, ledger, source_kind)
    else:
        native.write_mle_jsonl(output, train_rows, ledger, source_kind)
    existing = _read_jsonl(output)
    _validate_train_supervision(existing, ledger, source_kind)
    return {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_rows": len(source_rows),
        "train_source_rows": len(train_rows),
        "path": str(output),
        "sha256": _sha256(output),
        "rows": len(existing),
    }


def _validate_train_supervision(rows: list[dict[str, Any]], ledger: dict[str, dict[str, Any]], source_kind: str) -> None:
    if not rows:
        raise ValueError(f"empty MLE supervision for {source_kind}")
    for row in rows:
        uid = str(row.get("problem_uid", ""))
        if uid not in ledger or ledger[uid]["split"] != "train":
            raise ValueError(f"non-train problem in MLE supervision: {uid}")
        if str(row.get("split")) != "train":
            raise ValueError(f"non-train split in MLE supervision: {uid}")
        if str(row.get("source_kind")) != source_kind:
            raise ValueError(f"source-kind mismatch in MLE supervision: {uid}")
        if native.has_health_failure(str(row.get("completion", ""))):
            raise ValueError(f"unhealthy MLE completion: {uid}")


def _validate_parquet_factor(
    args: argparse.Namespace,
    ledger: dict[str, dict[str, Any]],
    source_kind: str,
) -> dict[str, Any]:
    paths = _parquet_paths(args, source_kind)
    reports = {
        "mle_train": native.audit_parquet(paths["mle_train"], ledger, "train", source_kind),
        "mle_val": native.audit_parquet(paths["mle_val"], ledger, "val", source_kind),
        "grpo_train": native.audit_parquet(paths["grpo_train"], ledger, "train", source_kind),
        "grpo_val": native.audit_parquet(paths["grpo_val"], ledger, "val", source_kind),
    }
    if reports["grpo_train"]["rows"] != 8000 or reports["grpo_val"]["rows"] != 500:
        raise ValueError(f"GRPO factor has wrong ledger sizes: {source_kind}")
    for objective in ("grpo_train", "grpo_val"):
        frame = __import__("pandas").read_parquet(paths[objective])
        if not set(frame["extra_info"].map(lambda value: value.get("objective"))) == {"grpo"}:
            raise ValueError(f"missing GRPO objective provenance: {source_kind}/{objective}")
        if frame["completion"].notna().any():
            raise ValueError(f"GRPO parquet unexpectedly contains completions: {source_kind}/{objective}")
        expected_split = "train" if objective.endswith("train") else "val"
        expected_uids = {uid for uid, row in ledger.items() if row["split"] == expected_split}
        if set(frame["problem_uid"].astype(str)) != expected_uids:
            raise ValueError(f"GRPO problem ledger mismatch: {source_kind}/{objective}")
    return reports


def _validate_all(args: argparse.Namespace) -> dict[str, Any]:
    ledger_rows = _read_jsonl(args.ledger)
    ledger = native.validate_ledger(ledger_rows)
    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    supervision = {}
    parquet = {}
    for source_kind in SOURCE_KINDS:
        source = _mle_supervision_path(args, source_kind)
        supervision[source_kind] = {
            "path": str(source),
            "sha256": _sha256(source),
            "rows": len(_read_jsonl(source)),
        }
        _validate_train_supervision(_read_jsonl(source), ledger, source_kind)
        parquet[source_kind] = _validate_parquet_factor(args, ledger, source_kind)
    return {
        "ledger": {"path": str(args.ledger), "sha256": _sha256(args.ledger), "rows": len(ledger_rows)},
        "supervision": supervision,
        "parquet": parquet,
    }


def _model_path(args: argparse.Namespace) -> str:
    return str(args.base_model)


def _sft_command(args: argparse.Namespace, source_kind: str, seed: int) -> list[str]:
    output_tag = f"factorial_intervention_{source_kind}_mle_seed{seed}_{args.tag}"
    return [
        "python",
        "train/train_line_level_sft.py",
        "--mode",
        "train",
        "--tag",
        output_tag,
        "--k",
        "4",
        "--sampling",
        "entrance-diverse",
        "--base-model",
        _model_path(args),
        "--checkpoint-root",
        str(args.checkpoint_root / "sft"),
        "--gpu-id",
        "{gpu}",
        "--epochs",
        str(args.sft_epochs),
        "--batch-size",
        str(args.sft_batch_size),
        "--grad-accum",
        str(args.sft_grad_accum),
        "--save-every",
        str(args.sft_save_every),
        "--supervision-path",
        str(_mle_supervision_path(args, source_kind)),
        "--prompt-style",
        "native",
        "--seed",
        str(seed),
    ]


def _grpo_command(args: argparse.Namespace, source_kind: str, seed: int) -> list[str]:
    paths = _parquet_paths(args, source_kind)
    output = args.checkpoint_root / "grpo" / f"{source_kind}_seed{seed}"
    experiment = f"factorial_intervention-{source_kind}-grpo-seed-{seed}"
    return [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={paths['grpo_train']}",
        f"data.val_files={paths['grpo_val']}",
        f"data.train_batch_size={args.train_batch_size}",
        "data.val_batch_size=500",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"actor_rollout_ref.model.path={_model_path(args)}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.optim.type=adamw",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.train_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.grad_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        f"+actor_rollout_ref.actor.fsdp_config.model_dtype={args.actor_model_dtype}",
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
        f"trainer.seed={seed}",
        "trainer.critic_warmup=0",
        "trainer.logger=[console]",
        "+trainer.val_before_train=False",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=2",
        "trainer.total_training_steps=null",
        f"trainer.save_freq={args.grpo_save_every}",
        "trainer.test_freq=-1",
        "trainer.default_hdfs_dir=null",
        f"trainer.default_local_dir={output}",
        "trainer.project_name=TinyZero-M12",
        f"trainer.experiment_name={experiment}",
    ]


def _runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    gpu_ids = _gpu_ids(args)
    runs = []
    index = 0
    for source_kind in SOURCE_KINDS:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                command = _sft_command(args, source_kind, seed) if objective == "mle" else _grpo_command(args, source_kind, seed)
                command = [gpu_ids[index % len(gpu_ids)] if token == "{gpu}" else token for token in command]
                runs.append({
                    "run_id": f"{source_kind}-{objective}-seed{seed}",
                    "source_kind": source_kind,
                    "objective": objective,
                    "seed": seed,
                    "gpu_id": gpu_ids[index % len(gpu_ids)],
                    "command": command,
                    "output_root": str(
                        (args.checkpoint_root / "sft" / f"factorial_intervention_{source_kind}_mle_seed{seed}_{args.tag}")
                        if objective == "mle"
                        else args.checkpoint_root / "grpo" / f"{source_kind}_seed{seed}"
                    ),
                })
                index += 1
    return runs


def _ray_tmpdir(args: argparse.Namespace, run: dict[str, Any]) -> Path:
    """Return a short unique Ray root for one condition/seed process."""
    source_code = {"solver-rich": "sr", "on-policy": "op"}[str(run["source_kind"])]
    objective_code = {"mle": "m", "grpo": "g"}[str(run["objective"])]
    path = args.ray_root / f"{source_code}{objective_code}{int(run['seed'])}"
    # Ray's session suffix is currently about 80 bytes including socket names.
    # Fail before spawning a worker if a custom root would reintroduce the
    # kernel limit, rather than producing a less actionable Ray traceback.
    if len(str(path.resolve())) + 82 > 107:
        raise ValueError(f"Ray root is too long for AF_UNIX sockets: {path}")
    return path


def _prior_statuses(args: argparse.Namespace) -> tuple[dict[str, int], str | None]:
    """Load the latest launcher statuses without changing the frozen design."""
    candidates = (_run_final_manifest_path(args), _run_manifest_path(args))
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        statuses = payload.get("statuses")
        if isinstance(statuses, dict):
            return {str(key): int(value) for key, value in statuses.items()}, str(path)
    return {}, None


def _mle_run_complete(run: dict[str, Any]) -> bool:
    """Require the formal SFT manifest before trusting a prior zero exit."""
    manifest_path = Path(str(run["output_root"])) / "manifest.json"
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        payload.get("status") == "complete"
        and payload.get("formal_complete") is True
        and int(payload.get("optimizer_steps", 0)) == int(payload.get("target_optimizer_steps", -1))
    )


def _completed_run_ids(manifest: dict[str, Any], prior: dict[str, int]) -> set[str]:
    completed = set()
    for run in manifest["runs"]:
        run_id = str(run["run_id"])
        if prior.get(run_id) != 0:
            continue
        if str(run["objective"]) == "mle" and not _mle_run_complete(run):
            raise RuntimeError(f"launcher recorded successful MLE without a complete manifest: {run_id}")
        completed.add(run_id)
    return completed


def _runtime_manifest(args: argparse.Namespace, frozen: dict[str, Any]) -> dict[str, Any]:
    """Overlay current resource/runtime commands without changing factor identity."""
    current = {str(run["run_id"]): run for run in _runs(args)}
    runs = []
    for frozen_run in frozen["runs"]:
        run_id = str(frozen_run["run_id"])
        if run_id not in current:
            raise ValueError(f"frozen run missing from current design: {run_id}")
        generated = current[run_id]
        for key in ("source_kind", "objective", "seed", "gpu_id", "output_root"):
            if str(frozen_run.get(key)) != str(generated.get(key)):
                raise ValueError(f"frozen/current M12 run mismatch for {run_id}: {key}")
        run = dict(frozen_run)
        run["command"] = list(generated["command"])
        runs.append(run)
    result = dict(frozen)
    result["runs"] = runs
    result["runtime_profile"] = {
        "actor_model_dtype": args.actor_model_dtype,
        "rollout_memory": args.rollout_memory,
        "rollout_max_batched_tokens": args.rollout_max_batched_tokens,
        "rollout_max_seqs": args.rollout_max_seqs,
        "note": "Resource-only GRPO override; data/objective/seed/output identities remain frozen.",
    }
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    ledger = native.validate_ledger(_read_jsonl(args.ledger))
    sources = {source_kind: _build_train_supervision(args, ledger, source_kind) for source_kind in SOURCE_KINDS}
    audit = _validate_all(args)
    payload = {
        "artifact": PROTOCOL_VERSION,
        "status": "prepared",
        "tag": args.tag,
        "ledger": audit["ledger"],
        "sources": sources,
        "parquet": audit["parquet"],
        "design": {
            "data_factor": ["solver-rich", "on-policy"],
            "objective_factor": ["mle", "grpo"],
            "seeds": list(SEEDS),
            "run_count": 8,
            "train_problems": 8000,
            "validation_problems": 500,
            "grpo_source_note": "Native GRPO samples responses itself; source-specific GRPO parquet rows are one per problem and differ only in provenance fingerprints.",
        },
        "runs": _runs(args),
        "config": {
            "base_model": str(args.base_model),
            "sft_epochs": args.sft_epochs,
            "grpo_epochs": 2,
            "rollout_n": args.rollout_n,
            "learning_rate": args.learning_rate,
            "actor_model_dtype": args.actor_model_dtype,
            "rollout_memory": args.rollout_memory,
            "rollout_max_batched_tokens": args.rollout_max_batched_tokens,
            "rollout_max_seqs": args.rollout_max_seqs,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "gpu_ids": _gpu_ids(args),
        },
    }
    _atomic_json(_manifest_path(args), payload)
    return payload


def audit(args: argparse.Namespace) -> dict[str, Any]:
    payload = _validate_all(args)
    manifest = _manifest_path(args)
    if manifest.is_file():
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        if frozen.get("ledger", {}).get("sha256") != payload["ledger"]["sha256"]:
            raise ValueError("factor manifest ledger hash does not match current ledger")
        if len(frozen.get("runs", [])) != 8:
            raise ValueError("factor manifest does not contain exactly eight runs")
    result = {"artifact": PROTOCOL_VERSION, "status": "complete", **payload}
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return result


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    audit(args)
    frozen = json.loads(_manifest_path(args).read_text(encoding="utf-8"))
    manifest = _runtime_manifest(args, frozen)
    for run in manifest["runs"]:
        print(f"{run['run_id']}: CUDA_VISIBLE_DEVICES={run['gpu_id']} conda run -n {args.env_name} {shlex.join(run['command'])}")
    return manifest


def launch(args: argparse.Namespace) -> dict[str, Any]:
    manifest = dry_run(args)
    run_manifest = _run_manifest_path(args)
    final_manifest = _run_final_manifest_path(args)
    prior_statuses, prior_manifest = _prior_statuses(args)
    completed = _completed_run_ids(manifest, prior_statuses)
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.ray_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **manifest,
        "status": "running",
        "resume": {
            "prior_manifest": prior_manifest,
            "prior_statuses": prior_statuses,
            "completed_before_launch": sorted(completed),
            "ray_root": str(args.ray_root),
        },
    }
    _replace_json(run_manifest, payload)
    statuses: dict[str, int] = {run_id: 0 for run_id in completed}
    for offset in range(0, len(manifest["runs"]), 2):
        processes = []
        handles = []
        batch = [
            run for run in manifest["runs"][offset : offset + 2]
            if str(run["run_id"]) not in completed
        ]
        for run in batch:
            log_path = args.log_root / f"{run['run_id']}.log"
            env = os.environ.copy()
            ray_tmpdir = _ray_tmpdir(args, run)
            env.update({
                "CUDA_VISIBLE_DEVICES": str(run["gpu_id"]),
                "VLLM_ATTENTION_BACKEND": "XFORMERS",
                "RAY_TMPDIR": str(ray_tmpdir),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(
                ["conda", "run", "-n", args.env_name, *run["command"]],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((run["run_id"], process))
        for run_id, process in processes:
            statuses[run_id] = process.wait()
        for handle in handles:
            handle.close()
        payload["statuses"] = dict(statuses)
        _replace_json(run_manifest, payload)
        if any(code != 0 for code in statuses.values()):
            break
    payload["statuses"] = dict(statuses)
    payload["status"] = "complete" if len(statuses) == len(manifest["runs"]) and all(code == 0 for code in statuses.values()) else "failed"
    _replace_json(final_manifest, payload)
    if payload["status"] != "complete":
        raise subprocess.CalledProcessError(1, "M12 factorial launch", output=statuses)
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "audit":
        result = audit(args)
    elif args.mode == "dry-run":
        result = dry_run(args)
    else:
        result = launch(args)
    if args.mode not in {"audit", "dry-run"}:
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
