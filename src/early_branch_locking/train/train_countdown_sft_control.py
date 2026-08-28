#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""training_breadth_control - SFT control separating narrow training data from on-policy dynamics.

Hypothesis: if the contraction is driven mainly by the narrowness of the
successful trajectories that training reinforces, then supervised fine-tuning of
the base model on a late checkpoint's own correct samples should reproduce a
comparable contraction; if instead on-policy dynamics are necessary, the SFT
model should retain broader access at comparable one-sample accuracy.
Inputs: Countdown raw JSONL for one source checkpoint; dataset/test.parquet;
the base model directory.
Outputs: data/analysis_results/rlvr_passk/metrics/solution_coverage2_sft_dataset_{tag}.jsonl;
data/analysis_results/rlvr_passk/metrics/solution_coverage2_sft_dataset_stats_{tag}.csv;
checkpoints/countdown_sft/{tag}/ (a plain HF directory);
data/analysis_results/rlvr_passk/metrics/solution_coverage2_sft_manifest_{tag}.json
Modes: ``build`` writes the prompt-masked dataset; ``train`` fine-tunes; the two
source checkpoints (a late, narrow one and an early, broad one) isolate data
breadth as the independent variable.
Status: complete

Training defaults to LoRA with a merged export so that the resulting directory
can be consumed unchanged by countdown/01_collect_rollouts.py. Full-parameter tuning
of a 3B actor needs the sharded trainer and is intentionally not attempted here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import METRICS_DIR, RAW_DIR, REPO_ROOT, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    build_prompt_text,
    extract_ground_truth,
    get_prompt_content,
    load_jsonl,
    load_parquet_sorted,
)
from early_branch_locking.core.structure_utils import first_operator  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "train"), default="build")
    parser.add_argument("--source-raw", type=Path, default=RAW_DIR / "countdown_raw_global_step_275_n320.jsonl")
    parser.add_argument("--base-model", type=Path, default=REPO_ROOT / "model" / "qwen253B")
    parser.add_argument("--num-problems", type=int, default=150)
    parser.add_argument("--max-per-problem", type=int, default=8)
    parser.add_argument("--max-per-canonical", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tag", default="step275")
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints" / "countdown_sft")
    return parser.parse_args(argv)


def dataset_path(args: argparse.Namespace) -> Path:
    return args.out_dir / f"solution_coverage2_sft_dataset_{args.tag}.jsonl"


def run_build(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompts, targets = {}, {}
    for pid, record in enumerate(records):
        prompts[pid] = build_prompt_text(get_prompt_content(record), tokenizer)
        numbers, target, _feasible = extract_ground_truth(record)
        targets[pid] = (numbers, target)
    per_problem: dict[int, list[dict]] = defaultdict(list)
    per_canonical: dict[tuple[int, str], int] = defaultdict(int)
    for row in load_jsonl(args.source_raw):
        pid = int(row.get("problem_index", -1))
        if pid < 0 or pid >= args.num_problems or not row.get("overall_ok"):
            continue
        canonical = str(row.get("canonical_expr") or "")
        if not canonical or per_canonical[(pid, canonical)] >= args.max_per_canonical:
            continue
        if len(per_problem[pid]) >= args.max_per_problem:
            continue
        per_canonical[(pid, canonical)] += 1
        per_problem[pid].append(
            {
                "problem_index": pid,
                "prompt": prompts[pid],
                "completion": str(row.get("completion") or ""),
                "canonical_expr": canonical,
                "first_op": first_operator(canonical),
                "source_raw": str(args.source_raw.name),
            }
        )
    rows = [item for pid in sorted(per_problem) for item in per_problem[pid]]
    if not rows:
        raise RuntimeError(f"no correct samples found in {args.source_raw}")
    random.Random(args.seed).shuffle(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_path(args)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    frame = pd.DataFrame(rows)
    stats = pd.DataFrame(
        [
            {
                "tag": args.tag,
                "source_raw": str(args.source_raw),
                "n_examples": len(frame),
                "n_problems": int(frame["problem_index"].nunique()),
                "unique_canonical": int(frame["canonical_expr"].nunique()),
                "unique_canonical_per_problem": float(
                    frame.groupby("problem_index")["canonical_expr"].nunique().mean()
                ),
                "unique_first_op_per_problem": float(
                    frame.groupby("problem_index")["first_op"].nunique().mean()
                ),
                "max_per_problem": args.max_per_problem,
                "max_per_canonical": args.max_per_canonical,
                "seed": args.seed,
            }
        ]
    )
    stats.to_csv(args.out_dir / f"solution_coverage2_sft_dataset_stats_{args.tag}.csv", index=False)
    print(json.dumps({"dataset": str(path), "examples": len(frame)}, sort_keys=True))


def run_train(args: argparse.Namespace) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    path = dataset_path(args)
    if not path.exists():
        raise FileNotFoundError(f"dataset missing; run --mode build first: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class Countdown(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict:
            row = rows[index]
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                target_ids = target_ids + [tokenizer.eos_token_id]
            ids = (prompt_ids + target_ids)[: args.max_seq_len]
            labels = ([-100] * len(prompt_ids) + target_ids)[: args.max_seq_len]
            return {"input_ids": ids, "labels": labels}

    def collate(batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        pad = tokenizer.pad_token_id
        input_ids, labels, mask = [], [], []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad] * padding)
            labels.append(item["labels"] + [-100] * padding)
            mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    adapter = "full"
    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        adapter = f"lora_r{args.lora_rank}"
    loader = DataLoader(Countdown(), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=False)
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    model.train()
    step, accumulated, done = 0, 0, False
    losses: list[float] = []
    while not done:
        for batch in loader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            losses.append(float(loss.item()) * args.grad_accum)
            accumulated += 1
            if accumulated % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"[c2] step {step}/{total_steps} loss={sum(losses[-10:]) / min(10, len(losses)):.4f}", flush=True)
                if step >= total_steps:
                    done = True
                    break
    output = args.checkpoint_root / args.tag
    output.mkdir(parents=True, exist_ok=True)
    export = model.merge_and_unload() if not args.full_finetune else model
    export.config.use_cache = True
    export.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    manifest = {
        "experiment_id": "training_breadth_control",
        "tag": args.tag,
        "base_model": str(args.base_model),
        "source_raw": str(args.source_raw),
        "dataset": str(path),
        "n_examples": len(rows),
        "adapter": adapter,
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "optimizer_steps": step,
        "final_loss_mean_last10": float(sum(losses[-10:]) / min(10, len(losses))) if losses else None,
        "checkpoint": str(output),
        "seed": args.seed,
        "downstream_evaluation": [
            f"countdown/01_collect_rollouts.py --model_path {output} --gpu_id 0 --n_samples 320 --num_problems 150",
            "countdown/03_solution_coverage.py --n_samples 320 --num_problems 150 --save_per_problem",
            "countdown/public_grpo_replication.py",
        ],
        "interpretation": (
            "comparable contraction supports a data-breadth account and requires weakening the "
            "rollout-exposure claim; preserved breadth supports the on-policy account"
        ),
    }
    manifest_path = args.out_dir / f"solution_coverage2_sft_manifest_{args.tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(output), "manifest": str(manifest_path), "steps": step}, sort_keys=True))


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "build":
        run_build(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
