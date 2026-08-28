
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rollout_collection - Countdown sampling and pass@k collection.
Hypothesis: Countdown RLVR changes both correctness mass and the diversity of sampled solutions.
Inputs: dataset/test.parquet; base/checkpoint model paths; sampling parameters.
Outputs: data/analysis_results/rlvr_passk/metrics/countdown_summary_n320.csv; data/analysis_results/rlvr_passk/raw/countdown_raw_*_n320.jsonl
Status: paper-main
"""
"""
passk_countdown_collect.py

在 Countdown-Feasible 上收集采样（raw jsonl）并计算 pass@k 指标。
支持：
- base 模型（可选）
- actor checkpoints: checkpoints/TinyZero/countdown-qwen2.5-3b/actor/global_step_*
- master/worker 多 GPU 并行（每 GPU 一个模型）
- N_SAMPLES 可设为 64/128/256/1024

重要：raw 输出包含：
- completion
- feasible_ok / expr_ok / overall_ok
- answer_label / trace_label
用于后续 branch_set_collection/entrance_entropy。

用法（master，自动扫所有模型）：
  python passk_countdown_collect.py

用法（worker，跑一个模型）：
  python passk_countdown_collect.py --model_path ... --gpu_id 0 --n_samples 256
"""

import os
import sys
import json
import math
import argparse
import subprocess
import glob
import time
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR, METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402

# checkpoints
BASE_CHECKPOINT_DIR = (
    COUNTDOWN_ACTOR_DIR
)

# 你原来的 base
DEFAULT_BASE_MODEL_PATH = "model/qwen253B"

RAW_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# 兼容清理
EOS_STRINGS = ["<|endoftext|>", "<|im_end|>"]

from early_branch_locking.core.countdown_utils import (  # noqa: E402
    parse_countdown_completion,
    evaluate_countdown_expression,
)

from early_branch_locking.core.countdown_shared import (  # noqa: E402
    load_parquet_sorted,
    extract_ground_truth,
    get_prompt_content,
    build_prompt_text,
    pass_at_k,
    bootstrap_ci_mean,
    evaluate_countdown_completion,
)


class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


def output_suffix(seed: int | None, tag: str) -> str:
    """Keep legacy names unless a seeded/tagged run was explicitly requested."""
    if seed is None and not tag:
        return ""
    effective_seed = SEED if seed is None else seed
    suffix = f"_seed{effective_seed}"
    return f"{suffix}_{tag}" if tag else suffix


def raw_output_path(model_name: str, n_samples: int, seed: int | None, tag: str) -> Path:
    return RAW_DIR / f"countdown_raw_{model_name}_n{n_samples}{output_suffix(seed, tag)}.jsonl"


def metrics_output_path(model_name: str, n_samples: int, seed: int | None, tag: str) -> Path:
    return METRICS_DIR / f"countdown_metrics_{model_name}_n{n_samples}{output_suffix(seed, tag)}.csv"


def metrics_glob(n_samples: int, seed: int | None, tag: str) -> str:
    return f"countdown_metrics_*_n{n_samples}{output_suffix(seed, tag)}.csv"


def get_worker_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--model_label",
        type=str,
        default="",
        help="explicit output/checkpoint label; required for custom checkpoints that share a basename",
    )
    p.add_argument("--gpu_id", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=256)  # countdown 推荐 256
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None,
                   help="seed the run and add a seed suffix; omit for legacy filenames")
    p.add_argument("--tag", type=str, default="",
                   help="optional run tag; tagged runs receive an isolated filename suffix")
    p.add_argument("--sample_chunk_size", type=int, default=0,
                   help="split each n-sample request into smaller vLLM calls; 0 disables chunking")
    p.add_argument(
        "--problem-indices",
        default="",
        help="comma-separated original test-set indices; preserves those IDs in raw output",
    )
    p.add_argument("--enforce_eager", action="store_true", default=False,
                   help="disable CUDA graph capture to avoid vLLM instability")
    return p.parse_args()


def worker_main():
    args = get_worker_args()
    # A worker may be launched by a multi-process controller that has already
    # isolated it to one physical GPU.  Respect that launch-time binding: CUDA
    # visibility must be fixed before importing/initializing vLLM, and
    # overwriting an inherited ``CUDA_VISIBLE_DEVICES=1`` with ``1`` inside
    # the child can make both workers resolve to physical GPU 0.  The legacy
    # direct invocation remains supported when no binding was inherited.
    inherited_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not inherited_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    from vllm import LLM, SamplingParams

    seed = SEED if args.seed is None else args.seed

    model_path = args.model_path
    ckpt_name = os.path.basename(model_path.rstrip("/"))
    checkpoint_label = args.model_label.strip() or ckpt_name
    if "/" in checkpoint_label or not checkpoint_label:
        raise ValueError(f"invalid model_label: {checkpoint_label!r}")

    print(
        f"\n[Worker GPU-{args.gpu_id}; visible={os.environ.get('CUDA_VISIBLE_DEVICES')}] "
        f"Model: {ckpt_name} label={checkpoint_label}"
    )
    print(f"[Worker GPU-{args.gpu_id}] n_samples={args.n_samples} num_problems={args.num_problems}")

    # 初始化 vLLM
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            seed=seed,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[Worker GPU-{args.gpu_id}] Model load failed: {e}")
        sys.exit(1)

    # Read the full fixed evaluation ordering when an explicit subset is
    # requested.  ``problem_index`` remains the original test-set index rather
    # than the position inside the subset, which is essential for paired
    # follow-up experiments such as commitment migration.
    requested_indices = [int(x) for x in args.problem_indices.split(",") if x.strip()]
    if len(set(requested_indices)) != len(requested_indices) or any(x < 0 for x in requested_indices):
        raise ValueError("--problem-indices must be unique non-negative integers")
    if requested_indices:
        all_records = load_parquet_sorted(TEST_PARQUET, n=max(max(requested_indices) + 1, args.num_problems), sort_key="sample_id")
        if max(requested_indices) >= len(all_records):
            raise ValueError(f"--problem-indices contains unavailable index {max(requested_indices)}")
        indexed_records = [(pid, all_records[pid]) for pid in requested_indices]
    else:
        records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
        indexed_records = list(enumerate(records))

    prompts = []
    meta = []
    for problem_index, rec in indexed_records:
        prompt_content = get_prompt_content(rec)
        prompt_text = build_prompt_text(prompt_content, tokenizer)
        numbers, target, feasible_label = extract_ground_truth(rec)
        prompts.append(prompt_text)
        meta.append(
            dict(
                sample_id=rec.get("sample_id", -1),
                problem_index=problem_index,
                numbers=numbers,
                target=target,
                feasible_label=feasible_label,
            )
        )

    # stop: 仅依赖 EOS/长度截断，避免因 stop 丢失后续标签
    stop_strs = None

    raw_path = raw_output_path(checkpoint_label, args.n_samples, args.seed, args.tag)
    partial_raw_path = raw_path.with_name(raw_path.name + ".partial")
    chunk_size = args.n_samples if args.sample_chunk_size == 0 else args.sample_chunk_size
    if chunk_size <= 0 or chunk_size > args.n_samples:
        raise ValueError(
            f"sample_chunk_size must be 0 or in [1, {args.n_samples}], got {args.sample_chunk_size}"
        )
    n_chunks = math.ceil(args.n_samples / chunk_size)
    print(
        f"[Worker GPU-{args.gpu_id}] Generating {len(prompts)}×{args.n_samples} "
        f"in {n_chunks} chunk(s) of at most {chunk_size} ..."
    )
    print(f"[Worker GPU-{args.gpu_id}] Writing raw -> {raw_path}")

    correct_flags = defaultdict(list)
    feasible_ok_flags = defaultdict(list)
    expr_ok_flags = defaultdict(list)

    with partial_raw_path.open("w", encoding="utf-8") as f:
        global_idx = 0
        for chunk_idx, chunk_start in enumerate(range(0, args.n_samples, chunk_size)):
            current_chunk_size = min(chunk_size, args.n_samples - chunk_start)
            # Vary the per-call seed so chunks form one reproducible independent sample set.
            chunk_seed = seed + chunk_idx
            sampling_params = SamplingParams(
                n=current_chunk_size,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                seed=chunk_seed,
                stop=stop_strs,
                stop_token_ids=[tokenizer.eos_token_id]
                if tokenizer.eos_token_id is not None
                else None,
            )
            print(
                f"[Worker GPU-{args.gpu_id}] Chunk {chunk_idx + 1}/{n_chunks}: "
                f"samples {chunk_start}..{chunk_start + current_chunk_size - 1}"
            )
            outputs = llm.generate(prompts, sampling_params)
            if len(outputs) != len(prompts):
                raise RuntimeError(f"vLLM outputs mismatch: {len(outputs)} vs {len(prompts)}")

            for out, m in zip(outputs, meta):
                prob_idx = int(m["problem_index"])
                for local_sample_idx, cand in enumerate(out.outputs):
                    sample_idx = chunk_start + local_sample_idx
                    text = cand.text or ""
                    # 兼容清理
                    for eos in EOS_STRINGS:
                        if eos in text:
                            text = text.split(eos)[0]
                    text = text.strip()

                    ev = evaluate_countdown_completion(
                        text=text,
                        numbers=m["numbers"],
                        target=m["target"],
                        feasible_label=m["feasible_label"],
                        parse_countdown_completion=parse_countdown_completion,
                        evaluate_countdown_expression=evaluate_countdown_expression,
                    )

                    correct_flags[prob_idx].append(bool(ev.overall_ok))
                    feasible_ok_flags[prob_idx].append(bool(ev.feasible_ok))
                    expr_ok_flags[prob_idx].append(bool(ev.expr_ok))

                    rec_out = dict(
                        global_index=global_idx,
                        problem_index=prob_idx,
                        sample_index=sample_idx,
                        checkpoint=checkpoint_label,
                        model_path=model_path,
                        sample_id=m["sample_id"],
                        numbers=m["numbers"],
                        target=m["target"],
                        feasible_label=m["feasible_label"],
                        completion=text,
                        feasible_pred=ev.feasible_pred,
                        parse_status=ev.parse_status,
                        has_feasible_tag=ev.has_feasible_tag,
                        has_answer_tag=ev.has_answer_tag,
                        tag_order_ok=ev.tag_order_ok,
                        feasible_ok=ev.feasible_ok,
                        expr_ok=ev.expr_ok,
                        overall_ok=ev.overall_ok,
                        answer_label=ev.answer_label,
                        trace_label=ev.trace_label,
                        expr_status=ev.expr_status,
                        canonical_expr=ev.canonical_expr,
                        opseq_label=ev.opseq_label,
                    )
                    if args.seed is not None or args.tag:
                        rec_out.update(seed=seed, tag=args.tag)
                    f.write(json.dumps(rec_out, ensure_ascii=False, cls=NumPyEncoder) + "\n")
                    global_idx += 1

            f.flush()

    partial_raw_path.replace(raw_path)

    # metrics: pass@k
    ks = [1, 4, 16, 64, 128, 256, 384, 512]
    per_k_vals = {k: [] for k in ks}
    per_k_ci = {}

    for prob_idx, flags in correct_flags.items():
        n = len(flags)
        c = sum(flags)
        for k in ks:
            if k <= n:
                per_k_vals[k].append(pass_at_k(n, c, k))

    metrics = dict(
        checkpoint=checkpoint_label,
        model_path=model_path,
        num_problems=len(correct_flags),
        n_samples=args.n_samples,
    )
    if args.seed is not None or args.tag:
        metrics.update(seed=seed, tag=args.tag)

    for k in ks:
        if len(per_k_vals[k]) == 0:
            continue
        m, lo, hi = bootstrap_ci_mean(per_k_vals[k], n_boot=2000, alpha=0.05, seed=seed)
        metrics[f"pass@{k}"] = m
        metrics[f"pass@{k}_ci_lo"] = lo
        metrics[f"pass@{k}_ci_hi"] = hi

    # 额外：feasible_ok 与 expr_ok 的 pass@k（用于诊断“只是学会格式”）
    for name, flags_map in [("feasible_ok", feasible_ok_flags), ("expr_ok", expr_ok_flags)]:
        for k in ks:
            vals = []
            for prob_idx, flags in flags_map.items():
                n = len(flags)
                c = sum(flags)
                if k <= n:
                    vals.append(pass_at_k(n, c, k))
            if vals:
                m, lo, hi = bootstrap_ci_mean(vals, n_boot=2000, alpha=0.05, seed=seed)
                metrics[f"{name}_pass@{k}"] = m
                metrics[f"{name}_pass@{k}_ci_lo"] = lo
                metrics[f"{name}_pass@{k}_ci_hi"] = hi

    # 写 metrics
    metrics_path = metrics_output_path(checkpoint_label, args.n_samples, args.seed, args.tag)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"[Worker GPU-{args.gpu_id}] Metrics -> {metrics_path}")
    print(f"[Worker GPU-{args.gpu_id}] Done: pass@1={metrics.get('pass@1', float('nan')):.4f} pass@{min(64,args.n_samples)}={metrics.get(f'pass@{min(64,args.n_samples)}', float('nan')):.4f}")


def master_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_base", action="store_true", default=True)
    p.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None,
                   help="seed the run and add a seed suffix; omit for legacy filenames")
    p.add_argument("--tag", type=str, default="",
                   help="optional run tag; tagged runs receive an isolated filename suffix")
    p.add_argument("--sample_chunk_size", type=int, default=0,
                   help="split each n-sample request into smaller vLLM calls; 0 disables chunking")
    p.add_argument("--enforce_eager", action="store_true", default=False,
                   help="disable CUDA graph capture to avoid vLLM instability")
    p.add_argument("--only_steps", type=str, default="")  # 例如 "50,125,200,275"
    return p.parse_args()


def collect_and_aggregate_metrics(n_samples: int, seed: int | None, tag: str):
    """
    聚合 countdown_metrics_*_n{n_samples}.csv
    """
    files = sorted(METRICS_DIR.glob(metrics_glob(n_samples, seed, tag)))
    if not files:
        print("[Master] No metric files to aggregate.")
        return

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        if df.empty:
            continue
        ckpt = df["checkpoint"].iloc[0]
        if ckpt.startswith("global_step_"):
            try:
                step = int(ckpt.split("_")[-1])
            except Exception:
                step = 0
        else:
            step = 0
        df["step"] = step
        rows.append(df)

    full = pd.concat(rows, ignore_index=True).sort_values("step").reset_index(drop=True)
    outp = METRICS_DIR / f"countdown_summary_n{n_samples}{output_suffix(seed, tag)}.csv"
    full.to_csv(outp, index=False)
    print(f"[Master] Summary -> {outp}")
    cols = [c for c in ["checkpoint","step","num_problems","n_samples","pass@1","pass@64","pass@128","pass@256"] if c in full.columns]
    print(full[cols])


def master_main():
    args = master_args()
    from natsort import natsorted


    model_paths = []
    if args.eval_base and args.base_model_path:
        model_paths.append(str(args.base_model_path))

    ckpt_dirs = [d for d in glob.glob(str(BASE_CHECKPOINT_DIR / "global_step_*")) if os.path.isdir(d)]
    ckpt_dirs = natsorted(ckpt_dirs)

    only_steps = set()
    if args.only_steps.strip():
        for s in args.only_steps.split(","):
            s = s.strip()
            if s:
                only_steps.add(int(s))

    for d in ckpt_dirs:
        name = os.path.basename(d)
        if only_steps:
            if name.startswith("global_step_"):
                st = int(name.split("_")[-1])
                if st not in only_steps:
                    continue
        model_paths.append(d)

    # 过滤已算过的
    tasks = []
    for mp in model_paths:
        ckpt_name = os.path.basename(mp.rstrip("/"))
        metrics_path = metrics_output_path(ckpt_name, args.n_samples, args.seed, args.tag)
        if metrics_path.exists():
            print(f"[Master] Skip {ckpt_name} (already done)")
            continue
        tasks.append(mp)

    print(f"[Master] To process: {len(tasks)} models, n_samples={args.n_samples}")

    if not tasks:
        collect_and_aggregate_metrics(args.n_samples, args.seed, args.tag)
        return

    num_gpus = torch.cuda.device_count()
    print(f"[Master] GPUs detected: {num_gpus}")
    free_gpus = list(range(num_gpus))
    running = {}  # gpu_id -> proc
    idx = 0

    while idx < len(tasks) or running:
        while free_gpus and idx < len(tasks):
            gpu = free_gpus.pop(0)
            mp = tasks[idx]
            idx += 1
            ckpt_name = os.path.basename(mp.rstrip("/"))

            cmd = [
                sys.executable, __file__,
                "--model_path", mp,
                "--gpu_id", str(gpu),
                "--n_samples", str(args.n_samples),
                "--num_problems", str(args.num_problems),
                "--temperature", str(args.temperature),
                "--top_p", str(args.top_p),
                "--max_new_tokens", str(args.max_new_tokens),
                "--sample_chunk_size", str(args.sample_chunk_size),
                "--max_model_len", str(args.max_model_len),
                "--dtype", str(args.dtype),
                "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            ]
            if args.seed is not None:
                cmd.extend(["--seed", str(args.seed)])
            if args.tag:
                cmd.extend(["--tag", args.tag])
            if args.enforce_eager:
                cmd.append("--enforce_eager")
            print(f"[Master] Launch {ckpt_name} on GPU {gpu}")
            proc = subprocess.Popen(cmd)
            running[gpu] = proc

        if running:
            done = []
            for gpu, proc in running.items():
                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        print(f"[Master] Worker GPU {gpu} failed code={ret}")
                    done.append(gpu)
            for gpu in done:
                del running[gpu]
                free_gpus.append(gpu)
                free_gpus.sort()
            if not done:
                time.sleep(1)

    print("[Master] All done.")
    collect_and_aggregate_metrics(args.n_samples, args.seed, args.tag)


if __name__ == "__main__":
    # worker 模式
    if "--model_path" in sys.argv:
        worker_main()
    else:
        master_main()
