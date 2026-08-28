
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""teacher_forced_likelihood - Teacher-forced solution log-probability oracle.
Hypothesis: conditional solution support is not sufficient to explain unconditional Countdown success after format probability is separated.
Inputs: Countdown raw JSONL; dataset/test.parquet; base and checkpoint causal language models.
Outputs: data/analysis_results/rlvr_passk/metrics/teacher_forced_likelihood_solution_logprob.csv; data/analysis_results/rlvr_passk/metrics/teacher_forced_likelihood_solution_logprob_per_problem.parquet
Status: paper-main
"""
"""
teacher_forced_likelihood_solution_logprob_countdown.py

ExpD: 条件 oracle（剥离格式概率）
- 枚举每题解集 S(x)
- 固定前缀 <feasible>yes</feasible><answer>，只对 answer span 内的表达式做 teacher-forcing
- 估计条件分布 P(e | x, tags) 并计算 support mass / 熵 / 集中度
- 再用 raw 里的格式率 p_fmt 近似无条件成功率：p_success ≈ p_fmt * support_mass_cond
- 计算 oracle pass@k（条件/无条件）
"""

import argparse
import json
import math
import sys
from collections import defaultdict, Counter
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_DATA_ROOT as ANALYSIS_ROOT, METRICS_DIR as OUT_DIR, TEST_PARQUET  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_shared import (  # noqa: E402
    load_parquet_sorted,
    extract_ground_truth,
    canonicalize_expression,
    tolerant_parse_completion,
    enumerate_solution_list,
    load_jsonl,
    step_of,
    logsumexp,
    gini_from_probs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_problems", type=int, default=200)
    p.add_argument("--n_samples", type=int, required=True,
                   help="n_samples used in raw collection for format rate estimation")
    p.add_argument("--raw_dir", type=str, default=str(ANALYSIS_ROOT / "raw"))
    p.add_argument("--k_list", type=str, default="1,4,16,64,128,256,512")
    p.add_argument("--base_model_path", type=str, default="model/qwen253B")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/TinyZero/countdown-qwen2.5-3b/actor")
    p.add_argument("--only_steps", type=str, default="")
    p.add_argument("--answer_tag_prefix", type=str, default="<feasible>yes</feasible>\n<answer>",
                   help="prefix inserted before answer expression when scoring")
    p.add_argument("--include_answer_close", action="store_true", default=False,
                   help="include </answer> in the scored target (stricter)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    return p.parse_args()


def enumerate_solution_set(numbers: List[int], target: int) -> List[str]:
    """Wrapper that returns sorted list for backward compat."""
    return enumerate_solution_list(numbers, target)


def load_format_rate_by_pid(raw_path: Path, num_problems: int) -> Optional[Dict[int, float]]:
    if not raw_path.exists():
        return None
    ok_counts = Counter()
    total_counts = Counter()
    for rec in load_jsonl(raw_path):
        try:
            pid = int(rec.get("problem_index", -1))
        except Exception:
            continue
        if pid < 0 or pid >= num_problems:
            continue
        total_counts[pid] += 1
        status = rec.get("parse_status")
        if status is None:
            comp = rec.get("completion", "") or ""
            parsed = tolerant_parse_completion(comp)
            feasible_pred = (parsed.get("feasible_pred") or "").strip().lower()
            answer_block = (parsed.get("answer_block") or "").strip()
            text_lower = comp.lower()
            has_feasible = "<feasible>" in text_lower
            has_answer = "<answer>" in text_lower
            status = "OK" if (has_feasible and has_answer and feasible_pred and answer_block) else "BAD"
        if status == "OK":
            ok_counts[pid] += 1
    rates = {}
    for pid, total in total_counts.items():
        rates[pid] = ok_counts.get(pid, 0) / total if total > 0 else 0.0
    return rates


def score_sequences(model, tokenizer, prompts: List[str], targets: List[str], batch_size: int, device: str):
    logps = []
    model.eval()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]
        full_texts = [p + t for p, t in zip(batch_prompts, batch_targets)]

        enc_full = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc_prompt = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)

        input_ids = enc_full["input_ids"].to(device)
        attn = enc_full["attention_mask"].to(device)
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]

        for b in range(input_ids.shape[0]):
            seq_len = int(attn[b].sum().item())
            prompt_len = prompt_lens[b]
            start = max(prompt_len - 1, 0)
            end = seq_len - 2
            if end < start:
                logps.append(float("-inf"))
                continue
            token_logp = log_probs[b, start : end + 1, :].gather(
                -1, labels[b, start : end + 1].unsqueeze(-1)
            ).squeeze(-1)
            logps.append(float(token_logp.sum().item()))
    return logps


def main():
    args = parse_args()
    device = args.device
    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    raw_dir = Path(args.raw_dir)

    # build model list
    model_paths = []
    if args.base_model_path:
        model_paths.append(args.base_model_path)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_paths = sorted(ckpt_dir.glob("global_step_*"))
    only_steps = set()
    if args.only_steps.strip():
        for s in args.only_steps.split(","):
            s = s.strip()
            if s:
                only_steps.add(int(s))
    for p in ckpt_paths:
        if only_steps:
            try:
                st = int(p.name.split("_")[-1])
            except Exception:
                continue
            if st not in only_steps:
                continue
        model_paths.append(str(p))

    if not model_paths:
        raise RuntimeError("No model paths to evaluate")

    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    # enumerate solutions per problem
    sol_sets = {}
    for pid, rec in enumerate(records):
        numbers, target, feasible_label = extract_ground_truth(rec)
        if feasible_label != "yes":
            sol_sets[pid] = []
            continue
        sol_sets[pid] = enumerate_solution_set(numbers, target)

    all_rows = []
    per_problem_rows = []

    for model_path in model_paths:
        ckpt_name = Path(model_path).name
        print(f"[teacher_forced_likelihood] Loading model: {ckpt_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
        ).to(device)

        raw_path = raw_dir / f"countdown_raw_{ckpt_name}_n{args.n_samples}.jsonl"
        format_rate_by_pid = load_format_rate_by_pid(raw_path, args.num_problems)
        format_rate_missing = False
        if format_rate_by_pid is None:
            format_rate_missing = True
            format_rate_by_pid = {pid: 1.0 for pid in range(args.num_problems)}
            print(f"[teacher_forced_likelihood] WARNING: format rate raw not found for {ckpt_name}: {raw_path}")

        # build prompts & targets for all solutions (conditioned on tags)
        prompts = []
        targets = []
        meta = []  # (pid, expr)
        for pid, rec in enumerate(records):
            prompt_content = rec.get("prompt")
            if hasattr(prompt_content, "tolist"):
                prompt_content = prompt_content.tolist()
            if isinstance(prompt_content, list):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_content, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = str(prompt_content)
            prompt_base = prompt_text + args.answer_tag_prefix
            sols = sol_sets.get(pid, [])
            for expr in sols:
                target_text = expr + ("</answer>" if args.include_answer_close else "")
                prompts.append(prompt_base)
                targets.append(target_text)
                meta.append((pid, expr))

        if not prompts:
            logps = []
        else:
            logps = score_sequences(model, tokenizer, prompts, targets, args.batch_size, device)

        # group logps by problem
        by_prob = defaultdict(list)
        for (pid, expr), lp in zip(meta, logps):
            by_prob[pid].append((expr, lp))

        support_mass_cond_list = []
        support_mass_list = []
        support_log_cond_list = []
        support_log_list = []
        entropy_list = []
        gini_list = []
        top1_list = []
        sol_count_list = []
        format_rate_list = []
        oracle_passk = {k: [] for k in k_list}
        oracle_passk_cond = {k: [] for k in k_list}

        for pid in range(args.num_problems):
            expr_lps = by_prob.get(pid, [])
            lps = [lp for _, lp in expr_lps]
            sol_count = len(sol_sets.get(pid, []))
            if sol_count == 0 or not lps:
                support_log_cond = float("-inf")
                support_mass_cond = 0.0
                probs = []
            else:
                support_log_cond = logsumexp(lps)
                support_mass_cond = math.exp(support_log_cond) if support_log_cond > -1000 else 0.0
                probs = [math.exp(lp - support_log_cond) for lp in lps]

            format_rate = format_rate_by_pid.get(pid, 0.0)
            support_mass = format_rate * support_mass_cond
            support_log = math.log(support_mass) if support_mass > 0 else float("-inf")

            support_mass_cond_list.append(support_mass_cond)
            support_mass_list.append(support_mass)
            support_log_cond_list.append(support_log_cond)
            support_log_list.append(support_log)
            sol_count_list.append(sol_count)
            format_rate_list.append(format_rate)

            if probs:
                entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                top1 = max(probs)
                gini = gini_from_probs(probs)
            else:
                entropy = 0.0
                top1 = 0.0
                gini = 0.0

            entropy_list.append(entropy)
            top1_list.append(top1)
            gini_list.append(gini)

            for k in k_list:
                oracle = 1.0 - (1.0 - support_mass) ** k
                oracle_passk[k].append(oracle)
                oracle_cond = 1.0 - (1.0 - support_mass_cond) ** k
                oracle_passk_cond[k].append(oracle_cond)

            if args.save_per_problem:
                per_problem_rows.append(dict(
                    checkpoint=ckpt_name,
                    problem_index=pid,
                    solution_count=sol_count,
                    format_rate=format_rate,
                    support_mass_cond=support_mass_cond,
                    support_mass_cond_log=support_log_cond,
                    support_mass=support_mass,
                    support_mass_log=support_log,
                    solution_entropy_cond=entropy,
                    solution_gini_cond=gini,
                    top1_solution_mass_cond=top1,
                ))

        row = dict(
            checkpoint=ckpt_name,
            num_problems=args.num_problems,
            format_rate_missing=format_rate_missing,
            format_rate_mean=float(np.mean(format_rate_list)) if format_rate_list else float("nan"),
            solution_count_mean=float(np.mean(sol_count_list)) if sol_count_list else float("nan"),
            support_mass_cond_mean=float(np.mean(support_mass_cond_list)) if support_mass_cond_list else float("nan"),
            support_mass_mean=float(np.mean(support_mass_list)) if support_mass_list else float("nan"),
            support_mass_cond_log_mean=float(np.mean(support_log_cond_list)) if support_log_cond_list else float("nan"),
            support_mass_log_mean=float(np.mean(support_log_list)) if support_log_list else float("nan"),
            solution_entropy_cond_mean=float(np.mean(entropy_list)) if entropy_list else float("nan"),
            solution_gini_cond_mean=float(np.mean(gini_list)) if gini_list else float("nan"),
            top1_solution_mass_cond_mean=float(np.mean(top1_list)) if top1_list else float("nan"),
        )
        for k in k_list:
            row[f"oracle_pass@{k}_mean"] = float(np.mean(oracle_passk[k])) if oracle_passk[k] else float("nan")
            row[f"oracle_pass@{k}_cond_mean"] = float(np.mean(oracle_passk_cond[k])) if oracle_passk_cond[k] else float("nan")
        all_rows.append(row)

        # free
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df["step"] = df["checkpoint"].apply(step_of)
    df = df.sort_values("step").reset_index(drop=True)

    out_csv = OUT_DIR / "teacher_forced_likelihood_solution_logprob.csv"
    df.to_csv(out_csv, index=False)
    print(f"[teacher_forced_likelihood] wrote: {out_csv}")
    print(df)

    if args.save_per_problem:
        out_parq = OUT_DIR / "teacher_forced_likelihood_solution_logprob_per_problem.parquet"
        pd.DataFrame(per_problem_rows).to_parquet(out_parq, index=False)
        print(f"[teacher_forced_likelihood] wrote per-problem: {out_parq}")


if __name__ == "__main__":
    main()
