#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""phase_aware_mixing - Training-free logit mixing.
Hypothesis: a small amount of source-model logit mass can recover alternative valid branches without parameter updates.
Inputs: source and RL checkpoint model paths; dataset/test.parquet; mixing schedule and sampling settings.
Outputs: data/analysis_results/rlvr_passk/metrics/phase_aware_mixing_logit_mix_summary_source_ablation.csv; data/analysis_results/rlvr_passk/metrics/phase_aware_mixing_logit_mix_per_problem_expu_source_ablation_base_g1.parquet
Status: paper-appendix
"""
from __future__ import annotations
"""
phase_aware_mixing_logit_mix_countdown.py

Experiment U — Training-Free Logit Mixing for Diversity Recovery
=================================================================

Mix two models' next-token logits at inference time:

    logit_mixed = (1 - α(t)) * logit_RL + α(t) * logit_source

Implemented strategies:
  1. global
  2. phase_aware
  3. answer_only
  4. token_mask
  5. entropy_adaptive

Outputs:
  metrics/phase_aware_mixing_logit_mix_summary_{tag}.csv
  metrics/phase_aware_mixing_logit_mix_per_problem_{tag}.parquet
  raw/phase_aware_mixing_logit_mix_raw_{tag}_{variant}.jsonl
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (  # noqa: E402
    COUNTDOWN_DATA_ROOT as ANALYSIS_ROOT,
    METRICS_DIR,
    RAW_DIR,
    TEST_PARQUET,
)

RAW_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_utils import (  # noqa: E402
    evaluate_countdown_expression,
    parse_countdown_completion,
)
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    bootstrap_ci_mean,
    build_prompt_text,
    entropy_from_counts,
    enumerate_solution_set,
    evaluate_countdown_completion,
    extract_ground_truth,
    get_prompt_content,
    load_parquet_sorted,
    pass_at_k,
)

SEED = 42
EPS = 1e-8
EOS_STRINGS = ["<|endoftext|>", "<|im_end|>"]
FORMAT_TOKEN_PATTERNS = [
    "<think>",
    "</think>",
    "<feasible>",
    "</feasible>",
    "<answer>",
    "</answer>",
    "yes",
    "no",
    "NO_SOLUTION",
    "<|im_end|>",
    "<|endoftext|>",
]
VALID_STRATEGIES = {
    "global",
    "phase_aware",
    "answer_only",
    "token_mask",
    "entropy_adaptive",
}


@dataclass(frozen=True)
class PromptExample:
    pid: int
    prompt_text: str
    prompt_ids: List[int]
    numbers: List[int]
    target: int
    feasible_label: str
    solution_set: Set[str]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    strategy: str
    alpha: float
    alpha_think: float
    alpha_answer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExpU: training-free logit mixing")
    parser.add_argument("--rl_model_path", type=str, required=True)
    parser.add_argument("--source_model_path", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.5,0.7")
    parser.add_argument(
        "--strategies",
        type=str,
        default="global,answer_only,token_mask,entropy_adaptive",
    )
    parser.add_argument("--alpha_think", type=float, default=0.0)
    parser.add_argument("--alpha_answer", type=float, default=0.5)
    parser.add_argument("--entropy_scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="Single GPU id used by this process; use multiple processes to exploit multiple GPUs.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of problems generated in parallel; total sequence batch = batch_size * n_samples.",
    )
    parser.add_argument("--tag", type=str, default="logit_mix_v1")
    parser.add_argument("--save_per_problem", action="store_true", default=False)
    parser.add_argument("--save_raw", action="store_true", default=False)
    parser.add_argument("--skip_controls", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_strategy_list(text: str) -> List[str]:
    strategies = [part.strip() for part in text.split(",") if part.strip()]
    invalid = [name for name in strategies if name not in VALID_STRATEGIES]
    if invalid:
        raise ValueError(f"Unknown strategies: {invalid}")
    return strategies


def single_gpu_id(gpu_arg: str) -> str:
    gpu_ids = [part.strip() for part in gpu_arg.split(",") if part.strip()]
    if len(gpu_ids) != 1:
        raise ValueError(
            "ExpU runs one process per GPU. Pass a single --gpu_id and parallelize jobs externally."
        )
    return gpu_ids[0]


def build_format_token_set(tokenizer) -> Set[int]:
    token_ids: Set[int] = set()
    for pattern in FORMAT_TOKEN_PATTERNS:
        for text in (pattern, f" {pattern}"):
            token_ids.update(tokenizer.encode(text, add_special_tokens=False))
    for token_id in (tokenizer.eos_token_id, tokenizer.bos_token_id):
        if token_id is not None:
            token_ids.add(token_id)
    return token_ids


def validate_tokenizer_compatibility(rl_tokenizer, source_tokenizer, probe_text: str) -> None:
    if len(rl_tokenizer) != len(source_tokenizer):
        raise ValueError(
            f"Tokenizer vocab mismatch: RL={len(rl_tokenizer)} source={len(source_tokenizer)}"
        )
    for attr in ("eos_token_id", "bos_token_id"):
        if getattr(rl_tokenizer, attr, None) != getattr(source_tokenizer, attr, None):
            raise ValueError(f"Tokenizer special token mismatch on {attr}")
    probe_strings = FORMAT_TOKEN_PATTERNS + [probe_text]
    for text in probe_strings:
        rl_ids = rl_tokenizer.encode(text, add_special_tokens=False)
        source_ids = source_tokenizer.encode(text, add_special_tokens=False)
        if rl_ids != source_ids:
            snippet = text if len(text) < 120 else text[:120] + "..."
            raise ValueError(f"Tokenizer encoding mismatch for probe text: {snippet}")


def build_prompt_data(records: Sequence[dict], tokenizer) -> List[PromptExample]:
    prompt_data: List[PromptExample] = []
    for pid, record in enumerate(records):
        prompt_content = get_prompt_content(record)
        prompt_text = build_prompt_text(prompt_content, tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        numbers, target, feasible_label = extract_ground_truth(record)
        solution_set = enumerate_solution_set(numbers, target) if feasible_label == "yes" else set()
        prompt_data.append(
            PromptExample(
                pid=pid,
                prompt_text=prompt_text,
                prompt_ids=prompt_ids,
                numbers=numbers,
                target=target,
                feasible_label=feasible_label,
                solution_set=solution_set,
            )
        )
    return prompt_data


class PhaseTracker:
    THINK = 0
    FORMAT = 1
    ANSWER = 2
    DONE = 3

    def __init__(self, batch_size: int, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._phases = [self.THINK] * batch_size
        self._decoded = [""] * batch_size

    def update(self, token_ids: torch.Tensor, finished: torch.Tensor) -> None:
        ids = token_ids.detach().cpu().tolist()
        finished_list = finished.detach().cpu().tolist()
        for idx, token_id in enumerate(ids):
            if finished_list[idx]:
                continue
            token_text = self._tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            self._decoded[idx] += token_text
            self._phases[idx] = self._phase_from_text(self._decoded[idx].lower())

    def get_phases(self) -> List[int]:
        return list(self._phases)

    def done_mask(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([phase == self.DONE for phase in self._phases], device=device)

    @staticmethod
    def _phase_from_text(text: str) -> int:
        if "</answer>" in text:
            return PhaseTracker.DONE
        if "<answer>" in text:
            return PhaseTracker.ANSWER
        if "<feasible>" in text:
            return PhaseTracker.FORMAT
        return PhaseTracker.THINK


def build_format_mask(vocab_size: int, format_token_ids: Set[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for token_id in format_token_ids:
        if 0 <= token_id < vocab_size:
            mask[token_id] = True
    return mask


def apply_format_mask(
    mixed_logits: torch.Tensor,
    logits_rl: torch.Tensor,
    format_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if format_mask is None:
        return mixed_logits
    mixed_logits[:, format_mask] = logits_rl[:, format_mask]
    return mixed_logits


def compute_mixed_logits(
    logits_rl: Optional[torch.Tensor],
    logits_source: Optional[torch.Tensor],
    variant: VariantConfig,
    phase_tracker: PhaseTracker,
    format_mask: Optional[torch.Tensor],
    entropy_scale: float,
) -> torch.Tensor:
    if logits_rl is None:
        return logits_source
    if logits_source is None:
        return logits_rl
    if variant.strategy == "global":
        return (1.0 - variant.alpha) * logits_rl + variant.alpha * logits_source
    if variant.strategy == "token_mask":
        mixed = (1.0 - variant.alpha) * logits_rl + variant.alpha * logits_source
        return apply_format_mask(mixed, logits_rl, format_mask)
    if variant.strategy == "entropy_adaptive":
        log_probs = F.log_softmax(logits_rl, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        max_entropy = math.log(logits_rl.shape[-1])
        alpha = variant.alpha * torch.clamp(entropy_scale * entropy / max_entropy, max=1.0)
        mixed = (1.0 - alpha) * logits_rl + alpha * logits_source
        return apply_format_mask(mixed, logits_rl, format_mask)
    phases = phase_tracker.get_phases()
    alpha = torch.zeros(logits_rl.shape[0], 1, device=logits_rl.device, dtype=logits_rl.dtype)
    for idx, phase in enumerate(phases):
        if phase == PhaseTracker.THINK and variant.strategy == "phase_aware":
            alpha[idx, 0] = variant.alpha_think
        elif phase == PhaseTracker.ANSWER:
            alpha[idx, 0] = variant.alpha_answer
    mixed = (1.0 - alpha) * logits_rl + alpha * logits_source
    return apply_format_mask(mixed, logits_rl, format_mask)


def left_pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    max_len = max(len(seq) for seq in sequences)
    padded = []
    masks = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append([pad_id] * pad_len + list(seq))
        masks.append([0] * pad_len + [1] * len(seq))
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        max_len,
    )


def expand_prompt_batch(prompt_batch: Sequence[PromptExample], n_samples: int) -> Tuple[List[List[int]], List[int]]:
    sequences: List[List[int]] = []
    owners: List[int] = []
    for owner, item in enumerate(prompt_batch):
        for _ in range(n_samples):
            sequences.append(item.prompt_ids)
            owners.append(owner)
    return sequences, owners


def forward_last_logits(
    model,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
) -> Tuple[torch.Tensor, Any]:
    if past_key_values is None:
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=True)
    else:
        outputs = model(
            input_ids=generated_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return outputs.logits[:, -1, :], outputs.past_key_values


def sample_next_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    scaled = logits / temperature
    if top_p >= 1.0:
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scaled, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        raise RuntimeError("NaN/Inf detected in sampling probabilities")
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def decode_generated(
    generated_ids: torch.Tensor,
    prompt_width: int,
    tokenizer,
) -> List[str]:
    eos_id = tokenizer.eos_token_id
    outputs: List[str] = []
    for token_ids in generated_ids[:, prompt_width:].detach().cpu().tolist():
        if eos_id is not None and eos_id in token_ids:
            token_ids = token_ids[: token_ids.index(eos_id)]
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for eos_text in EOS_STRINGS:
            if eos_text in text:
                text = text.split(eos_text)[0]
        outputs.append(text.strip())
    return outputs


@torch.no_grad()
def generate_mixed_batch(
    rl_model,
    source_model,
    tokenizer,
    prompt_batch: Sequence[PromptExample],
    variant: VariantConfig,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    format_token_ids: Set[int],
    entropy_scale: float,
    device: torch.device,
) -> List[List[str]]:
    prompt_sequences, owners = expand_prompt_batch(prompt_batch, n_samples)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    generated_ids, attention_mask, prompt_width = left_pad_sequences(prompt_sequences, pad_id, device)
    phase_tracker = PhaseTracker(len(prompt_sequences), tokenizer)
    finished = torch.zeros(len(prompt_sequences), dtype=torch.bool, device=device)
    format_mask = build_format_mask(rl_model.config.vocab_size, format_token_ids, device)
    use_rl = not (variant.strategy == "global" and variant.alpha >= 1.0 - EPS)
    use_source = not (variant.strategy == "global" and variant.alpha <= EPS)
    rl_past = None
    source_past = None
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        if bool(finished.all()):
            break
        logits_rl = None
        logits_source = None
        if use_rl:
            logits_rl, rl_past = forward_last_logits(rl_model, generated_ids, attention_mask, rl_past)
        if use_source:
            logits_source, source_past = forward_last_logits(
                source_model,
                generated_ids,
                attention_mask,
                source_past,
            )
        mixed_logits = compute_mixed_logits(
            logits_rl=logits_rl,
            logits_source=logits_source,
            variant=variant,
            phase_tracker=phase_tracker,
            format_mask=format_mask if variant.strategy in {"phase_aware", "answer_only", "token_mask", "entropy_adaptive"} else None,
            entropy_scale=entropy_scale,
        )
        if finished.any():
            mixed_logits = mixed_logits.clone()
            mixed_logits[finished] = float("-inf")
            mixed_logits[finished, pad_id] = 0.0
        next_tokens = sample_next_tokens(mixed_logits, temperature=temperature, top_p=top_p)
        finished_before = finished.clone()
        phase_tracker.update(next_tokens, finished)
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
        new_mask = (~finished_before).long().unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
        if eos_id is not None:
            finished |= next_tokens.eq(eos_id)
        finished |= phase_tracker.done_mask(device)

    decoded = decode_generated(generated_ids, prompt_width, tokenizer)
    grouped = [[] for _ in prompt_batch]
    for text, owner in zip(decoded, owners):
        grouped[owner].append(text)
    return grouped


def evaluate_completions(
    completions: Sequence[str],
    numbers: List[int],
    target: int,
    feasible_label: str,
    solution_set: Set[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = len(completions)
    correct = 0
    format_ok = 0
    solution_counter: Counter = Counter()
    raw_rows: List[Dict[str, Any]] = []
    for sample_index, completion in enumerate(completions):
        eval_result = evaluate_countdown_completion(
            text=completion,
            numbers=numbers,
            target=target,
            feasible_label=feasible_label,
            parse_countdown_completion=parse_countdown_completion,
            evaluate_countdown_expression=evaluate_countdown_expression,
        )
        if eval_result.overall_ok:
            correct += 1
        if eval_result.parse_status == "OK":
            format_ok += 1
        if eval_result.canonical_expr and eval_result.canonical_expr in solution_set:
            solution_counter[eval_result.canonical_expr] += 1
        raw_rows.append(
            dict(
                sample_index=sample_index,
                completion=completion,
                overall_ok=bool(eval_result.overall_ok),
                feasible_ok=bool(eval_result.feasible_ok),
                expr_ok=bool(eval_result.expr_ok),
                parse_status=eval_result.parse_status,
                canonical_expr=eval_result.canonical_expr,
            )
        )
    solution_count = len(solution_set)
    metrics = dict(
        n=n,
        correct=correct,
        correct_mass=(correct / n) if n else 0.0,
        unique_solution=len(solution_counter),
        coverage=(len(solution_counter) / solution_count) if solution_count else 0.0,
        top1_sol_mass=(max(solution_counter.values()) / n) if solution_counter and n else 0.0,
        solution_entropy=entropy_from_counts(solution_counter) if solution_counter else 0.0,
        format_ok=format_ok,
        format_rate=(format_ok / n) if n else 0.0,
        solution_count=solution_count,
    )
    return metrics, raw_rows


def build_variants(args: argparse.Namespace, alphas: Sequence[float], strategies: Sequence[str]) -> List[VariantConfig]:
    variants: List[VariantConfig] = []
    if not args.skip_controls:
        variants.append(VariantConfig("pure_rl", "global", 0.0, 0.0, 0.0))
        variants.append(VariantConfig("pure_source", "global", 1.0, 0.0, 1.0))
    for strategy in strategies:
        if strategy == "phase_aware":
            seen = set()
            for answer_alpha in [args.alpha_answer, *alphas]:
                if round(answer_alpha, 8) in seen:
                    continue
                seen.add(round(answer_alpha, 8))
                variants.append(
                    VariantConfig(
                        name=f"phase_aware_t{args.alpha_think:.2f}_a{answer_alpha:.2f}",
                        strategy="phase_aware",
                        alpha=answer_alpha,
                        alpha_think=args.alpha_think,
                        alpha_answer=answer_alpha,
                    )
                )
            continue
        for alpha in alphas:
            variants.append(
                VariantConfig(
                    name=f"{strategy}_a{alpha:.2f}",
                    strategy=strategy,
                    alpha=alpha,
                    alpha_think=0.0,
                    alpha_answer=alpha,
                )
            )
    return variants


def aggregate_variant_metrics(
    per_problem_rows: Sequence[Dict[str, Any]],
    correct_counts: Dict[int, int],
    attempt_counts: Dict[int, int],
) -> Dict[str, float]:
    aggregate: Dict[str, float] = {"n_problems": len(per_problem_rows)}
    metric_keys = [
        "correct_mass",
        "coverage",
        "unique_solution",
        "top1_sol_mass",
        "solution_entropy",
        "format_rate",
    ]
    for key in metric_keys:
        values = [row[key] for row in per_problem_rows]
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
    for k in (1, 4, 16, 64):
        values = []
        for pid, correct in correct_counts.items():
            attempts = attempt_counts.get(pid, 0)
            if attempts >= k and attempts > 0:
                values.append(pass_at_k(attempts, correct, k))
        if values:
            mean, lo, hi = bootstrap_ci_mean(values)
            aggregate[f"pass@{k}"] = mean
            aggregate[f"pass@{k}_ci_lo"] = lo
            aggregate[f"pass@{k}_ci_hi"] = hi
    return aggregate


def write_raw_rows(raw_file, variant: VariantConfig, problem_index: int, raw_rows: Iterable[Dict[str, Any]]) -> None:
    for row in raw_rows:
        payload = {"variant": variant.name, "strategy": variant.strategy, "problem_index": problem_index, **row}
        raw_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_variant(
    variant: VariantConfig,
    prompt_data: Sequence[PromptExample],
    rl_model,
    source_model,
    tokenizer,
    args: argparse.Namespace,
    format_token_ids: Set[int],
    device: torch.device,
    rl_name: str,
    source_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print(f"\n[phase_aware_mixing] Running variant: {variant.name} (strategy={variant.strategy})")
    per_problem_rows: List[Dict[str, Any]] = []
    correct_counts: Dict[int, int] = {}
    attempt_counts: Dict[int, int] = {}
    raw_path = RAW_DIR / f"phase_aware_mixing_logit_mix_raw_{args.tag}_{variant.name}.jsonl"
    raw_file = raw_path.open("w", encoding="utf-8") if args.save_raw else None
    try:
        for batch_start in range(0, len(prompt_data), args.batch_size):
            batch = prompt_data[batch_start : batch_start + args.batch_size]
            grouped = generate_mixed_batch(
                rl_model=rl_model,
                source_model=source_model,
                tokenizer=tokenizer,
                prompt_batch=batch,
                variant=variant,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                format_token_ids=format_token_ids,
                entropy_scale=args.entropy_scale,
                device=device,
            )
            for item, completions in zip(batch, grouped):
                metrics, raw_rows = evaluate_completions(
                    completions,
                    item.numbers,
                    item.target,
                    item.feasible_label,
                    item.solution_set,
                )
                correct_counts[item.pid] = metrics["correct"]
                attempt_counts[item.pid] = metrics["n"]
                per_problem_rows.append({"problem_index": item.pid, **metrics})
                if raw_file is not None:
                    write_raw_rows(raw_file, variant, item.pid, raw_rows)
            done = min(batch_start + len(batch), len(prompt_data))
            if done % 10 == 0 or done == len(prompt_data):
                print(f"  [{variant.name}] {done}/{len(prompt_data)} problems done")
    finally:
        if raw_file is not None:
            raw_file.close()
    aggregate = aggregate_variant_metrics(per_problem_rows, correct_counts, attempt_counts)
    summary_row = dict(
        variant=variant.name,
        rl_model=rl_name,
        source_model=source_name,
        alpha=variant.alpha,
        alpha_think=variant.alpha_think,
        alpha_answer=variant.alpha_answer,
        strategy=variant.strategy,
        n_samples=args.n_samples,
        **aggregate,
    )
    pass1 = aggregate.get("pass@1", float("nan"))
    pass64 = aggregate.get("pass@64", float("nan"))
    coverage = aggregate.get("coverage_mean", float("nan"))
    format_rate = aggregate.get("format_rate_mean", float("nan"))
    print(
        f"  [{variant.name}] pass@1={pass1:.4f} pass@64={pass64:.4f} "
        f"coverage={coverage:.4f} format_rate={format_rate:.4f}"
    )
    return summary_row, per_problem_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu_id = single_gpu_id(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ExpU")
    device = torch.device("cuda")
    dtype_torch = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    alphas = parse_float_list(args.alphas)
    strategies = parse_strategy_list(args.strategies)
    rl_name = Path(args.rl_model_path).name
    source_name = Path(args.source_model_path).name
    tag = args.tag or f"mix_{rl_name}_with_{source_name}"
    args.tag = tag

    print(f"[phase_aware_mixing] RL model: {rl_name}")
    print(f"[phase_aware_mixing] Source model: {source_name}")
    print(f"[phase_aware_mixing] Strategies: {strategies}")
    print(f"[phase_aware_mixing] Alphas: {alphas}")
    print(f"[phase_aware_mixing] GPU: {gpu_id}")

    # The local Qwen tokenizer.json is rejected by the current fast-tokenizer
    # backend in this environment.  Slow tokenization is deterministic and
    # keeps the two model paths on the same compatibility check.
    rl_tokenizer = AutoTokenizer.from_pretrained(args.rl_model_path, trust_remote_code=True, use_fast=False)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_path, trust_remote_code=True, use_fast=False)
    for tokenizer in (rl_tokenizer, source_tokenizer):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_data = build_prompt_data(records, rl_tokenizer)
    probe_text = prompt_data[0].prompt_text if prompt_data else "<answer>1+2</answer>"
    validate_tokenizer_compatibility(rl_tokenizer, source_tokenizer, probe_text)
    format_token_ids = build_format_token_set(rl_tokenizer)
    print(f"[phase_aware_mixing] Format token IDs: {len(format_token_ids)}")

    print(f"[phase_aware_mixing] Loading RL model: {args.rl_model_path}")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.rl_model_path,
        dtype=dtype_torch,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    print(f"[phase_aware_mixing] Loading source model: {args.source_model_path}")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model_path,
        dtype=dtype_torch,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    rl_model.eval()
    source_model.eval()
    if rl_model.config.vocab_size != source_model.config.vocab_size:
        raise ValueError(
            f"Model vocab mismatch: RL={rl_model.config.vocab_size} source={source_model.config.vocab_size}"
        )

    variants = build_variants(args, alphas, strategies)
    print(f"[phase_aware_mixing] Total variants: {len(variants)}")
    summary_rows: List[Dict[str, Any]] = []
    per_problem_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    for variant in variants:
        summary_row, variant_problem_rows = run_variant(
            variant=variant,
            prompt_data=prompt_data,
            rl_model=rl_model,
            source_model=source_model,
            tokenizer=rl_tokenizer,
            args=args,
            format_token_ids=format_token_ids,
            device=device,
            rl_name=rl_name,
            source_name=source_name,
        )
        summary_rows.append(summary_row)
        if args.save_per_problem:
            for row in variant_problem_rows:
                per_problem_rows.append(
                    {
                        "variant": variant.name,
                        "strategy": variant.strategy,
                        "alpha": variant.alpha,
                        "alpha_think": variant.alpha_think,
                        "alpha_answer": variant.alpha_answer,
                        **row,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = METRICS_DIR / f"phase_aware_mixing_logit_mix_summary_{tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[phase_aware_mixing] Saved summary -> {summary_path}")
    display_cols = [
        "variant",
        "strategy",
        "alpha",
        "alpha_think",
        "alpha_answer",
        "format_rate_mean",
        "correct_mass_mean",
        "coverage_mean",
        "unique_solution_mean",
        "pass@1",
        "pass@64",
    ]
    display_cols = [col for col in display_cols if col in summary_df.columns]
    print("\n=== Key Results ===")
    print(summary_df[display_cols].to_string(index=False))
    if args.save_per_problem and per_problem_rows:
        per_problem_df = pd.DataFrame(per_problem_rows)
        per_problem_path = METRICS_DIR / f"phase_aware_mixing_logit_mix_per_problem_{tag}.parquet"
        per_problem_df.to_parquet(per_problem_path, index=False)
        print(f"[phase_aware_mixing] Saved per-problem -> {per_problem_path}")
    elapsed = time.time() - start_time
    print(f"[phase_aware_mixing] Total runtime: {elapsed / 60.0:.2f} min")
    del rl_model, source_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[phase_aware_mixing] Done.")


if __name__ == "__main__":
    main()
