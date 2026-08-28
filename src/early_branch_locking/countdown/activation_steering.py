#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""activation_steering - Phase-aware activation steering.
Hypothesis: isotropic or PCA-directed residual perturbations in late layers can restore branch diversity at inference time.
Inputs: checkpoint model path; dataset/test.parquet; steering layer, noise and sampling settings.
Outputs: data/analysis_results/rlvr_passk/metrics/activation_steering_isotropic_grid_summary.csv; data/analysis_results/rlvr_passk/metrics/activation_steering_activation_steering_summary_FINAL_ISO_L28to32_a0.3_N256.csv
Status: paper-appendix
"""
from __future__ import annotations
"""Experiment V: phase-aware activation steering on Countdown."""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking.core.countdown_shared import load_parquet_sorted
from early_branch_locking.core.countdown_shared import step_of
from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR as ACTOR_DIR, METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402
from early_branch_locking.core.prompt_data import (
    build_prompt_data,
    ensure_tokenizer_padding,
    parse_layer_list,
    single_gpu_id,
    torch_dtype_from_name,
)
from early_branch_locking.core.op1_utils import get_layers_container

SEED = 42
PASS_KS = (1, 4, 16, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp V: dynamic activation steering")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--steer_layers", type=str, default="28,29,30,31,32")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default="isotropic", choices=["isotropic", "pca_dir", "contrastive_mean"])
    parser.add_argument("--pca_vector_path", type=str, default="")
    parser.add_argument("--pca_rank", type=int, default=5)
    parser.add_argument("--max_steer_tokens", type=int, default=-1)
    parser.add_argument("--alpha_decay", type=str, default="none", choices=VALID_ALPHA_DECAYS)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--disable_phase_gating", action="store_true", default=False)
    parser.add_argument("--save_per_problem", action="store_true", default=False)
    parser.add_argument("--save_raw", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def aggregate_all(problem_rows: List[dict]) -> Dict[str, float]:
    valid_ks = [k for k in PASS_KS if k <= max(row["attempt_count"] for row in problem_rows)]
    summary = {}
    for key in ("correct_count", "feasible_ok_count", "expr_ok_count"):
        summary.update(aggregate_problem_rows(problem_rows, key, valid_ks))
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu_id = single_gpu_id(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Exp V.")
    device = torch.device("cuda")
    dtype = torch_dtype_from_name(args.dtype)
    steer_layers = parse_layer_list(args.steer_layers)
    tag = args.tag or f"{args.noise_type}_{Path(args.model_path).name}_a{args.alpha:g}"
    print(f"[activation_steering] model={Path(args.model_path).name} gpu={gpu_id} noise={args.noise_type} alpha={args.alpha}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ensure_tokenizer_padding(tokenizer)
    records_all = load_parquet_sorted(TEST_PARQUET, n=None, sort_key="sample_id")
    start_index = max(args.start_index, 0)
    end_index = len(records_all) if args.end_index is None or args.end_index < 0 else min(args.end_index, len(records_all))
    if end_index <= start_index:
        raise ValueError(f"Invalid problem slice: start={start_index}, end={end_index}")
    records = records_all[start_index:end_index]
    if args.num_problems > 0:
        records = records[: args.num_problems]
    prompt_data = build_prompt_data(records, tokenizer)
    vector_bank = None
    if args.noise_type != "isotropic":
        if not args.pca_vector_path:
            raise ValueError(f"{args.noise_type} requires --pca_vector_path.")
        vector_bank = load_vector_bank(args.pca_vector_path, device=device, dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    layer_count = len(get_layers_container(model))
    invalid = [layer for layer in steer_layers if layer < 0 or layer >= layer_count]
    if invalid:
        raise ValueError(f"Invalid steer layers {invalid}; model has {layer_count} layers.")
    config = SteeringConfig(
        alpha=args.alpha,
        noise_type=args.noise_type,
        pca_rank=args.pca_rank,
        disable_phase_gating=args.disable_phase_gating,
        max_steer_tokens=args.max_steer_tokens,
        alpha_decay=args.alpha_decay,
    )
    problem_rows: List[dict] = []
    raw_rows: List[dict] = []
    start_time = time.time()
    for batch_start in range(0, len(prompt_data), args.batch_size):
        batch = prompt_data[batch_start : batch_start + args.batch_size]
        grouped, token_counts = generate_steered_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_batch=batch,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            steer_layers=steer_layers,
            config=config,
            bank=vector_bank,
            device=device,
        )
        for item, completions, counts in zip(batch, grouped, token_counts):
            row, item_raw_rows = evaluate_problem(item, completions, counts)
            problem_rows.append(row)
            raw_rows.extend(item_raw_rows)
        done = min(batch_start + len(batch), len(prompt_data))
        if done % 10 == 0 or done == len(prompt_data):
            print(f"[activation_steering] processed {done}/{len(prompt_data)} problems", flush=True)
    summary = aggregate_all(problem_rows)
    summary_row = {
        "tag": tag,
        "variant": f"{args.noise_type}_L{args.steer_layers}_a{args.alpha:g}",
        "strategy": args.noise_type,
        "checkpoint": Path(args.model_path).name,
        "model": Path(args.model_path).name,
        "step": step_of(Path(args.model_path).name),
        "noise_type": args.noise_type,
        "steer_layers": args.steer_layers,
        "alpha": args.alpha,
        "pca_rank": args.pca_rank,
        "max_steer_tokens": args.max_steer_tokens,
        "alpha_decay": args.alpha_decay,
        "gating_mode": "ungated" if args.disable_phase_gating else "phase_aware",
        "n_samples": args.n_samples,
        "start_index": start_index,
        "end_index": start_index + len(problem_rows),
        "num_problems": len(problem_rows),
        "n_problems": len(problem_rows),
        "vector_path": args.pca_vector_path,
        **summary,
    }
    summary_df = pd.DataFrame([summary_row])
    summary_path = METRICS_DIR / f"activation_steering_activation_steering_summary_{tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[activation_steering] saved summary -> {summary_path}", flush=True)
    if args.save_per_problem:
        per_problem_df = pd.DataFrame(problem_rows)
        per_problem_path = METRICS_DIR / f"activation_steering_activation_steering_per_problem_{tag}.parquet"
        per_problem_df.to_parquet(per_problem_path, index=False)
        print(f"[activation_steering] saved per-problem -> {per_problem_path}", flush=True)
    if args.save_raw:
        raw_path = RAW_DIR / f"activation_steering_activation_steering_raw_{tag}.jsonl"
        write_raw_rows(
            raw_path,
            [
                {
                    **row,
                    "variant": summary_row["variant"],
                    "strategy": args.noise_type,
                    "checkpoint": summary_row["checkpoint"],
                    "model": summary_row["model"],
                }
                for row in raw_rows
            ],
        )
        print(f"[activation_steering] saved raw -> {raw_path}", flush=True)
    display_cols = [col for col in ("variant", "gating_mode", "format_rate_mean", "correct_mass_mean", "coverage_mean", "pass@1", "pass@64", "pass@256") if col in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False), flush=True)
    elapsed = time.time() - start_time
    print(f"[activation_steering] runtime={elapsed / 60.0:.2f} min", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---- merged runtime mode ----
"""Runtime helpers for Exp V activation steering."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from early_branch_locking.core.prompt_data import PromptExample
from early_branch_locking.core.op1_utils import get_layers_container

EOS_STRINGS = ("<|endoftext|>", "<|im_end|>")
UNLIMITED_STEER_TOKENS = -1
ALPHA_DECAY_NONE = "none"
ALPHA_DECAY_LINEAR = "linear"
VALID_ALPHA_DECAYS = (ALPHA_DECAY_NONE, ALPHA_DECAY_LINEAR)


@dataclass(frozen=True)
class SteeringVectorBank:
    mean_delta: Optional[torch.Tensor]
    principal_components: Optional[torch.Tensor]
    explained_variance: Optional[torch.Tensor]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SteeringConfig:
    alpha: float
    noise_type: str
    pca_rank: int
    disable_phase_gating: bool
    max_steer_tokens: int = UNLIMITED_STEER_TOKENS
    alpha_decay: str = ALPHA_DECAY_NONE


def load_vector_bank(path: str, device: torch.device, dtype: torch.dtype) -> SteeringVectorBank:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return SteeringVectorBank(payload.to(device=device, dtype=dtype), None, None, {})
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported steering payload type: {type(payload)!r}")
    mean_delta = payload["mean_delta"] if "mean_delta" in payload else payload.get("delta_mean", payload.get("vector"))
    components = payload["principal_components"] if "principal_components" in payload else payload.get("components")
    variance = payload["explained_variance"] if "explained_variance" in payload else payload.get("component_variance")
    return SteeringVectorBank(
        mean_delta=None if mean_delta is None else mean_delta.to(device=device, dtype=dtype),
        principal_components=None if components is None else components.to(device=device, dtype=dtype),
        explained_variance=None if variance is None else variance.to(device=device, dtype=dtype),
        metadata={key: value for key, value in payload.items() if not torch.is_tensor(value)},
    )


def left_pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    max_len = max(len(seq) for seq in sequences)
    padded = [[pad_id] * (max_len - len(seq)) + list(seq) for seq in sequences]
    masks = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in sequences]
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        max_len,
    )


def sample_next_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    scaled = logits / max(temperature, 1e-6)
    if top_p >= 1.0:
        return torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)
    sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(sorted_probs, dim=-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scaled, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        raise RuntimeError("NaN/Inf detected in sampling probabilities.")
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class PhaseTracker:
    THINK = 0
    FORMAT = 1
    ANSWER = 2
    DONE = 3

    def __init__(self, batch_size: int, tokenizer, disable_phase_gating: bool) -> None:
        self._decoded = [""] * batch_size
        self._phases = [self.THINK] * batch_size
        self._step_counts = torch.zeros(batch_size, dtype=torch.long)
        self._tokenizer = tokenizer
        self._disable_phase_gating = disable_phase_gating

    def update(self, token_ids: torch.Tensor, finished: torch.Tensor) -> None:
        token_texts = self._tokenizer.batch_decode(
            token_ids.unsqueeze(-1),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        finished_list = finished.detach().cpu().tolist()
        for idx, token_text in enumerate(token_texts):
            if finished_list[idx]:
                continue
            self._step_counts[idx] += 1
            self._decoded[idx] += token_text
            self._phases[idx] = self._phase_from_text(self._decoded[idx].lower())

    def active_mask(self, device: torch.device, max_steer_tokens: int = UNLIMITED_STEER_TOKENS) -> torch.Tensor:
        if self._disable_phase_gating:
            phase_ok = torch.tensor([phase != self.DONE for phase in self._phases], device=device)
        else:
            phase_ok = torch.tensor([phase == self.THINK for phase in self._phases], device=device)
        return phase_ok & self._token_window_mask(device, max_steer_tokens)

    def alpha_scale(self, device: torch.device, max_steer_tokens: int, decay: str) -> torch.Tensor:
        if decay == ALPHA_DECAY_NONE or max_steer_tokens == UNLIMITED_STEER_TOKENS:
            return torch.ones(len(self._phases), device=device)
        if decay != ALPHA_DECAY_LINEAR:
            raise ValueError(f"Unsupported alpha_decay: {decay}")
        counts = self._step_counts.to(device=device, dtype=torch.float32)
        scale = 1.0 - counts / float(max_steer_tokens)
        return scale.clamp(min=0.0, max=1.0)

    def done_mask(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([phase == self.DONE for phase in self._phases], device=device)

    def step_counts(self, device: torch.device) -> torch.Tensor:
        return self._step_counts.to(device=device)

    def _token_window_mask(self, device: torch.device, max_steer_tokens: int) -> torch.Tensor:
        if max_steer_tokens == UNLIMITED_STEER_TOKENS:
            return torch.ones(len(self._phases), dtype=torch.bool, device=device)
        if max_steer_tokens < 0:
            raise ValueError(f"Invalid max_steer_tokens: {max_steer_tokens}")
        return self._step_counts.to(device=device) < max_steer_tokens

    @staticmethod
    def _phase_from_text(text: str) -> int:
        if "</answer>" in text:
            return PhaseTracker.DONE
        if "<answer>" in text:
            return PhaseTracker.ANSWER
        if "<feasible>" in text or "</think>" in text:
            return PhaseTracker.FORMAT
        return PhaseTracker.THINK


class ResidualSteeringHook:
    def __init__(self, tracker: PhaseTracker, config: SteeringConfig, bank: Optional[SteeringVectorBank]) -> None:
        self._tracker = tracker
        self._config = config
        self._bank = bank

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] != 1 or self._config.alpha == 0:
            return output
        mask = self._tracker.active_mask(hidden.device, self._config.max_steer_tokens)
        if not bool(mask.any()):
            return output
        alpha = self._current_alpha(hidden.device, hidden.dtype)
        steered = hidden + alpha * self._sample(hidden) * mask.view(-1, 1, 1).to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def _current_alpha(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = self._tracker.alpha_scale(device, self._config.max_steer_tokens, self._config.alpha_decay)
        return (self._config.alpha * scale).view(-1, 1, 1).to(dtype)

    def _sample(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._config.noise_type == "isotropic":
            return self._normalize(torch.randn_like(hidden))
        if self._bank is None or self._bank.mean_delta is None:
            raise ValueError("Directional steering requires a valid vector bank.")
        if self._config.noise_type == "contrastive_mean":
            return self._normalize(self._bank.mean_delta.view(1, 1, -1).expand_as(hidden))
        components = self._bank.principal_components
        if components is None:
            scale = torch.randn((hidden.shape[0], 1), device=hidden.device, dtype=hidden.dtype)
            base = self._bank.mean_delta.view(1, -1).expand(hidden.shape[0], -1)
            return self._normalize((scale * base).unsqueeze(1))
        rank = min(self._config.pca_rank, components.shape[0])
        coeffs = torch.randn((hidden.shape[0], rank), device=hidden.device, dtype=hidden.dtype)
        if self._bank.explained_variance is not None:
            coeffs = coeffs * torch.sqrt(self._bank.explained_variance[:rank]).view(1, -1)
        noise = torch.einsum("br,rh->bh", coeffs, components[:rank]).unsqueeze(1)
        return self._normalize(noise)

    @staticmethod
    def _normalize(noise: torch.Tensor) -> torch.Tensor:
        denom = noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return noise / denom


@torch.no_grad()
def generate_steered_batch(
    model,
    tokenizer,
    prompt_batch: Sequence[PromptExample],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    steer_layers: Sequence[int],
    config: SteeringConfig,
    bank: Optional[SteeringVectorBank],
    device: torch.device,
) -> Tuple[List[List[str]], List[List[int]]]:
    sequences: List[List[int]] = []
    owners: List[int] = []
    for owner, item in enumerate(prompt_batch):
        for _ in range(n_samples):
            sequences.append(item.prompt_ids)
            owners.append(owner)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    generated_ids, attention_mask, prompt_width = left_pad_sequences(sequences, pad_id, device)
    tracker = PhaseTracker(len(sequences), tokenizer, config.disable_phase_gating)
    hooks = []
    if config.alpha != 0:
        model_layers = get_layers_container(model)
        hook = ResidualSteeringHook(tracker, config, bank)
        hooks = [model_layers[layer].register_forward_hook(hook) for layer in steer_layers]
    finished = torch.zeros(len(sequences), dtype=torch.bool, device=device)
    past_key_values = None
    eos_id = tokenizer.eos_token_id
    try:
        for _ in range(max_new_tokens):
            if bool(finished.all()):
                break
            if past_key_values is None:
                outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=True)
            else:
                outputs = model(input_ids=generated_ids[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            if finished.any():
                logits = logits.clone()
                logits[finished] = float("-inf")
                logits[finished, pad_id] = 0.0
            next_tokens = sample_next_tokens(logits, temperature=temperature, top_p=top_p)
            finished_before = finished.clone()
            tracker.update(next_tokens, finished_before)
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, (~finished_before).long().unsqueeze(-1)], dim=-1)
            if eos_id is not None:
                finished |= next_tokens.eq(eos_id)
            finished |= tracker.done_mask(device)
    finally:
        for handle in hooks:
            handle.remove()
    texts, token_counts = decode_generated(generated_ids, prompt_width, tokenizer)
    grouped_texts = [[] for _ in prompt_batch]
    grouped_counts = [[] for _ in prompt_batch]
    for text, token_count, owner in zip(texts, token_counts, owners):
        grouped_texts[owner].append(text)
        grouped_counts[owner].append(token_count)
    return grouped_texts, grouped_counts


def decode_generated(generated_ids: torch.Tensor, prompt_width: int, tokenizer) -> Tuple[List[str], List[int]]:
    outputs: List[str] = []
    token_counts: List[int] = []
    eos_id = tokenizer.eos_token_id
    for token_ids in generated_ids[:, prompt_width:].detach().cpu().tolist():
        if eos_id is not None and eos_id in token_ids:
            token_ids = token_ids[: token_ids.index(eos_id)]
        text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for eos_text in EOS_STRINGS:
            if eos_text in text:
                text = text.split(eos_text)[0]
        outputs.append(text.strip())
        token_counts.append(len(token_ids))
    return outputs, token_counts


# ---- merged evaluation mode ----
"""Evaluation helpers for Exp V activation steering."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion
from early_branch_locking.core.countdown_shared import bootstrap_ci_mean, entropy_from_counts, evaluate_countdown_completion, pass_at_k
from early_branch_locking.core.prompt_data import PromptExample


def evaluate_problem(item: PromptExample, completions: Sequence[str], token_counts: Sequence[int]) -> Tuple[dict, List[dict]]:
    correct = feasible_ok = expr_ok = format_ok = 0
    solution_counter: Counter[str] = Counter()
    raw_rows: List[dict] = []
    for sample_index, (completion, token_count) in enumerate(zip(completions, token_counts)):
        ev = evaluate_countdown_completion(
            completion,
            item.numbers,
            item.target,
            item.feasible_label,
            parse_countdown_completion=parse_countdown_completion,
            evaluate_countdown_expression=evaluate_countdown_expression,
        )
        correct += int(ev.overall_ok)
        feasible_ok += int(ev.feasible_ok)
        expr_ok += int(ev.expr_ok)
        format_ok += int(ev.parse_status == "OK")
        if ev.canonical_expr and ev.canonical_expr in item.solution_set:
            solution_counter[ev.canonical_expr] += 1
        raw_rows.append(
            {
                "sample_id": item.sample_id,
                "problem_index": item.pid,
                "sample_index": sample_index,
                "numbers": item.numbers,
                "target": item.target,
                "feasible_label": item.feasible_label,
                "completion": completion,
                "completion_tokens": token_count,
                "feasible_pred": ev.feasible_pred,
                "feasible_ok": bool(ev.feasible_ok),
                "expr_ok": bool(ev.expr_ok),
                "overall_ok": bool(ev.overall_ok),
                "parse_status": ev.parse_status,
                "has_feasible_tag": ev.has_feasible_tag,
                "has_answer_tag": ev.has_answer_tag,
                "canonical_expr": ev.canonical_expr,
                "opseq_label": ev.opseq_label,
                "answer_label": ev.answer_label,
                "trace_label": ev.trace_label,
                "expr_status": ev.expr_status,
                "tag_order_ok": ev.tag_order_ok,
            }
        )
    attempts = len(completions)
    total_tokens = int(sum(token_counts))
    solution_count = len(item.solution_set)
    unique_solution = len(solution_counter)
    support_hits = int(sum(solution_counter.values()))
    top1_support = max(solution_counter.values()) if solution_counter else 0
    row = {
        "problem_index": item.pid,
        "sample_id": item.sample_id,
        "numbers": item.numbers,
        "target": item.target,
        "feasible_label": item.feasible_label,
        "solver_feasible": bool(solution_count > 0),
        "attempt_count": attempts,
        "correct_count": correct,
        "feasible_ok_count": feasible_ok,
        "expr_ok_count": expr_ok,
        "format_ok_count": format_ok,
        "correct_mass": correct / attempts if attempts else 0.0,
        "feasible_ok_mass": feasible_ok / attempts if attempts else 0.0,
        "feasible_ok_rate": feasible_ok / attempts if attempts else 0.0,
        "expr_ok_mass": expr_ok / attempts if attempts else 0.0,
        "expr_ok_rate": expr_ok / attempts if attempts else 0.0,
        "format_rate": format_ok / attempts if attempts else 0.0,
        "support_mass": support_hits / attempts if attempts else 0.0,
        "off_support_mass": 1.0 - (support_hits / attempts) if attempts else 0.0,
        "coverage": unique_solution / solution_count if solution_count else 0.0,
        "unique_solution": unique_solution,
        "top1_solution_mass": top1_support / attempts if attempts else 0.0,
        "top1_solution_mass_cond": top1_support / support_hits if support_hits else 0.0,
        "solution_entropy": entropy_from_counts(solution_counter) if solution_counter else 0.0,
        "solution_count": solution_count,
        "total_output_tokens": total_tokens,
        "avg_output_tokens": total_tokens / attempts if attempts else 0.0,
    }
    return row, raw_rows


def aggregate_problem_rows(problem_rows: Sequence[dict], count_key: str, ks: Iterable[int]) -> Dict[str, float]:
    attempts = {row["problem_index"]: int(row["attempt_count"]) for row in problem_rows}
    counts = {row["problem_index"]: int(row[count_key]) for row in problem_rows}
    values = {
        "n_problems": len(problem_rows),
        "total_output_tokens": float(sum(int(row["total_output_tokens"]) for row in problem_rows)),
        "mean_output_tokens_per_completion": float(np.mean([row["avg_output_tokens"] for row in problem_rows])) if problem_rows else 0.0,
    }
    mean_keys = (
        "correct_mass",
        "feasible_ok_mass",
        "expr_ok_mass",
        "format_rate",
        "support_mass",
        "off_support_mass",
        "coverage",
        "unique_solution",
        "top1_solution_mass",
        "top1_solution_mass_cond",
        "solution_entropy",
        "solution_count",
        "avg_output_tokens",
    )
    for key in mean_keys:
        values[f"{key}_mean"] = float(np.mean([row[key] for row in problem_rows])) if problem_rows else 0.0
    values["solver_feasible_rate"] = float(np.mean([float(row["solver_feasible"]) for row in problem_rows])) if problem_rows else 0.0
    values["feasible_ok_rate_mean"] = values["feasible_ok_mass_mean"]
    values["expr_ok_rate_mean"] = values["expr_ok_mass_mean"]
    for k in ks:
        estimates = [pass_at_k(attempts[pid], counts[pid], k) for pid in counts if attempts[pid] >= k]
        mean, lo, hi = bootstrap_ci_mean(estimates) if estimates else (0.0, 0.0, 0.0)
        prefix = "" if count_key == "correct_count" else count_key.replace("_count", "") + "_"
        values[f"{prefix}pass@{k}"] = mean
        values[f"{prefix}pass@{k}_ci_lo"] = lo
        values[f"{prefix}pass@{k}_ci_hi"] = hi
    return values


def write_raw_rows(path: Path, rows: Iterable[dict]) -> None:
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


# ---- merged in-process multi-job mode ----
def parse_multi_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Exp V steering grid in one process.")
    parser.add_argument("--actor_dir", type=str, default=str(ACTOR_DIR))
    parser.add_argument("--source_step", type=int, default=50)
    parser.add_argument("--target_step", type=int, default=275)
    parser.add_argument("--windows", type=str, default="5,10,15,20")
    parser.add_argument("--alphas", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--steer_layers", type=str, default="28,29,30,31,32")
    parser.add_argument("--pca_layer", type=int, default=30)
    parser.add_argument("--pca_top_k", type=int, default=5)
    parser.add_argument("--pca_num_problems", type=int, default=32)
    parser.add_argument("--activation_window_tokens", type=int, default=10)
    parser.add_argument("--pca_output_path", type=str, default="")
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--tag_prefix", type=str, default="activation_steering2_transient_pca_early10")
    parser.add_argument("--skip_pca_extract", action="store_true")
    parser.add_argument("--force_pca_extract", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    return parser.parse_args()


def _csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _multi_model_path(actor_dir: str, step: int) -> Path:
    return Path(actor_dir) / f"global_step_{step}"


def _multi_pca_path(args: argparse.Namespace) -> Path:
    if args.pca_output_path:
        return Path(args.pca_output_path).expanduser().resolve()
    return METRICS_DIR / (
        f"pca_dir_L{args.pca_layer}_early{args.activation_window_tokens}"
        f"_src{args.source_step}_tgt{args.target_step}.pt"
    )


def _run_pca_for_multi(args: argparse.Namespace, pca_path: Path) -> None:
    if args.skip_pca_extract:
        if not pca_path.is_file():
            raise FileNotFoundError(f"PCA payload does not exist: {pca_path}")
        return
    if pca_path.is_file() and not args.force_pca_extract:
        print(f"[activation_steering] reuse PCA payload -> {pca_path}", flush=True)
        return
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = list(sys.argv)
    try:
        sys.argv = [
            old_argv[0],
            "--source_model_path", str(_multi_model_path(args.actor_dir, args.source_step)),
            "--target_model_path", str(_multi_model_path(args.actor_dir, args.target_step)),
            "--output_path", str(pca_path),
            "--layer", str(args.pca_layer),
            "--top_k", str(args.pca_top_k),
            "--num_problems", str(args.pca_num_problems),
            "--activation_window_tokens", str(args.activation_window_tokens),
            "--max_new_tokens", str(args.max_new_tokens),
            "--dtype", args.dtype,
            "--attn_implementation", args.attn_implementation,
            "--source_gpu_id", args.gpu_id,
            "--target_gpu_id", args.gpu_id,
        ]
        run_extract_pca()
    finally:
        sys.argv = old_argv


def _run_multi() -> None:
    args = parse_multi_args()
    pca_path = _multi_pca_path(args)
    _run_pca_for_multi(args, pca_path)
    model_path = _multi_model_path(args.actor_dir, args.target_step)
    tags = []
    old_argv = list(sys.argv)
    try:
        for window in _csv_ints(args.windows):
            for alpha in _csv_floats(args.alphas):
                tag = f"{args.tag_prefix}_T{window}_a{alpha:g}"
                run_argv = [
                    old_argv[0],
                    "--model_path", str(model_path),
                    "--num_problems", str(args.num_problems),
                    "--start_index", str(args.start_index),
                    "--end_index", str(args.end_index),
                    "--n_samples", str(args.n_samples),
                    "--steer_layers", args.steer_layers,
                    "--alpha", str(alpha),
                    "--noise_type", "pca_dir",
                    "--pca_vector_path", str(pca_path),
                    "--pca_rank", str(args.pca_top_k),
                    "--max_steer_tokens", str(window),
                    "--alpha_decay", "linear",
                    "--temperature", str(args.temperature),
                    "--top_p", str(args.top_p),
                    "--max_new_tokens", str(args.max_new_tokens),
                    "--gpu_id", args.gpu_id,
                    "--dtype", args.dtype,
                    "--batch_size", str(args.batch_size),
                    "--attn_implementation", args.attn_implementation,
                    "--tag", tag,
                    "--save_per_problem",
                ]
                if args.save_raw:
                    run_argv.append("--save_raw")
                print(f"[activation_steering] multi run {tag} on GPU {args.gpu_id}", flush=True)
                sys.argv = run_argv
                main()
                tags.append(tag)
    finally:
        sys.argv = old_argv
    if not tags:
        raise ValueError("The Exp V multi grid is empty.")
    frames = [pd.read_csv(METRICS_DIR / f"activation_steering_activation_steering_summary_{tag}.csv") for tag in tags]
    summary_path = METRICS_DIR / f"activation_steering_transient_steering_driver_{args.tag_prefix}.csv"
    manifest_path = METRICS_DIR / f"activation_steering_transient_steering_driver_{args.tag_prefix}.json"
    pd.concat(frames, ignore_index=True).to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps({"tags": tags, "pca_path": str(pca_path), "summary_path": str(summary_path)}, indent=2),
        encoding="utf-8",
    )
    print(f"[activation_steering] saved multi summary -> {summary_path}", flush=True)


# ---- merged extract_pca mode ----
"""Extract contrastive/PCA steering directions from source vs target models."""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from early_branch_locking.core.countdown_shared import load_parquet_sorted
from early_branch_locking.core.prompt_data import (
    TEST_PARQUET,
    build_prompt_data,
    ensure_tokenizer_padding,
    torch_dtype_from_name,
)
from early_branch_locking.core.op1_utils import get_layers_container

SEED = 42
ALL_THINK_TOKENS = -1


def parse_args_extract_pca() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Exp V activation PCA payload")
    parser.add_argument("--source_model_path", type=str, required=True)
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_problems", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--activation_window_tokens", type=int, default=ALL_THINK_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--source_gpu_id", type=int, default=0)
    parser.add_argument("--target_gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def validate_tokenizers(source_tokenizer, target_tokenizer, probe_text: str) -> None:
    if len(source_tokenizer) != len(target_tokenizer):
        raise ValueError("Tokenizer vocab mismatch between source and target.")
    for attr in ("eos_token_id", "bos_token_id"):
        if getattr(source_tokenizer, attr, None) != getattr(target_tokenizer, attr, None):
            raise ValueError(f"Tokenizer mismatch on {attr}.")
    if source_tokenizer.encode(probe_text, add_special_tokens=False) != target_tokenizer.encode(probe_text, add_special_tokens=False):
        raise ValueError("Tokenizer mismatch on probe text.")


@torch.no_grad()
def generate_reference_batch(
    model,
    tokenizer,
    prompt_batch: Sequence,
    max_new_tokens: int,
    activation_window_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[Dict[str, List[int]]]:
    prompt_lengths = [len(item.prompt_ids) for item in prompt_batch]
    sequences = [item.prompt_ids for item in prompt_batch]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids, attention_mask, _ = left_pad_sequences(sequences, pad_id, device)
    tracker = PhaseTracker(len(prompt_batch), tokenizer, disable_phase_gating=False)
    finished = torch.zeros(len(prompt_batch), dtype=torch.bool, device=device)
    generated_lists = [[] for _ in prompt_batch]
    active_positions = [[] for _ in prompt_batch]
    generated_counts = [0] * len(prompt_batch)
    past_key_values = None
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        if bool(finished.all()):
            break
        if past_key_values is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        else:
            outputs = model(input_ids=input_ids[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        if finished.any():
            logits = logits.clone()
            logits[finished] = float("-inf")
            logits[finished, pad_id] = 0.0
        active_before = tracker.active_mask(device)
        next_tokens = sample_next_tokens(logits, temperature=temperature, top_p=top_p)
        finished_before = finished.clone()
        for idx, token_id in enumerate(next_tokens.detach().cpu().tolist()):
            if finished_before[idx]:
                continue
            within_window = activation_window_tokens == ALL_THINK_TOKENS or generated_counts[idx] < activation_window_tokens
            if active_before[idx] and within_window:
                active_positions[idx].append(prompt_lengths[idx] + generated_counts[idx] - 1)
            generated_lists[idx].append(int(token_id))
            generated_counts[idx] += 1
        tracker.update(next_tokens, finished_before)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([attention_mask, (~finished_before).long().unsqueeze(-1)], dim=-1)
        if eos_id is not None:
            finished |= next_tokens.eq(eos_id)
        finished |= tracker.done_mask(device)
    return [{"token_ids": item.prompt_ids + gen_ids, "prediction_positions": positions} for item, gen_ids, positions in zip(prompt_batch, generated_lists, active_positions)]


@torch.no_grad()
def capture_layer_hidden(model, token_ids: List[int], layer_idx: int) -> torch.Tensor:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = get_layers_container(model)[layer_idx].register_forward_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return captured["hidden"][0].detach().cpu().float()


def compute_pca_payload(deltas: torch.Tensor, top_k: int) -> Dict[str, torch.Tensor]:
    mean_delta = deltas.mean(dim=0)
    centered = deltas - mean_delta
    if centered.shape[0] < 2:
        return {"mean_delta": mean_delta, "principal_components": mean_delta.unsqueeze(0), "explained_variance": torch.ones(1)}
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    k = min(top_k, vh.shape[0])
    variance = (singular_values[:k] ** 2) / max(centered.shape[0] - 1, 1)
    return {
        "mean_delta": mean_delta,
        "principal_components": vh[:k].contiguous(),
        "explained_variance": variance.contiguous(),
    }


def run_extract_pca() -> None:
    args = parse_args_extract_pca()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch_dtype_from_name(args.dtype)
    source_device = torch.device(f"cuda:{args.source_gpu_id}")
    target_device = torch.device(f"cuda:{args.target_gpu_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for activation extraction.")
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_path, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    ensure_tokenizer_padding(source_tokenizer)
    ensure_tokenizer_padding(target_tokenizer)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_data = build_prompt_data(records, source_tokenizer)
    probe_text = prompt_data[0].prompt_text if prompt_data else "<think>test</think>"
    validate_tokenizers(source_tokenizer, target_tokenizer, probe_text)
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model_path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(source_device)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(target_device)
    source_model.eval()
    target_model.eval()
    deltas: List[torch.Tensor] = []
    for batch_start in range(0, len(prompt_data), args.batch_size):
        batch = prompt_data[batch_start : batch_start + args.batch_size]
        refs = generate_reference_batch(
            model=source_model,
            tokenizer=source_tokenizer,
            prompt_batch=batch,
            max_new_tokens=args.max_new_tokens,
            activation_window_tokens=args.activation_window_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=source_device,
        )
        for item, ref in zip(batch, refs):
            if not ref["prediction_positions"]:
                continue
            source_hidden = capture_layer_hidden(source_model, ref["token_ids"], args.layer)
            target_hidden = capture_layer_hidden(target_model, ref["token_ids"], args.layer)
            positions = [pos for pos in ref["prediction_positions"] if 0 <= pos < source_hidden.shape[0]]
            if not positions:
                continue
            deltas.append(source_hidden[positions] - target_hidden[positions])
        done = min(batch_start + len(batch), len(prompt_data))
        print(f"[extract_pca] processed {done}/{len(prompt_data)} prompts", flush=True)
    if not deltas:
        raise RuntimeError("No think-phase activation deltas were collected.")
    delta_matrix = torch.cat(deltas, dim=0)
    payload = compute_pca_payload(delta_matrix, args.top_k)
    payload.update(
        {
            "layer": args.layer,
            "num_problems": args.num_problems,
            "num_vectors": int(delta_matrix.shape[0]),
            "activation_window_tokens": args.activation_window_tokens,
            "source_model": Path(args.source_model_path).name,
            "target_model": Path(args.target_model_path).name,
        }
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[extract_pca] saved payload -> {output_path}", flush=True)
    print(f"[extract_pca] mean_delta_shape={tuple(payload['mean_delta'].shape)} components={tuple(payload['principal_components'].shape)}", flush=True)

def _run_selected():
    if "--multi" in sys.argv:
        sys.argv.remove("--multi")
        return _run_multi()
    if "--extract-pca" in sys.argv:
        sys.argv.remove("--extract-pca")
        return run_extract_pca()
    for index, argument in enumerate(sys.argv):
        if argument == "--mode" and index + 1 < len(sys.argv):
            mode = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            if mode == "extract_pca":
                return run_extract_pca()
            if mode not in {"steering", "run"}:
                raise ValueError(f"Unknown --mode: {mode}")
            break
    return main()

if __name__ == "__main__":
    _run_selected()
