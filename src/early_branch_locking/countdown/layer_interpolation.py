#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""layer_interpolation - Layer-localized checkpoint interpolation.
Hypothesis: interpolating only the layers responsible for branch access can recover diversity without discarding late-checkpoint accuracy.
Inputs: early and late checkpoint weights; dataset/test.parquet; sampling and layer-block settings.
Outputs: data/analysis_results/rlvr_passk/metrics/layer_interpolation_layer_soup_summary_layer_soup_v1.csv; data/analysis_results/rlvr_passk/metrics/layer_interpolation_layer_soup_per_problem_layer_soup_v1.parquet
Status: paper-appendix
"""
from __future__ import annotations
"""
layer_interpolation_layer_soup_countdown.py

Experiment T — Layer-Localized Checkpoint Interpolation
========================================================

Given an early checkpoint (high diversity, e.g. step50) and a late
checkpoint (high accuracy but collapsed, e.g. step275), constructs
"layer soup" models by interpolating weights only within specified
layer blocks:

    θ_soup[block] = θ_late + α · (θ_early - θ_late)   for layers in block
    θ_soup[other] = θ_late                              elsewhere

Sweeps over:
  - α ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
  - layer_blocks ∈ {0-12, 12-24, 16-24, 20-28, 24-32, 28-36, 32-36, 0-36}
  - global WiSE-FT baseline: α-interpolation of ALL parameters

Evaluates each soup model via vLLM batch sampling:
  - pass@1, pass@k
  - coverage (from exact solution enumeration)
  - unique_solution, solution_entropy

Outputs:
  metrics/layer_interpolation_layer_soup_summary_{tag}.csv
  metrics/layer_interpolation_layer_soup_per_problem_{tag}.parquet (optional)
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
    parse_countdown_completion,
    evaluate_countdown_expression,
)
from early_branch_locking.core.countdown_shared import (  # noqa: E402
    build_prompt_text,
    canonicalize_expression,
    entropy_from_counts,
    enumerate_solution_set,
    evaluate_countdown_completion,
    extract_ground_truth,
    get_prompt_content,
    load_parquet_sorted,
    pass_at_k,
    bootstrap_ci_mean,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
EOS_STRINGS = ["<|endoftext|>", "<|im_end|>"]


def tensor_parallel_size_from_gpu_arg(gpu_id: str) -> int:
    return max(1, len([device for device in gpu_id.split(",") if device.strip()]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ExpT: layer-localized checkpoint interpolation")
    p.add_argument("--early_model_path", type=str, required=True,
                   help="High-diversity checkpoint (e.g. global_step_50)")
    p.add_argument("--late_model_path", type=str, required=True,
                   help="High-accuracy, low-diversity checkpoint (e.g. global_step_275)")
    p.add_argument("--num_problems", type=int, default=150)
    p.add_argument("--n_samples", type=int, default=64,
                   help="Samples per problem for each soup variant")
    # Interpolation config
    p.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.5,0.7,1.0",
                   help="Comma-separated interpolation coefficients")
    p.add_argument("--layer_blocks", type=str,
                   default="0-12,12-24,16-24,20-28,24-32,28-36,32-36",
                   help="Comma-separated layer blocks as start-end")
    p.add_argument("--include_global", action="store_true", default=True,
                   help="Include global WiSE-FT baseline (all layers)")
    p.add_argument("--skip_global", action="store_true", default=False)
    p.add_argument("--include_controls", action="store_true", default=True,
                   help="Include pure early / pure late as controls")
    p.add_argument("--skip_controls", action="store_true", default=False)
    # Generation params
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", type=str, default="layer_soup_v1")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    p.add_argument("--save_raw", action="store_true", default=False)
    p.add_argument("--tmp_dir", type=str, default="")
    return p.parse_args()


def parse_layer_blocks(text: str) -> List[Tuple[int, int]]:
    blocks = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            blocks.append((int(a.strip()), int(b.strip())))
    return blocks


def parse_alphas(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_soup_variant(
    model_path: str,
    variant_name: str,
    records: List[dict],
    prompt_texts: List[str],
    sol_sets: Dict[int, Set[str]],
    args,
    tokenizer,
) -> Tuple[Dict[int, dict], dict]:
    """
    Evaluate a single soup model variant using vLLM.
    Returns (per_problem_metrics, aggregate_metrics).
    """
    from vllm import LLM, SamplingParams

    print(f"  [layer_interpolation] Evaluating variant: {variant_name} ...")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size_from_gpu_arg(args.gpu_id),
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    outputs = llm.generate(prompt_texts, params)

    per_problem: Dict[int, dict] = {}
    raw_rows: List[dict] = []

    for pid, out in enumerate(outputs):
        rec = records[pid]
        numbers, target, feasible_label = extract_ground_truth(rec)
        sol_set = sol_sets.get(pid, set())

        correct = 0
        sol_counter: Counter = Counter()
        format_ok = 0

        for si, seq in enumerate(out.outputs):
            text = seq.text or ""
            for eos in EOS_STRINGS:
                if eos in text:
                    text = text.split(eos)[0]
            text = text.strip()

            ev = evaluate_countdown_completion(
                text, numbers, target, feasible_label,
                parse_countdown_completion=parse_countdown_completion,
                evaluate_countdown_expression=evaluate_countdown_expression,
            )
            if ev.overall_ok:
                correct += 1
            if ev.parse_status == "OK":
                format_ok += 1
            if ev.canonical_expr and ev.canonical_expr in sol_set:
                sol_counter[ev.canonical_expr] += 1

            if args.save_raw:
                raw_rows.append(dict(
                    variant=variant_name,
                    problem_index=pid,
                    sample_index=si,
                    completion=text,
                    overall_ok=ev.overall_ok,
                    canonical_expr=ev.canonical_expr,
                    parse_status=ev.parse_status,
                ))

        n = len(out.outputs)
        unique_sol = len(sol_counter)
        sol_count = len(sol_set)

        per_problem[pid] = dict(
            n=n,
            correct_mass=correct / n if n > 0 else 0.0,
            coverage=unique_sol / sol_count if sol_count > 0 else 0.0,
            unique_solution=unique_sol,
            solution_count=sol_count,
            top1_sol_mass=(max(sol_counter.values()) / n) if sol_counter else 0.0,
            solution_entropy=entropy_from_counts(sol_counter) if sol_counter else 0.0,
            format_rate=format_ok / n if n > 0 else 0.0,
        )

    # Save raw if requested
    if args.save_raw and raw_rows:
        raw_path = RAW_DIR / f"layer_interpolation_raw_{variant_name}_{args.tag}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Aggregate
    agg = {}
    for key in ["correct_mass", "coverage", "unique_solution", "top1_sol_mass",
                 "solution_entropy", "format_rate"]:
        vals = [per_problem[pid][key] for pid in per_problem]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")

    agg["n_problems"] = len(per_problem)

    # Pass@k
    ks = [1, 4, 16, 64]
    for k in ks:
        pk_vals = []
        for pid in per_problem:
            n = per_problem[pid]["n"]
            c = int(per_problem[pid]["correct_mass"] * n + 0.5)
            if k <= n:
                pk_vals.append(pass_at_k(n, c, k))
        if pk_vals:
            m, lo, hi = bootstrap_ci_mean(pk_vals)
            agg[f"pass@{k}"] = m
            agg[f"pass@{k}_ci_lo"] = lo
            agg[f"pass@{k}_ci_hi"] = hi

    if hasattr(llm, "shutdown"):
        llm.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return per_problem, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    alphas = parse_alphas(args.alphas)
    layer_blocks = parse_layer_blocks(args.layer_blocks)
    dtype_torch = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    early_name = Path(args.early_model_path).name
    late_name = Path(args.late_model_path).name
    tag = args.tag or f"{early_name}_to_{late_name}"

    tmp_base = Path(args.tmp_dir) if args.tmp_dir else (ANALYSIS_ROOT / "tmp_models")
    tmp_dir = tmp_base / f"layer_interpolation_{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.early_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")

    prompt_texts = []
    for rec in records:
        prompt_content = get_prompt_content(rec)
        prompt_texts.append(build_prompt_text(prompt_content, tokenizer))

    sol_sets: Dict[int, Set[str]] = {}
    for pid, rec in enumerate(records):
        numbers, target, feasible_label = extract_ground_truth(rec)
        if feasible_label == "yes":
            sol_sets[pid] = enumerate_solution_set(numbers, target)
        else:
            sol_sets[pid] = set()

    # ------------------------------------------------------------------
    # 2. Load state dicts
    # ------------------------------------------------------------------
    print("[layer_interpolation] Loading early and late model state dicts...")
    early_model = AutoModelForCausalLM.from_pretrained(
        args.early_model_path, torch_dtype=dtype_torch, trust_remote_code=True,
    )
    early_sd = {k: v.cpu().clone() for k, v in early_model.state_dict().items()}
    del early_model
    gc.collect()

    late_model = AutoModelForCausalLM.from_pretrained(
        args.late_model_path, torch_dtype=dtype_torch, trust_remote_code=True,
    )
    late_sd = {k: v.cpu().clone() for k, v in late_model.state_dict().items()}

    # Detect model type and number of layers
    config = AutoConfig.from_pretrained(args.early_model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "qwen2")
    num_layers = getattr(config, "num_hidden_layers", 36)
    if model_type not in ("qwen2", "gpt2"):
        print(f"[layer_interpolation] Warning: unknown model_type '{model_type}', defaulting to qwen2")
        model_type = "qwen2"

    del late_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[layer_interpolation] Model has {num_layers} layers, type={model_type}")

    # Validate layer blocks
    valid_blocks = [(s, e) for s, e in layer_blocks if 0 <= s < e <= num_layers]

    # ------------------------------------------------------------------
    # 3. Build all variants
    # ------------------------------------------------------------------
    variants: List[Tuple[str, str]] = []  # (variant_name, model_path)

    # Controls
    if args.include_controls and not args.skip_controls:
        variants.append(("pure_early", args.early_model_path))
        variants.append(("pure_late", args.late_model_path))

    config_source = args.late_model_path

    def _build_save_variant(vname: str, soup_sd: Dict[str, torch.Tensor]):
        vdir = tmp_dir / vname
        if not vdir.exists():
            save_perturbed_model(soup_sd, config_source, vdir)
        variants.append((vname, str(vdir)))

    # Global WiSE-FT
    if args.include_global and not args.skip_global:
        for alpha in alphas:
            vname = f"global_a{alpha:.2f}"
            soup_sd = build_global_soup_state_dict(late_sd, early_sd, alpha)
            _build_save_variant(vname, soup_sd)
            del soup_sd
            gc.collect()

    # Layer-localized soups
    for (layer_start, layer_end) in valid_blocks:
        for alpha in alphas:
            vname = f"block_{layer_start}_{layer_end}_a{alpha:.2f}"
            soup_sd = build_layer_soup_state_dict(
                late_sd, early_sd, alpha, layer_start, layer_end, model_type,
            )
            _build_save_variant(vname, soup_sd)
            del soup_sd
            gc.collect()

    # Free state dicts
    del early_sd, late_sd
    gc.collect()

    print(f"[layer_interpolation] Total variants to evaluate: {len(variants)}")

    # ------------------------------------------------------------------
    # 4. Evaluate each variant
    # ------------------------------------------------------------------
    summary_rows: List[dict] = []
    all_per_problem: List[dict] = []

    for vname, vpath in variants:
        per_problem, agg = evaluate_soup_variant(
            vpath, vname, records, prompt_texts, sol_sets, args, tokenizer,
        )

        # Parse variant metadata
        meta = dict(
            variant=vname,
            early_model=early_name,
            late_model=late_name,
            n_samples=args.n_samples,
        )

        if vname.startswith("block_"):
            parts = vname.split("_")
            # block_{start}_{end}_a{alpha}
            meta["layer_start"] = int(parts[1])
            meta["layer_end"] = int(parts[2])
            meta["alpha"] = float(parts[3][1:])
            meta["interpolation_type"] = "layer_localized"
        elif vname.startswith("global_"):
            meta["layer_start"] = 0
            meta["layer_end"] = num_layers
            meta["alpha"] = float(vname.split("_a")[1])
            meta["interpolation_type"] = "global_wiseft"
        elif vname == "pure_early":
            meta["alpha"] = 1.0
            meta["interpolation_type"] = "control"
        elif vname == "pure_late":
            meta["alpha"] = 0.0
            meta["interpolation_type"] = "control"
        else:
            meta["interpolation_type"] = "unknown"

        summary_rows.append({**meta, **agg})

        if args.save_per_problem:
            for pid, metrics in per_problem.items():
                all_per_problem.append(dict(
                    variant=vname,
                    problem_index=pid,
                    **{k: v for k, v in meta.items() if k != "variant"},
                    **metrics,
                ))

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(summary_rows)
    out_csv = METRICS_DIR / f"layer_interpolation_layer_soup_summary_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[layer_interpolation] Saved summary → {out_csv}")

    # Print key results
    print("\n=== Key Results ===")
    display_cols = [
        "variant", "interpolation_type", "alpha",
        "correct_mass_mean", "coverage_mean", "unique_solution_mean",
        "format_rate_mean",
    ]
    pk_cols = [c for c in df.columns if c.startswith("pass@") and "_ci_" not in c]
    display_cols.extend(pk_cols)
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    if args.save_per_problem and all_per_problem:
        pp_df = pd.DataFrame(all_per_problem)
        pp_path = METRICS_DIR / f"layer_interpolation_layer_soup_per_problem_{tag}.parquet"
        pp_df.to_parquet(pp_path, index=False)
        print(f"[layer_interpolation] Saved per-problem → {pp_path}")

    # ------------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------------
    print(f"[layer_interpolation] Cleaning up temporary models in {tmp_dir} ...")
    cleanup_perturbed_dir(tmp_dir)
    print("[layer_interpolation] Done.")


# ---- merged layer_soup_helpers ----
"""
structure_utilities.py

Shared utilities for weight-space thicket probing (ExpS) and
layer-localized interpolation (ExpT).

Provides:
- Weight perturbation samplers (isotropic, directional, subspace)
- Functional-radius calibration (binary search for σ given target KL)
- vPLSD / VCC computation from teacher-forced logprobs
- Headroom-normalized support mass helpers
- Perturbation model save / load for vLLM evaluation
"""

import gc
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Weight perturbation sampling
# ---------------------------------------------------------------------------

class IsotropicSampler:
    """Sample isotropic Gaussian perturbations in full parameter space."""

    def __init__(self, sigma: float, seed: int = 42):
        self.sigma = sigma
        self.seed = seed

    def sample(
        self,
        state_dict: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        """Return a perturbed state_dict: θ + σ·ε where ε ~ N(0, I)."""
        rng = torch.Generator()
        rng.manual_seed(self.seed + index)
        perturbed = {}
        for key, param in state_dict.items():
            noise = torch.randn(param.shape, generator=rng, dtype=torch.float32)
            perturbed[key] = (param.float() + self.sigma * noise).to(param.dtype)
        return perturbed


class DirectionalSampler:
    """
    Sample perturbations along a specified direction (e.g. rollback).

    direction_sd = θ_early - θ_late (pre-computed, stored as state_dict delta).
    The perturbation is: θ + α · direction, where α ~ N(0, σ²).
    """

    def __init__(
        self,
        direction_sd: Dict[str, torch.Tensor],
        sigma: float,
        seed: int = 42,
    ):
        self.direction_sd = direction_sd
        self.sigma = sigma
        self.seed = seed

    def sample(
        self,
        state_dict: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self.seed + index)
        alpha = torch.randn(1, generator=rng).item() * self.sigma
        perturbed = {}
        for key, param in state_dict.items():
            if key in self.direction_sd:
                delta = self.direction_sd[key].float()
                perturbed[key] = (param.float() + alpha * delta).to(param.dtype)
            else:
                perturbed[key] = param.clone()
        return perturbed


class SubspaceSampler:
    """
    Sample perturbations in a low-rank subspace derived from
    training trajectory (PCA of checkpoint deltas).

    basis: list of state_dict deltas (each a Dict[str, Tensor]), already
           orthonormalized. Perturbation = θ + σ · Σ_i c_i · basis_i
           where c ~ N(0, I_rank).
    """

    def __init__(
        self,
        basis: List[Dict[str, torch.Tensor]],
        sigma: float,
        seed: int = 42,
    ):
        self.basis = basis
        self.sigma = sigma
        self.seed = seed
        self.rank = len(basis)

    def sample(
        self,
        state_dict: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self.seed + index)
        coeffs = torch.randn(self.rank, generator=rng).tolist()

        perturbed = {}
        for key, param in state_dict.items():
            delta = torch.zeros_like(param, dtype=torch.float32)
            for i, basis_sd in enumerate(self.basis):
                if key in basis_sd:
                    delta = delta + coeffs[i] * basis_sd[key].float()
            perturbed[key] = (param.float() + self.sigma * delta).to(param.dtype)
        return perturbed


# ---------------------------------------------------------------------------
# Trajectory subspace construction
# ---------------------------------------------------------------------------

def compute_checkpoint_deltas(
    checkpoint_paths: List[str],
    reference_path: str,
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, torch.Tensor]]:
    """
    Compute weight deltas: δ_i = θ_i - θ_ref for each checkpoint.
    Returns list of state_dict deltas (on CPU, float32).
    """
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_path, torch_dtype=dtype, trust_remote_code=True,
    )
    ref_sd = {k: v.cpu().float() for k, v in ref_model.state_dict().items()}
    del ref_model
    gc.collect()

    deltas = []
    for ckpt_path in checkpoint_paths:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=dtype, trust_remote_code=True,
        )
        delta = {}
        for k, v in model.state_dict().items():
            if k in ref_sd:
                delta[k] = v.cpu().float() - ref_sd[k]
        del model
        gc.collect()
        deltas.append(delta)

    return deltas


def build_pca_basis(
    deltas: List[Dict[str, torch.Tensor]],
    max_rank: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build orthonormal basis from checkpoint deltas via PCA.

    Flattens each delta into a vector, stacks into matrix [n_deltas, d],
    does SVD, returns top-k right singular vectors as state_dict format.
    """
    if not deltas:
        return []

    # Get parameter keys from first delta
    keys = list(deltas[0].keys())
    shapes = {k: deltas[0][k].shape for k in keys}
    sizes = {k: deltas[0][k].numel() for k in keys}
    total_dim = sum(sizes.values())

    # Flatten deltas into matrix
    n = len(deltas)
    matrix = torch.zeros(n, total_dim, dtype=torch.float32)
    for i, delta in enumerate(deltas):
        offset = 0
        for k in keys:
            flat = delta[k].flatten()
            matrix[i, offset : offset + sizes[k]] = flat
            offset += sizes[k]

    # SVD
    # For n << d, compute economy SVD via n×n gram matrix
    if n < total_dim:
        gram = matrix @ matrix.T  # [n, n]
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        rank = min(max_rank, n)
        # Right singular vectors: V = M^T U S^{-1}
        basis_vectors = []
        for r in range(rank):
            if eigenvalues[r] < 1e-12:
                break
            v = matrix.T @ eigenvectors[:, r]  # [d]
            v = v / (v.norm() + 1e-12)
            basis_vectors.append(v)
    else:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, len(S))
        basis_vectors = [Vh[r] for r in range(rank)]

    # Unflatten back to state_dict format
    basis_sds = []
    for vec in basis_vectors:
        sd = {}
        offset = 0
        for k in keys:
            sd[k] = vec[offset : offset + sizes[k]].reshape(shapes[k]).clone()
            offset += sizes[k]
        basis_sds.append(sd)

    return basis_sds


# ---------------------------------------------------------------------------
# Functional-radius calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_prefix_kl(
    model_a,
    model_b,
    tokenizer,
    calibration_texts: List[str],
    prefix_char_lens: List[int],
    device: torch.device,
    max_tokens: int = 32,
) -> float:
    """
    Estimate average KL(p_a || p_b) at the next few tokens after prefix,
    averaged over calibration examples.

    This gives a functional distance measure between two models
    at a specific prefix position.
    """
    total_kl = 0.0
    count = 0

    for text, prefix_len in zip(calibration_texts, prefix_char_lens):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # Find token position corresponding to prefix_len chars
        prefix_text = text[:prefix_len]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        start_pos = len(prefix_ids)
        end_pos = min(start_pos + max_tokens, seq_len - 1)

        if end_pos <= start_pos:
            continue

        logits_a = model_a(input_ids).logits[0, start_pos:end_pos, :]
        logits_b = model_b(input_ids).logits[0, start_pos:end_pos, :]

        log_p_a = F.log_softmax(logits_a, dim=-1)
        log_p_b = F.log_softmax(logits_b, dim=-1)
        p_a = torch.exp(log_p_a)

        kl = (p_a * (log_p_a - log_p_b)).sum(dim=-1).mean().item()
        total_kl += kl
        count += 1

    return total_kl / max(count, 1)


def calibrate_sigma(
    base_sd: Dict[str, torch.Tensor],
    sampler_cls,
    sampler_kwargs: dict,
    model_loader,
    tokenizer,
    calibration_texts: List[str],
    prefix_char_lens: List[int],
    device: torch.device,
    target_kl: float = 0.1,
    sigma_lo: float = 1e-5,
    sigma_hi: float = 0.1,
    n_probes: int = 3,
    max_iters: int = 12,
    tolerance: float = 0.02,
) -> float:
    """
    Binary search for σ such that E[KL(p_θ || p_{θ+ε})] ≈ target_kl.

    model_loader: callable(state_dict) -> model on device
    """
    for iteration in range(max_iters):
        sigma_mid = (sigma_lo + sigma_hi) / 2.0
        kwargs = {**sampler_kwargs, "sigma": sigma_mid}
        sampler = sampler_cls(**kwargs)

        kls = []
        for probe_idx in range(n_probes):
            perturbed_sd = sampler.sample(base_sd, index=probe_idx + iteration * 1000)
            perturbed_model = model_loader(perturbed_sd)
            base_model = model_loader(base_sd)

            kl = estimate_prefix_kl(
                base_model, perturbed_model, tokenizer,
                calibration_texts, prefix_char_lens, device,
            )
            kls.append(kl)

            del perturbed_model, base_model
            gc.collect()
            torch.cuda.empty_cache()

        mean_kl = float(np.mean(kls))

        if abs(mean_kl - target_kl) < tolerance * target_kl:
            return sigma_mid
        elif mean_kl > target_kl:
            sigma_hi = sigma_mid
        else:
            sigma_lo = sigma_mid

    return (sigma_lo + sigma_hi) / 2.0


# ---------------------------------------------------------------------------
# vPLSD / VCC computation from logprobs
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_solutions_conditioned(
    model,
    tokenizer,
    prompt_base: str,
    candidate_exprs: List[str],
    device: torch.device,
    max_batch: int = 16,
) -> List[float]:
    """
    Teacher-force score each candidate expression continuation.
    Returns list of log-probabilities log p(expr | prompt_base).
    """
    if not candidate_exprs:
        return []

    model.eval()
    logps = []

    for start in range(0, len(candidate_exprs), max_batch):
        batch_targets = candidate_exprs[start : start + max_batch]
        full_texts = [prompt_base + t for t in batch_targets]

        enc_full = tokenizer(
            full_texts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
        enc_prompt = tokenizer(
            [prompt_base] * len(batch_targets),
            return_tensors="pt", padding=True, add_special_tokens=False,
        )

        input_ids = enc_full["input_ids"].to(device)
        attn = enc_full["attention_mask"].to(device)
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]

        for b in range(input_ids.shape[0]):
            seq_len = int(attn[b].sum().item())
            p_len = int(prompt_lens[b])
            s = max(p_len - 1, 0)
            e = seq_len - 2
            if e < s:
                logps.append(float("-inf"))
                continue
            token_lp = log_probs[b, s : e + 1, :].gather(
                -1, labels[b, s : e + 1].unsqueeze(-1),
            ).squeeze(-1)
            logps.append(float(token_lp.sum().item()))

    return logps


def compute_support_mass(logps: List[float]) -> float:
    """Total probability mass over all candidates: sum exp(logp_i)."""
    if not logps:
        return 0.0
    max_lp = max(logps)
    if max_lp < -500:
        return 0.0
    return float(sum(math.exp(lp - max_lp) for lp in logps) * math.exp(max_lp))


def compute_headroom_normalized_mass(
    mass_perturbed: float,
    mass_base: float,
    eta: float = 1e-8,
) -> float:
    """
    Headroom-normalized support mass improvement:
    (M_perturbed - M_base) / (1 - M_base + η)
    """
    return (mass_perturbed - mass_base) / (1.0 - mass_base + eta)


def compute_vcc(
    logps: List[float],
    candidate_exprs: List[str],
    solution_set: Set[str],
    top_b: int = 10,
) -> int:
    """
    Verified Class Coverage: number of distinct verified solutions
    among the top-B candidates by probability.
    """
    if not logps or not candidate_exprs:
        return 0

    # Sort by logprob descending
    paired = sorted(zip(logps, candidate_exprs), key=lambda x: -x[0])
    seen = set()
    for lp, expr in paired[:top_b]:
        if expr in solution_set:
            seen.add(expr)
    return len(seen)


def compute_plsd_single(
    mass_perturbed: float,
    mass_base: float,
    headroom_alpha: float = 0.1,
) -> bool:
    """
    Check if a single perturbation passes the PLSD threshold:
    M_perturbed >= M_base + α * (1 - M_base)
    """
    threshold = mass_base + headroom_alpha * (1.0 - mass_base)
    return mass_perturbed >= threshold


# ---------------------------------------------------------------------------
# Perturbation model persistence for vLLM
# ---------------------------------------------------------------------------

def save_perturbed_model(
    perturbed_sd: Dict[str, torch.Tensor],
    config_source_path: str,
    output_dir: Path,
) -> None:
    """Save perturbed weights so vLLM can load them."""
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(config_source_path)
    for fname in source_path.iterdir():
        if fname.is_file() and not fname.name.startswith("model"):
            shutil.copy2(str(fname), str(output_dir / fname.name))

    try:
        from safetensors.torch import save_file
        save_file(perturbed_sd, str(output_dir / "model.safetensors"))
        index_path = output_dir / "model.safetensors.index.json"
        if index_path.exists():
            index_path.unlink()
    except ImportError:
        torch.save(perturbed_sd, str(output_dir / "pytorch_model.bin"))

    # Remove shard files
    for pat in [
        "model-*.safetensors", "pytorch_model-*.bin",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    ]:
        for f in output_dir.glob(pat):
            f.unlink()


def cleanup_perturbed_dir(path: Path) -> None:
    """Remove a temporary perturbed model directory."""
    if path.exists():
        shutil.rmtree(str(path), ignore_errors=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_plsd_results(
    results: List[Dict[str, Any]],
    group_keys: List[str],
) -> List[Dict[str, Any]]:
    """
    Group results by group_keys and compute:
    - plsd_rate: fraction of perturbations passing threshold
    - mean headroom-normalized mass improvement
    - mean VCC
    """
    from collections import defaultdict

    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in results:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)

    aggregated = []
    for key, rows in groups.items():
        n = len(rows)
        plsd_hits = sum(1 for r in rows if r.get("plsd_pass", False))
        agg = {k: v for k, v in zip(group_keys, key)}
        agg["n_eval_pairs"] = n
        agg["n_perturbations"] = len({r.get("perturbation_index") for r in rows})
        agg["n_problems"] = len({r.get("problem_index") for r in rows})
        agg["plsd_rate"] = plsd_hits / n if n > 0 else 0.0
        agg["headroom_mass_mean"] = float(
            np.mean([r.get("headroom_mass", 0.0) for r in rows])
        )
        agg["vcc_mean"] = float(np.mean([r.get("vcc", 0) for r in rows]))
        agg["support_mass_perturbed_mean"] = float(
            np.mean([r.get("support_mass_perturbed", 0.0) for r in rows])
        )
        agg["support_mass_base_mean"] = float(
            np.mean([r.get("support_mass_base", 0.0) for r in rows])
        )
        aggregated.append(agg)

    return aggregated


# ---------------------------------------------------------------------------
# Layer-localized interpolation helpers
# ---------------------------------------------------------------------------

def build_layer_soup_state_dict(
    late_sd: Dict[str, torch.Tensor],
    early_sd: Dict[str, torch.Tensor],
    alpha: float,
    layer_start: int,
    layer_end: int,
    model_type: str = "qwen2",
) -> Dict[str, torch.Tensor]:
    """
    Layer-localized checkpoint interpolation:
    θ_soup = θ_late + α · (θ_early - θ_late) only for layers [layer_start, layer_end)
    Other parameters stay at θ_late.

    α = 0 → pure late (collapse model)
    α = 1 → early layers in [start, end), late elsewhere
    """
    if model_type == "qwen2":
        layer_prefix = "model.layers."
    else:
        layer_prefix = "transformer.h."

    soup_sd = {}
    for key, late_param in late_sd.items():
        # Check if this key belongs to the target layer range
        if key.startswith(layer_prefix):
            rest = key[len(layer_prefix):]
            dot_pos = rest.find(".")
            if dot_pos > 0:
                try:
                    layer_idx = int(rest[:dot_pos])
                except ValueError:
                    layer_idx = -1
            else:
                layer_idx = -1

            if layer_start <= layer_idx < layer_end and key in early_sd:
                # Interpolate: late + α * (early - late) = (1-α)*late + α*early
                early_param = early_sd[key]
                interpolated = (
                    (1.0 - alpha) * late_param.float()
                    + alpha * early_param.float()
                ).to(late_param.dtype)
                soup_sd[key] = interpolated
            else:
                soup_sd[key] = late_param.clone()
        else:
            soup_sd[key] = late_param.clone()

    return soup_sd


def build_global_soup_state_dict(
    late_sd: Dict[str, torch.Tensor],
    early_sd: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Global WiSE-FT interpolation:
    θ_soup = (1-α) * θ_late + α * θ_early
    """
    soup_sd = {}
    for key, late_param in late_sd.items():
        if key in early_sd:
            interpolated = (
                (1.0 - alpha) * late_param.float()
                + alpha * early_sd[key].float()
            ).to(late_param.dtype)
            soup_sd[key] = interpolated
        else:
            soup_sd[key] = late_param.clone()
    return soup_sd

if __name__ == "__main__":
    main()
