
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cross_checkpoint - Chimera checkpoint causal sweep.
Hypothesis: the location of early-versus-late checkpoint weights identifies where solution diversity is lost.
Inputs: step50 and step275 checkpoint weights; dataset/test.parquet; vLLM sampling configuration.
Outputs: data/analysis_results/rlvr_passk/metrics/cross_checkpoint_chimera_summary_step50_vs_step275_v3.csv; data/analysis_results/rlvr_passk/metrics/cross_checkpoint_chimera_per_problem_step50_vs_step275_v3.parquet
Status: paper-main
"""
from __future__ import annotations
"""
cross_checkpoint_chimera_eval_countdown.py

Experiment H v3 — Chimera Model Causal Sweep (Step50 ↔ Step275)
================================================================

Constructs chimera (hybrid) models by splicing two RLVR checkpoints'
weights at different layer boundaries, then evaluates each chimera's
generation diversity via vLLM batch sampling.

Key change from v2: uses step50 (high diversity, format-aware) and
step275 (high accuracy, diversity-collapsed) instead of base model.
The base model (step 0) cannot produce valid Countdown format and is
unsuitable for chimera analysis.

Chimera configurations
----------------------
For each cut_layer L in the sweep:

  forward:  explore layers [0..L-1] + collapse layers [L..N-1] + collapse lm_head
            (tests: do collapse model's late layers cause diversity collapse?)

  reverse:  collapse layers [0..L-1] + explore layers [L..N-1] + explore lm_head
            (tests: can explore model's late layers restore diversity?)

Single-layer replacement (NEW in v3):
  inject_one: collapse model everywhere, except layer L is from explore model
            (tests: can a single diverse layer restore some diversity?)

  remove_one: explore model everywhere, except layer L is from collapse model
            (tests: does a single collapsed layer destroy diversity?)

Range replacement (NEW in v3):
  inject_range: collapse model everywhere, except layers [L_start..L_end] from explore
            (tests: can a small block of diverse layers restore diversity?)

Plus pure explore and pure collapse as controls.

Terminology
-----------
  explore_model  = step50  (high diversity, moderate accuracy)
  collapse_model = step275 (high accuracy, low diversity)

Outputs
-------
- metrics/cross_checkpoint_chimera_summary_{tag}.csv
- metrics/cross_checkpoint_chimera_per_problem_{tag}.parquet  (optional)
- raw/countdown_raw_chimera_{tag}_{variant}.jsonl  (optional)
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
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
    load_parquet_sorted,
    extract_ground_truth,
    get_prompt_content,
    build_prompt_text,
    evaluate_countdown_completion,
    canonicalize_expression,
    entropy_from_counts,
    pass_at_k,
    bootstrap_ci_mean,
    enumerate_solution_set,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEED = 42
EOS_STRINGS = ["<|endoftext|>", "<|im_end|>"]


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
def configure_vllm_worker_multiprocessing() -> None:
    current = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if current == "spawn":
        return
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# ---------------------------------------------------------------------------
# temp paths
# ---------------------------------------------------------------------------
def get_temp_model_base_dir(tmp_dir: str, tag: str) -> Path:
    base_dir = Path(tmp_dir) if tmp_dir else (ANALYSIS_ROOT / "tmp_models")
    chimera_dir_base = base_dir / f"chimera_{tag}"
    chimera_dir_base.mkdir(parents=True, exist_ok=True)
    print(f"[cross_checkpoint] Temporary chimera directory: {chimera_dir_base}")
    return chimera_dir_base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ExpH v3: chimera model causal sweep (explore ↔ collapse)")
    p.add_argument("--explore_model_path", type=str, required=True,
                   help="High-diversity checkpoint (e.g. global_step_50)")
    p.add_argument("--collapse_model_path", type=str, required=True,
                   help="High-accuracy, low-diversity checkpoint (e.g. global_step_275)")
    p.add_argument("--num_problems", type=int, default=150)
    p.add_argument("--n_samples", type=int, default=64,
                   help="number of samples per problem for each chimera variant")
    # Block-level sweep
    p.add_argument("--cut_layers", type=str, default="4,8,12,16,20,24,28,30,32,33,34,35,36",
                   help="comma-separated cut layer indices for block-level chimera boundary")
    p.add_argument("--include_forward", action="store_true", default=True,
                   help="include forward chimeras: explore[0:L] + collapse[L:end+head]")
    p.add_argument("--include_reverse", action="store_true", default=True,
                   help="include reverse chimeras: collapse[0:L] + explore[L:end+head]")
    p.add_argument("--skip_forward", action="store_true", default=False,
                   help="disable forward chimeras even though they are enabled by default")
    p.add_argument("--skip_reverse", action="store_true", default=False,
                   help="disable reverse chimeras even though they are enabled by default")
    # Single-layer replacement (NEW)
    p.add_argument("--inject_one_layers", type=str, default="28,30,32,33,34,35",
                   help="layers to test single-layer injection (explore->collapse)")
    p.add_argument("--remove_one_layers", type=str, default="28,30,32,33,34,35",
                   help="layers to test single-layer removal (collapse->explore)")
    # Range replacement (NEW)
    p.add_argument("--inject_ranges", type=str, default="32-36,33-36,34-36,30-36,28-36,24-36",
                   help="ranges to test range injection, format: start-end,start-end,...")
    # Head swap
    p.add_argument("--include_head_only", action="store_true", default=True,
                   help="include head-only chimera: explore layers + collapse lm_head")
    p.add_argument("--include_head_reverse", action="store_true", default=True,
                   help="include reverse head-only: collapse layers + explore lm_head")
    p.add_argument("--skip_head_only", action="store_true", default=False,
                   help="disable head-only chimera even though it is enabled by default")
    p.add_argument("--skip_head_reverse", action="store_true", default=False,
                   help="disable reverse head-only chimera even though it is enabled by default")
    # Norm swap (NEW)
    p.add_argument("--include_norm_swap", action="store_true", default=True,
                   help="include norm-swap: collapse model + explore final layer norm")
    p.add_argument("--skip_norm_swap", action="store_true", default=False,
                   help="disable norm-swap chimera even though it is enabled by default")
    p.add_argument("--skip_controls", action="store_true", default=False,
                   help="disable pure explore / pure collapse controls")
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
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    p.add_argument("--save_raw", action="store_true", default=False)
    p.add_argument("--tmp_dir", type=str, default="",
                   help="directory for temporary chimera model files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# chimera model construction
# ---------------------------------------------------------------------------
def get_num_layers_from_config(config) -> int:
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    raise ValueError("Cannot determine number of layers from config")


def _get_layer_idx(key: str, layer_prefix: str) -> Optional[int]:
    if not key.startswith(layer_prefix):
        return None
    rest = key[len(layer_prefix):]
    dot_pos = rest.find(".")
    if dot_pos < 0:
        return None
    try:
        return int(rest[:dot_pos])
    except ValueError:
        return None


def build_chimera_state_dict(
    explore_sd: Dict[str, torch.Tensor],
    collapse_sd: Dict[str, torch.Tensor],
    num_layers: int,
    mode: str,
    mode_params: dict,
    model_type: str = "qwen2",
) -> Dict[str, torch.Tensor]:
    """
    Build a chimera state_dict by splicing explore and collapse weights.

    Parameters
    ----------
    explore_sd : high-diversity model state_dict
    collapse_sd : high-accuracy model state_dict
    num_layers : total number of transformer layers
    mode : one of
        "forward"       -> explore[0:cut] + collapse[cut:end] + collapse head
        "reverse"       -> collapse[0:cut] + explore[cut:end] + explore head
        "head_only"     -> explore[all layers] + collapse head
        "head_reverse"  -> collapse[all layers] + explore head
        "inject_one"    -> collapse everywhere, except layer L from explore
        "remove_one"    -> explore everywhere, except layer L from collapse
        "inject_range"  -> collapse everywhere, except layers [start..end) from explore
        "norm_swap"     -> collapse everywhere, except final norm from explore
    mode_params : dict with mode-specific parameters
        For forward/reverse: {"cut_layer": int}
        For inject_one/remove_one: {"layer": int}
        For inject_range: {"start": int, "end": int}
        For head_only/head_reverse/norm_swap: {}
    """
    if model_type == "qwen2":
        layer_prefix = "model.layers."
        head_keys = {"lm_head.weight"}
        embed_keys = {"model.embed_tokens.weight"}
        norm_keys = {"model.norm.weight"}
    else:
        layer_prefix = "transformer.h."
        head_keys = {"lm_head.weight"}
        embed_keys = {"transformer.wte.weight", "transformer.wpe.weight"}
        norm_keys = {"transformer.ln_f.weight", "transformer.ln_f.bias"}

    chimera_sd = {}

    # Determine which source each component comes from
    # We iterate over collapse_sd keys as the canonical key set
    all_keys = set(collapse_sd.keys()) | set(explore_sd.keys())

    for key in all_keys:
        layer_idx = _get_layer_idx(key, layer_prefix)
        is_head = key in head_keys
        is_embed = key in embed_keys
        is_norm = key in norm_keys
        is_layer = layer_idx is not None

        # Default: which model is the "base" for this mode?
        if mode == "forward":
            cut = mode_params["cut_layer"]
            if is_layer:
                src = explore_sd if layer_idx < cut else collapse_sd
            elif is_head:
                src = collapse_sd
            elif is_embed:
                src = explore_sd
            elif is_norm:
                src = collapse_sd
            else:
                src = collapse_sd

        elif mode == "reverse":
            cut = mode_params["cut_layer"]
            if is_layer:
                src = collapse_sd if layer_idx < cut else explore_sd
            elif is_head:
                src = explore_sd
            elif is_embed:
                src = collapse_sd
            elif is_norm:
                src = explore_sd
            else:
                src = explore_sd

        elif mode == "head_only":
            # explore layers + collapse head
            if is_head:
                src = collapse_sd
            else:
                src = explore_sd

        elif mode == "head_reverse":
            # collapse layers + explore head
            if is_head:
                src = explore_sd
            else:
                src = collapse_sd

        elif mode == "inject_one":
            # collapse everywhere, except one layer from explore
            target_layer = mode_params["layer"]
            if is_layer and layer_idx == target_layer:
                src = explore_sd
            else:
                src = collapse_sd

        elif mode == "remove_one":
            # explore everywhere, except one layer from collapse
            target_layer = mode_params["layer"]
            if is_layer and layer_idx == target_layer:
                src = collapse_sd
            else:
                src = explore_sd

        elif mode == "inject_range":
            # collapse everywhere, except layers [start, end) from explore
            start = mode_params["start"]
            end = mode_params["end"]
            if is_layer and start <= layer_idx < end:
                src = explore_sd
            else:
                src = collapse_sd

        elif mode == "norm_swap":
            # collapse everywhere, except final norm from explore
            if is_norm:
                src = explore_sd
            else:
                src = collapse_sd

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if key in src:
            chimera_sd[key] = src[key].clone().cpu()
        elif key in collapse_sd:
            chimera_sd[key] = collapse_sd[key].clone().cpu()
        elif key in explore_sd:
            chimera_sd[key] = explore_sd[key].clone().cpu()

    return chimera_sd


def save_chimera_model(
    chimera_sd: Dict[str, torch.Tensor],
    config_source_path: str,
    output_dir: Path,
):
    """Save a chimera model to disk so vLLM can load it."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer files from source
    source_path = Path(config_source_path)
    for fname in source_path.iterdir():
        if fname.is_file() and not fname.name.startswith("model"):
            shutil.copy2(str(fname), str(output_dir / fname.name))

    try:
        from safetensors.torch import save_file
        save_file(chimera_sd, str(output_dir / "model.safetensors"))
        index_path = output_dir / "model.safetensors.index.json"
        if index_path.exists():
            index_path.unlink()
    except ImportError:
        torch.save(chimera_sd, str(output_dir / "pytorch_model.bin"))

    # Remove shard files that might confuse loading
    for pat in ["model-*.safetensors", "pytorch_model-*.bin",
                "model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        for f in output_dir.glob(pat):
            f.unlink()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate_completions_diversity(
    records: List[dict],
    completions_by_pid: Dict[int, List[str]],
    sol_sets: Dict[int, Set[str]],
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    per_problem: Dict[int, Dict[str, float]] = {}

    for pid, completions in completions_by_pid.items():
        rec = records[pid]
        numbers, target, feasible_label = extract_ground_truth(rec)
        sol_set = sol_sets.get(pid, set())
        n = len(completions)

        if n == 0:
            per_problem[pid] = dict(
                n=0, correct_mass=0.0, coverage=0.0,
                unique_solution=0, top1_sol_mass=0.0,
                solution_entropy=0.0, solution_count=len(sol_set),
            )
            continue

        correct = 0
        sol_counts: Counter = Counter()
        for comp in completions:
            for eos in EOS_STRINGS:
                if eos in comp:
                    comp = comp.split(eos)[0]
            comp = comp.strip()

            ev = evaluate_countdown_completion(
                text=comp,
                numbers=numbers,
                target=target,
                feasible_label=feasible_label,
                parse_countdown_completion=parse_countdown_completion,
                evaluate_countdown_expression=evaluate_countdown_expression,
            )
            if ev.overall_ok:
                correct += 1
            canon = ev.canonical_expr
            if canon and canon in sol_set:
                sol_counts[canon] += 1

        correct_mass = correct / n
        support_hits = sum(sol_counts.values())
        unique_solution = len(sol_counts)
        sol_count = len(sol_set)
        coverage = (unique_solution / sol_count) if sol_count > 0 else 0.0
        top1_sol_mass = (max(sol_counts.values()) / n) if support_hits > 0 else 0.0
        solution_entropy = entropy_from_counts(sol_counts) if support_hits > 0 else 0.0

        per_problem[pid] = dict(
            n=n,
            correct_mass=correct_mass,
            coverage=coverage,
            unique_solution=unique_solution,
            top1_sol_mass=top1_sol_mass,
            solution_entropy=solution_entropy,
            solution_count=sol_count,
        )

    # aggregate
    agg = {}
    for key in ["correct_mass", "coverage", "unique_solution",
                "top1_sol_mass", "solution_entropy"]:
        vals = [per_problem[pid][key] for pid in per_problem]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
    agg["n_problems"] = len(per_problem)

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

    return per_problem, agg


def run_variant(
    model_path: str,
    variant_name: str,
    records: List[dict],
    prompt_texts: List[str],
    sol_sets: Dict[int, Set[str]],
    args,
    tokenizer,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    from vllm import LLM, SamplingParams

    print(f"\n[cross_checkpoint] Running variant: {variant_name} ...")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    outputs = llm.generate(prompt_texts, sampling_params)

    completions_by_pid: Dict[int, List[str]] = {}
    for pid, out in enumerate(outputs):
        completions_by_pid[pid] = [seq.text or "" for seq in out.outputs]

    if args.save_raw:
        raw_path = RAW_DIR / f"countdown_raw_chimera_{args.tag}_{variant_name}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for pid, comps in completions_by_pid.items():
                for si, comp in enumerate(comps):
                    f.write(json.dumps(dict(
                        variant=variant_name,
                        problem_index=pid,
                        sample_index=si,
                        completion=comp,
                    ), ensure_ascii=False) + "\n")

    per_problem, agg = evaluate_completions_diversity(
        records, completions_by_pid, sol_sets
    )

    if hasattr(llm, "shutdown"):
        llm.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return per_problem, agg


# ---------------------------------------------------------------------------
# helper: parse ranges like "32-36,33-36"
# ---------------------------------------------------------------------------
def parse_ranges(text: str) -> List[Tuple[int, int]]:
    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ranges.append((int(a.strip()), int(b.strip())))
        else:
            v = int(part)
            ranges.append((v, v + 1))
    return ranges


def parse_int_list(text: str) -> List[int]:
    result = []
    for part in text.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    configure_vllm_worker_multiprocessing()

    cut_layers = parse_int_list(args.cut_layers)
    inject_one_layers = parse_int_list(args.inject_one_layers)
    remove_one_layers = parse_int_list(args.remove_one_layers)
    inject_ranges = parse_ranges(args.inject_ranges)

    explore_name = Path(args.explore_model_path).name
    collapse_name = Path(args.collapse_model_path).name
    tag = args.tag or f"{explore_name}_vs_{collapse_name}_n{args.n_samples}"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.explore_model_path, trust_remote_code=True)
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
    print("[cross_checkpoint] Loading explore and collapse models for weight extraction ...")

    dtype_torch = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    explore_model = AutoModelForCausalLM.from_pretrained(
        args.explore_model_path, torch_dtype=dtype_torch, trust_remote_code=True,
    )
    explore_sd = {k: v.cpu().clone() for k, v in explore_model.state_dict().items()}
    num_layers = len([k for k in explore_sd if k.startswith("model.layers.") and k.endswith(".self_attn.q_proj.weight")])
    if num_layers == 0:
        # fallback
        config = AutoConfig.from_pretrained(args.explore_model_path, trust_remote_code=True)
        num_layers = get_num_layers_from_config(config)
    del explore_model
    gc.collect()

    collapse_model = AutoModelForCausalLM.from_pretrained(
        args.collapse_model_path, torch_dtype=dtype_torch, trust_remote_code=True,
    )
    collapse_sd = {k: v.cpu().clone() for k, v in collapse_model.state_dict().items()}
    del collapse_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[cross_checkpoint] Model has {num_layers} layers.")

    config = AutoConfig.from_pretrained(args.explore_model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "qwen2")
    if model_type not in ("qwen2", "gpt2"):
        print(f"[cross_checkpoint] Warning: unknown model_type '{model_type}', defaulting to qwen2")
        model_type = "qwen2"

    # Validate layers
    valid_cuts = [c for c in cut_layers if 0 < c <= num_layers]
    valid_inject_one = [l for l in inject_one_layers if 0 <= l < num_layers]
    valid_remove_one = [l for l in remove_one_layers if 0 <= l < num_layers]
    valid_inject_ranges = [(s, e) for s, e in inject_ranges if 0 <= s < e <= num_layers]

    # ------------------------------------------------------------------
    # 3. Build variants
    # ------------------------------------------------------------------
    chimera_dir_base = get_temp_model_base_dir(args.tmp_dir, tag)

    # Use explore model path as config source (both should have same config)
    config_source = args.explore_model_path

    variants: List[Tuple[str, str]] = []  # (variant_name, model_path)

    # Pure controls
    if not args.skip_controls:
        variants.append(("pure_explore", args.explore_model_path))
        variants.append(("pure_collapse", args.collapse_model_path))

    def _build_and_save(vname: str, mode: str, mode_params: dict):
        vdir = chimera_dir_base / vname
        if not vdir.exists():
            print(f"[cross_checkpoint] Building chimera: {vname} ...")
            sd = build_chimera_state_dict(
                explore_sd, collapse_sd, num_layers, mode, mode_params, model_type
            )
            save_chimera_model(sd, config_source, vdir)
            del sd
            gc.collect()
        variants.append((vname, str(vdir)))

    # Forward: explore[0:L] + collapse[L:end]
    if args.include_forward and not args.skip_forward:
        for cut in valid_cuts:
            _build_and_save(f"fwd_cut{cut}", "forward", {"cut_layer": cut})

    # Reverse: collapse[0:L] + explore[L:end]
    if args.include_reverse and not args.skip_reverse:
        for cut in valid_cuts:
            _build_and_save(f"rev_cut{cut}", "reverse", {"cut_layer": cut})

    # Head-only: explore layers + collapse head
    if args.include_head_only and not args.skip_head_only:
        _build_and_save("head_only_collapse", "head_only", {})

    # Head-reverse: collapse layers + explore head
    if args.include_head_reverse and not args.skip_head_reverse:
        _build_and_save("head_only_explore", "head_reverse", {})

    # Norm swap: collapse + explore's final norm
    if args.include_norm_swap and not args.skip_norm_swap:
        _build_and_save("norm_swap_explore", "norm_swap", {})

    # Single-layer injection: collapse + one explore layer
    for l in valid_inject_one:
        _build_and_save(f"inject_L{l}", "inject_one", {"layer": l})

    # Single-layer removal: explore + one collapse layer
    for l in valid_remove_one:
        _build_and_save(f"remove_L{l}", "remove_one", {"layer": l})

    # Range injection: collapse + explore layers [start, end)
    for s, e in valid_inject_ranges:
        _build_and_save(f"inject_R{s}_{e}", "inject_range", {"start": s, "end": e})

    # Free state dicts
    del explore_sd, collapse_sd
    gc.collect()

    print(f"[cross_checkpoint] Total variants to evaluate: {len(variants)}")

    # ------------------------------------------------------------------
    # 4. Evaluate each variant
    # ------------------------------------------------------------------
    summary_rows: List[dict] = []
    all_per_problem: List[dict] = []

    for vname, vpath in variants:
        per_problem, agg = run_variant(
            vpath, vname, records, prompt_texts, sol_sets, args, tokenizer
        )

        summary_rows.append(dict(
            variant=vname,
            explore_model=explore_name,
            collapse_model=collapse_name,
            n_samples=args.n_samples,
            **agg,
        ))

        if args.save_per_problem:
            for pid, metrics in per_problem.items():
                all_per_problem.append(dict(
                    variant=vname,
                    problem_index=pid,
                    **metrics,
                ))

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(summary_rows)
    out_csv = METRICS_DIR / f"cross_checkpoint_chimera_summary_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[cross_checkpoint] Saved summary → {out_csv}")
    print(df.to_string(index=False))

    if args.save_per_problem and all_per_problem:
        df_pp = pd.DataFrame(all_per_problem)
        out_parq = METRICS_DIR / f"cross_checkpoint_chimera_per_problem_{tag}.parquet"
        df_pp.to_parquet(out_parq, index=False)
        print(f"[cross_checkpoint] Saved per-problem → {out_parq}")

    # ------------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------------
    print(f"[cross_checkpoint] Cleaning up temporary chimera models in {chimera_dir_base} ...")
    shutil.rmtree(str(chimera_dir_base), ignore_errors=True)
    print("[cross_checkpoint] Done.")


if __name__ == "__main__":
    main()
