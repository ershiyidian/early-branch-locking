
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""representation_geometry - Representation geometry of correct solutions.
Hypothesis: RLVR training reduces geometric separability and effective rank among alternative valid solution traces.
Inputs: teacher-forced correct Countdown completions; dataset/test.parquet; checkpoint model paths.
Outputs: data/analysis_results/rlvr_passk/metrics/representation_geometry_repr_geometry_summary_geom_v3_teacher50.csv; data/analysis_results/rlvr_passk/metrics/representation_geometry_repr_geometry_per_problem_geom_v3_teacher50.parquet
Status: paper-main
"""
from __future__ import annotations
"""
representation_geometry_representation_geometry.py

Experiment L v3 — Representation Geometry of Correct Solutions
==============================================================

Key change from v2: uses step50 as teacher (not base model).
The base model cannot produce valid Countdown format, so its completions
cannot serve as teacher trajectories. Step 50 is the earliest checkpoint
with both valid format and high solution diversity.

For each checkpoint, for each problem with multiple correct solution classes,
we extract hidden states (teacher-forced) at landmark positions and measure
whether different solutions are geometrically separable.

Metrics per (checkpoint, layer, position, problem):
    mean_cosine_dist    – average pairwise cosine distance between
                          hidden states of *different* solution classes
    within_cosine_dist  – average pairwise cosine distance *within*
                          the same solution class
    separability_ratio  – mean_cosine_dist / within_cosine_dist
    effective_rank      – participation ratio of PCA eigenvalues
    cluster_purity      – k-means cluster-label match

If training causes representation collapse, we expect:
    • separability_ratio to DECREASE
    • effective_rank to DECREASE
    • cluster_purity to DECREASE

The checkpoint sweep should start from step50 (not base), since base
cannot generate the task format and thus teacher-forced hidden states
from base on formatted completions are not meaningful.

Outputs
-------
- metrics/representation_geometry_repr_geometry_summary_{tag}.csv
- metrics/representation_geometry_repr_geometry_per_problem_{tag}.parquet  (optional)
"""

import argparse
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (  # noqa: E402
    COUNTDOWN_ACTOR_DIR as ACTOR_DIR,
    METRICS_DIR,
    RAW_DIR,
    TEST_PARQUET,
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_shared import (  # noqa: E402
    load_parquet_sorted,
    extract_ground_truth,
    get_prompt_content,
    build_prompt_text,
    collect_model_paths as _collect_model_paths,
    step_of,
)
from early_branch_locking.core.prefix_utils import locate_positions_from_text  # noqa: E402
from early_branch_locking.core.op1_utils import (  # noqa: E402
    load_raw_indexed,
    pick_diverse_success_completions,
    get_solution_class_count,
    get_op_token_ids,
    apply_final_norm,
    get_layers_container,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEED = 42

POSITION_NAMES = ["think_end", "feasible_end", "answer_start", "op1_before"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ExpL v3: representation geometry (step50 as teacher)")
    p.add_argument("--teacher_raw_path", type=str, required=True,
                   help="Raw jsonl from teacher checkpoint (should be step50, NOT base)")
    p.add_argument("--actor_dir", type=str, default=str(ACTOR_DIR))
    p.add_argument("--only_steps", type=str, default="",
                   help="Comma-separated step numbers to evaluate (e.g. '50,100,150,200,275')")
    p.add_argument("--model_paths", type=str, nargs="*", default=None,
                   help="Explicit model paths to evaluate (overrides actor_dir scanning)")
    p.add_argument("--num_problems", type=int, default=150)
    p.add_argument("--max_per_solution", type=int, default=5,
                   help="max completions per solution class per problem")
    p.add_argument("--min_classes", type=int, default=2)
    p.add_argument("--min_samples_per_class", type=int, default=2)
    p.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,28,30,32,33,34,35,36",
                   help="Layer indices to extract hidden states from")
    p.add_argument("--positions", type=str, default=",".join(POSITION_NAMES))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    # v3 flags
    p.add_argument("--logit_space", action="store_true", default=False,
                   help="additionally compute geometry in logit space")
    p.add_argument("--logit_op_only", action="store_true", default=False,
                   help="restrict logit-space geometry to operator token subspace")
    p.add_argument("--weight_delta", action="store_true", default=False,
                   help="compute weight-delta direction analysis between consecutive checkpoints")
    p.add_argument("--weight_delta_ref_path", type=str, default="",
                   help="Reference model for weight delta (default: first model in sweep)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# geometry metrics
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - dot / (na * nb)


def mean_between_class_distance(
    vecs: List[np.ndarray],
    labels: List[int],
) -> float:
    dists = []
    n = len(vecs)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                dists.append(cosine_distance(vecs[i], vecs[j]))
    return float(np.mean(dists)) if dists else float("nan")


def mean_within_class_distance(
    vecs: List[np.ndarray],
    labels: List[int],
) -> float:
    dists = []
    n = len(vecs)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                dists.append(cosine_distance(vecs[i], vecs[j]))
    return float(np.mean(dists)) if dists else float("nan")


def effective_rank(vecs: List[np.ndarray]) -> float:
    if len(vecs) < 2:
        return float("nan")
    X = np.stack(vecs, axis=0)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    lambdas = s ** 2
    total = lambdas.sum()
    if total < 1e-12:
        return float("nan")
    return float(total ** 2 / (lambdas ** 2).sum())


def cluster_purity(
    vecs: List[np.ndarray],
    labels: List[int],
    n_classes: int,
) -> float:
    if len(vecs) < n_classes or n_classes < 2:
        return float("nan")

    from sklearn.cluster import KMeans
    from collections import Counter as _Counter

    X = np.stack(vecs, axis=0)
    y_true = np.array(labels)

    km = KMeans(n_clusters=n_classes, random_state=SEED, n_init=10, max_iter=300)
    y_pred = km.fit_predict(X)

    cluster_counts: Dict[int, _Counter] = {}
    for c in range(n_classes):
        mask = y_pred == c
        cluster_counts[c] = _Counter(y_true[mask].tolist())

    remaining_clusters = list(range(n_classes))
    remaining_labels = set(np.unique(y_true).tolist())
    matched = 0
    for _ in range(n_classes):
        best_c, best_l, best_count = -1, -1, -1
        for c in remaining_clusters:
            for l in remaining_labels:
                cnt = cluster_counts[c].get(l, 0)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    best_l = l
        if best_c >= 0 and best_l >= 0:
            matched += best_count
            remaining_clusters.remove(best_c)
            remaining_labels.discard(best_l)

    return matched / len(vecs) if len(vecs) > 0 else float("nan")


# ---------------------------------------------------------------------------
# logit-space projection
# ---------------------------------------------------------------------------

def project_hidden_to_logits(
    model,
    h_vec: np.ndarray,
    device: torch.device,
    op_token_ids: Optional[List[int]] = None,
) -> np.ndarray:
    model_dtype = next(model.lm_head.parameters()).dtype
    h_t = torch.tensor(h_vec, dtype=model_dtype).unsqueeze(0).to(device)
    with torch.no_grad():
        h_normed = apply_final_norm(model, h_t)
        logits = model.lm_head(h_normed)
    if op_token_ids is not None:
        return logits[0, op_token_ids].float().cpu().numpy()
    return logits[0].float().cpu().numpy()


def compute_logit_space_geometry(
    model,
    vecs: List[np.ndarray],
    labels: List[int],
    device: torch.device,
    op_token_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    logit_vecs = []
    for h_vec in vecs:
        lv = project_hidden_to_logits(model, h_vec, device, op_token_ids)
        logit_vecs.append(lv)

    n_classes = len(set(labels))
    between = mean_between_class_distance(logit_vecs, labels)
    within = mean_within_class_distance(logit_vecs, labels)
    er = effective_rank(logit_vecs)
    purity = cluster_purity(logit_vecs, labels, n_classes) if n_classes >= 2 else float("nan")

    sep_ratio = float("nan")
    if not math.isnan(between) and not math.isnan(within) and within > 1e-12:
        sep_ratio = between / within

    return dict(
        logit_between_class_cosine_dist=between,
        logit_within_class_cosine_dist=within,
        logit_separability_ratio=sep_ratio,
        logit_effective_rank=er,
        logit_cluster_purity=purity,
    )


def compute_weight_delta_metrics(
    ref_model,
    current_model,
    vecs_by_class: Dict[int, List[np.ndarray]],
    device: torch.device,
) -> Dict[str, float]:
    with torch.no_grad():
        W_ref = ref_model.lm_head.weight.to(device).float()
        W_current = current_model.lm_head.weight.to(device).float()
        W_delta = W_current - W_ref

    frobenius = float(torch.norm(W_delta, p="fro").item())

    class_ids = sorted(vecs_by_class.keys())
    if len(class_ids) < 2:
        return dict(
            weight_delta_frobenius=frobenius,
            weight_delta_cosine_sim=float("nan"),
            weight_delta_norm_ratio=float("nan"),
        )

    class_means: Dict[int, torch.Tensor] = {}
    for cid in class_ids:
        arr = np.stack(vecs_by_class[cid], axis=0).mean(axis=0)
        class_means[cid] = torch.tensor(arr, dtype=torch.float32, device=device)

    delta_projections: Dict[int, torch.Tensor] = {}
    ref_projections: Dict[int, torch.Tensor] = {}
    for cid in class_ids:
        h = class_means[cid]
        delta_projections[cid] = W_delta @ h
        ref_projections[cid] = W_ref @ h

    cos_sims = []
    for i in range(len(class_ids)):
        for j in range(i + 1, len(class_ids)):
            ci, cj = class_ids[i], class_ids[j]
            cos = torch.nn.functional.cosine_similarity(
                delta_projections[ci].unsqueeze(0), delta_projections[cj].unsqueeze(0)
            ).item()
            cos_sims.append(cos)

    norm_ratios = []
    for cid in class_ids:
        delta_norm = torch.norm(delta_projections[cid]).item()
        ref_norm = torch.norm(ref_projections[cid]).item()
        if ref_norm > 1e-12:
            norm_ratios.append(delta_norm / ref_norm)

    return dict(
        weight_delta_frobenius=frobenius,
        weight_delta_cosine_sim=float(np.mean(cos_sims)) if cos_sims else float("nan"),
        weight_delta_norm_ratio=float(np.mean(norm_ratios)) if norm_ratios else float("nan"),
    )


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

def _collect_model_paths_from_args(args) -> List[str]:
    return _collect_model_paths(
        actor_dir=args.actor_dir,
        only_steps=args.only_steps,
        explicit_paths=args.model_paths,
    )


def ckpt_name(path: str) -> str:
    p = Path(path)
    if p.name.startswith("global_step_"):
        return p.name
    return p.name


# ---------------------------------------------------------------------------
# hidden state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states_batch(
    model,
    tokenizer,
    texts: List[str],
    prompt_lens: List[int],
    prompt_char_lens: List[int],
    layers: List[int],
    position_names: List[str],
    device: torch.device,
) -> List[Dict[str, Dict[int, np.ndarray]]]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states

    results = []
    for b in range(input_ids.shape[0]):
        seq_len = int(attention_mask[b].sum().item())
        ids_b = input_ids[b, :seq_len].tolist()
        pos_map = locate_positions_from_text(
            texts[b],
            prompt_char_lens[b],
            tokenizer,
            position_names,
            input_ids=ids_b,
        )

        sample_result: Dict[str, Dict[int, np.ndarray]] = {}
        for pos_name in position_names:
            tok_idx = pos_map.get(pos_name)
            if tok_idx is None or tok_idx < 0 or tok_idx >= seq_len:
                sample_result[pos_name] = None
                continue
            layer_dict = {}
            for li in layers:
                hs_idx = li + 1
                if hs_idx >= len(hidden_states):
                    continue
                h = hidden_states[hs_idx][b, tok_idx, :].float().cpu().numpy()
                layer_dict[li] = h
            sample_result[pos_name] = layer_dict
        results.append(sample_result)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    position_names = [x.strip() for x in args.positions.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # 1. Build completion records from teacher (step50)
    # ------------------------------------------------------------------
    teacher_raw_path = Path(args.teacher_raw_path)
    print(f"[representation_geometry] Teacher raw: {teacher_raw_path}")
    by_teacher = load_raw_indexed(teacher_raw_path)
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")

    model_paths = _collect_model_paths_from_args(args)
    if not model_paths:
        raise RuntimeError("No model paths found. Use --only_steps or --model_paths.")

    print(f"[representation_geometry] Models to evaluate: {[Path(p).name for p in model_paths]}")

    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # completion_records grouped by pid
    completion_records: Dict[int, List[Dict[str, Any]]] = {}
    problems_used: List[int] = []

    for pid, rec in enumerate(records):
        if pid >= args.num_problems:
            break
        n_classes = get_solution_class_count(by_teacher, pid)
        if n_classes < args.min_classes:
            continue

        diverse = pick_diverse_success_completions(
            by_teacher, pid, max_per_solution=args.max_per_solution
        )
        if not diverse:
            continue

        sol_ids_present = set(d["solution_id"] for d in diverse)
        if len(sol_ids_present) < args.min_classes:
            continue

        prompt_content = get_prompt_content(rec)
        prompt_text = build_prompt_text(prompt_content, tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        cr_list = []
        for d in diverse:
            cr_list.append(dict(
                completion=d["completion"],
                canonical_expr=d["canonical_expr"],
                opseq_label=d["opseq_label"],
                solution_id=d["solution_id"],
                prompt_text=prompt_text,
                prompt_len=prompt_len,
            ))
        completion_records[pid] = cr_list
        problems_used.append(pid)

    print(f"[representation_geometry] Problems with >= {args.min_classes} solution classes: {len(problems_used)}")
    total_completions = sum(len(v) for v in completion_records.values())
    print(f"[representation_geometry] Total completions to process: {total_completions}")
    if not problems_used:
        raise RuntimeError("No problems found. Lower --min_classes or use richer teacher raw.")

    # flatten for batch processing
    flat_records: List[Dict[str, Any]] = []
    flat_pids: List[int] = []
    for pid in problems_used:
        for cr in completion_records[pid]:
            flat_records.append(cr)
            flat_pids.append(pid)

    full_texts = [cr["prompt_text"] + cr["completion"] for cr in flat_records]
    prompt_lens = [cr["prompt_len"] for cr in flat_records]
    prompt_char_lens = [len(cr["prompt_text"]) for cr in flat_records]

    # ------------------------------------------------------------------
    # 2. Iterate over checkpoints
    # ------------------------------------------------------------------
    summary_rows: List[dict] = []
    per_problem_rows: List[dict] = []

    # v3: optionally load reference model for weight_delta
    ref_model_for_delta = None
    if args.weight_delta:
        ref_path = args.weight_delta_ref_path if args.weight_delta_ref_path else model_paths[0]
        print(f"[representation_geometry] Loading reference model for weight_delta: {Path(ref_path).name}")
        ref_model_for_delta = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
        ref_model_for_delta.eval()

    # op token ids
    op_token_ids_for_logit: Optional[List[int]] = None
    if args.logit_space and args.logit_op_only:
        op_token_ids_for_logit = get_op_token_ids(tokenizer)
        print(f"[representation_geometry] Logit space restricted to op tokens: {op_token_ids_for_logit}")

    for model_path in model_paths:
        ckpt = ckpt_name(model_path)
        step = step_of(ckpt)
        print(f"\n[representation_geometry] Loading {ckpt} ...")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
        model.eval()
        max_layer_idx = int(getattr(model.config, "num_hidden_layers", 0)) - 1
        active_layers = [li for li in layers if 0 <= li <= max_layer_idx]
        if not active_layers:
            print(f"[representation_geometry] Warning: no valid layers for {ckpt}, skipping")
            model.cpu()
            del model
            torch.cuda.empty_cache()
            continue

        # extract hidden states
        all_hidden: List[Dict[str, Dict[int, np.ndarray]]] = []
        for start in range(0, len(full_texts), args.batch_size):
            end = min(start + args.batch_size, len(full_texts))
            batch_results = extract_hidden_states_batch(
                model, tokenizer,
                full_texts[start:end],
                prompt_lens[start:end],
                prompt_char_lens[start:end],
                active_layers, position_names, device,
            )
            all_hidden.extend(batch_results)

        assert len(all_hidden) == len(flat_records)

        # group by pid
        by_pid: Dict[int, List[int]] = defaultdict(list)
        for idx, pid in enumerate(flat_pids):
            by_pid[pid].append(idx)

        # compute geometry metrics
        for pos_name in position_names:
            for li in active_layers:
                prob_between: List[float] = []
                prob_within: List[float] = []
                prob_sep_ratio: List[float] = []
                prob_eff_rank: List[float] = []
                prob_purity: List[float] = []
                prob_logit_between: List[float] = []
                prob_logit_within: List[float] = []
                prob_logit_sep_ratio: List[float] = []
                prob_logit_eff_rank: List[float] = []
                prob_logit_purity: List[float] = []
                prob_wd_cosine_sim: List[float] = []
                prob_wd_norm_ratio: List[float] = []

                for pid in problems_used:
                    idxs = by_pid.get(pid, [])
                    vecs: List[np.ndarray] = []
                    labels: List[int] = []

                    for idx in idxs:
                        h_dict = all_hidden[idx].get(pos_name)
                        if h_dict is None:
                            continue
                        h_vec = h_dict.get(li)
                        if h_vec is None:
                            continue
                        vecs.append(h_vec)
                        labels.append(flat_records[idx]["solution_id"])

                    if len(vecs) < 2 or len(set(labels)) < 2:
                        continue

                    n_classes = len(set(labels))

                    # hidden space geometry
                    between = mean_between_class_distance(vecs, labels)
                    within = mean_within_class_distance(vecs, labels)
                    er = effective_rank(vecs)
                    purity = cluster_purity(vecs, labels, n_classes)

                    if not math.isnan(between):
                        prob_between.append(between)
                    if not math.isnan(within):
                        prob_within.append(within)
                    if not math.isnan(between) and not math.isnan(within) and within > 1e-12:
                        prob_sep_ratio.append(between / within)
                    if not math.isnan(er):
                        prob_eff_rank.append(er)
                    if not math.isnan(purity):
                        prob_purity.append(purity)

                    # logit space geometry
                    logit_metrics_row = {}
                    if args.logit_space:
                        logit_geom = compute_logit_space_geometry(
                            model, vecs, labels, device, op_token_ids_for_logit
                        )
                        for k_name in ["logit_between_class_cosine_dist",
                                       "logit_within_class_cosine_dist",
                                       "logit_separability_ratio",
                                       "logit_effective_rank",
                                       "logit_cluster_purity"]:
                            v = logit_geom[k_name]
                            if not math.isnan(v):
                                locals().setdefault(f"prob_{k_name}", []).append(v)
                        # Collect into named lists
                        lb = logit_geom["logit_between_class_cosine_dist"]
                        lw = logit_geom["logit_within_class_cosine_dist"]
                        lsr = logit_geom["logit_separability_ratio"]
                        ler = logit_geom["logit_effective_rank"]
                        lp = logit_geom["logit_cluster_purity"]
                        if not math.isnan(lb):
                            prob_logit_between.append(lb)
                        if not math.isnan(lw):
                            prob_logit_within.append(lw)
                        if not math.isnan(lsr):
                            prob_logit_sep_ratio.append(lsr)
                        if not math.isnan(ler):
                            prob_logit_eff_rank.append(ler)
                        if not math.isnan(lp):
                            prob_logit_purity.append(lp)
                        logit_metrics_row = logit_geom

                    # weight delta
                    wd_metrics_row = {}
                    if args.weight_delta and ref_model_for_delta is not None:
                        vecs_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
                        for v, lab in zip(vecs, labels):
                            vecs_by_class[lab].append(v)
                        wd_metrics = compute_weight_delta_metrics(
                            ref_model_for_delta, model, vecs_by_class, device
                        )
                        wd_cs = wd_metrics["weight_delta_cosine_sim"]
                        wd_nr = wd_metrics["weight_delta_norm_ratio"]
                        if not math.isnan(wd_cs):
                            prob_wd_cosine_sim.append(wd_cs)
                        if not math.isnan(wd_nr):
                            prob_wd_norm_ratio.append(wd_nr)
                        wd_metrics_row = wd_metrics

                    if args.save_per_problem:
                        sep_ratio_val = between / within if (not math.isnan(within) and within > 1e-12 and not math.isnan(between)) else float("nan")
                        pp_row = dict(
                            checkpoint=ckpt,
                            step=step,
                            layer=li,
                            position=pos_name,
                            problem_index=pid,
                            n_classes=n_classes,
                            n_samples=len(vecs),
                            between_class_cosine_dist=between,
                            within_class_cosine_dist=within,
                            separability_ratio=sep_ratio_val,
                            effective_rank=er,
                            cluster_purity=purity,
                        )
                        pp_row.update(logit_metrics_row)
                        pp_row.update(wd_metrics_row)
                        per_problem_rows.append(pp_row)

                # summary row
                summary_row = dict(
                    checkpoint=ckpt,
                    step=step,
                    layer=li,
                    position=pos_name,
                    between_class_cosine_dist_mean=float(np.mean(prob_between)) if prob_between else float("nan"),
                    within_class_cosine_dist_mean=float(np.mean(prob_within)) if prob_within else float("nan"),
                    separability_ratio_mean=float(np.mean(prob_sep_ratio)) if prob_sep_ratio else float("nan"),
                    effective_rank_mean=float(np.mean(prob_eff_rank)) if prob_eff_rank else float("nan"),
                    cluster_purity_mean=float(np.mean(prob_purity)) if prob_purity else float("nan"),
                    n_problems=len(prob_between),
                )
                if args.logit_space:
                    summary_row["logit_between_class_cosine_dist_mean"] = float(np.mean(prob_logit_between)) if prob_logit_between else float("nan")
                    summary_row["logit_within_class_cosine_dist_mean"] = float(np.mean(prob_logit_within)) if prob_logit_within else float("nan")
                    summary_row["logit_separability_ratio_mean"] = float(np.mean(prob_logit_sep_ratio)) if prob_logit_sep_ratio else float("nan")
                    summary_row["logit_effective_rank_mean"] = float(np.mean(prob_logit_eff_rank)) if prob_logit_eff_rank else float("nan")
                    summary_row["logit_cluster_purity_mean"] = float(np.mean(prob_logit_purity)) if prob_logit_purity else float("nan")
                if args.weight_delta:
                    summary_row["weight_delta_cosine_sim_mean"] = float(np.mean(prob_wd_cosine_sim)) if prob_wd_cosine_sim else float("nan")
                    summary_row["weight_delta_norm_ratio_mean"] = float(np.mean(prob_wd_norm_ratio)) if prob_wd_norm_ratio else float("nan")
                    if ref_model_for_delta is not None:
                        with torch.no_grad():
                            W_ref_lmh = ref_model_for_delta.lm_head.weight.float().to(device)
                            W_curr_lmh = model.lm_head.weight.float().to(device)
                            summary_row["weight_delta_frobenius"] = float(torch.norm(W_curr_lmh - W_ref_lmh, p="fro").item())
                    else:
                        summary_row["weight_delta_frobenius"] = float("nan")

                summary_rows.append(summary_row)

        model.cpu()
        del model
        torch.cuda.empty_cache()
        print(f"[representation_geometry] {ckpt}: done")

    # free ref model
    if ref_model_for_delta is not None:
        ref_model_for_delta.cpu()
        del ref_model_for_delta
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    df = pd.DataFrame(summary_rows)
    df = df.sort_values(["step", "position", "layer"]).reset_index(drop=True)
    tag = args.tag or f"n{len(problems_used)}"
    out_csv = METRICS_DIR / f"representation_geometry_repr_geometry_summary_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[representation_geometry] Saved summary → {out_csv}")
    print(df.to_string(index=False))

    if args.save_per_problem and per_problem_rows:
        df_pp = pd.DataFrame(per_problem_rows)
        out_parq = METRICS_DIR / f"representation_geometry_repr_geometry_per_problem_{tag}.parquet"
        df_pp.to_parquet(out_parq, index=False)
        print(f"[representation_geometry] Saved per-problem → {out_parq}")


if __name__ == "__main__":
    main()
