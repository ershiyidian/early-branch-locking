#!/usr/bin/env python3
"""Open-math per-token base/RLVR NLL localization.

Selection is made from the existing execution_training score ledger, restricted to
base-origin, official-correct, non-leaking trajectories with a detected first
calculation boundary.  Both models teacher-force the same stored completion;
the output is therefore a localization diagnostic, not a generation-level
causal comparison.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.core.branch_protocol import milestone_record
from early_branch_locking.core.math_trace_utils import extract_first_calc_branch
from early_branch_locking.math_transfer.analyze_segment_suppression import minimal_style_normalize

DEFAULT_SCORES = ROOT / "data" / "rlvr" / "outputs" / "e2" / "qwen_7b_base_rl" / "scores.jsonl"
DEFAULT_OUT = ROOT / "data" / "rlvr" / "outputs" / "e2_token_profile"
MODEL_DIRS = {"math_base_7b": "math_base_7b", "math_simple_rl_7b": "math_simple_rl_7b"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--models-root", type=Path, default=ROOT / "model")
    parser.add_argument("--base-model", default="math_base_7b")
    parser.add_argument("--rl-model", default="math_simple_rl_7b")
    parser.add_argument("--benchmarks", default="gsm8k,math500")
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--relative-window", type=int, default=10)
    return parser.parse_args(argv)


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_rows(rows: list[dict], benchmark: str, base_model: str, cap: int) -> list[dict]:
    candidates = []
    for source_row, row in enumerate(rows):
        if str(row.get("benchmark", "")) != benchmark or str(row.get("model", "")) != base_model:
            continue
        if not truthy(row.get("official_correct")) or truthy(row.get("answer_leakage_before_p2")):
            continue
        if row.get("p1_char_end") is None or (isinstance(row.get("p1_char_end"), float) and math.isnan(row.get("p1_char_end"))):
            continue
        if str(row.get("base_score_status", "")) != "ok" or str(row.get("rl_score_status", "")) != "ok":
            continue
        completion = str(row.get("completion", ""))
        question = str(row.get("question", ""))
        if not completion.strip() or not question.strip():
            continue
        item = dict(row)
        item["source_row"] = source_row
        item["selection_id"] = f"{benchmark}:{row.get('problem_id')}:{row.get('sample_index')}:{row.get('completion_sha1', hashlib.sha1(completion.encode()).hexdigest())}"
        candidates.append(item)
    candidates.sort(key=lambda row: (str(row.get("problem_id", "")), int(row.get("sample_index", 0) or 0), str(row.get("completion_sha1", "")), int(row["source_row"])))
    selected = []
    seen_problems = set()
    for row in candidates:
        pid = str(row.get("problem_id", ""))
        if pid in seen_problems:
            continue
        selected.append(row)
        seen_problems.add(pid)
        if len(selected) >= cap:
            break
    if len(selected) < cap:
        seen_ids = {row["selection_id"] for row in selected}
        for row in candidates:
            if row["selection_id"] in seen_ids:
                continue
            selected.append(row)
            if len(selected) >= cap:
                break
    return selected


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def expand_variants(selected: list[dict]) -> tuple[list[dict], dict]:
    """Expand selected raw trajectories into auditable style variants.

    The raw score-ledger provenance and leakage filter remain the selection
    authority.  A normalized copy is retained only when the minimal
    typography/whitespace transform changes text without changing the first
    calculation branch, removing the first-calculation boundary, or creating
    an early answer-leakage flag.  Its milestone record is recomputed from the
    transformed text.
    """
    expanded = []
    counts = {
        "selected_raw_rows": len(selected),
        "raw_variants": 0,
        "normalized_candidates": 0,
        "normalized_variants": 0,
        "normalized_unchanged": 0,
        "normalized_branch_changed": 0,
        "normalized_no_p1": 0,
        "normalized_leakage": 0,
    }
    for row in selected:
        completion = str(row.get("completion", ""))
        ground_truth = str(row.get("ground_truth", row.get("gt", "")))
        raw_marks = milestone_record(completion, ground_truth)
        raw_branch = extract_first_calc_branch(completion)
        raw_variant = dict(row)
        raw_variant.update(
            {
                "completion": completion,
                "ground_truth": ground_truth,
                "style_variant": "raw",
                "variant_milestones": raw_marks,
                "variant_branch": raw_branch,
                "source_p1_char_end": row.get("p1_char_end"),
                "source_answer_leakage_before_p2": truthy(row.get("answer_leakage_before_p2")),
            }
        )
        expanded.append(raw_variant)
        counts["raw_variants"] += 1

        normalized = minimal_style_normalize(completion)
        if normalized == completion:
            counts["normalized_unchanged"] += 1
            continue
        counts["normalized_candidates"] += 1
        normalized_marks = milestone_record(normalized, ground_truth)
        normalized_branch = extract_first_calc_branch(normalized)
        if normalized_branch != raw_branch:
            counts["normalized_branch_changed"] += 1
            continue
        if _missing(normalized_marks.get("p1_char_end")):
            counts["normalized_no_p1"] += 1
            continue
        if truthy(normalized_marks.get("answer_leakage_before_p2")):
            counts["normalized_leakage"] += 1
            continue
        normalized_variant = dict(row)
        normalized_variant.update(
            {
                "completion": normalized,
                "ground_truth": ground_truth,
                "style_variant": "minimal_normalized",
                "variant_milestones": normalized_marks,
                "variant_branch": normalized_branch,
                "source_p1_char_end": row.get("p1_char_end"),
                "source_answer_leakage_before_p2": truthy(row.get("answer_leakage_before_p2")),
            }
        )
        expanded.append(normalized_variant)
        counts["normalized_variants"] += 1
    counts["expanded_variants"] = len(expanded)
    return expanded, counts


def prompt_for(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def make_ids(tokenizer, question: str, completion: str, max_input_tokens: int) -> tuple[list[int], int, int, bool]:
    prompt_ids = tokenizer.encode(prompt_for(question), add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    full = prompt_ids + completion_ids
    truncated = len(full) > max_input_tokens
    if truncated:
        full = full[:max_input_tokens]
        completion_ids = full[len(prompt_ids):]
    return full, len(prompt_ids), len(completion_ids), truncated


def score_one(model, tokenizer, row: dict, max_input_tokens: int, device: torch.device) -> dict:
    full, prompt_len, completion_tokens, truncated = make_ids(tokenizer, row["question"], row["completion"], max_input_tokens)
    if completion_tokens <= 0 or prompt_len >= len(full):
        return {"nll": np.asarray([], dtype=float), "completion_tokens": completion_tokens, "prompt_tokens": prompt_len, "truncated": truncated, "score_status": "empty"}
    ids = torch.tensor([full], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    labels = ids[:, 1:]
    begin = prompt_len - 1
    end = begin + completion_tokens
    values = logp[0, begin:end].gather(-1, labels[0, begin:end].unsqueeze(-1)).squeeze(-1)
    return {"nll": (-values.detach().cpu().numpy()).astype(float), "completion_tokens": completion_tokens, "prompt_tokens": prompt_len, "truncated": truncated, "score_status": "ok"}


def boundary_token(tokenizer, completion: str, char_end) -> int | None:
    if _missing(char_end):
        return None
    return len(tokenizer.encode(completion[: int(char_end)], add_special_tokens=False))


def validate_tokenizers(base_tokenizer, rl_tokenizer, rows: list[dict]) -> dict:
    special_mismatches = {}
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        left = getattr(base_tokenizer, attr, None)
        right = getattr(rl_tokenizer, attr, None)
        if left != right:
            special_mismatches[attr] = {"base": left, "rl": right}
    if len(base_tokenizer) != len(rl_tokenizer):
        raise ValueError(f"tokenizer vocabulary mismatch: base={len(base_tokenizer)} rl={len(rl_tokenizer)}")
    probes = ["<|im_start|>assistant\n", " 1 + 2 = 3", "\\boxed{42}"]
    for probe in probes:
        if base_tokenizer.encode(probe, add_special_tokens=False) != rl_tokenizer.encode(probe, add_special_tokens=False):
            raise ValueError(f"tokenizer encoding mismatch for probe {probe!r}")
    for row in rows:
        question = prompt_for(str(row["question"]))
        completion = str(row["completion"])
        base_ids = base_tokenizer.encode(question + completion, add_special_tokens=False)
        rl_ids = rl_tokenizer.encode(question + completion, add_special_tokens=False)
        if base_ids != rl_ids:
            raise ValueError(
                f"tokenizer encoding mismatch for selected completion "
                f"{row['selection_id']} style={row.get('style_variant', 'raw')}"
            )
    return special_mismatches


def bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float, int, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(len(values)), int(len(values))


def main(argv=None):
    args = parse_args(argv)
    if not args.scores.exists():
        raise FileNotFoundError(args.scores)
    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    rows = load_scores(args.scores)
    selected = {bench: select_rows(rows, bench, args.base_model, args.max_rows) for bench in benchmarks}
    selected_all = [row for bench in benchmarks for row in selected[bench]]
    expanded, variant_counts = expand_variants(selected_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(str(args.gpu).split(",")[0]))
    raw_rows: list[dict] = []
    model_paths = {alias: args.models_root / MODEL_DIRS[alias] for alias in (args.base_model, args.rl_model)}
    base_tokenizer = AutoTokenizer.from_pretrained(model_paths[args.base_model], local_files_only=True, trust_remote_code=False, use_fast=False)
    rl_tokenizer = AutoTokenizer.from_pretrained(model_paths[args.rl_model], local_files_only=True, trust_remote_code=False, use_fast=False)
    special_token_mismatches = validate_tokenizers(base_tokenizer, rl_tokenizer, expanded)
    sequence_audit: list[dict] = []
    for alias in (args.base_model, args.rl_model):
        model_path = model_paths[alias]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        tokenizer = base_tokenizer if alias == args.base_model else rl_tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, low_cpu_mem_usage=True,
            **({"attn_implementation": "sdpa"} if device.type == "cuda" else {}),
        ).to(device).eval()
        print(f"[D5] model={alias} variants={len(expanded)} device={device}", flush=True)
        for index, row in enumerate(expanded, start=1):
            result = score_one(model, tokenizer, row, args.max_input_tokens, device)
            milestones = row["variant_milestones"]
            boundary = boundary_token(tokenizer, row["completion"], milestones.get("p1_char_end"))
            if boundary is None:
                raise ValueError(
                    f"missing first-calculation boundary for {row['selection_id']} "
                    f"style={row['style_variant']}"
                )
            boundary = min(max(boundary, 0), len(result["nll"]))
            variant_sha1 = hashlib.sha1(row["completion"].encode("utf-8")).hexdigest()
            sequence_audit.append(
                {
                    "selection_id": row["selection_id"],
                    "style_variant": row["style_variant"],
                    "model": alias,
                    "completion_tokens": int(result["completion_tokens"]),
                    "prompt_tokens": int(result["prompt_tokens"]),
                    "p1_token_boundary": int(boundary),
                    "truncated": bool(result["truncated"]),
                    "score_status": result["score_status"],
                }
            )
            for token_index, nll in enumerate(result["nll"]):
                raw_rows.append({
                    "benchmark": str(row["benchmark"]),
                    "selection_id": row["selection_id"],
                    "problem_id": str(row.get("problem_id", "")),
                    "source_row": int(row["source_row"]),
                    "sample_index": int(row.get("sample_index", -1)),
                    "source_file": str(row.get("source_file", "")),
                    "completion_sha1": str(row.get("completion_sha1", "")),
                    "variant_completion_sha1": variant_sha1,
                    "official_correct": True,
                    "answer_leakage_before_p2": False,
                    "model": alias,
                    "style_variant": str(row["style_variant"]),
                    "token_index": int(token_index),
                    "relative_pos": int(token_index - boundary),
                    "nll": float(nll),
                    "p1_char_end": int(milestones["p1_char_end"]),
                    "source_p1_char_end": int(row["source_p1_char_end"]),
                    "p1_token_boundary": int(boundary),
                    "completion_tokens": int(result["completion_tokens"]),
                    "prompt_tokens": int(result["prompt_tokens"]),
                    "truncated": bool(result["truncated"]),
                    "score_status": result["score_status"],
                })
            if index == 1 or index % 10 == 0 or index == len(expanded):
                print(f"[D5] model={alias} scored={index}/{len(expanded)}", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    raw = pd.DataFrame(raw_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "e2_token_profile_raw_v1.jsonl"
    raw.to_json(raw_path, orient="records", lines=True)
    output_paths = []
    for bench in benchmarks:
        subset = raw[raw.benchmark.eq(bench)] if not raw.empty else pd.DataFrame()
        base = subset[subset.model.eq(args.base_model)][["selection_id", "style_variant", "relative_pos", "nll", "problem_id"]].rename(columns={"nll": "base_nll"})
        rl = subset[subset.model.eq(args.rl_model)][["selection_id", "style_variant", "relative_pos", "nll"]].rename(columns={"nll": "rl_nll"})
        merged = base.merge(rl, on=["selection_id", "style_variant", "relative_pos"], how="inner")
        if merged.empty:
            profile = pd.DataFrame()
        else:
            merged["delta_nll_rl_minus_base"] = merged["rl_nll"] - merged["base_nll"]
            merged = merged[merged.relative_pos.between(-args.relative_window, args.relative_window)]
            profile_rows = []
            for (style_variant, rel), group in merged.groupby(["style_variant", "relative_pos"], sort=True):
                base_mean, base_lo, base_hi, n, _ = bootstrap(group.groupby("problem_id", as_index=False).base_nll.mean().base_nll.to_numpy(), args.bootstrap_draws, args.bootstrap_seed + int(rel) + 100)
                rl_mean, rl_lo, rl_hi, _, _ = bootstrap(group.groupby("problem_id", as_index=False).rl_nll.mean().rl_nll.to_numpy(), args.bootstrap_draws, args.bootstrap_seed + int(rel) + 200)
                delta_mean, delta_lo, delta_hi, _, _ = bootstrap(group.groupby("problem_id", as_index=False).delta_nll_rl_minus_base.mean().delta_nll_rl_minus_base.to_numpy(), args.bootstrap_draws, args.bootstrap_seed + int(rel) + 300)
                profile_rows.append({"benchmark": bench, "style_variant": str(style_variant), "relative_pos": int(rel), "n_token_pairs": int(len(group)), "n_problems": int(group.problem_id.nunique()), "base_mean_nll": base_mean, "base_ci_lo": base_lo, "base_ci_hi": base_hi, "rl_mean_nll": rl_mean, "rl_ci_lo": rl_lo, "rl_ci_hi": rl_hi, "delta_nll_rl_minus_base": delta_mean, "delta_ci_lo": delta_lo, "delta_ci_hi": delta_hi, "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": args.bootstrap_seed, "boundary": "variant-specific milestone_record p1_char_end re-encoded with slow tokenizer"})
            profile = pd.DataFrame(profile_rows)
        output_path = args.out_dir / f"e2_token_profile_{bench}.csv"
        profile.to_csv(output_path, index=False)
        output_paths.append(output_path)
    selection_manifest = {
        bench: {"selected_rows": len(selected[bench]), "unique_problems": len({row.get("problem_id") for row in selected[bench]}), "selection": "base-origin model row; official_correct; answer_leakage_before_p2=false; p1_char_end present; one earliest deterministic row per problem before cap"}
        for bench in benchmarks
    }
    manifest = {
        "artifact": "D5_e2_token_profile_v1",
        "schema_version": 2,
        "status": "complete" if not raw.empty else "empty",
        "benchmarks": benchmarks,
        "base_model": args.base_model,
        "rl_model": args.rl_model,
        "selected": selection_manifest,
        "variants": variant_counts,
        "style_variants": ["raw", "minimal_normalized"],
        "normalization_policy": {
            "name": "minimal_style_normalize",
            "operations": [
                "replace curly single/double quotes with ASCII quotes",
                "collapse horizontal spaces/tabs to one space",
                "collapse runs of three or more newlines to two",
                "strip leading/trailing whitespace",
            ],
            "retention": "include only changed text whose first-calculation branch and non-leakage status are preserved and whose milestone_record has p1_char_end",
            "boundary": "recompute milestone_record and tokenize the normalized completion independently",
        },
        # Keep the historical raw_token_rows field for the JSONL row count,
        # and expose style-specific counts now that the file contains both
        # raw and minimal-normalized variants.
        "raw_token_rows": int(len(raw)),
        "token_rows": int(len(raw)),
        "raw_style_token_rows": int(raw.style_variant.eq("raw").sum()) if not raw.empty else 0,
        "minimal_normalized_style_token_rows": int(raw.style_variant.eq("minimal_normalized").sum()) if not raw.empty else 0,
        "raw_token_rows_semantics": "all style variants stored in e2_token_profile_raw_v1.jsonl",
        "scored_variant_model_sequences": len(sequence_audit),
        "truncated_variant_model_sequences": int(sum(item["truncated"] for item in sequence_audit)),
        "non_ok_variant_model_sequences": int(sum(item["score_status"] != "ok" for item in sequence_audit)),
        "sequence_audit": sequence_audit,
        "tokenizer": {
            "use_fast": False,
            "boundary_method": "character-prefix re-encoding",
            "offset_mapping": False,
            "vocabulary_and_selected_content_encoding_match": True,
            "special_token_mismatches": special_token_mismatches,
            "special_token_mismatch_allowed": "stored completions are teacher-forced without appending a model-specific EOS token",
        },
        "leakage_filter": "answer_leakage_before_p2 == false",
        "correctness_filter": "official_correct == true and both stored aggregate score_status fields == ok",
        "delta_definition": "RL NLL - base NLL on the same stored completion token",
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.bootstrap_seed, "unit": "problem_id"},
        "interpretation": "teacher-forced per-token localization diagnostic; not a generation-level causal estimate",
        "input": str(args.scores),
        "input_sha256": sha256(args.scores),
        "aggregate_style_variants": sorted(raw.style_variant.unique().tolist()) if not raw.empty else [],
        "outputs": [raw_path.name, *[path.name for path in output_paths]],
    }
    manifest_path = args.out_dir / "e2_token_profile_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": {key: len(value) for key, value in selected.items()}, "raw_rows": len(raw), "outputs": [str(path) for path in output_paths]}, sort_keys=True))


if __name__ == "__main__":
    main()
