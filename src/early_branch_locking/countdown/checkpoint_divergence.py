#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""checkpoint_divergence - Divergence-token localization.
Hypothesis: checkpoint trajectories first diverge at identifiable reasoning positions that predict later solution-access collapse.
Inputs: teacher-forced Countdown trajectories; dataset/test.parquet; checkpoint tokenizer/model paths.
Outputs: data/analysis_results/rlvr_passk/metrics/checkpoint_divergence_divergence_overview.csv; data/analysis_results/rlvr_passk/metrics/checkpoint_divergence_divergence_summary_expw_greedy_main_src50_cmp275_greedy.csv
Status: paper-appendix
"""
from __future__ import annotations
"""Locate divergence tokens on Countdown by teacher-forcing multiple checkpoints."""

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking.core.countdown_shared import load_parquet_sorted, step_of
from early_branch_locking.core.prompt_data import (
    METRICS_DIR,
    RAW_DIR,
    TEST_PARQUET,
    build_prompt_data,
    ensure_tokenizer_padding,
    single_gpu_id,
    torch_dtype_from_name,
)
from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR as ACTOR_DIR  # noqa: E402

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Countdown divergence-token localization")
    parser.add_argument("--source_model_path", type=str, required=True)
    parser.add_argument("--compare_model_paths", nargs="+", required=True)
    parser.add_argument("--completion_source", choices=["greedy", "raw_shortest_success"], default="greedy")
    parser.add_argument("--source_raw_path", type=str, default="")
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--generate_batch_size", type=int, default=8)
    parser.add_argument("--teacher_force_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--min_disagree_teachers", type=int, default=1)
    parser.add_argument("--allow_source_sampling_mismatch", action="store_true", default=False)
    parser.add_argument("--cpu_threads", type=int, default=56)
    parser.add_argument("--save_raw", action="store_true", default=False)
    parser.add_argument("--save_per_problem", action="store_true", default=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_tag(args: argparse.Namespace) -> str:
    source_name = Path(args.source_model_path).name
    compare_names = "_".join(Path(path).name.replace("global_step_", "s") for path in args.compare_model_paths)
    tag = args.tag or f"{args.completion_source}_{source_name}_vs_{compare_names}"
    return tag.replace(",", "_")


def load_prompt_subset(tokenizer, args: argparse.Namespace):
    records_all = load_parquet_sorted(TEST_PARQUET, n=None, sort_key="sample_id")
    start_index = max(args.start_index, 0)
    end_index = len(records_all) if args.end_index < 0 else min(args.end_index, len(records_all))
    records = records_all[start_index:end_index]
    if args.num_problems > 0:
        records = records[: args.num_problems]
    prompt_data = build_prompt_data(records, tokenizer)
    return prompt_data, start_index, start_index + len(prompt_data)


def collect_completions(model, tokenizer, prompt_data, args: argparse.Namespace):
    if args.completion_source == "greedy":
        return (
            list(prompt_data),
            generate_greedy_completions(
                model,
                tokenizer,
                prompt_data,
                batch_size=args.generate_batch_size,
                max_new_tokens=args.max_new_tokens,
                device=next(model.parameters()).device,
            ),
        )
    if not args.source_raw_path:
        raise ValueError("--source_raw_path is required for raw_shortest_success mode.")
    return load_shortest_success_completions(args.source_raw_path, prompt_data)


def summarize_all(traces, source_eval, compare_evals, tokenizer, args: argparse.Namespace):
    problem_rows = []
    raw_rows = []
    compare_names = list(compare_evals.keys())
    for index, trace in enumerate(traces):
        row, token_rows = summarize_trace(
            trace,
            token_metadata(trace, tokenizer),
            source_eval[index],
            {name: payload[index] for name, payload in compare_evals.items()},
            require_source_match=not args.allow_source_sampling_mismatch,
            min_disagree_teachers=args.min_disagree_teachers,
        )
        row["source_checkpoint"] = Path(args.source_model_path).name
        row["compare_checkpoints"] = ",".join(compare_names)
        problem_rows.append(row)
        raw_rows.extend(token_rows)
    return problem_rows, raw_rows


def save_outputs(tag: str, args: argparse.Namespace, summary_row: dict, problem_rows, raw_rows) -> None:
    pd.DataFrame([summary_row]).to_csv(METRICS_DIR / f"checkpoint_divergence_divergence_summary_{tag}.csv", index=False)
    if args.save_per_problem:
        pd.DataFrame(problem_rows).to_parquet(METRICS_DIR / f"checkpoint_divergence_divergence_per_problem_{tag}.parquet", index=False)
    if args.save_raw:
        write_jsonl(RAW_DIR / f"checkpoint_divergence_divergence_tokens_{tag}.jsonl", raw_rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(args.cpu_threads, 1))
    gpu_id = single_gpu_id(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint_divergence.")
    device = torch.device("cuda")
    dtype = torch_dtype_from_name(args.dtype)
    tag = make_tag(args)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_path, trust_remote_code=True)
    compare_tokenizers = [AutoTokenizer.from_pretrained(path, trust_remote_code=True) for path in args.compare_model_paths]
    ensure_tokenizer_padding(source_tokenizer)
    for tokenizer in compare_tokenizers:
        ensure_tokenizer_padding(tokenizer)
    prompt_data, start_index, end_index = load_prompt_subset(source_tokenizer, args)
    probe_text = prompt_data[0].prompt_text if prompt_data else "<think>1+2</think>"
    validate_tokenizers(source_tokenizer, compare_tokenizers, probe_text)
    source_model = load_model(args.source_model_path, dtype, args.attn_implementation, device)
    prompt_data, completions = collect_completions(source_model, source_tokenizer, prompt_data, args)
    traces = build_completion_traces(prompt_data, completions, source_tokenizer, args.completion_source)
    source_eval = teacher_force_argmax(source_model, source_tokenizer, traces, args.teacher_force_batch_size, device)
    compare_evals = {}
    del source_model
    gc.collect()
    torch.cuda.empty_cache()
    for model_path in args.compare_model_paths:
        model = load_model(model_path, dtype, args.attn_implementation, device)
        compare_evals[Path(model_path).name] = teacher_force_argmax(model, source_tokenizer, traces, args.teacher_force_batch_size, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    problem_rows, raw_rows = summarize_all(traces, source_eval, compare_evals, source_tokenizer, args)
    summary_row = {
        "tag": tag,
        "source_checkpoint": Path(args.source_model_path).name,
        "compare_checkpoints": ",".join(Path(path).name for path in args.compare_model_paths),
        "completion_source": args.completion_source,
        "step": step_of(Path(args.source_model_path).name),
        "num_compare_teachers": len(args.compare_model_paths),
        "min_disagree_teachers": args.min_disagree_teachers,
        "require_source_argmax_match": int(not args.allow_source_sampling_mismatch),
        "start_index": start_index,
        "end_index": end_index,
        **aggregate_summary_rows(problem_rows),
    }
    save_outputs(tag, args, summary_row, problem_rows, raw_rows)
    display = pd.DataFrame([summary_row])[[
        "tag",
        "completion_source",
        "num_problems",
        "completion_correct_mean",
        "divergence_fraction_mean",
        "first_divergence_index_mean",
        "divergence_think_count_mean",
        "divergence_answer_count_mean",
    ]]
    print(display.to_string(index=False), flush=True)


# ---- merged trace_utils mode ----
"""Model/runtime utilities for countdown divergence-token localization."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM

from early_branch_locking.core.prefix_utils import answer_span, char_pos_to_token_len, extract_answer_op_positions
from early_branch_locking.core.prompt_data import PromptExample
from early_branch_locking.core.op1_utils import load_raw_indexed, pick_shortest_success_completion


@dataclass(frozen=True)
class CompletionTrace:
    item: PromptExample
    completion: str
    source_mode: str
    full_text: str
    full_token_ids: List[int]
    prompt_char_count: int
    prompt_token_count: int


def validate_tokenizers(primary_tokenizer, other_tokenizers: Sequence, probe_text: str) -> None:
    base_ids = primary_tokenizer.encode(probe_text, add_special_tokens=False)
    for tokenizer in other_tokenizers:
        if len(primary_tokenizer) != len(tokenizer):
            raise ValueError("Tokenizer vocab mismatch across teacher checkpoints.")
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            if getattr(primary_tokenizer, attr, None) != getattr(tokenizer, attr, None):
                raise ValueError(f"Tokenizer mismatch on {attr}.")
        if tokenizer.encode(probe_text, add_special_tokens=False) != base_ids:
            raise ValueError("Tokenizer encoding mismatch on probe text.")


def load_shortest_success_completions(raw_path: str, prompt_data: Sequence[PromptExample]) -> tuple[List[PromptExample], List[str]]:
    by_problem = load_raw_indexed(Path(raw_path))
    kept_items: List[PromptExample] = []
    completions: List[str] = []
    for item in prompt_data:
        completion = pick_shortest_success_completion(by_problem, item.pid)
        if not completion:
            continue
        kept_items.append(item)
        completions.append(completion)
    if not kept_items:
        raise ValueError(f"No successful completions found in {raw_path}")
    return kept_items, completions


@torch.no_grad()
def generate_greedy_completions(
    model,
    tokenizer,
    prompt_data: Sequence[PromptExample],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> List[str]:
    completions: List[str] = []
    for start in range(0, len(prompt_data), batch_size):
        batch = prompt_data[start : start + batch_size]
        prompts = [item.prompt_text for item in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        outputs = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = enc["input_ids"].shape[1]
        for row in outputs.detach().cpu().tolist():
            generated = row[prompt_width:]
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated:
                generated = generated[: generated.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            completions.append(text.strip())
    return completions


def build_completion_traces(
    prompt_data: Sequence[PromptExample],
    completions: Sequence[str],
    tokenizer,
    source_mode: str,
) -> List[CompletionTrace]:
    traces: List[CompletionTrace] = []
    for item, completion in zip(prompt_data, completions):
        full_text = item.prompt_text + completion
        full_token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        traces.append(
            CompletionTrace(
                item=item,
                completion=completion,
                source_mode=source_mode,
                full_text=full_text,
                full_token_ids=full_token_ids,
                prompt_char_count=len(item.prompt_text),
                prompt_token_count=len(item.prompt_ids),
            )
        )
    return traces


@torch.no_grad()
def teacher_force_argmax(
    model: AutoModelForCausalLM,
    tokenizer,
    traces: Sequence[CompletionTrace],
    batch_size: int,
    device: torch.device,
) -> List[dict]:
    outputs: List[dict] = []
    for start in range(0, len(traces), batch_size):
        batch = traces[start : start + batch_size]
        enc = tokenizer(
            [trace.full_text for trace in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        logits = model(**enc).logits
        for batch_index, trace in enumerate(batch):
            seq_len = int(enc["attention_mask"][batch_index].sum().item())
            pad_offset = int(enc["input_ids"].shape[1] - seq_len)
            prompt_len = trace.prompt_token_count
            start = pad_offset + prompt_len
            end = pad_offset + seq_len
            gt_ids = enc["input_ids"][batch_index, start:end].detach().cpu().tolist()
            pred_ids = logits[batch_index, start - 1 : end - 1].argmax(dim=-1).detach().cpu().tolist()
            outputs.append({"gt_ids": gt_ids, "pred_ids": pred_ids})
    return outputs


def load_model(model_path: str, dtype: torch.dtype, attn_implementation: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    return model


def token_metadata(trace: CompletionTrace, tokenizer) -> List[dict]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer with offset mapping is required for divergence localization.")
    enc = tokenizer(
        trace.full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(pair) for pair in enc["offset_mapping"]]
    if list(enc["input_ids"]) != trace.full_token_ids:
        raise ValueError("Tokenization mismatch while building token metadata.")
    completion_answer = answer_span(trace.completion)
    answer_start = (
        trace.prompt_char_count + completion_answer[0]
        if completion_answer is not None
        else len(trace.full_text)
    )
    answer_close = (
        trace.prompt_char_count + completion_answer[1]
        if completion_answer is not None
        else len(trace.full_text)
    )
    completion_feasible = trace.completion.lower().find("<feasible>")
    feasible_pos = (
        trace.prompt_char_count + completion_feasible
        if completion_feasible >= 0
        else -1
    )
    op_tokens = _answer_op_token_positions(trace.full_text, trace.completion, trace.prompt_char_count, tokenizer)
    metadata: List[dict] = []
    for absolute_index in range(trace.prompt_token_count, len(trace.full_token_ids)):
        token_start, _ = offsets[absolute_index]
        phase = _phase_name(token_start, feasible_pos, answer_start, answer_close)
        region = _answer_region(absolute_index, phase, op_tokens)
        metadata.append(
            {
                "absolute_index": absolute_index,
                "completion_index": absolute_index - trace.prompt_token_count,
                "token_id": trace.full_token_ids[absolute_index],
                "token_text": tokenizer.decode(
                    [trace.full_token_ids[absolute_index]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "phase": phase,
                "answer_region": region,
            }
        )
    return metadata


def _answer_op_token_positions(full_text: str, completion: str, prompt_char_count: int, tokenizer) -> List[int]:
    positions = []
    for char_pos in extract_answer_op_positions(completion):
        token_pos = char_pos_to_token_len(full_text, prompt_char_count + char_pos, tokenizer)
        if token_pos is not None:
            positions.append(token_pos)
    return positions


def _phase_name(token_start: int, feasible_pos: int, answer_start: int, answer_close: int) -> str:
    if feasible_pos < 0 or token_start < feasible_pos:
        return "think"
    if token_start < answer_start:
        return "format"
    if token_start < answer_close:
        return "answer"
    return "post_answer"


def _answer_region(absolute_index: int, phase: str, op_tokens: Sequence[int]) -> str:
    if phase != "answer":
        return phase
    if not op_tokens or absolute_index < op_tokens[0]:
        return "answer_before_op1"
    if len(op_tokens) == 1 or absolute_index < op_tokens[1]:
        return "answer_after_op1"
    if len(op_tokens) == 2 or absolute_index < op_tokens[2]:
        return "answer_after_op2"
    return "answer_after_op3"


# ---- merged trace_eval mode ----
"""Evaluation and aggregation helpers for countdown divergence-token experiments."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion
from early_branch_locking.core.countdown_shared import evaluate_countdown_completion

SUMMARY_VALUE_KEYS = (
    "source_match_rate",
    "completion_correct",
    "completion_feasible_ok",
    "completion_expr_ok",
    "completion_format_ok",
    "completion_tokens",
    "divergence_count",
    "divergence_fraction",
    "mean_disagree_teachers",
    "max_disagree_teachers",
    "first_divergence_index",
    "divergence_think_count",
    "divergence_format_count",
    "divergence_answer_count",
    "divergence_answer_before_op1_count",
    "divergence_answer_after_op1_count",
    "divergence_answer_after_op2_count",
    "divergence_answer_after_op3_count",
)


def summarize_trace(
    trace: CompletionTrace,
    meta: Sequence[dict],
    source_eval: dict,
    compare_evals: Dict[str, dict],
    require_source_match: bool,
    min_disagree_teachers: int,
) -> tuple[dict, List[dict]]:
    completion_eval = evaluate_countdown_completion(
        trace.completion,
        trace.item.numbers,
        trace.item.target,
        trace.item.feasible_label,
        parse_countdown_completion=parse_countdown_completion,
        evaluate_countdown_expression=evaluate_countdown_expression,
    )
    raw_rows: List[dict] = []
    summary = init_summary(trace, completion_eval)
    first_divergence = None
    for index, token_meta in enumerate(meta):
        source_match = source_eval["pred_ids"][index] == source_eval["gt_ids"][index]
        compare_info = compare_predictions(index, source_eval["gt_ids"][index], compare_evals)
        is_divergence = compare_info["num_disagree"] >= min_disagree_teachers and (source_match or not require_source_match)
        if is_divergence and first_divergence is None:
            first_divergence = token_meta
        update_summary(summary, token_meta, is_divergence, compare_info["num_disagree"], source_match)
        raw_rows.append(
            {
                "sample_id": trace.item.sample_id,
                "problem_index": trace.item.pid,
                "source_mode": trace.source_mode,
                "completion_token_index": token_meta["completion_index"],
                "token_id": token_meta["token_id"],
                "token_text": token_meta["token_text"],
                "phase": token_meta["phase"],
                "answer_region": token_meta["answer_region"],
                "source_matches_completion": bool(source_match),
                "num_disagree_teachers": compare_info["num_disagree"],
                "disagree_teachers": compare_info["disagree_teachers"],
                "is_divergence": bool(is_divergence),
            }
        )
    finalize_summary(summary, first_divergence)
    return summary, raw_rows


def compare_predictions(index: int, gt_token_id: int, compare_evals: Dict[str, dict]) -> dict:
    disagree_teachers = [
        name
        for name, payload in compare_evals.items()
        if payload["pred_ids"][index] != gt_token_id
    ]
    return {"num_disagree": len(disagree_teachers), "disagree_teachers": disagree_teachers}


def init_summary(trace: CompletionTrace, completion_eval) -> dict:
    return {
        "problem_index": trace.item.pid,
        "sample_id": trace.item.sample_id,
        "source_mode": trace.source_mode,
        "completion_correct": float(completion_eval.overall_ok),
        "completion_feasible_ok": float(completion_eval.feasible_ok),
        "completion_expr_ok": float(completion_eval.expr_ok),
        "completion_format_ok": float(completion_eval.parse_status == "OK"),
        "completion_parse_status": completion_eval.parse_status,
        "completion_tokens": len(trace.full_token_ids) - trace.prompt_token_count,
        "source_match_count": 0.0,
        "divergence_count": 0.0,
        "mean_disagree_teachers": 0.0,
        "max_disagree_teachers": 0.0,
        "divergence_think_count": 0.0,
        "divergence_format_count": 0.0,
        "divergence_answer_count": 0.0,
        "divergence_answer_before_op1_count": 0.0,
        "divergence_answer_after_op1_count": 0.0,
        "divergence_answer_after_op2_count": 0.0,
        "divergence_answer_after_op3_count": 0.0,
    }


def update_summary(summary: dict, token_meta: dict, is_divergence: bool, num_disagree: int, source_match: bool) -> None:
    summary["source_match_count"] += float(source_match)
    summary["mean_disagree_teachers"] += float(num_disagree)
    summary["max_disagree_teachers"] = max(summary["max_disagree_teachers"], float(num_disagree))
    if not is_divergence:
        return
    summary["divergence_count"] += 1.0
    phase_key = f"divergence_{token_meta['phase']}_count"
    region_key = f"divergence_{token_meta['answer_region']}_count"
    if phase_key in summary:
        summary[phase_key] += 1.0
    if token_meta["answer_region"].startswith("answer_") and region_key in summary:
        summary[region_key] += 1.0


def finalize_summary(summary: dict, first_divergence: dict | None) -> None:
    tokens = max(summary["completion_tokens"], 1.0)
    summary["source_match_rate"] = summary["source_match_count"] / tokens
    summary["divergence_fraction"] = summary["divergence_count"] / tokens
    summary["mean_disagree_teachers"] = summary["mean_disagree_teachers"] / tokens
    if first_divergence is None:
        summary["first_divergence_index"] = np.nan
        summary["first_divergence_phase"] = ""
        summary["first_divergence_region"] = ""
        return
    summary["first_divergence_index"] = float(first_divergence["completion_index"])
    summary["first_divergence_phase"] = first_divergence["phase"]
    summary["first_divergence_region"] = first_divergence["answer_region"]


def aggregate_summary_rows(rows: Sequence[dict]) -> Dict[str, float]:
    if not rows:
        return {"num_problems": 0}
    summary = {"num_problems": len(rows)}
    for key in SUMMARY_VALUE_KEYS:
        values = [float(row[key]) for row in rows if not np.isnan(row[key])]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
    defined = [float(not np.isnan(row["first_divergence_index"])) for row in rows]
    summary["first_divergence_defined_rate"] = float(np.mean(defined)) if defined else 0.0
    return summary


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def _parse_multi_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint_divergence divergence localization over multiple GPU specs.")
    parser.add_argument("--specs", nargs="+", required=True, help="Specs such as 50:275 or 100:150,200,275.")
    parser.add_argument("--actor-dir", default=str(ACTOR_DIR))
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--completion-source", choices=["greedy", "raw_shortest_success"], default="greedy")
    parser.add_argument("--num-problems", type=int, default=150)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--generate-batch-size", type=int, default=8)
    parser.add_argument("--teacher-force-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-disagree-teachers", type=int, default=1)
    parser.add_argument("--tag-prefix", default="checkpoint_divergence_divergence_suite")
    return parser.parse_args()


def _merge_multi_summaries(tags: list[str], tag_prefix: str) -> None:
    frames = [pd.read_csv(METRICS_DIR / f"checkpoint_divergence_divergence_summary_{tag}.csv") for tag in tags]
    merged = pd.concat(frames, ignore_index=True)
    summary_path = METRICS_DIR / f"checkpoint_divergence_divergence_multi_{tag_prefix}.csv"
    merged.to_csv(summary_path, index=False)
    (METRICS_DIR / f"checkpoint_divergence_divergence_multi_{tag_prefix}.json").write_text(
        json.dumps({"tags": tags, "summary_path": str(summary_path)}, indent=2),
        encoding="utf-8",
    )


def _run_multi() -> None:
    args = _parse_multi_args()
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one device.")
    tags = []
    old_argv = list(sys.argv)
    try:
        for index, spec in enumerate(args.specs):
            source_step, compare_text = spec.split(":", 1)
            compare_steps = [item for item in compare_text.split(",") if item.strip()]
            gpu_id = gpu_ids[index % len(gpu_ids)]
            tag = f"{args.tag_prefix}_src{source_step}_cmp{'-'.join(compare_steps)}_{args.completion_source}"
            run_argv = [
                old_argv[0],
                "--source_model_path", str(Path(args.actor_dir) / f"global_step_{source_step}"),
                "--compare_model_paths", *[str(Path(args.actor_dir) / f"global_step_{step}") for step in compare_steps],
                "--completion_source", args.completion_source,
                "--num_problems", str(args.num_problems),
                "--start_index", str(args.start_index),
                "--end_index", str(args.end_index),
                "--generate_batch_size", str(args.generate_batch_size),
                "--teacher_force_batch_size", str(args.teacher_force_batch_size),
                "--max_new_tokens", str(args.max_new_tokens),
                "--gpu_id", gpu_id,
                "--dtype", args.dtype,
                "--attn_implementation", args.attn_implementation,
                "--min_disagree_teachers", str(args.min_disagree_teachers),
                "--tag", tag,
                "--save_per_problem",
                "--save_raw",
            ]
            if args.completion_source == "raw_shortest_success":
                run_argv.extend(["--source_raw_path", str(RAW_DIR / f"countdown_raw_global_step_{source_step}_n320.jsonl")])
            print(f"[checkpoint_divergence] multi run {tag} on GPU {gpu_id}", flush=True)
            sys.argv = run_argv
            main()
            tags.append(tag)
    finally:
        sys.argv = old_argv
    _merge_multi_summaries(tags, args.tag_prefix)


def _run_selected():
    if "--multi" in sys.argv:
        sys.argv.remove("--multi")
        return _run_multi()
    return main()

if __name__ == "__main__":
    _run_selected()
