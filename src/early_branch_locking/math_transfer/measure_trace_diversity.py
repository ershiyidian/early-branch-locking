#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""math-trace-diversity - Score and generate math reasoning traces.
Hypothesis: answer correctness and trace diversity are distinct axes that should be measured with the same scorer for transfer comparisons.
Inputs: math raw JSONL; GSM8K/MATH500 records; parser or official correctness mode.
Outputs: data/rlvr/outputs/math_transfer/olmo3_trace_diversity_summary_20260617.csv; data/rlvr/outputs/math_transfer/qwen_olmo_transfer_macro_20260617.csv
Status: paper-main
"""
from __future__ import annotations
"""Score Limit-of-RLVR math raw JSONL with trace-diversity metrics."""

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import METRICS_DIR as COUNTDOWN_METRICS_DIR, RAW_DIR as COUNTDOWN_RAW_DIR
from early_branch_locking.core.math_trace_utils import SampleEval, evaluate_completion, problem_metrics, summarize_problem_rows

DEFAULT_KS = (1, 4, 16, 64)
DEFAULT_METRICS_ROOT = COUNTDOWN_METRICS_DIR
DEFAULT_RAW_ROOT = COUNTDOWN_RAW_DIR
MODE_PARSER = "parser"
MODE_OFFICIAL = "official"
MODE_BOTH = "both"


@dataclass(frozen=True)
class ScoreConfig:
    raw_path: Path
    dataset: str
    model_label: str
    tag: str
    correctness: str
    ks: tuple[int, ...]
    metrics_root: Path
    scored_raw_root: Path


@dataclass(frozen=True)
class ModeContext:
    config: ScoreConfig
    mode: str


@dataclass(frozen=True)
class SampleContext:
    record: dict
    mode_context: ModeContext
    sample_index: int
    completion: str
    eval_item: SampleEval


@dataclass(frozen=True)
class RecordScores:
    record: dict
    mode_context: ModeContext
    completions: Sequence[str]
    evals: Sequence[SampleEval]


def parse_args() -> ScoreConfig:
    parser = argparse.ArgumentParser(description="Score math raw JSONL with migrated trace metrics.")
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--correctness", choices=[MODE_PARSER, MODE_OFFICIAL, MODE_BOTH], default=MODE_BOTH)
    parser.add_argument("--ks", default=",".join(str(item) for item in DEFAULT_KS))
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--scored-raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    args = parser.parse_args()
    return ScoreConfig(
        raw_path=args.raw_path,
        dataset=args.dataset,
        model_label=args.model_label,
        tag=args.tag,
        correctness=args.correctness,
        ks=parse_ks(args.ks),
        metrics_root=args.metrics_root,
        scored_raw_root=args.scored_raw_root,
    )


def parse_ks(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--ks must contain at least one integer")
    return values


def main() -> None:
    config = parse_args()
    validate_config(config)
    output_dir = config.metrics_root / f"expx_trace_diversity_{config.tag}"
    scored_raw_dir = config.scored_raw_root / f"expx_trace_diversity_{config.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_raw_dir.mkdir(parents=True, exist_ok=True)
    results = [score_mode(ModeContext(config, mode), output_dir, scored_raw_dir) for mode in selected_modes(config)]
    write_combined_summary(output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def validate_config(config: ScoreConfig) -> None:
    if not config.raw_path.exists():
        raise FileNotFoundError(f"Raw JSONL not found: {config.raw_path}")
    if config.raw_path.suffix != ".jsonl":
        raise ValueError(f"Raw path must be a .jsonl file: {config.raw_path}")


def selected_modes(config: ScoreConfig) -> tuple[str, ...]:
    if config.correctness == MODE_BOTH:
        return (MODE_PARSER, MODE_OFFICIAL)
    return (config.correctness,)


def score_mode(context: ModeContext, output_dir: Path, scored_raw_dir: Path) -> dict:
    records = load_records(context.config.raw_path)
    prefix = f"{context.config.dataset}_{context.config.model_label}_{context.mode}"
    scored_raw_path = scored_raw_dir / f"{prefix}.jsonl"
    rows = []
    with scored_raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = score_record(record, context, handle)
            rows.append(row)
    write_per_problem_csv(output_dir / f"{prefix}_per_problem.csv", rows)
    summary = summarize_problem_rows(rows, context.config.ks)
    summary.update(summary_metadata(context, scored_raw_path))
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_records(raw_path: Path) -> list[dict]:
    records = []
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                records.append(json.loads(line))
            else:
                raise ValueError(f"Empty line in {raw_path} at {line_number}")
    if not records:
        raise ValueError(f"No records loaded from {raw_path}")
    return records


def score_record(record: dict, context: ModeContext, handle) -> dict:
    completions = completion_list(record)
    evals = [evaluate_completion(text, ground_truth(record)) for text in completions]
    if context.mode == MODE_OFFICIAL:
        evals = apply_official_scores(evals, record)
    write_scored_samples(handle, RecordScores(record, context, completions, evals))
    row = problem_metrics(evals, ground_truth(record), context.config.ks)
    row.update(problem_metadata(record, context))
    if context.mode == MODE_OFFICIAL:
        row.update(official_diagnostics(record, evals))
    return row


def ground_truth(record: dict) -> str:
    if "gt" in record:
        return str(record["gt"])
    if "answer" in record and not isinstance(record["answer"], list):
        return str(record["answer"])
    if "final_answer" in record:
        return str(record["final_answer"])
    raise KeyError(f"Missing ground-truth key in record idx={record.get('idx')}")


def completion_list(record: dict) -> list[str]:
    completions = record.get("code")
    if not isinstance(completions, list):
        raise TypeError(f"Record idx={record.get('idx')} has non-list code field")
    if not completions:
        raise ValueError(f"Record idx={record.get('idx')} has no completions")
    return [str(item) for item in completions]


def apply_official_scores(evals: Sequence[SampleEval], record: dict) -> list[SampleEval]:
    scores = record.get("score")
    preds = record.get("pred")
    if not isinstance(scores, list) or len(scores) != len(evals):
        raise ValueError(f"Record idx={record.get('idx')} has invalid score list")
    if not isinstance(preds, list) or len(preds) != len(evals):
        raise ValueError(f"Record idx={record.get('idx')} has invalid pred list")
    return [replace(item, is_correct=bool(score)) for item, score in zip(evals, scores, strict=True)]


def write_scored_samples(handle, scores: RecordScores) -> None:
    for sample_index, item in enumerate(scores.evals):
        payload = sample_payload(make_sample_context(scores, sample_index, item))
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_sample_context(scores: RecordScores, sample_index: int, item: SampleEval) -> SampleContext:
    return SampleContext(
        record=scores.record,
        mode_context=scores.mode_context,
        sample_index=sample_index,
        completion=scores.completions[sample_index],
        eval_item=item,
    )


def sample_payload(context: SampleContext) -> dict:
    config = context.mode_context.config
    return {
        "dataset": config.dataset,
        "model_label": config.model_label,
        "correctness_mode": context.mode_context.mode,
        "problem_id": problem_id(context.record, config.dataset),
        "sample_index": context.sample_index,
        "ground_truth": ground_truth(context.record),
        "completion": context.completion,
        "answer": context.eval_item.answer,
        "numeric_trace": context.eval_item.numeric_trace,
        "first_calc_branch": context.eval_item.first_calc_branch,
        "is_correct": context.eval_item.is_correct,
        "parsed": context.eval_item.parsed,
        "completion_chars": context.eval_item.completion_chars,
    }


def problem_id(record: dict, dataset: str) -> str:
    if "idx" not in record:
        raise KeyError("Record is missing idx")
    return f"{dataset}_{record['idx']}"


def problem_metadata(record: dict, context: ModeContext) -> dict:
    config = context.config
    return {
        "problem_id": problem_id(record, config.dataset),
        "dataset": config.dataset,
        "model_label": config.model_label,
        "correctness_mode": context.mode,
    }


def official_diagnostics(record: dict, evals: Sequence[SampleEval]) -> dict:
    scores = [bool(item) for item in record["score"]]
    parser_scores = [evaluate_completion(text, ground_truth(record)).is_correct for text in completion_list(record)]
    return {
        "majority_accuracy": official_majority_accuracy(record),
        "parser_official_agreement_rate": mean(left == right for left, right in zip(parser_scores, scores, strict=True)),
        "parser_correct_rate": mean(parser_scores),
        "official_correct_rate": mean(item.is_correct for item in evals),
    }


def official_majority_accuracy(record: dict) -> float:
    preds = [str(item) for item in record["pred"] if str(item)]
    if not preds:
        return 0.0
    majority = max(set(preds), key=preds.count)
    matching = [bool(score) for pred, score in zip(record["pred"], record["score"], strict=True) if str(pred) == majority]
    return float(any(matching))


def mean(values) -> float:
    items = [float(item) for item in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def write_per_problem_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_metadata(context: ModeContext, scored_raw_path: Path) -> dict:
    config = context.config
    return {
        "dataset": config.dataset,
        "model_label": config.model_label,
        "correctness_mode": context.mode,
        "input_raw_path": str(config.raw_path),
        "scored_raw_path": str(scored_raw_path),
        "ks": list(config.ks),
    }


def write_combined_summary(output_dir: Path) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output_dir.glob("*_summary.json"))]
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- merged generate mode ----
"""Sample GSM8K/MATH completions and compute trace-diversity metrics."""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyarrow.ipc as pa_ipc
from transformers import AutoModelForCausalLM, AutoTokenizer

from early_branch_locking.core.math_trace_utils import evaluate_completion, problem_metrics, summarize_problem_rows

METRICS_ROOT = COUNTDOWN_METRICS_DIR
RAW_ROOT = COUNTDOWN_RAW_DIR
DEFAULT_KS = (1, 4, 16, 32, 64, 128)
DEFAULT_SEED = 42


@dataclass(frozen=True)
class MathProblem:
    problem_id: str
    question: str
    ground_truth: str
    dataset: str


def parse_args_generate() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExpX math trace diversity worker")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_label", type=str, required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math500", "amc23"], required=True)
    parser.add_argument("--num_problems", type=int, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--cpu_threads", type=int, default=56)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--tag", type=str, default="formal")
    parser.add_argument("--hf_endpoint", type=str, default="")
    parser.add_argument("--local_files_only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def run_generate() -> None:
    args = parse_args_generate()
    configure_runtime(args)
    problems = load_math_problems(args.dataset, args.num_problems, args.start_index)
    tokenizer, model = load_hf_model(args)
    output_dir = METRICS_ROOT / f"expx_trace_diversity_{args.tag}"
    raw_dir = RAW_ROOT / f"expx_trace_diversity_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = run_sampling(args, tokenizer, model, problems, output_dir, raw_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def configure_runtime(args: argparse.Namespace) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    else:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.set_num_threads(max(args.cpu_threads, 1))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ExpX trace diversity.")


def load_hf_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        trust_remote_code=False,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).to("cuda")
    model.eval()
    return tokenizer, model


def load_math_problems(dataset: str, num_problems: int, start_index: int) -> list[MathProblem]:
    if dataset == "gsm8k":
        return _load_gsm8k(num_problems, start_index)
    return _load_math500(num_problems, start_index)


def _load_gsm8k(num_problems: int, start_index: int) -> list[MathProblem]:
    arrow_path = _gsm8k_arrow_path()
    rows = []
    with pa_ipc.open_stream(arrow_path) as reader:
        table = reader.read_all()
    df = table.to_pandas()
    for idx, rec in enumerate(df.iloc[start_index : start_index + num_problems].to_dict("records")):
        answer = str(rec["answer"]).split("####")[-1].strip()
        rows.append(MathProblem(f"gsm8k_{start_index + idx}", rec["question"], answer, "gsm8k"))
    return rows


def _gsm8k_arrow_path() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "datasets" / "openai___gsm8k"
    candidates = sorted(base.glob("main/**/gsm8k-test.arrow"))
    if not candidates:
        raise FileNotFoundError(f"GSM8K cached arrow not found under {base}")
    return candidates[-1]


def _load_math500(num_problems: int, start_index: int) -> list[MathProblem]:
    arrow_path = _math500_arrow_path()
    rows = []
    with pa_ipc.open_stream(arrow_path) as reader:
        table = reader.read_all()
    df = table.to_pandas()
    for idx, rec in enumerate(df.iloc[start_index : start_index + num_problems].to_dict("records")):
        answer = str(rec.get("answer") or rec.get("solution") or "")
        rows.append(MathProblem(f"math500_{start_index + idx}", rec["problem"], answer, "math500"))
    return rows


def _math500_arrow_path() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "datasets" / "HuggingFaceH4___math-500"
    candidates = sorted(base.glob("default/**/math-500-test.arrow"))
    if not candidates:
        raise FileNotFoundError(f"Math-500 cached arrow not found under {base}")
    return candidates[-1]


def run_sampling(args, tokenizer, model, problems, output_dir: Path, raw_dir: Path) -> dict:
    raw_path = raw_dir / f"{args.dataset}_{args.model_label}.jsonl"
    problem_rows = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for problem in problems:
            completions = generate_problem_completions(args, tokenizer, model, problem)
            evals = [evaluate_completion(text, problem.ground_truth) for text in completions]
            write_raw_rows(raw_f, args, problem, completions, evals)
            raw_f.flush()
            row = problem_metrics(evals, problem.ground_truth, DEFAULT_KS)
            row.update(problem_id=problem.problem_id, dataset=args.dataset, model_label=args.model_label)
            problem_rows.append(row)
            print_progress(args, problem, row)
    return save_metrics(args, output_dir, raw_path, problem_rows)


def generate_problem_completions(args, tokenizer, model, problem: MathProblem) -> list[str]:
    prompt = build_prompt(problem.question)
    prompts = [prompt] * args.n_samples
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        outputs.extend(generate_batch(args, tokenizer, model, prompts[start : start + args.batch_size]))
    return outputs


def generate_batch(args, tokenizer, model, prompts: list[str]) -> list[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = encoded["input_ids"].shape[1]
    return decode_completions(tokenizer, generated[:, prompt_width:])


def decode_completions(tokenizer, token_rows) -> list[str]:
    texts = []
    for row in token_rows.detach().cpu().tolist():
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id)]
        texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return texts


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def write_raw_rows(raw_f, args, problem, completions, evals) -> None:
    for sample_idx, (completion, ev) in enumerate(zip(completions, evals)):
        payload = {
            "dataset": args.dataset,
            "model_label": args.model_label,
            "problem_id": problem.problem_id,
            "sample_index": sample_idx,
            "ground_truth": problem.ground_truth,
            "completion": completion,
            "answer": ev.answer,
            "numeric_trace": ev.numeric_trace,
            "first_calc_branch": ev.first_calc_branch,
            "is_correct": ev.is_correct,
            "parsed": ev.parsed,
            "completion_chars": ev.completion_chars,
        }
        raw_f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def print_progress(args, problem, row) -> None:
    max_k = max(k for k in DEFAULT_KS if f"pass@{k}" in row)
    msg = (
        f"[{args.model_label}/{args.dataset}] {problem.problem_id} "
        f"pass@1={row.get('pass@1', 0):.3f} pass@{max_k}={row[f'pass@{max_k}']:.3f} "
        f"H_trace={row['numeric_trace_entropy']:.3f}"
    )
    print(msg, flush=True)


def save_metrics(args, output_dir: Path, raw_path: Path, rows: list[dict]) -> dict:
    prefix = f"{args.dataset}_{args.model_label}"
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_per_problem.csv", index=False)
    summary = summarize_problem_rows(rows, DEFAULT_KS)
    summary.update(_summary_metadata(args, raw_path))
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_combined_summary(output_dir)
    return summary


def _summary_metadata(args, raw_path: Path) -> dict:
    return {
        "dataset": args.dataset,
        "model_label": args.model_label,
        "model_name_or_path": args.model_name_or_path,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "raw_path": str(raw_path),
    }


def write_combined_summary(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

def _run_multi_generate() -> None:
    parser = argparse.ArgumentParser(description="Generate trace-diversity metrics for multiple math jobs.")
    parser.add_argument("--multi-jobs", required=True, help="Semicolon-separated dataset:model_path:model_label entries.")
    parser.add_argument("--num-problems", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--tag", default="formal")
    args = parser.parse_args()
    old_argv = list(sys.argv)
    try:
        for entry in args.multi_jobs.split(";"):
            dataset, model_path, model_label = [item.strip() for item in entry.split(":", 2)]
            sys.argv = [
                old_argv[0], "--model_name_or_path", model_path, "--model_label", model_label,
                "--dataset", dataset, "--num_problems", str(args.num_problems),
                "--n_samples", str(args.n_samples), "--gpu_id", args.gpu_id, "--tag", args.tag,
            ]
            run_generate()
    finally:
        sys.argv = old_argv


def _run_selected():
    if "--multi" in sys.argv:
        sys.argv.remove("--multi")
        return _run_multi_generate()
    for index, argument in enumerate(sys.argv):
        if argument == "--source" and index + 1 < len(sys.argv):
            source_mode = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            if source_mode == "generate":
                return run_generate()
            if source_mode != "rescore":
                raise ValueError(f"Unknown --source: {source_mode}")
            break
    return main()

if __name__ == "__main__":
    _run_selected()
