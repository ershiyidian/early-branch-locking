#!/usr/bin/env python3
"""decoding_robustness: decoding, prompt, length, scale, and family robustness.

Hypothesis: Early branch concentration may be a model-family property rather
than a decoding artifact.
Inputs: local math checkpoint and benchmark test files.
Procedure: sample fixed problem subsets at pre-registered temperature/top-p
conditions and main/minimal prompts, retaining full candidate-level records.
Metrics: accuracy, first-calc/strategy entropy, correct branch discovery,
format/no-calculation noise, and branch-distribution JSD-ready summaries.
Outputs: data/rlvr/outputs/e9/<model>/{raw.jsonl,per_problem.parquet,
summary.csv,config.json}.
Statistical unit: problem, with samples nested under problem and condition.
Known limitations: this is a parameter/family robustness experiment, not a
multi-seed claim; model pairs retain their checkpoint identity.
Status: formal training-free sampling runner; no training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.core.math_trace_utils import evaluate_completion
from early_branch_locking.core.branch_protocol import entropy, milestone_record

MODEL_ROOT = ROOT / "model"
MODEL_DIRS = {
    "math_base_7b": "math_base_7b",
    "math_simple_rl_7b": "math_simple_rl_7b",
    "math_base_14b": "math_base_14b",
    "math_simple_rl_14b": "math_simple_rl_14b",
    "math_olmo3_sft_7b": "Olmo-3-7B-Instruct-SFT",
    "math_olmo3_dpo_7b": "Olmo-3-7B-Instruct-DPO",
    "math_olmo3_rlvr_7b": "Olmo-3-7B-Instruct-RLVR",
}
RAW_ROOT = ROOT / "data" / "rlvr" / "outputs" / "math"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_DIRS), default="")
    parser.add_argument("--models", default="", help="Comma-separated model aliases; runs each model into its own directory.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--models-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--input-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "rlvr" / "outputs" / "e9_fixed")
    parser.add_argument("--benchmarks", default="gsm8k,math500,minerva_math,olympiadbench")
    parser.add_argument("--problems-per-benchmark", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperatures", default="0.6,0.9")
    parser.add_argument("--top-ps", default="0.90,0.95")
    parser.add_argument("--prompt-variants", default="main,minimal")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--verify-decode", type=int, default=20, help="Persist this many raw decoded records for sanity inspection.")
    return parser.parse_args(argv)


def prompt(question: str, variant: str) -> str:
    if variant == "minimal":
        return f"{question.strip()}\nSolve carefully and give the final answer in \\boxed{{}}.\n"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def raw_path(root: Path, model: str, benchmark: str) -> Path:
    files = sorted((root / model / benchmark).glob("*_merged.jsonl"))
    files = [path for path in files if "_1_seed" not in path.name]
    if not files:
        raise FileNotFoundError(f"No merged raw input for {model}/{benchmark}")
    return files[-1]


def load_problems(root: Path, model: str, benchmarks: list[str], limit: int) -> list[dict]:
    output = []
    for benchmark in benchmarks:
        with raw_path(root, model, benchmark).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows[:limit]:
            output.append({"benchmark": benchmark, "problem_id": f"{benchmark}:{row.get('idx')}", "question": row.get("question", row.get("problem", "")), "ground_truth": str(row.get("gt", row.get("ground_truth", "")))})
    return output


def load_model(args, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(args.models_root / MODEL_DIRS[model_name], local_files_only=True, trust_remote_code=False, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.models_root / MODEL_DIRS[model_name], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda")
    model.eval()
    return tokenizer, model


def generate_batch(model, tokenizer, texts: list[str], n_samples: int, temperature: float, top_p: float, max_new_tokens: int) -> list[str]:
    input_rows = [tokenizer.encode(text, add_special_tokens=False) for text in texts for _ in range(n_samples)]
    width = max(len(row) for row in input_rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(input_rows), width), pad_id, dtype=torch.long, device=model.device)
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(input_rows):
        input_ids[index, width - len(row) :] = torch.tensor(row, dtype=torch.long, device=model.device)
        attention[index, width - len(row) :] = 1
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=attention, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id)
    completions = []
    for index, row in enumerate(output.detach().cpu().tolist()):
        generated = row[width:]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated:
            generated = generated[: generated.index(tokenizer.eos_token_id)]
        completions.append((tokenizer.decode(generated, skip_special_tokens=True).strip(), len(input_rows[index]), width, len(generated)))
    return completions


def aggregate(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["benchmark"], row["problem_id"], row["temperature"], row["top_p"], row["prompt_variant"])].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        branches = [row["first_calc_branch"] for row in rows]
        correct = [row["official_correct"] for row in rows]
        output.append({"benchmark": key[0], "problem_id": key[1], "temperature": key[2], "top_p": key[3], "prompt_variant": key[4], "n_samples": len(rows), "correct_rate": float(np.mean(correct)), "format_failure_rate": float(np.mean([not row["format_valid"] for row in rows])), "no_calc_rate": float(np.mean([row["first_calc_branch"] == "NO_VALID_FIRST_CALC" for row in rows])), "branch_entropy": entropy(branches), "observed_branch_count": len(set(branches)), "correct_branch_count": len({row["first_calc_branch"] for row in rows if row["official_correct"]}), "mean_completion_tokens": float(np.mean([row["completion_tokens"] for row in rows]))})
    return output


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(args, model_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("decoding_robustness requires CUDA")
    torch.cuda.set_device(int(str(args.gpu).split(",")[0]))
    seed_everything(args.seed)
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    problems = load_problems(args.input_root, model_name, benchmarks, args.problems_per_benchmark)
    prompt_variants = [item.strip() for item in args.prompt_variants.split(",") if item.strip()]
    invalid_variants = sorted(set(prompt_variants) - {"main", "minimal"})
    if invalid_variants:
        raise ValueError(f"unknown prompt variants: {invalid_variants}")
    conditions = [(float(t), float(p), variant) for t in args.temperatures.split(",") for p in args.top_ps.split(",") for variant in prompt_variants]
    tokenizer, model = load_model(args, model_name)
    records = []
    for condition_index, (temperature, top_p, variant) in enumerate(conditions):
        seed_everything(args.seed + condition_index)
        for start in range(0, len(problems), 4):
            chunk = problems[start : start + 4]
            completions = generate_batch(model, tokenizer, [prompt(row["question"], variant) for row in chunk], args.n_samples, temperature, top_p, args.max_new_tokens)
            offset = 0
            for problem in chunk:
                for sample_index in range(args.n_samples):
                    completion, prompt_tokens_true, padded_width, gen_token_len = completions[offset]
                    offset += 1
                    evaluation = evaluate_completion(completion, problem["ground_truth"])
                    branch = evaluation.first_calc_branch if evaluation.first_calc_branch != "<no_calc>" else "NO_VALID_FIRST_CALC"
                    records.append({"model": model_name, **problem, "sample_index": sample_index, "condition_index": condition_index, "temperature": temperature, "top_p": top_p, "prompt_variant": variant, "completion": completion, "completion_tokens": len(tokenizer.encode(completion, add_special_tokens=False)), "prompt_tokens_true": prompt_tokens_true, "padded_width": padded_width, "gen_token_len": gen_token_len, "official_correct": bool(evaluation.is_correct), "parsed": bool(evaluation.parsed), "format_valid": bool(evaluation.parsed and re.search(r"(?:\\boxed|####|answer)", completion, re.IGNORECASE)), "first_calc_branch": branch, "p1_available": milestone_record(completion, problem["ground_truth"]).get("p1_char_end") is not None})
        print(f"condition {condition_index + 1}/{len(conditions)} complete", flush=True)
    padding_side = tokenizer.padding_side
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    output = args.output_root / (model_name + ("_" + args.tag if args.tag else ""))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    per_problem = aggregate(records)
    pd.DataFrame(per_problem).to_parquet(output / "per_problem.parquet", index=False)
    pd.DataFrame(per_problem).to_csv(output / "summary.csv", index=False)
    config = {"experiment_id": "decoding_robustness", "model": model_name, "tag": args.tag, "benchmarks": benchmarks, "problems_per_benchmark": args.problems_per_benchmark, "n_samples": args.n_samples, "temperatures": [float(value) for value in args.temperatures.split(",")], "top_ps": [float(value) for value in args.top_ps.split(",")], "prompt_variants": prompt_variants, "max_new_tokens": args.max_new_tokens, "seed": args.seed, "statistical_unit": "problem", "multi_seed_claim": False, "padding_side": padding_side, "verify_decode": args.verify_decode}
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    sanity = records[: max(0, args.verify_decode)]
    with (output / "decode_sanity.jsonl").open("w", encoding="utf-8") as handle:
        for row in sanity:
            handle.write(json.dumps({key: row[key] for key in ("benchmark", "problem_id", "condition_index", "temperature", "top_p", "prompt_variant", "prompt_tokens_true", "padded_width", "gen_token_len", "completion", "format_valid", "official_correct")}, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "records": len(records), "problem_conditions": len(per_problem), "decode_sanity": len(sanity)}, sort_keys=True))


def main(argv=None) -> None:
    args = parse_args(argv)
    models = [item.strip() for item in (args.models or args.model).split(",") if item.strip()]
    if not models:
        raise ValueError("provide --model or --models")
    invalid = sorted(set(models) - set(MODEL_DIRS))
    if invalid:
        raise ValueError(f"unknown models: {invalid}")
    for model_name in models:
        run_one(args, model_name)


if __name__ == "__main__":
    main()
