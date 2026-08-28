#!/usr/bin/env python3
"""execution_training: teacher-forced segment-wise probability suppression.

Hypothesis: RL probability suppression may be stronger before an early strategy
commitment than during execution of the same correct trace.
Inputs: branch_decomposition labels, their source merged JSONL rows, and a tokenizer-compatible
base/RL model pair.
Procedure: select balanced correct trajectories (at most one per problem and
first-calculation branch by default), score the exact same token sequence with
both models, and split completion log-likelihood at frozen P1/P2 offsets.
Metrics: raw and length-normalized commitment/execution/P1-P2/P2-answer log
likelihood, four-way base/RL scoring, branch-frequency strata, and minimal
style-normalization sensitivity.
Outputs: data/rlvr/outputs/e2/<pair>/scores.jsonl, per_problem.parquet,
summary.csv, config.json, and alignment_audit.csv.
Statistical unit: problem, with trajectory nested below problem and branch.
Known limitations: teacher forcing measures relative sequence probability, not
free-generation probability; cross-tokenizer pairs are rejected.
Status: formal training-free scoring; no model is trained.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.core.math_trace_utils import extract_first_calc_branch
from early_branch_locking.core.branch_protocol import bootstrap_mean, milestone_record

E1_ROOT = ROOT / "data" / "rlvr" / "outputs" / "e1"
MODEL_ROOT = ROOT / "model"
PAIR_CONFIG = {
    "qwen_7b_base_rl": ("math_base_7b", "math_simple_rl_7b"),
    "qwen_14b_base_rl": ("math_base_14b", "math_simple_rl_14b"),
    "olmo_sft_rlvr": ("math_olmo3_sft_7b", "math_olmo3_rlvr_7b"),
    "olmo_dpo_rlvr": ("math_olmo3_dpo_7b", "math_olmo3_rlvr_7b"),
}
MODEL_DIRS = {
    "math_base_7b": "math_base_7b",
    "math_simple_rl_7b": "math_simple_rl_7b",
    "math_base_14b": "math_base_14b",
    "math_simple_rl_14b": "math_simple_rl_14b",
    "math_olmo3_sft_7b": "Olmo-3-7B-Instruct-SFT",
    "math_olmo3_dpo_7b": "Olmo-3-7B-Instruct-DPO",
    "math_olmo3_rlvr_7b": "Olmo-3-7B-Instruct-RLVR",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("score", "aggregate"), default="score")
    parser.add_argument("--pair", choices=sorted(PAIR_CONFIG), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--e1-root", type=Path, default=E1_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "rlvr" / "outputs" / "e2")
    parser.add_argument("--models-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--benchmarks", default="gsm8k,math500,minerva_math,olympiadbench,amc23,aime24")
    parser.add_argument("--per-branch", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--include-normalized", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    # CPU aggregation options are intentionally optional for the normal score
    # mode, preserving the historical scoring CLI.
    parser.add_argument("--model", default="")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--compare-model", default="")
    parser.add_argument("--compare-out", type=Path, default=None)
    return parser.parse_args(argv)


def prompt_for(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def minimal_style_normalize(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_e1_labels(path: Path, pair: str, benchmarks: Sequence[str], per_branch: int, max_rows: int, smoke: bool) -> list[dict]:
    base_model, rl_model = PAIR_CONFIG[pair]
    wanted_models = {base_model, rl_model}
    wanted_benchmarks = set(benchmarks)
    selected: list[dict] = []
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    with (path / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["model"] not in wanted_models or row["benchmark"] not in wanted_benchmarks or not row["official_correct"]:
                continue
            grouped[(row["model"], row["benchmark"], row["problem_id"], row["first_calc_branch"])].append(row)
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row["sample_index"], row["completion_sha1"]))
        for row in rows[:per_branch]:
            row = dict(row)
            row["branch_sample_count"] = len(rows)
            row["branch_frequency_stratum"] = "rare" if len(rows) <= 2 else ("mid" if len(rows) <= 8 else "common")
            selected.append(row)
    if smoke:
        selected = selected[: min(8, len(selected))]
    elif len(selected) > max_rows:
        # Keep problem/branch strata deterministic while bounding a long run.
        selected = sorted(selected, key=lambda row: hashlib.sha1(f"{row['problem_id']}:{row['model']}:{row['sample_index']}".encode()).hexdigest())[:max_rows]
        selected.sort(key=lambda row: (row["benchmark"], row["problem_id"], row["model"], row["sample_index"]))
    return selected


def load_source_rows(selected: Sequence[dict]) -> dict[tuple[str, int], dict]:
    needed: dict[str, set[int]] = defaultdict(set)
    for row in selected:
        needed[row["source_file"]].add(int(row["source_row"]))
    source: dict[tuple[str, int], dict] = {}
    for relative, indices in needed.items():
        with (ROOT / relative).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in indices and line.strip():
                    source[(relative, index)] = json.loads(line)
    return source


def expand_selected(selected: Sequence[dict], source: dict[tuple[str, int], dict], include_normalized: bool) -> list[dict]:
    expanded = []
    for label in selected:
        row = source[(label["source_file"], int(label["source_row"]))]
        completions = row.get("code")
        if not isinstance(completions, list):
            completions = row.get("completions") or row.get("responses") or []
        completion = str(completions[int(label["sample_index"])])
        variants = [("raw", completion)]
        if include_normalized:
            normalized = minimal_style_normalize(completion)
            if normalized != completion and extract_first_calc_branch(normalized) == label["first_calc_branch"]:
                variants.append(("minimal_normalized", normalized))
        for style, text in variants:
            milestones = milestone_record(text, str(row.get("gt", row.get("ground_truth", ""))))
            expanded.append(
                {
                    **label,
                    "question": str(row.get("question", row.get("problem", ""))),
                    "ground_truth": str(row.get("gt", row.get("ground_truth", ""))),
                    "completion": text,
                    "style_variant": style,
                    "milestones": milestones,
                }
            )
    return expanded


def token_boundary(tokenizer, text: str, char_end: int | None) -> int | None:
    if char_end is None:
        return None
    return len(tokenizer.encode(text[:char_end], add_special_tokens=False))


def build_sequence(tokenizer, prompt: str, completion: str, max_input_tokens: int) -> tuple[list[int], int, list[int], bool]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    full = prompt_ids + completion_ids
    truncated = len(full) > max_input_tokens
    if truncated:
        full = full[:max_input_tokens]
        completion_ids = full[len(prompt_ids) :]
    return full, len(prompt_ids), completion_ids, truncated


def score_token_segments(model, tokenizer, prompt: str, completion: str, milestones: dict, max_input_tokens: int) -> dict:
    ids, prompt_len, completion_ids, truncated = build_sequence(tokenizer, prompt, completion, max_input_tokens)
    if not completion_ids or prompt_len >= len(ids):
        return {"truncated": truncated, "completion_tokens": len(completion_ids), "score_status": "empty"}
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[0]
        target = input_ids[0, 1:]
        token_logprob = logits[:-1].float().gather(1, target[:, None]).squeeze(1)
        token_logprob = token_logprob - torch.logsumexp(logits[:-1].float(), dim=-1)
        completion_logprob = token_logprob[prompt_len - 1 :].detach().cpu().numpy()
    completion_tokens = len(completion_ids)
    p0 = token_boundary(tokenizer, completion, milestones.get("p0_char_end"))
    p1 = token_boundary(tokenizer, completion, milestones.get("p1_char_end")) or completion_tokens
    p2 = token_boundary(tokenizer, completion, milestones.get("p2_char_end"))
    p1 = min(max(p1, 0), completion_tokens)
    p2 = min(max(p2, p1), completion_tokens) if p2 is not None else completion_tokens

    def agg(start: int, end: int) -> tuple[float | None, int]:
        start, end = max(0, start), min(completion_tokens, end)
        values = completion_logprob[start:end]
        if len(values) == 0:
            return None, 0
        return float(values.sum()), int(len(values))

    segments = {
        "commitment": (0, p1),
        "execution": (p1, completion_tokens),
        "p1_to_p2": (p1, p2),
        "p2_to_answer": (p2, completion_tokens),
    }
    result = {"truncated": truncated, "completion_tokens": completion_tokens, "prompt_tokens": prompt_len, "p0_token_end": p0, "p1_token_end": p1, "p2_token_end": p2, "score_status": "ok"}
    for name, (start, end) in segments.items():
        value, count = agg(start, end)
        result[f"{name}_ll"] = value
        result[f"{name}_tokens"] = count
        result[f"{name}_ll_per_token"] = value / count if value is not None and count else None
    return result


def load_model(model_alias: str, models_root: Path, device: str):
    path = models_root / MODEL_DIRS[model_alias]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)
    model.eval()
    return tokenizer, model


def score_all(expanded: Sequence[dict], alias: str, tokenizer, model, max_input_tokens: int) -> list[dict]:
    outputs = []
    for index, row in enumerate(expanded, start=1):
        prompt = prompt_for(row["question"])
        result = score_token_segments(model, tokenizer, prompt, row["completion"], row["milestones"], max_input_tokens)
        outputs.append({"key": f"{row['completion_sha1']}:{row['style_variant']}", "model": alias, **result})
        if index % 100 == 0:
            print(f"[{alias}] scored {index}/{len(expanded)}", flush=True)
    return outputs


def aggregate(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["benchmark"], row["problem_id"], row["style_variant"])].append(row)
    metrics = ("commitment_ll_per_token", "execution_ll_per_token", "p1_to_p2_ll_per_token", "p2_to_answer_ll_per_token", "commitment_ll", "execution_ll")
    output = []
    for key, values in sorted(grouped.items()):
        item = {"pair": key[0], "benchmark": key[1], "problem_id": key[2], "style_variant": key[3], "n_trajectories": len(values)}
        for metric in metrics:
            for source in ("base", "rl"):
                field = f"{source}_{metric}"
                numbers = [row[field] for row in values if row.get(field) is not None]
                item[f"{field}_mean"] = float(np.mean(numbers)) if numbers else None
        if item.get("base_commitment_ll_per_token_mean") is not None and item.get("rl_commitment_ll_per_token_mean") is not None:
            item["rl_minus_base_commitment_ll_per_token"] = item["rl_commitment_ll_per_token_mean"] - item["base_commitment_ll_per_token_mean"]
        if item.get("base_execution_ll_per_token_mean") is not None and item.get("rl_execution_ll_per_token_mean") is not None:
            item["rl_minus_base_execution_ll_per_token"] = item["rl_execution_ll_per_token_mean"] - item["base_execution_ll_per_token_mean"]
        output.append(item)
    return output


def aggregate_scores_file(scores_path: Path, model: str, out: Path) -> list[dict]:
    """Reproduce the historical execution_training CPU aggregation from ``scores.jsonl``.

    This is deliberately kept as a separate function from :func:`aggregate`,
    which builds the per-problem parquet rows during the scoring stage.  The
    CSV schema and bootstrap settings are unchanged from the former
    former aggregate entry point so existing outputs can be replayed
    and byte-compared without rerunning either model.
    """

    rows = [json.loads(line) for line in scores_path.open(encoding="utf-8") if line.strip()]
    rows = [row for row in rows if row.get("model") == model]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark"]), str(row["style_variant"]))].append(row)
    output: list[dict] = []
    for (benchmark, style), group in sorted(grouped.items()):
        # Keep the problem as the inferential unit even when each problem has
        # multiple branch/normalization trajectories.
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_problem[str(row["problem_id"])].append(row)
        problem_means = []
        for problem_id, problem_rows in sorted(by_problem.items()):
            problem_means.append(
                {
                    "problem_id": problem_id,
                    "commitment": float(
                        np.mean([float(row["base_minus_rl_commitment_ll_per_token"]) for row in problem_rows])
                    ),
                    "execution": float(
                        np.mean([float(row["base_minus_rl_execution_ll_per_token"]) for row in problem_rows])
                    ),
                }
            )
        commitment = [item["commitment"] for item in problem_means]
        execution = [item["execution"] for item in problem_means]
        c_mean, c_lo, c_hi = bootstrap_mean(commitment, seed=1729, draws=1000)
        e_mean, e_lo, e_hi = bootstrap_mean(execution, seed=1729, draws=1000)
        output.append(
            {
                "model": model,
                "benchmark": benchmark,
                "style_variant": style,
                "n_rows": len(group),
                "n_problems": len({row["problem_id"] for row in group}),
                "commitment_diff_mean": c_mean,
                "commitment_ci_lo": c_lo,
                "commitment_ci_hi": c_hi,
                "execution_diff_mean": e_mean,
                "execution_ci_lo": e_lo,
                "execution_ci_hi": e_hi,
                "commitment_gt_execution_fraction": float(
                    np.mean([c > e for c, e in zip(commitment, execution)])
                ),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "aggregate":
        if not args.model or args.out is None:
            raise SystemExit("--mode aggregate requires --model and --out")
        scores_path = args.scores or (args.output_root / args.pair / "scores.jsonl")
        rows = aggregate_scores_file(scores_path, args.model, args.out)
        print(f"rows={len(rows)} output={args.out}")
        if args.compare and args.compare_model and args.compare_out:
            compare = aggregate_scores_file(args.compare, args.compare_model, args.compare_out)
            print(f"compare_rows={len(compare)} output={args.compare_out}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("execution_training requires CUDA")
    torch.cuda.set_device(int(str(args.gpu).split(",")[0]))
    pair = args.pair
    base_alias, rl_alias = PAIR_CONFIG[pair]
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    selected = load_e1_labels(args.e1_root, pair, benchmarks, args.per_branch, args.max_rows, args.smoke)
    source = load_source_rows(selected)
    expanded = expand_selected(selected, source, args.include_normalized)
    for row in expanded:
        row["pair"] = pair
        row["prompt_template"] = "qwen_boxed_v1"
    print(f"selected={len(selected)} expanded={len(expanded)} pair={pair}", flush=True)
    device = torch.device("cuda")
    tokenizer_base, model_base = load_model(base_alias, args.models_root, device)
    base_scores = score_all(expanded, base_alias, tokenizer_base, model_base, args.max_input_tokens)
    base_vocab = len(tokenizer_base)
    del model_base, tokenizer_base
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer_rl, model_rl = load_model(rl_alias, args.models_root, device)
    if len(tokenizer_rl) != base_vocab:
        raise RuntimeError(f"Tokenizer vocabulary mismatch: {base_alias} vs {rl_alias}")
    rl_scores = score_all(expanded, rl_alias, tokenizer_rl, model_rl, args.max_input_tokens)
    del model_rl, tokenizer_rl
    gc.collect()
    torch.cuda.empty_cache()
    by_key = {(row["key"], row["model"]): row for row in base_scores + rl_scores}
    merged = []
    audit = []
    for row in expanded:
        key = f"{row['completion_sha1']}:{row['style_variant']}"
        base = by_key[(key, base_alias)]
        rl = by_key[(key, rl_alias)]
        output = {key_name: value for key_name, value in row.items() if key_name != "milestones"}
        for prefix, score in (("base", base), ("rl", rl)):
            output.update({f"{prefix}_{name}": value for name, value in score.items() if name not in ("key", "model")})
        output["base_minus_rl_commitment_ll_per_token"] = (output.get("base_commitment_ll_per_token") or 0.0) - (output.get("rl_commitment_ll_per_token") or 0.0)
        output["base_minus_rl_execution_ll_per_token"] = (output.get("base_execution_ll_per_token") or 0.0) - (output.get("rl_execution_ll_per_token") or 0.0)
        merged.append(output)
        audit.append({"key": key, "style_variant": row["style_variant"], "branch_before": row["first_calc_branch"], "branch_after": extract_first_calc_branch(row["completion"]), "base_truncated": base["truncated"], "rl_truncated": rl["truncated"], "base_status": base["score_status"], "rl_status": rl["score_status"]})
    output_root = args.output_root / pair
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    pd.DataFrame(aggregate(merged)).to_parquet(output_root / "per_problem.parquet", index=False)
    pd.DataFrame(merged).groupby(["benchmark", "style_variant"], dropna=False).agg({"problem_id": "nunique", "completion_sha1": "count", "base_minus_rl_commitment_ll_per_token": "mean", "base_minus_rl_execution_ll_per_token": "mean"}).reset_index().to_csv(output_root / "summary.csv", index=False)
    pd.DataFrame(audit).to_csv(output_root / "alignment_audit.csv", index=False)
    config = {"experiment_id": "execution_training", "pair": pair, "base_model": base_alias, "rl_model": rl_alias, "benchmarks": benchmarks, "per_branch": args.per_branch, "max_rows": args.max_rows, "include_normalized": args.include_normalized, "max_input_tokens": args.max_input_tokens, "seed": args.seed, "prompt_template": "qwen_boxed_v1", "statistical_unit": "problem", "tokenizer_vocab_size": base_vocab}
    (output_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output_root), "rows": len(merged), "problems": len({row['problem_id'] for row in merged})}, sort_keys=True))


if __name__ == "__main__":
    main()
