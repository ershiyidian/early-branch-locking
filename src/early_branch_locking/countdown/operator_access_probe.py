
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""operator_access - Format-free first-operator probe.
Hypothesis: checkpoint training changes the first-operator distribution before the answer format is considered.
Inputs: teacher success prefixes; dataset/test.parquet; base and checkpoint model paths.
Outputs: data/analysis_results/rlvr_passk/metrics/operator_access_op1_probe_summary_op1_teacher50_n150.csv; data/analysis_results/rlvr_passk/metrics/operator_access_op1_probe_layers_op1_logitlens_n150.csv
Status: paper-main
"""
from __future__ import annotations
"""
operator_access_op1_probe_countdown.py

Format-free op1 probe:
- Build prefix_before_op1 from teacher success trajectories.
- Feed clean prompt + prefix into model.
- Measure next-token distribution over {+,-,*,/}.
- Optional logit lens across layers.
"""

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR as ACTOR_DIR, METRICS_DIR, TEST_PARQUET  # noqa: E402

METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.op1_utils import (  # noqa: E402
    build_prefix_records,
    build_format_free_inputs,
    load_problem_ids_from_sets,
    get_device,
    get_op_token_ids,
    entropy_from_probs,
    kl_divergence,
    summarise_op_probs,
    apply_final_norm,
    ckpt_name_from_path,
)
from early_branch_locking.core.countdown_shared import collect_model_paths  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_raw_path", type=str, required=True)
    p.add_argument("--base_model_path", type=str, default="model/qwen253B")
    p.add_argument("--model_paths", type=str, nargs="*", default=None)
    p.add_argument("--actor_dir", type=str, default=str(ACTOR_DIR))
    p.add_argument("--only_steps", type=str, default="")
    p.add_argument("--include_base", action="store_true", default=False)
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--logit_lens", action="store_true", default=False)
    p.add_argument("--sets_path", type=str, default="")
    p.add_argument("--set_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    return p.parse_args()


def _collect_model_paths_from_args(args) -> List[str]:
    return collect_model_paths(
        base_model_path=args.base_model_path,
        include_base=args.include_base,
        actor_dir=args.actor_dir,
        only_steps=args.only_steps,
        explicit_paths=args.model_paths,
    )


def compute_op_probs(
    model,
    tokenizer,
    input_texts: List[str],
    op_ids: List[int],
    batch_size: int,
    device: torch.device,
    logit_lens: bool,
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    op_probs = []
    layer_probs: Optional[List[List[torch.Tensor]]] = None

    for start in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=logit_lens, use_cache=False)
        logits = outputs.logits[:, -1, :]
        p = torch.softmax(logits[:, op_ids], dim=-1)
        op_probs.append(p.float().detach().cpu())

        if logit_lens:
            if layer_probs is None:
                layer_probs = [[] for _ in range(len(outputs.hidden_states))]
            for li, hs in enumerate(outputs.hidden_states):
                hs_last = apply_final_norm(model, hs[:, -1, :])
                layer_logits = model.lm_head(hs_last)
                p_layer = torch.softmax(layer_logits[:, op_ids], dim=-1)
                layer_probs[li].append(p_layer.float().detach().cpu())

    op_probs_arr = torch.cat(op_probs, dim=0).numpy()
    if logit_lens and layer_probs is not None:
        layer_probs_arr = [torch.cat(x, dim=0).detach().numpy() for x in layer_probs]
    else:
        layer_probs_arr = None
    return op_probs_arr, layer_probs_arr


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    teacher_raw_path = Path(args.teacher_raw_path)
    prefix_by_pid, _ = build_prefix_records(teacher_raw_path, args.num_problems)
    pids, prompts, _ = build_format_free_inputs(TEST_PARQUET, prefix_by_pid, args.num_problems)
    prefixes = [prefix_by_pid[pid] for pid in pids]

    if args.sets_path and args.set_name:
        set_ids = load_problem_ids_from_sets(Path(args.sets_path), args.set_name)
        if set_ids:
            keep = set(set_ids)
            filtered = [(pid, p, pref) for pid, p, pref in zip(pids, prompts, prefixes) if pid in keep]
            pids, prompts, prefixes = zip(*filtered) if filtered else ([], [], [])
            pids, prompts, prefixes = list(pids), list(prompts), list(prefixes)

    if args.max_examples and args.max_examples > 0 and len(pids) > args.max_examples:
        idx = list(range(len(pids)))
        random.shuffle(idx)
        idx = idx[: args.max_examples]
        pids = [pids[i] for i in idx]
        prompts = [prompts[i] for i in idx]
        prefixes = [prefixes[i] for i in idx]

    if not pids:
        raise ValueError("No valid prefix records found for op1 probe.")

    input_texts = [p + pref for p, pref in zip(prompts, prefixes)]

    device = get_device(args.gpu_id)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    model_paths = _collect_model_paths_from_args(args)
    if not model_paths:
        raise ValueError("No model paths provided.")

    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    op_ids = get_op_token_ids(tokenizer)

    base_probs = None
    base_layer_probs = None

    rows = []
    layer_rows = []
    per_problem_rows = []

    if args.include_base or args.base_model_path in model_paths:
        base_path = args.base_model_path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        base_model.eval()
        base_probs, base_layer_probs = compute_op_probs(
            base_model,
            tokenizer,
            input_texts,
            op_ids,
            args.batch_size,
            device,
            args.logit_lens,
        )
        base_model.cpu()
        del base_model
        torch.cuda.empty_cache()

    for model_path in model_paths:
        ckpt = ckpt_name_from_path(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model.eval()
        probs, layer_probs = compute_op_probs(
            model,
            tokenizer,
            input_texts,
            op_ids,
            args.batch_size,
            device,
            args.logit_lens,
        )

        summary = summarise_op_probs(probs)
        row = {
            "checkpoint": ckpt,
            "num_examples": len(pids),
        }
        row.update(summary)
        if base_probs is not None:
            row["op1_kl_base_mean"] = float(kl_divergence(base_probs, probs).mean())
        rows.append(row)

        if args.logit_lens and layer_probs is not None:
            for li, p_layer in enumerate(layer_probs):
                layer_summary = summarise_op_probs(p_layer)
                layer_row = {
                    "checkpoint": ckpt,
                    "layer": li,
                    "num_examples": len(pids),
                }
                layer_row.update(layer_summary)
                if base_layer_probs is not None:
                    layer_row["op1_kl_base_mean"] = float(
                        kl_divergence(base_layer_probs[li], p_layer).mean()
                    )
                layer_rows.append(layer_row)

        if args.save_per_problem:
            for pid, p in zip(pids, probs):
                per_problem_rows.append(
                    dict(
                        checkpoint=ckpt,
                        problem_index=pid,
                        op1_p_plus=float(p[0]),
                        op1_p_minus=float(p[1]),
                        op1_p_mul=float(p[2]),
                        op1_p_div=float(p[3]),
                    )
                )

        model.cpu()
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    tag = args.tag or f"n{len(pids)}"
    out_path = METRICS_DIR / f"operator_access_op1_probe_summary_{tag}.csv"
    df.to_csv(out_path, index=False)

    if layer_rows:
        df_layers = pd.DataFrame(layer_rows)
        out_layers = METRICS_DIR / f"operator_access_op1_probe_layers_{tag}.csv"
        df_layers.to_csv(out_layers, index=False)

    if per_problem_rows:
        df_pp = pd.DataFrame(per_problem_rows)
        out_pp = METRICS_DIR / f"operator_access_op1_probe_per_problem_{tag}.parquet"
        df_pp.to_parquet(out_pp, index=False)

    print(f"Saved summary -> {out_path}")
    if layer_rows:
        print(f"Saved layers -> {out_layers}")
    if per_problem_rows:
        print(f"Saved per-problem -> {out_pp}")


# ---- merged trajectory mode ----
"""Compute comparable Op1 entropy trajectory across RLVR checkpoints."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from early_branch_locking.core.countdown_shared import collect_model_paths, enumerate_solution_list, extract_ground_truth, load_parquet_sorted, step_of  # noqa: E402
from early_branch_locking.core.op1_utils import build_clean_prompt, entropy_from_probs, get_op_token_ids, summarise_op_probs  # noqa: E402

OPS = ("+", "-", "*", "/")


def parse_args_trajectory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExpF2: full Op1 entropy trajectory.")
    parser.add_argument("--actor_dir", default=str(ACTOR_DIR))
    parser.add_argument("--model_paths", nargs="*", default=None)
    parser.add_argument("--only_steps", default="50,75,100,125,150,175,200,225,250,275")
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--candidate_scan", type=int, default=2047)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", default="1")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def first_op(expr: str) -> str:
    for char in expr:
        if char in OPS:
            return char
    return ""


def prefix_before_first_op(expr: str) -> str:
    for index, char in enumerate(expr):
        if char in OPS:
            return expr[:index]
    raise ValueError(f"Expression has no operator: {expr}")


def select_prompts(args: argparse.Namespace) -> tuple[List[str], List[dict]]:
    records = load_parquet_sorted(TEST_PARQUET, n=args.candidate_scan, sort_key="sample_id")
    prompts, metadata = [], []
    for pid, record in enumerate(records):
        numbers, target, feasible = extract_ground_truth(record)
        if feasible != "yes":
            continue
        solutions = enumerate_solution_list(numbers, target)
        valid_ops = sorted({first_op(expr) for expr in solutions if first_op(expr)})
        if len(valid_ops) < 2:
            continue
        prefix = prefix_before_first_op(solutions[0])
        prompts.append(build_clean_prompt(numbers, target) + prefix)
        metadata.append({"problem_index": pid, "num_solutions": len(solutions), "valid_ops": "".join(valid_ops)})
        if len(prompts) >= args.num_problems:
            break
    if len(prompts) < args.num_problems:
        raise ValueError(f"Only found {len(prompts)} prompts; requested {args.num_problems}.")
    return prompts, metadata


@torch.no_grad()
def compute_probs(model, tokenizer, prompts: List[str], op_ids: List[int], batch_size: int, device) -> np.ndarray:
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {key: value.to(device) for key, value in enc.items()}
        logits = model(**enc, use_cache=False).logits[:, -1, :]
        chunks.append(torch.softmax(logits[:, op_ids], dim=-1).float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def load_model(path: str, dtype: torch.dtype, device):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    return model


def summarize_checkpoint(name: str, probs: np.ndarray, metadata: List[dict]) -> dict:
    summary = {"checkpoint": name, "step": step_of(name), "num_examples": len(metadata)}
    summary.update(summarise_op_probs(probs))
    summary["op1_entropy_std"] = float(entropy_from_probs(probs).std())
    for index, op_name in enumerate(("plus", "minus", "mul", "div")):
        summary[f"mean_p_{op_name}"] = float(probs[:, index].mean())
    return summary


def run_trajectory() -> None:
    args = parse_args_trajectory()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for operator_access2.")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    prompts, metadata = select_prompts(args)
    model_paths = collect_model_paths(actor_dir=args.actor_dir, only_steps=args.only_steps, explicit_paths=args.model_paths)
    if not model_paths:
        raise ValueError("No model paths selected.")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    op_ids = get_op_token_ids(tokenizer)
    rows = []
    for model_path in model_paths:
        model = load_model(model_path, dtype, device)
        probs = compute_probs(model, tokenizer, prompts, op_ids, args.batch_size, device)
        rows.append(summarize_checkpoint(Path(model_path).name, probs, metadata))
        model.cpu()
        del model
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows).sort_values("step")
    out.to_csv(METRICS_DIR / "operator_access2_op1_entropy_full_trajectory.csv", index=False)
    print(out.to_string(index=False), flush=True)


# ---- merged hierarchy mode ----
"""Probe Op1/Op2/Op3 next-operator entropy on solver-derived prefixes."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from early_branch_locking.core.countdown_shared import collect_model_paths, enumerate_solution_list, extract_ground_truth, load_parquet_sorted, step_of  # noqa: E402
from early_branch_locking.core.op1_utils import build_clean_prompt, entropy_from_probs, get_op_token_ids  # noqa: E402

OPS = ("+", "-", "*", "/")


def parse_args_hierarchy() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExpF3: Op1/Op2/Op3 operator hierarchy probe.")
    parser.add_argument("--actor_dir", default=str(ACTOR_DIR))
    parser.add_argument("--only_steps", default="50,150,275")
    parser.add_argument("--model_paths", nargs="*", default=None)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", default="1")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def operator_positions(expr: str) -> List[int]:
    return [index for index, char in enumerate(expr) if char in OPS]


def build_examples(num_problems: int) -> List[dict]:
    records = load_parquet_sorted(TEST_PARQUET, n=num_problems, sort_key="sample_id")
    examples = []
    for pid, record in enumerate(records):
        numbers, target, feasible = extract_ground_truth(record)
        if feasible != "yes":
            continue
        clean = build_clean_prompt(numbers, target)
        for expr in enumerate_solution_list(numbers, target):
            examples.extend(expression_examples(pid, clean, expr))
    if not examples:
        raise ValueError("No operator hierarchy examples found.")
    return examples


def expression_examples(pid: int, clean_prompt: str, expr: str) -> List[dict]:
    positions = operator_positions(expr)
    rows = []
    for op_index, pos in enumerate(positions[:3], start=1):
        rows.append(
            {
                "problem_index": pid,
                "canonical_expr": expr,
                "op_position": f"op{op_index}",
                "target_op": expr[pos],
                "input_text": clean_prompt + expr[:pos],
            }
        )
    return rows


@torch.no_grad()
def compute_probs(model, tokenizer, texts: List[str], op_ids: List[int], batch_size: int, device) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {key: value.to(device) for key, value in enc.items()}
        logits = model(**enc, use_cache=False).logits[:, -1, :]
        chunks.append(torch.softmax(logits[:, op_ids], dim=-1).float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def load_model(path: str, dtype: torch.dtype, device):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    return model


def per_example_rows(checkpoint: str, examples: List[dict], probs: np.ndarray) -> List[dict]:
    rows = []
    ent = entropy_from_probs(probs)
    for example, prob, entropy in zip(examples, probs, ent):
        target_index = OPS.index(example["target_op"])
        rows.append(
            {
                **{key: value for key, value in example.items() if key != "input_text"},
                "checkpoint": checkpoint,
                "step": step_of(checkpoint),
                "entropy": float(entropy),
                "top_op": OPS[int(np.argmax(prob))],
                "target_prob": float(prob[target_index]),
                "target_is_top": bool(int(np.argmax(prob)) == target_index),
                "p_plus": float(prob[0]),
                "p_minus": float(prob[1]),
                "p_mul": float(prob[2]),
                "p_div": float(prob[3]),
            }
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["entropy", "target_prob", "target_is_top", "p_plus", "p_minus", "p_mul", "p_div"]
    return df.groupby(["checkpoint", "step", "op_position"], sort=False)[metrics].mean().reset_index()


def run_hierarchy() -> None:
    args = parse_args_hierarchy()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for operator_access3.")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    examples = build_examples(args.num_problems)
    model_paths = collect_model_paths(actor_dir=args.actor_dir, only_steps=args.only_steps, explicit_paths=args.model_paths)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    op_ids = get_op_token_ids(tokenizer)
    rows = []
    texts = [example["input_text"] for example in examples]
    for model_path in model_paths:
        model = load_model(model_path, dtype, device)
        probs = compute_probs(model, tokenizer, texts, op_ids, args.batch_size, device)
        rows.extend(per_example_rows(Path(model_path).name, examples, probs))
        model.cpu()
        del model
        torch.cuda.empty_cache()
    per_df = pd.DataFrame(rows)
    summary = summarize(per_df)
    summary.to_csv(METRICS_DIR / "operator_accesentrance_entropy_training_operator_hierarchy.csv", index=False)
    per_df.to_parquet(METRICS_DIR / "operator_accesentrance_entropy_training_operator_hierarchy_per_prefix.parquet", index=False)
    print(summary.to_string(index=False), flush=True)

def _run_selected():
    selector = None
    selector_index = None
    flag = "--mode"
    for index, argument in enumerate(sys.argv):
        if argument == flag:
            selector_index = index
            selector = sys.argv[index + 1] if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--") else "__flag__"
            break
        if argument.startswith(flag + "="):
            selector_index = index
            selector = argument.split("=", 1)[1]
            break
    if selector_index is not None:
        if selector == "__flag__":
            sys.argv.pop(selector_index)
        else:
            del sys.argv[selector_index:selector_index + 2]
        if selector == "trajectory":
            return run_trajectory()
        if selector == "hierarchy":
            return run_hierarchy()
        if selector not in {"probe", "__flag__"}:
            raise ValueError(f"Unknown --mode: {selector}")
    return main()

if __name__ == "__main__":
    _run_selected()
