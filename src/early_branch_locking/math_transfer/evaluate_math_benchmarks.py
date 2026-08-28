#!/usr/bin/env python3
"""math-eval - Canonical math evaluation entry point.
Hypothesis: a single maintained math entry point keeps model, benchmark and output conventions consistent across the evaluation pipeline.
Inputs: dataset/math_eval; local model directories; optional resource manifest and sampling settings.
Outputs: data/rlvr/outputs/math_summary.csv; data/rlvr/outputs/run_manifest.latest.json
Status: paper-appendix
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (
    DATASET_DIR as DATA_DIR,
    EXTERNAL_DIR,
    MATH_LOGS_DIR,
    MATH_OUTPUTS_DIR,
    MATH_PLOTS_DIR,
    MODEL_DIR as MODELS_DIR,
    REPO_ROOT,
)

OUTPUTS_DIR = MATH_OUTPUTS_DIR
LOGS_DIR = MATH_LOGS_DIR
PLOTS_DIR = MATH_PLOTS_DIR

MATH_BENCHMARKS = ("gsm8k", "math500", "minerva_math", "olympiadbench", "amc23", "aime24")
CHUNK_FILE_RE = re.compile(r"(?P<prefix>.+)_s\d+_e-?\d+\.jsonl$")


@dataclass(frozen=True)
class MathModelSpec:
    alias: str
    model_dir: str
    prompt_type: str
    tp_size: int
    benchmarks: tuple[str, ...]
    max_tokens: int = 16000
    temperature: float = 0.6
    top_p: float = 0.95
    apply_chat_template: bool = True
    use_vllm: bool = True

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.model_dir


def math_model_specs() -> list[MathModelSpec]:
    simple = tuple(MATH_BENCHMARKS)
    aime = ("aime24",)
    dapo = ("aime24", "amc23")
    return [
        MathModelSpec("math_base_7b", "math_base_7b", "qwen-boxed", 1, simple),
        MathModelSpec("math_simple_rl_7b", "math_simple_rl_7b", "qwen-boxed", 1, simple),
        MathModelSpec("math_base_14b", "math_base_14b", "qwen-boxed", 2, simple),
        MathModelSpec("math_simple_rl_14b", "math_simple_rl_14b", "qwen-boxed", 2, simple),
        MathModelSpec("math_base_32b", "math_base_32b", "qwen-boxed", 4, simple),
        MathModelSpec("math_simple_rl_32b", "math_simple_rl_32b", "qwen-boxed", 4, simple),
        MathModelSpec("math_base_qwen_math_7b", "math_base_qwen_math_7b", "qwen-boxed", 1, ("gsm8k", "math500", "amc23")),
        MathModelSpec("math_instruct_qwen_math_7b", "math_instruct_qwen_math_7b", "qwen-boxed", 1, aime),
        MathModelSpec("math_distill_qwen7b", "math_distill_qwen7b", "qwen-boxed", 1, ("gsm8k", "math500")),
        MathModelSpec("math_oat_zero_7b", "math_oat_zero_7b", "qwen-boxed", 1, ("gsm8k", "math500", "amc23")),
        MathModelSpec("math_base_32b_abel", "math_base_32b", "abel", 4, dapo),
        MathModelSpec("math_dapo_32b", "math_dapo_32b", "abel", 4, dapo),
        MathModelSpec("math_olmo3_sft_7b", "Olmo-3-7B-Instruct-SFT", "qwen-boxed", 1, simple),
        MathModelSpec("math_olmo3_dpo_7b", "Olmo-3-7B-Instruct-DPO", "qwen-boxed", 1, simple),
        MathModelSpec("math_olmo3_rlvr_7b", "Olmo-3-7B-Instruct-RLVR", "qwen-boxed", 1, simple),
        # The released base tokenizer has no chat_template.  Use the
        # evaluator's plain chain-of-thought completion prompt until a
        # separately justified base-native template exists.
        MathModelSpec("math_olmo3_base_7b", "Olmo-3-1025-7B", "cot", 1, simple, apply_chat_template=False, use_vllm=False),
        # S1 is an RL-over-distill lineage comparison, not a zero-RL comparison.
        # DeepScaleR's native prompt format is supplied by the model family;
        # keep the external evaluator's plain DeepSeek-style prompt here.
        MathModelSpec("deepscaler_base_1p5b", "deepscaler_base_1p5b", "deepseek-r1", 1, ("gsm8k", "math500"), max_tokens=16384, apply_chat_template=False),
        MathModelSpec("deepscaler_1p5b", "deepscaler_1p5b", "deepseek-r1", 1, ("gsm8k", "math500"), max_tokens=16384, apply_chat_template=False),
    ]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run math transfer evaluation for limit-of-RLVR.")
    parser.add_argument("--mode", choices=("eval", "aggregate", "summarize"), default="eval")
    parser.add_argument("--models", default="", help="Comma-separated model aliases; empty means all math models.")
    parser.add_argument("--math-benchmarks", default="", help="Comma-separated benchmark subset.")
    parser.add_argument("--n-sampling", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--math-vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--math-vllm-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--resource-manifest", type=Path, default=None)
    # Aggregate/summarize mode options; they are ignored by the GPU eval mode.
    parser.add_argument("--model", default="")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--output-root", type=Path, default=None, help="Root for math outputs; applies to eval and aggregate modes.")
    parser.add_argument("--summarize", action="store_true", help="Write math_summary.csv and pass@k plots")
    return parser.parse_args(argv)


def selected_aliases(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def default_resource_manifest() -> Path | None:
    configured = os.environ.get("LIMITOFRLVR_RESOURCE_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    manifest = MATH_OUTPUTS_DIR / "manifests" / "resource_manifest.json"
    return manifest if manifest.is_file() else None


def model_ready(alias: str, manifest_path: Path | None) -> bool:
    if manifest_path is None or not manifest_path.exists():
        return True
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return any((item.get("alias") or item.get("role")) == alias and item.get("status") == "done" for item in payload.get("models", []))


def effective_benchmarks(spec: MathModelSpec, requested: str) -> list[str]:
    if not requested:
        return list(spec.benchmarks)
    selected = {item.strip() for item in requested.split(",") if item.strip()}
    return [benchmark for benchmark in spec.benchmarks if benchmark in selected]


def math_dataset_size(benchmark: str) -> int:
    for name in ("test.jsonl", "test.json"):
        path = DATA_DIR / "math_eval" / benchmark / name
        if path.exists():
            if path.suffix == ".jsonl":
                return sum(1 for line in path.open(encoding="utf-8") if line.strip())
            return len(json.loads(path.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"Math dataset not found for benchmark={benchmark}")


def max_num_seqs(alias: str, requested: int | None) -> int | None:
    if requested is None:
        return None
    if "32b" in alias:
        return min(requested, 16)
    if "14b" in alias:
        return min(requested, 24)
    return requested


def max_batched_tokens(alias: str, requested: int | None) -> int | None:
    if requested is not None:
        return requested
    if "32b" in alias:
        return 6144
    if "14b" in alias:
        return 8192
    return None


def choose_devices(requested: str, tp_size: int) -> str:
    return requested or ",".join(str(index) for index in range(tp_size))


def build_math_command(spec: MathModelSpec, args: argparse.Namespace, benchmark: str, sample_count: int, output_root: Path, python_path: Path) -> list[str]:
    evaluator = EXTERNAL_DIR / "math" / "examples" / "math_eval" / "math_eval.py"
    if not evaluator.exists():
        evaluator = EXTERNAL_DIR / "limit-of-RLVR" / "math" / "examples" / "math_eval" / "math_eval.py"
    if not evaluator.exists():
        raise FileNotFoundError("Cannot locate external math evaluator")
    command = [
        str(python_path),
        str(evaluator),
        "--data_names", benchmark, "--data_dir", str(DATA_DIR / "math_eval"),
        "--model_name_or_path", str(spec.path), "--output_dir", str(output_root / "math" / spec.alias),
        "--prompt_type", spec.prompt_type, "--split", "test",
        "--num_test_sample", str(args.sample_limit), "--start", "0", "--end", str(sample_count),
        "--seed", str(args.seed), "--temperature", str(spec.temperature),
        "--n_sampling", str(args.n_sampling), "--top_p", str(spec.top_p),
        "--max_tokens_per_call", str(spec.max_tokens),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization), "--save_outputs",
        "--disable_custom_all_reduce", "--release_model_before_eval",
    ]
    if spec.use_vllm:
        command.append("--use_vllm")
    else:
        command.append("--use_safetensors")
    if spec.apply_chat_template:
        command.append("--apply_chat_template")
    seqs = max_num_seqs(spec.alias.lower(), args.math_vllm_max_num_seqs)
    tokens = max_batched_tokens(spec.alias.lower(), args.math_vllm_max_num_batched_tokens)
    if seqs is not None:
        command.extend(["--vllm_max_num_seqs", str(seqs)])
    if tokens is not None:
        command.extend(["--vllm_max_num_batched_tokens", str(tokens)])
    if args.enforce_eager:
        command.append("--enforce_eager")
    return command


# ---------------------------------------------------------------------------
# Shard aggregation formerly maintained in ``aggregate.py``.
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def estimate_pass_at_k(num_samples: int, num_correct: np.ndarray, k: int) -> np.ndarray:
    def estimate(n: int, correct: int, take: int) -> float:
        if n - correct < take:
            return 1.0
        return float(1.0 - np.prod(1.0 - take / np.arange(n - correct + 1, n + 1)))

    return np.array([estimate(num_samples, int(correct), k) for correct in num_correct])


def score_matrix(records: list[dict]) -> list[list[bool]]:
    rows = [[bool(item) for item in record.get("score", [])] for record in records]
    max_len = max((len(row) for row in rows), default=0)
    return [row + [row[-1]] * (max_len - len(row)) if row else [False] * max_len for row in rows]


def math_pass_k(rows: list[list[bool]]) -> dict[int, float]:
    if not rows or not rows[0]:
        return {}
    array = np.asarray(rows, dtype=np.bool_)
    n_sampling = array.shape[1]
    correct = np.sum(array, axis=1)
    result = {}
    k = 1
    while k <= n_sampling:
        result[k] = float(np.round(np.mean(estimate_pass_at_k(n_sampling, correct, k)) * 100, 1))
        k *= 2
    return result


def aggregated_metrics(records: list[dict], metric_files: list[Path]) -> dict:
    rows = score_matrix(records)
    array = np.asarray(rows, dtype=np.float32)
    scores = math_pass_k(rows)
    time_seconds = sum(float(load_json(path).get("time_use_in_second", 0.0)) for path in metric_files if path.exists())
    timeout_samples = sum(int(load_json(path).get("timeout_samples", 0)) for path in metric_files if path.exists())
    return {
        "num_samples": len(records),
        "num_scores": int(sum(len(row) for row in rows)),
        "timeout_samples": timeout_samples,
        "empty_samples": sum(int(not record.get("pred", [""])[-1]) for record in records if record.get("pred")),
        "acc": float(np.round(float(array.mean()) * 100, 1)) if array.size else 0.0,
        "pass_acc": float(np.round(np.mean([any(row) for row in rows]) * 100, 1)) if rows else 0.0,
        "pass@k": scores,
        "time_use_in_second": float(time_seconds),
        "time_use_in_minite": f"{int(time_seconds // 60)}:{int(time_seconds % 60):02d}",
        "is_aggregate": True,
    }


def _unique_chunks(paths: list[Path]) -> list[Path]:
    selected = []
    seen: set[tuple[int, ...]] = set()
    for path in sorted(paths):
        indices = tuple(sorted(int(record["idx"]) for record in load_jsonl(path)))
        if indices in seen:
            continue
        seen.add(indices)
        selected.append(path)
    return selected


def aggregate_job(model: str, benchmark: str, sample_limit: int, seed: int, output_root: Path | None = None) -> Path | None:
    root = output_root or MATH_OUTPUTS_DIR / "math"
    benchmark_dir = root / model / benchmark
    if not benchmark_dir.exists():
        return None
    token = f"_{sample_limit}_seed{seed}_"
    chunks = [path for path in benchmark_dir.glob("*.jsonl") if CHUNK_FILE_RE.match(path.name) and token in path.name]
    chunks = _unique_chunks(chunks)
    if not chunks:
        return None
    records_by_idx = {}
    for path in chunks:
        for record in load_jsonl(path):
            records_by_idx[int(record["idx"])] = record
    records = [records_by_idx[index] for index in sorted(records_by_idx)]
    metric_files = [path.with_name(f"{path.stem}_metrics.json") for path in chunks]
    metrics = aggregated_metrics(records, metric_files)
    prefix_match = CHUNK_FILE_RE.match(chunks[0].name)
    assert prefix_match is not None
    prefix = prefix_match.group("prefix")
    merged_jsonl = benchmark_dir / f"{prefix}_merged.jsonl"
    merged_metrics = benchmark_dir / f"{prefix}_merged_metrics.json"
    with merged_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    merged_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged_metrics


def _pass_columns(payload: dict) -> dict[str, float]:
    result = {}
    for key, value in (payload.get("pass@k") or {}).items():
        number = float(value)
        result[f"pass@{key}"] = number / 100.0 if number > 1 else number
    return result


def summarize_math(output_root: Path | None = None) -> Path:
    root = output_root or MATH_OUTPUTS_DIR / "math"
    rows = []
    for path in sorted(root.glob("*/*/*_merged_metrics.json")):
        payload = load_json(path)
        rows.append({
            "model": path.parents[1].name,
            "benchmark": path.parent.name,
            "acc": float(payload.get("acc", 0)) / 100.0 if float(payload.get("acc", 0)) > 1 else float(payload.get("acc", 0)),
            "pass_acc": float(payload.get("pass_acc", 0)) / 100.0 if float(payload.get("pass_acc", 0)) > 1 else float(payload.get("pass_acc", 0)),
            "num_samples": payload.get("num_samples", 0),
            "metric_path": str(path),
            **_pass_columns(payload),
        })
    frame = pd.DataFrame(rows)
    output_path = root.parent / "math_summary.csv"
    if not frame.empty:
        frame.sort_values(["benchmark", "model"]).to_csv(output_path, index=False)
        try:
            import matplotlib.pyplot as plt

            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            for benchmark, group in frame.groupby("benchmark"):
                ks = sorted(int(column.split("@", 1)[1]) for column in group.columns if column.startswith("pass@"))
                if not ks:
                    continue
                plt.figure(figsize=(9, 6))
                for _, row in group.sort_values("model").iterrows():
                    values = [row.get(f"pass@{k}") for k in ks]
                    plt.plot(ks, values, marker="o", label=row["model"])
                plt.xscale("log", base=2)
                plt.xlabel("k")
                plt.ylabel("pass@k")
                plt.title(f"Math pass@k - {benchmark}")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f"math_{benchmark}_passk.png", dpi=200)
                plt.close()
        except ImportError:
            pass
    return output_path


def run_math_job(spec: MathModelSpec, benchmarks: list[str], args: argparse.Namespace, manifest: Path | None) -> dict:
    if not spec.path.exists() or not model_ready(spec.alias, manifest):
        if args.allow_missing_models:
            return {"status": "missing_model", "model": spec.alias, "path": str(spec.path)}
        raise FileNotFoundError(f"Model not found or not ready: {spec.path}")
    python_path = REPO_ROOT / ".venvs" / "math-eval" / "bin" / "python"
    if not python_path.exists():
        # The repository also supports the documented TinyZero conda runtime;
        # keep the exact interpreter when the optional venv is absent.
        python_path = Path(sys.executable)
    output_root = args.output_root or OUTPUTS_DIR
    output_dir = output_root / "math" / spec.alias
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = choose_devices(args.cuda_visible_devices, spec.tp_size)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Let this vLLM build select its CUDA attention backend.  Hard-coding the
    # old TRITON_ATTN name (or CPU-only TORCH_SDPA) breaks current 4090 wheels.
    env.pop("VLLM_ATTENTION_BACKEND", None)
    env.setdefault("LIMITOFRLVR_MATH_EVAL_WORKERS", "16")
    summaries = []
    for benchmark in benchmarks:
        sample_count = min(args.sample_limit, math_dataset_size(benchmark))
        subprocess.run(build_math_command(spec, args, benchmark, sample_count, output_root, python_path), check=True, env=env)
        aggregate_job(spec.alias, benchmark, args.sample_limit, args.seed, output_root / "math")
        metric_files = sorted((output_dir / benchmark).glob("*_merged_metrics.json"))
        if metric_files:
            summaries.append(json.loads(metric_files[-1].read_text(encoding="utf-8")))
    return {"status": "done", "model": spec.alias, "summary": summaries}


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "summarize" or args.summarize:
        print(summarize_math(args.output_root))
        return
    if args.mode == "aggregate":
        output_root = args.output_root or MATH_OUTPUTS_DIR / "math"
        jobs = [(args.model, args.benchmark)] if args.model and args.benchmark else [
            (model.alias, benchmark)
            for model in math_model_specs()
            for benchmark in model.benchmarks
        ]
        for model, benchmark in jobs:
            path = aggregate_job(model, benchmark, args.sample_limit, args.seed, output_root)
            if path:
                print(path)
        return
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    aliases = selected_aliases(args.models)
    manifest = args.resource_manifest or default_resource_manifest()
    run_manifest = {"math": []}
    for spec in math_model_specs():
        if aliases and spec.alias not in aliases:
            continue
        benchmarks = effective_benchmarks(spec, args.math_benchmarks)
        if benchmarks:
            run_manifest["math"].append(run_math_job(spec, benchmarks, args, manifest))
    output_root = args.output_root or OUTPUTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.latest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
