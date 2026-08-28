#!/usr/bin/env python3
"""math-c2f-campaign - Cross-model C2F campaign scheduler.
Hypothesis: a common C2F protocol makes transfer effects comparable across model pairs, benchmarks and prefix budgets.
Inputs: existing math raw JSONL baselines; local model/resource availability; campaign matrix settings.
Outputs: data/rlvr/outputs/experiments/c2f/metrics/c2f_campaign_overview.csv
Status: paper-appendix
"""
from __future__ import annotations
"""C2F campaign scheduler.

Hypothesis: run the same C2F protocol across benchmark/prefix/model pairs so
the transfer effect can be compared without changing output naming.
Inputs:  existing math raw JSONL baselines and GPU/model availability.
Outputs: data/rlvr/outputs/experiments/c2f/metrics plus campaign logs.
Status:  paper-appendix
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import C2F_METRICS_DIR, C2F_RAW_DIR, MATH_LOGS_DIR, MATH_OUTPUTS_DIR, RLVR_ROOT
from early_branch_locking.math_transfer import run_candidate_to_final as c2f_run
from early_branch_locking.math_transfer.run_candidate_to_final import find_leak, load_benchmark_records, rebuild_prefix

PAIR_CHOICES = ("7b", "14b", "32b")
DEFAULT_BENCHMARKS = ("gsm8k", "math500", "minerva_math", "olympiadbench", "amc23", "aime24")
DEFAULT_PREFIXES = (50, 100, 150, 200)
LOG_ROOT = MATH_LOGS_DIR / "c2f_campaign"
LOG_SUMMARY_PATH = MATH_LOGS_DIR / "c2f_campaign.md"
CAMPAIGN_TAG = "campaign"
BENCHMARK_TOKEN_CAPS = {
    "gsm8k": 1024,
    "math500": 2560,
    "minerva_math": 2048,
    "olympiadbench": 4096,
    "amc23": 4096,
    "aime24": 4096,
}


@dataclass(frozen=True)
class PairSpec:
    name: str
    draft_model: str
    refine_model: str
    tp_size: int
    batch_size: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full C2F math campaign with GPU-aware waves")
    parser.add_argument("--mode", choices=("run", "summarize", "audit-leakage", "budget-recompute"), default="run")
    parser.add_argument("--pairs", type=str, default="7b,14b")
    parser.add_argument("--benchmarks", type=str, default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--prefixes", type=str, default=",".join(str(value) for value in DEFAULT_PREFIXES))
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--n-sampling", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--refine-max-tokens", type=int, default=16000)
    parser.add_argument("--draft-max-tokens", type=int, default=16000)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--wait-for-idle", action="store_true", default=True)
    parser.add_argument("--summarize", action="store_true",
                        help="aggregate existing campaign summaries without launching jobs")
    parser.add_argument("--raw-dir", type=Path, default=C2F_RAW_DIR)
    parser.add_argument("--math-root", type=Path, default=MATH_OUTPUTS_DIR / "math")
    parser.add_argument("--out-dir", type=Path, default=C2F_METRICS_DIR)
    parser.add_argument("--only-tag", default="")
    parser.add_argument("--recompute-leakfree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=None, help="Output CSV for budget-recompute mode")
    return parser.parse_args(argv)


def pair_specs() -> dict[str, PairSpec]:
    return {
        "7b": PairSpec("7b", "math_base_7b", "math_simple_rl_7b", 1, 8),
        "14b": PairSpec("14b", "math_base_14b", "math_simple_rl_14b", 2, 8),
        "32b": PairSpec("32b", "math_base_32b", "math_simple_rl_32b", 4, 4),
    }


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_prefixes(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_arg(raw)]


def effective_sample_limit(benchmark: str, requested_limit: int) -> int:
    records = load_benchmark_records(benchmark, requested_limit)
    return len(records)


def gpu_groups(gpu_list: list[str], tp_size: int) -> list[str]:
    if len(gpu_list) % tp_size != 0:
        raise ValueError(f"GPU count {len(gpu_list)} is not divisible by tp_size={tp_size}")
    groups = []
    for index in range(0, len(gpu_list), tp_size):
        groups.append(",".join(gpu_list[index:index + tp_size]))
    return groups


def merged_jsonl_path(model_alias: str, benchmark: str) -> Path:
    candidates = sorted((MATH_OUTPUTS_DIR / "math" / model_alias / benchmark).glob("*_merged.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Missing merged baseline for {model_alias}/{benchmark}")
    return candidates[-1]


def campaign_summary_files() -> list[Path]:
    return sorted(C2F_METRICS_DIR.glob("c2f_summary_campaign_*.csv"))


def load_campaign_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def campaign_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("variant") == variant), None)


def build_campaign_overview() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in campaign_summary_files():
        rows = load_campaign_rows(path)
        c2f = campaign_variant(rows, "c2f")
        draft = campaign_variant(rows, "draft")
        refine = campaign_variant(rows, "refine")
        if not c2f or not draft or not refine:
            continue

        def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
            value = row.get(key, "")
            return float(value) if value not in {"", "None", None} else default

        records.append({
            "draft_model": c2f.get("draft_model", ""),
            "refine_model": c2f.get("refine_model", ""),
            "benchmark": c2f.get("benchmark", ""),
            "prefix_tokens": int(number(c2f, "prefix_tokens")),
            "num_problems": int(number(c2f, "num_problems")),
            "draft_pass@1": number(draft, "pass@1"),
            "draft_pass@64": number(draft, "pass@64"),
            "refine_pass@1": number(refine, "pass@1"),
            "refine_pass@64": number(refine, "pass@64"),
            "c2f_pass@1": number(c2f, "pass@1"),
            "c2f_pass@64": number(c2f, "pass@64"),
            "draft_avg_tokens": number(draft, "avg_output_tokens_mean"),
            "refine_avg_tokens": number(refine, "avg_output_tokens_mean"),
            "c2f_avg_tokens": number(c2f, "avg_output_tokens_mean"),
            "summary_path": str(path),
        })
    return records


def summarize_campaign() -> Path:
    records = build_campaign_overview()
    output = C2F_METRICS_DIR / "c2f_campaign_overview.csv"
    if not records:
        raise FileNotFoundError(f"No campaign summaries found in {C2F_METRICS_DIR}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


# ---------------------------------------------------------------------------
# Offline campaign audits formerly maintained as separate entry points.
# ---------------------------------------------------------------------------


BENCHMARK_PATTERN = "gsm8k|math500|minerva_math|olympiadbench|amc23|aime24"
TAG_RE = re.compile(
    rf"^(?P<method>draft|refine|c2f)_campaign_(?P<draft>.+)_to_(?P<refine>.+)_(?P<benchmark>{BENCHMARK_PATTERN})_tokens(?P<prefix_tokens>\d+)\.jsonl$"
)


def parse_tag(fname: str) -> tuple[str, str, str, int]:
    match = TAG_RE.match(Path(fname).name)
    if match is None:
        raise ValueError(f"not a campaign raw filename: {fname}")
    return match.group("draft"), match.group("refine"), match.group("benchmark"), int(match.group("prefix_tokens"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _load_triple(raw_dir: Path, tag: str) -> dict[str, Path]:
    paths = {method: raw_dir / f"{method}_{tag}.jsonl" for method in ("draft", "refine", "c2f")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete campaign triple for {tag}: {missing}")
    return paths


def _completion_from_sources(draft_row: dict, merged_rows: list[dict]) -> str:
    value = draft_row.get("completion")
    if value is not None:
        return str(value)
    pid = int(draft_row["problem_index"])
    sample = int(draft_row["sample_index"])
    if not 0 <= pid < len(merged_rows):
        raise IndexError(f"draft problem_index out of range: {pid}")
    completions = merged_rows[pid].get("code", [])
    if not isinstance(completions, list) or not 0 <= sample < len(completions):
        raise IndexError(f"draft sample index out of range: pid={pid} sample={sample}")
    return str(completions[sample])


def audit_one_tag(tag_triple_paths: dict[str, Path], math_root: Path, tokenizer) -> tuple[dict, list[dict], dict]:
    draft_model, refine_model, benchmark, prefix_tokens = parse_tag(tag_triple_paths["c2f"].name)
    draft_rows = _read_jsonl(tag_triple_paths["draft"])
    c2f_rows = _read_jsonl(tag_triple_paths["c2f"])
    merged_rows = _read_jsonl(_merged_path_for_audit(math_root, draft_model, benchmark))
    draft_by_key = {(int(row["problem_index"]), int(row["sample_index"])): row for row in draft_rows}
    details: list[dict] = []
    filtered: dict[int, list[dict]] = defaultdict(list)
    leak_counts = Counter()
    single_char = 0
    for row in c2f_rows:
        pid = int(row["problem_index"])
        sample = int(row["sample_index"])
        merged = merged_rows[pid]
        gt = str(merged.get("gt", merged.get("ground_truth", "")))
        if len(gt.strip()) == 1:
            single_char += 1
        draft_row = draft_by_key.get((pid, sample), row)
        completion = _completion_from_sources(draft_row, merged_rows)
        prefix = rebuild_prefix(completion, tokenizer, prefix_tokens)
        leak = find_leak(prefix, gt)
        if leak is None:
            filtered[pid].append(row)
            leak_type, leak_pos = "", ""
        else:
            leak_type, leak_pos = leak
            leak_counts[leak_type] += 1
        details.append({"tag": tag_triple_paths["c2f"].stem.removeprefix("c2f_"), "problem_index": pid, "sample_index": sample, "leak_type": leak_type, "leak_char_pos": leak_pos, "gt_single_char": len(gt.strip()) == 1, "prefix_tail": prefix[-150:]})
    tag = tag_triple_paths["c2f"].stem.removeprefix("c2f_")
    summary = {"tag": tag, "draft_model": draft_model, "refine_model": refine_model, "benchmark": benchmark, "prefix_tokens": prefix_tokens, "n_rows": len(c2f_rows), "n_leaks": sum(leak_counts.values()), "leak_rate": sum(leak_counts.values()) / len(c2f_rows) if c2f_rows else 0.0, "boxed_count": leak_counts["boxed"], "gsm_hash_count": leak_counts["gsm_hash"], "conclusion_count": leak_counts["conclusion"], "gt_string_count": leak_counts["gt_string"], "gt_single_char_count": single_char, "num_problems": len({int(row["problem_index"]) for row in c2f_rows})}
    return summary, details, filtered


def _merged_path_for_audit(math_root: Path, draft_model: str, benchmark: str) -> Path:
    paths = sorted((math_root / draft_model / benchmark).glob("*_merged.jsonl"))
    paths = [path for path in paths if "_1_seed" not in path.name]
    if not paths:
        raise FileNotFoundError(f"missing merged baseline for {draft_model}/{benchmark}")
    return paths[-1]


def _leakfree_row(summary: dict, filtered: dict[int, list[dict]]) -> dict:
    counts = {}
    attempts = {}
    for pid, rows in filtered.items():
        attempts[pid] = len(rows)
        counts[pid] = sum(bool(row.get("correct", row.get("score", 0))) for row in rows)
    row = {"draft_model": summary["draft_model"], "refine_model": summary["refine_model"], "benchmark": summary["benchmark"], "prefix_tokens": summary["prefix_tokens"], "num_problems": summary["num_problems"], "n_rows_before": summary["n_rows"], "n_rows_after": sum(attempts.values()), "leak_rate": summary["leak_rate"], "mean_attempts_after_filter": sum(attempts.values()) / len(attempts) if attempts else 0.0}
    for k in (1, 2, 4, 8, 16, 32, 64):
        values_for_k = [c2f_run.pass_at_k(attempts[pid], counts[pid], k) for pid in attempts if attempts[pid] > 0]
        row[f"pass@{k}"] = sum(values_for_k) / len(values_for_k) if values_for_k else 0.0
    return row


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_leakage_audit(args: argparse.Namespace) -> None:
    campaign_paths = sorted(args.raw_dir.glob("c2f_campaign_*.jsonl"))
    tags = sorted({path.stem.removeprefix("c2f_") for path in campaign_paths})
    if args.only_tag:
        tags = [tag for tag in tags if tag == args.only_tag]
    if not tags:
        raise SystemExit("no matching campaign tags")
    from transformers import AutoTokenizer

    tokenizers = {}
    summaries: list[dict] = []
    leakfree: list[dict] = []
    printed = 0
    for tag in tags:
        triple = _load_triple(args.raw_dir, tag)
        _, refine_model, _, _ = parse_tag(triple["c2f"].name)
        if refine_model not in tokenizers:
            tokenizers[refine_model] = AutoTokenizer.from_pretrained(str(ROOT / "model" / refine_model), local_files_only=True, trust_remote_code=True, use_fast=False)
        summary, details, filtered = audit_one_tag(triple, args.math_root, tokenizers[refine_model])
        summaries.append(summary)
        if args.recompute_leakfree:
            leakfree.append(_leakfree_row(summary, filtered))
        if args.only_tag:
            for detail in details:
                if detail["leak_type"] and printed < 5:
                    print(json.dumps(detail, ensure_ascii=False))
                    printed += 1
    audit_path = args.out_dir / "c2f_leakage_audit.csv"
    _write_audit_csv(audit_path, summaries)
    if args.recompute_leakfree:
        _write_audit_csv(args.out_dir / "c2f_leakfree_summary.csv", leakfree)
    print(f"audit_rows={len(summaries)} output={audit_path}")
    for row in summaries:
        print(json.dumps(row, sort_keys=True))


def run_budget_recompute(args: argparse.Namespace) -> None:
    tags = sorted(path.stem.removeprefix("c2f_") for path in args.raw_dir.glob("c2f_campaign_*.jsonl"))
    tags = sorted(set(tags))
    if args.only_tag:
        tags = [tag for tag in tags if tag == args.only_tag]
    output: list[dict] = []
    for tag in tags:
        triple = _load_triple(args.raw_dir, tag)
        draft_model, refine_model, benchmark, prefix_tokens = parse_tag(triple["c2f"].name)
        raw = {method: _read_jsonl(path) for method, path in triple.items()}
        for row in c2f_run.budget_matched_rows(raw):
            output.append({"tag": tag, "draft_model": draft_model, "refine_model": refine_model, "benchmark": benchmark, "prefix_tokens": prefix_tokens, **row})
    if not output:
        raise SystemExit("no matching campaign triples")
    output_path = args.out or (args.out_dir / "c2f_budget_matched_campaign.csv")
    _write_audit_csv(output_path, output)
    print(f"rows={len(output)} tags={len(set(row['tag'] for row in output))} output={output_path}")


def load_summary_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def matches_summary(path: Path, draft_model: str, refine_model: str, benchmark: str, prefix_tokens: int, n_sampling: int, num_problems: int) -> bool:
    rows = load_summary_rows(path)
    for row in rows:
        if row["draft_model"] != draft_model:
            continue
        if row["refine_model"] != refine_model:
            continue
        if row["benchmark"] != benchmark:
            continue
        if int(float(row["n_sampling"])) != n_sampling:
            continue
        if int(float(row["num_problems"])) != num_problems:
            continue
        prefix_value = row.get("prefix_tokens", "")
        if prefix_value in {"", "None"}:
            continue
        if int(float(prefix_value)) == prefix_tokens:
            return True
    return False


def existing_summary_path(draft_model: str, refine_model: str, benchmark: str, prefix_tokens: int, n_sampling: int, num_problems: int) -> Path | None:
    for path in sorted(C2F_METRICS_DIR.glob("c2f_summary_*.csv")):
        if matches_summary(path, draft_model, refine_model, benchmark, prefix_tokens, n_sampling, num_problems):
            return path
    return None


def campaign_tag(pair: PairSpec, benchmark: str) -> str:
    return f"{CAMPAIGN_TAG}_{pair.draft_model}_to_{pair.refine_model}_{benchmark}_tokens"


def summary_path_for_tag(tag: str, prefix: int) -> Path:
    return C2F_METRICS_DIR / f"c2f_summary_{tag}{prefix}.csv"


def log_path(pair: PairSpec, benchmark: str, prefix: int) -> Path:
    return LOG_ROOT / f"{pair.name}_{benchmark}_tokens{prefix}.log"


def batch_size_for(pair: PairSpec, benchmark: str, token_cap: int) -> int:
    del benchmark
    if pair.tp_size != 1:
        return pair.batch_size
    if token_cap <= 1024:
        return 16
    if token_cap <= 2560:
        return 12
    return pair.batch_size


def should_enforce_eager(pair: PairSpec, args: argparse.Namespace) -> bool:
    return args.enforce_eager or pair.tp_size > 1


def wait_for_idle_experiments() -> None:
    while True:
        output = subprocess.run(
            ["pgrep", "-af", "math_transfer/run_candidate_to_final.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        active_lines = [line for line in output.stdout.splitlines() if line.strip()]
        if not active_lines:
            return
        print(f"[campaign] Waiting for {len(active_lines)} active C2F run(s) to finish...")
        time.sleep(30)


def launch_job(pair: PairSpec, benchmark: str, prefix: int, gpu_group: str, args: argparse.Namespace, limit: int) -> subprocess.Popen:
    tag = campaign_tag(pair, benchmark)
    token_cap = min(args.refine_max_tokens, BENCHMARK_TOKEN_CAPS.get(benchmark, args.refine_max_tokens))
    batch_size = batch_size_for(pair, benchmark, token_cap)
    eager = should_enforce_eager(pair, args)
    command = [
        sys.executable,
        str(RLVR_ROOT / "math_transfer" / "run_candidate_to_final.py"),
        "--draft-model", pair.draft_model,
        "--refine-model", pair.refine_model,
        "--benchmark", benchmark,
        "--draft-raw-path", str(merged_jsonl_path(pair.draft_model, benchmark)),
        "--refine-raw-path", str(merged_jsonl_path(pair.refine_model, benchmark)),
        "--prefix-mode", "tokens",
        "--prefix-tokens", str(prefix),
        "--n-sampling", str(args.n_sampling),
        "--sample-limit", str(limit),
        "--cuda-visible-devices", gpu_group,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--refine-max-tokens", str(token_cap),
        "--draft-max-tokens", str(min(args.draft_max_tokens, token_cap)),
        "--batch-size", str(batch_size),
        "--tag", tag,
    ]
    if eager:
        command.append("--enforce-eager")
    job_log = log_path(pair, benchmark, prefix)
    job_log.parent.mkdir(parents=True, exist_ok=True)
    handle = job_log.open("w")
    print(
        f"[campaign] Launch {pair.name} {benchmark} tokens{prefix} "
        f"on GPUs {gpu_group} with cap={token_cap} batch={batch_size} eager={eager}"
    )
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)


def wave_prefixes(prefixes: list[int], width: int) -> list[list[int]]:
    return [prefixes[index:index + width] for index in range(0, len(prefixes), width)]


def append_log_summary(pair: PairSpec, benchmark: str, prefixes: list[int], n_sampling: int, limit: int) -> None:
    marker = f"### C2F formal `{pair.draft_model}` → `{pair.refine_model}` on `{benchmark}`"
    log_path = LOG_SUMMARY_PATH
    if log_path.exists() and marker in log_path.read_text(encoding="utf-8"):
        print(f"[campaign] Log already contains {pair.name}/{benchmark}; skip duplicate append")
        return

    lines = ["", marker, ""]
    for prefix in prefixes:
        summary_path = existing_summary_path(pair.draft_model, pair.refine_model, benchmark, prefix, n_sampling, limit)
        if summary_path is None:
            raise FileNotFoundError(f"Missing summary for {pair.name}/{benchmark}/tokens{prefix}")
        rows = load_summary_rows(summary_path)
        c2f_row = next(row for row in rows if row["variant"] == "c2f")
        refine_row = next(row for row in rows if row["variant"] == "refine")
        draft_row = next(row for row in rows if row["variant"] == "draft")
        lines.append(
            f"- tokens{prefix}: draft pass@1={draft_row['pass@1']}, refine pass@1={refine_row['pass@1']}, "
            f"c2f pass@1={c2f_row['pass@1']}; draft pass@64={draft_row.get('pass@64', '')}, "
            f"refine pass@64={refine_row.get('pass@64', '')}, c2f pass@64={c2f_row.get('pass@64', '')}; "
            f"summary=`{summary_path}`"
        )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_pair_benchmark(pair: PairSpec, benchmark: str, prefixes: list[int], groups: list[str], args: argparse.Namespace) -> None:
    limit = effective_sample_limit(benchmark, args.sample_limit)
    pending = []
    for prefix in prefixes:
        summary_path = existing_summary_path(pair.draft_model, pair.refine_model, benchmark, prefix, args.n_sampling, limit)
        if summary_path is None:
            pending.append(prefix)
        else:
            print(f"[campaign] Skip existing {pair.name} {benchmark} tokens{prefix}: {summary_path}")

    for wave in wave_prefixes(pending, len(groups)):
        procs: list[tuple[int, subprocess.Popen]] = []
        assigned_groups = groups[:len(wave)]
        for prefix, gpu_group in zip(wave, assigned_groups, strict=True):
            proc = launch_job(pair, benchmark, prefix, gpu_group, args, limit)
            procs.append((prefix, proc))
        for prefix, proc in procs:
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"Job failed for {pair.name}/{benchmark}/tokens{prefix} with exit code {code}")
            expected = summary_path_for_tag(campaign_tag(pair, benchmark), prefix)
            if not expected.exists():
                raise FileNotFoundError(f"Expected summary missing: {expected}")

    append_log_summary(pair, benchmark, prefixes, args.n_sampling, limit)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "audit-leakage":
        run_leakage_audit(args)
        return
    if args.mode == "budget-recompute":
        run_budget_recompute(args)
        return
    if args.summarize or args.mode == "summarize":
        print(f"[campaign] Overview -> {summarize_campaign()}")
        return
    pairs = [pair_specs()[name] for name in parse_csv_arg(args.pairs)]
    benchmarks = parse_csv_arg(args.benchmarks)
    prefixes = parse_prefixes(args.prefixes)
    gpu_list = parse_csv_arg(args.gpus)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.wait_for_idle:
        wait_for_idle_experiments()

    for pair in pairs:
        groups = gpu_groups(gpu_list, pair.tp_size)
        for benchmark in benchmarks:
            run_pair_benchmark(pair, benchmark, prefixes, groups, args)


if __name__ == "__main__":
    main()
