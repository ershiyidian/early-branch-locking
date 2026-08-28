#!/usr/bin/env python3
"""Keep the six-benchmark olmo3_benchmark matrix moving without changing its protocol.

The first GSM8K and MATH500 processes were launched before this scheduler.
This process only observes them, finalizes completed raw shards, starts the
isolated official scorer, and assigns queued benchmarks to GPUs that have
actually been released. It never terminates or overwrites an active shard.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "math_transfer" / "olmo3_base_evaluation.py"
SCORER = ROOT / "math_transfer" / "olmo3_benchmark_score.py"
AGGREGATOR = ROOT / "math_transfer" / "olmo3_benchmark_aggregate.py"
TINYZERO_PYTHON = Path(sys.executable)
SCORER_PYTHON = Path(sys.executable)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "rlvr" / "outputs" / "experiments" / "olmo3_full_trajectory_v2"
LOG_ROOT = ROOT / "logs"
BENCHMARK_COUNTS = OrderedDict(
    (
        ("gsm8k", 500),
        ("math500", 500),
        ("minerva_math", 272),
        ("olympiadbench", 500),
        ("amc23", 40),
        ("aime24", 30),
    )
)
INITIAL_GPU = {"gsm8k": "1", "math500": "0"}
QUEUE = ["minerva_math", "olympiadbench", "amc23", "aime24"]
MEMORY_RELEASE_MIB = 2_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument(
        "--benchmark-root",
        action="append",
        default=[],
        metavar="BENCHMARK=PATH",
        help="Use PATH as the artifact root for this benchmark; unmapped queued work uses --output-root.",
    )
    parser.add_argument(
        "--benchmark-seed",
        action="append",
        default=[],
        metavar="BENCHMARK=SEED",
        help="Override the generation seed for a benchmark, preserving retry provenance on recovery.",
    )
    parser.add_argument("--once", action="store_true", help="Run one reconciliation pass and exit")
    return parser.parse_args(argv)


def parse_benchmark_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        benchmark, separator, root = value.partition("=")
        if not separator or benchmark not in BENCHMARK_COUNTS or not root:
            raise ValueError(f"Invalid --benchmark-root {value!r}; expected BENCHMARK=PATH")
        if benchmark in roots:
            raise ValueError(f"Duplicate --benchmark-root for {benchmark}")
        roots[benchmark] = Path(root).resolve()
    return roots


def parse_benchmark_seeds(values: list[str]) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for value in values:
        benchmark, separator, seed_text = value.partition("=")
        if not separator or benchmark not in BENCHMARK_COUNTS or not seed_text:
            raise ValueError(f"Invalid --benchmark-seed {value!r}; expected BENCHMARK=SEED")
        if benchmark in seeds:
            raise ValueError(f"Duplicate --benchmark-seed for {benchmark}")
        try:
            seeds[benchmark] = int(seed_text)
        except ValueError as error:
            raise ValueError(f"Invalid seed in --benchmark-seed {value!r}") from error
    return seeds


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (LOG_ROOT / "olmo3_orchestrator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def raw_path(benchmark: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / benchmark / f"records_s0_e{BENCHMARK_COUNTS[benchmark]}.jsonl"


def partial_path(benchmark: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / benchmark / f"records_s0_e{BENCHMARK_COUNTS[benchmark]}.jsonl.partial"


def score_dir(benchmark: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / benchmark / "score"


def score_metrics_path(benchmark: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return score_dir(benchmark, output_root) / f"{benchmark}_official_metrics.json"


def score_process_path(benchmark: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return score_dir(benchmark, output_root) / f"{benchmark}_scored.jsonl"


def runner_option(command: str, option: str) -> str | None:
    """Read one option from a runner command shown by ``ps``."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for index, token in enumerate(tokens[:-1]):
        if token == option:
            return tokens[index + 1]
    return None


def normalize_runner_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def runner_processes(benchmark: str, output_root: Path | None = None) -> list[str]:
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=False)
    needle = f"olmo3_base_evaluation.py --mode generate --benchmark {benchmark}"
    expected_root = output_root.resolve() if output_root is not None else None
    matches = []
    for line in result.stdout.splitlines():
        if needle not in line:
            continue
        if expected_root is not None:
            command_root = runner_option(line, "--output-root")
            if command_root is None or normalize_runner_path(command_root) != expected_root:
                continue
        matches.append(line)
    return matches


def assert_no_cross_root_runner(benchmark: str, output_root: Path) -> None:
    """Reject an existing same-benchmark runner pointed at another root."""
    all_processes = runner_processes(benchmark)
    expected_root = output_root.resolve()
    mismatches = [
        command
        for command in all_processes
        if runner_option(command, "--output-root") is None
        or normalize_runner_path(runner_option(command, "--output-root")) != expected_root
    ]
    if mismatches:
        roots = sorted(
            {
                runner_option(command, "--output-root") or "<default/unknown>"
                for command in mismatches
            }
        )
        raise RuntimeError(
            f"existing runner root mismatch benchmark={benchmark} expected={expected_root} "
            f"observed={roots}"
        )


def runner_process_present(benchmark: str, output_root: Path | None = None) -> bool:
    return bool(runner_processes(benchmark, output_root))


def runner_gpu(command: str) -> str | None:
    return runner_option(command, "--cuda-visible-devices")


def shard_state(benchmark: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> str:
    raw = raw_path(benchmark, output_root)
    partial = partial_path(benchmark, output_root)
    if raw.is_file() and partial.is_file():
        return "inconsistent"
    if raw.is_file():
        return "complete"
    if runner_process_present(benchmark, output_root):
        return "running"
    if partial.is_file():
        return "partial"
    return "missing"


def gpu_memory_used_mib(gpu: str) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def gpu_released(gpu: str) -> bool:
    used = gpu_memory_used_mib(gpu)
    return used is not None and used < MEMORY_RELEASE_MIB


def run_checked(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        log(result.stdout.strip())
    if result.stderr.strip():
        log(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def finalize(benchmark: str, output_root: Path) -> None:
    log(f"finalize_start benchmark={benchmark}")
    run_checked(
        [
            str(TINYZERO_PYTHON),
            str(RUNNER),
            "--mode",
            "finalize",
            "--benchmark",
            benchmark,
            "--raw",
            str(raw_path(benchmark, output_root)),
            "--n-sampling",
            "64",
        ]
    )
    log(f"finalize_complete benchmark={benchmark}")


def start_scorer(benchmark: str, output_root: Path) -> subprocess.Popen:
    output_dir = score_dir(benchmark, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "scorer.log"
    command = [
        str(SCORER_PYTHON),
        str(SCORER),
        "--raw",
        str(raw_path(benchmark, output_root)),
        "--benchmark",
        benchmark,
        "--output-dir",
        str(output_dir),
        "--workers",
        "16",
        "--overwrite",
    ]
    stream = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    log(f"score_started benchmark={benchmark} pid={process.pid}")
    return process


def start_generation(
    benchmark: str,
    gpu: str,
    output_root: Path,
    min_new_tokens: int,
    seed: int,
) -> subprocess.Popen:
    log_path = LOG_ROOT / f"olmo3_full_{benchmark}_gpu{gpu}_{output_root.name}.log"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        str(TINYZERO_PYTHON),
        str(RUNNER),
        "--mode",
        "generate",
        "--benchmark",
        benchmark,
        "--start",
        "0",
        "--end",
        str(BENCHMARK_COUNTS[benchmark]),
        "--n-sampling",
        "64",
        "--batch-size",
        "4",
        "--max-new-tokens",
        "16000",
        "--min-new-tokens",
        str(min_new_tokens),
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
        "--cuda-visible-devices",
        gpu,
    ]
    stream = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    log(f"generation_started benchmark={benchmark} gpu={gpu} pid={process.pid}")
    return process


def aggregate(output_root: Path, benchmark_roots: dict[str, Path]) -> None:
    log("aggregate_start")
    command = [str(TINYZERO_PYTHON), str(AGGREGATOR), "--output-root", str(output_root)]
    for benchmark, root in sorted(benchmark_roots.items()):
        command.extend(["--benchmark-root", f"{benchmark}={root}"])
    run_checked(command)
    log("aggregate_complete")


def restore_scheduler_state(
    artifact_root,
) -> tuple[dict[str, str | None], list[str]]:
    """Adopt runners that survived a supervisor restart before scheduling new work."""
    assignments: dict[str, str | None] = {"0": None, "1": None}
    queue: list[str] = []

    for benchmark, expected_gpu in INITIAL_GPU.items():
        root = artifact_root(benchmark)
        assert_no_cross_root_runner(benchmark, root)
        processes = runner_processes(benchmark, root)
        if not processes:
            assignments[expected_gpu] = benchmark
            continue
        observed_gpu = runner_gpu(processes[0])
        if observed_gpu is not None and observed_gpu != expected_gpu:
            raise RuntimeError(
                f"runner GPU mismatch benchmark={benchmark} expected={expected_gpu} observed={observed_gpu}"
            )
        assignments[expected_gpu] = benchmark

    for benchmark in QUEUE:
        root = artifact_root(benchmark)
        assert_no_cross_root_runner(benchmark, root)
        if shard_state(benchmark, root) == "complete":
            continue
        processes = runner_processes(benchmark, root)
        if not processes:
            queue.append(benchmark)
            continue
        observed_gpu = runner_gpu(processes[0])
        if observed_gpu not in assignments:
            raise RuntimeError(f"queued runner has unknown GPU benchmark={benchmark} gpu={observed_gpu}")
        if assignments[observed_gpu] is not None:
            raise RuntimeError(
                f"GPU has conflicting olmo3_benchmark runners gpu={observed_gpu} "
                f"existing={assignments[observed_gpu]} queued={benchmark}"
            )
        assignments[observed_gpu] = benchmark
        log(f"adopted_existing_generation benchmark={benchmark} gpu={observed_gpu} root={root}")
    return assignments, queue


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    if not TINYZERO_PYTHON.is_file() or not SCORER_PYTHON.is_file():
        raise FileNotFoundError("Configured runtime interpreter is missing")

    score_processes: dict[str, subprocess.Popen] = {}
    score_attempts: dict[str, int] = {}
    finalized: set[str] = set()
    aggregated = False

    output_root = args.output_root.resolve()
    benchmark_roots = parse_benchmark_roots(args.benchmark_root)
    benchmark_seeds = parse_benchmark_seeds(args.benchmark_seed)
    artifact_root = lambda benchmark: benchmark_roots.get(benchmark, output_root)
    generation_seed = lambda benchmark: benchmark_seeds.get(benchmark, 1729)
    gpu_assignments, queue = restore_scheduler_state(artifact_root)
    log(
        "scheduler_started "
        f"protocol=olmo3_benchmark-olmo3-base-full-trajectory output_root={output_root} "
        f"min_new_tokens={args.min_new_tokens}"
    )
    while True:
        for benchmark in BENCHMARK_COUNTS:
            root = artifact_root(benchmark)
            state = shard_state(benchmark, root)
            if state == "inconsistent":
                log(f"ERROR inconsistent_shard benchmark={benchmark}")
                continue
            if state == "complete" and benchmark not in finalized:
                try:
                    finalize(benchmark, root)
                    finalized.add(benchmark)
                except Exception as error:
                    log(f"ERROR finalize_failed benchmark={benchmark} error={error}")
                    continue
            if state == "complete" and benchmark not in score_processes:
                if score_metrics_path(benchmark, root).is_file() and score_process_path(benchmark, root).is_file():
                    continue
                if score_attempts.get(benchmark, 0) >= 3:
                    log(f"ERROR score_attempt_limit benchmark={benchmark}")
                    continue
                score_attempts[benchmark] = score_attempts.get(benchmark, 0) + 1
                score_processes[benchmark] = start_scorer(benchmark, root)

        for benchmark, process in list(score_processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del score_processes[benchmark]
            if return_code == 0:
                log(f"score_complete benchmark={benchmark}")
            else:
                log(f"ERROR score_failed benchmark={benchmark} returncode={return_code}")

        if args.once:
            return 0

        for gpu, benchmark in list(gpu_assignments.items()):
            if benchmark is not None:
                root = artifact_root(benchmark)
                state = shard_state(benchmark, root)
                if state == "complete":
                    gpu_assignments[gpu] = None
                elif not runner_process_present(benchmark, root) and state in {"partial", "missing"}:
                    if gpu_released(gpu):
                        start_generation(benchmark, gpu, root, args.min_new_tokens, generation_seed(benchmark))
                    continue
                else:
                    continue
            if gpu_assignments[gpu] is None and queue and gpu_released(gpu):
                next_benchmark = queue[0]
                next_root = artifact_root(next_benchmark)
                assert_no_cross_root_runner(next_benchmark, next_root)
                existing = runner_processes(next_benchmark, next_root)
                if existing:
                    observed_gpu = runner_gpu(existing[0])
                    raise RuntimeError(
                        f"unadopted existing runner benchmark={next_benchmark} "
                        f"gpu={observed_gpu} root={next_root}"
                    )
                queue.pop(0)
                gpu_assignments[gpu] = next_benchmark
                start_generation(
                    next_benchmark,
                    gpu,
                    next_root,
                    args.min_new_tokens,
                    generation_seed(next_benchmark),
                )

        all_scored = all(
            shard_state(benchmark, artifact_root(benchmark)) == "complete"
            and score_metrics_path(benchmark, artifact_root(benchmark)).is_file()
            and score_process_path(benchmark, artifact_root(benchmark)).is_file()
            for benchmark in BENCHMARK_COUNTS
        )
        if all_scored and not score_processes and not aggregated:
            try:
                aggregate(output_root, benchmark_roots)
                aggregated = True
            except Exception as error:
                log(f"ERROR aggregate_failed error={error}")
            if aggregated:
                return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
