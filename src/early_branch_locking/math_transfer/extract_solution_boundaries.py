"""A2 frontier-model semantic audit of the open-math first-calculation boundary.

The sample packet reuses the existing branch_decomposition boundary-audit packet but emits a new
blind representation. API responses are append-only and are resolved locally
from unit ids plus verbatim spans. The API mode is intentionally separate from
the historical local proxy audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.core.api_judge import (  # noqa: E402
    MissingAPICredential,
    call_boundary_judge,
    load_api_config,
    resolve_boundary_span,
)
from early_branch_locking.core.branch_protocol import (  # noqa: E402
    bootstrap_mean,
    completion_list,
    milestone_record,
)


BENCHMARKS = ("gsm8k", "math500", "minerva_math", "olympiadbench", "amc23", "aime24")
BASE_MODELS = {"math_base_7b", "math_base_14b", "math_olmo3_sft_7b", "math_olmo3_dpo_7b", "math_base_qwen_math_7b"}
RL_MODELS = {"math_simple_rl_7b", "math_simple_rl_14b", "math_olmo3_rlvr_7b", "math_oat_zero_7b"}
DEFAULT_OUTPUT = ROOT / "data" / "rlvr" / "outputs" / "experiments" / "e1_boundary_audit_api_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sample", "judge-primary", "reresolve", "replace-unresolved", "summarize"),
        default="sample",
    )
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "rlvr" / "outputs" / "e1" / "labels.jsonl")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--per-cell", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--include-packet",
        action="append",
        type=Path,
        default=[],
        help="prepend blind packet rows and sample only new item ids",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--allow-missing-credentials", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="retain successful rows and retry previously failed item ids")
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="also retry successful API rows whose local resolver did not resolve an exact span",
    )
    return parser.parse_args(argv)


def origin_of(model: str) -> str:
    if model in BASE_MODELS:
        return "base_origin"
    if model in RL_MODELS:
        return "rl_origin"
    return "other"


def _load_labels(path: Path, benchmarks: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("benchmark")) not in benchmarks:
                continue
            origin = origin_of(str(row.get("model", "")))
            if origin == "other" or row.get("p1_char_end") is None:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError("no eligible branch_decomposition labels for A2 packet")
    return rows


def _load_source_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    needed: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        needed[str(row["source_file"])].add(int(row["source_row"]))
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for relative, indices in needed.items():
        source = ROOT / relative
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in indices and line.strip():
                    output[(relative, index)] = json.loads(line)
    return output


def _units(text: str) -> list[dict[str, Any]]:
    """Split conservatively, preserving exact character offsets."""

    text = str(text or "")
    units: list[dict[str, Any]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_start = offset
        line_body = line.rstrip("\r\n")
        offset += len(line)
        if not line_body.strip():
            continue
        cursor = 0
        # Keep displayed equations intact; split prose only after sentence or
        # semicolon delimiters that are outside simple math delimiters.
        boundaries: list[int] = []
        depth = 0
        for index, char in enumerate(line_body):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            elif char in ".;!?。！？" and depth == 0:
                if index + 1 == len(line_body) or line_body[index + 1].isspace():
                    boundaries.append(index + 1)
        for end in boundaries + [len(line_body)]:
            segment = line_body[cursor:end]
            if segment.strip():
                left = line_start + cursor
                right = line_start + end
                units.append({"unit_id": f"U{len(units):03d}", "char_start": left, "char_end": right, "text": text[left:right]})
            cursor = end
    if not units and text:
        units.append({"unit_id": "U000", "char_start": 0, "char_end": len(text), "text": text})
    return units


def _item_id(row: dict[str, Any], sample_index: int) -> str:
    return hashlib.sha1(f"{row['model']}|{row['problem_id']}|{sample_index}".encode("utf-8")).hexdigest()[:16]


def sample(args: argparse.Namespace) -> Path:
    benchmarks = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    labels = _load_labels(args.labels, benchmarks)
    rng = np.random.default_rng(args.seed)
    included: list[dict[str, Any]] = []
    included_ids: set[str] = set()
    for packet_path in args.include_packet:
        if not packet_path.is_file():
            raise FileNotFoundError(packet_path)
        for line in packet_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("item_id", ""))
            if not item_id or item_id in included_ids:
                raise ValueError(f"duplicate or missing item id in --include-packet: {packet_path}")
            if not isinstance(item.get("units"), list):
                raise ValueError(f"included packet row has no units: {item_id}")
            included.append(item)
            included_ids.add(item_id)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        if _item_id(row, int(row["sample_index"])) in included_ids:
            continue
        cells[(str(row["benchmark"]), origin_of(str(row["model"])))].append(row)
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, pool in cells.items():
        indices = rng.permutation(len(pool))[: min(args.per_cell, len(pool))]
        candidates[key] = [pool[int(index)] for index in indices]
    selected: list[dict[str, Any]] = []
    if args.max_items > 0:
        positions = {key: 0 for key in candidates}
        while len(selected) < args.max_items:
            advanced = False
            for key in sorted(candidates):
                position = positions[key]
                if position >= len(candidates[key]):
                    continue
                selected.append(candidates[key][position])
                positions[key] = position + 1
                advanced = True
                if len(selected) == args.max_items:
                    break
            if not advanced:
                break
    else:
        selected = [row for key in sorted(candidates) for row in candidates[key]]
    source_rows = _load_source_rows(selected)
    args.output_root.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_root / "packet_blind.jsonl"
    units_path = args.output_root / "units.jsonl"
    if (packet_path.exists() or units_path.exists()) and not args.overwrite:
        raise FileExistsError(f"A2 packet exists below {args.output_root}; pass --overwrite")
    with packet_path.open("w", encoding="utf-8") as packet_handle, units_path.open("w", encoding="utf-8") as units_handle:
        for item in included:
            for unit in item["units"]:
                units_handle.write(json.dumps({"item_id": item["item_id"], **unit}, ensure_ascii=False, sort_keys=True) + "\n")
            packet_handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
        for row in selected:
            raw = source_rows[(str(row["source_file"]), int(row["source_row"]))]
            completions = completion_list(raw)
            sample_index = int(row["sample_index"])
            response = completions[sample_index] if sample_index < len(completions) else ""
            item_id = _item_id(row, sample_index)
            units = _units(response)
            for unit in units:
                units_handle.write(json.dumps({"item_id": item_id, **unit}, ensure_ascii=False, sort_keys=True) + "\n")
            packet_handle.write(
                json.dumps(
                    {
                        "item_id": item_id,
                        "question": str(raw.get("question", raw.get("problem", ""))),
                        "response": response,
                        "units": units,
                        "blind_fields_removed": ["model", "origin", "rule_p1_char_end", "source_file", "source_row", "heuristic_boundaries"],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
    manifest = args.output_root / "manifest.json"
    packet_rows = sum(1 for _ in packet_path.open(encoding="utf-8"))
    unit_rows = sum(1 for _ in units_path.open(encoding="utf-8"))
    item_cells = {
        _item_id(row, int(row["sample_index"])): (str(row["benchmark"]), origin_of(str(row["model"])))
        for row in labels
    }
    packet_cell_counts = Counter(
        item_cells[str(item["item_id"])]
        for item in included
        if str(item["item_id"]) in item_cells
    )
    packet_cell_counts.update(
        (str(row["benchmark"]), origin_of(str(row["model"]))) for row in selected
    )
    manifest.write_text(
        json.dumps(
            {
                "experiment_id": "A2",
                "protocol_version": "A2-api-v3-env",
                "packet_path": str(packet_path),
                "units_path": str(units_path),
                "n_items": len(included) + len(selected),
                "row_counts": {"packet_blind": packet_rows, "units": unit_rows},
                "input_hashes": {
                    "labels": _sha256(args.labels),
                    "packet_blind": _sha256(packet_path),
                    "units": _sha256(units_path),
                },
                "cells": {
                    f"{benchmark}|{origin}": int(packet_cell_counts[(benchmark, origin)])
                    for benchmark, origin in sorted(cells)
                },
                "included_packet_paths": [str(path) for path in args.include_packet],
                "included_item_count": len(included),
                "new_item_count": len(selected),
                "seed": args.seed,
                "per_cell": args.per_cell,
                "blind_fields": ["model", "origin", "rule_p1_char_end", "source_file", "source_row", "heuristic_boundaries"],
                "judge_status": "frontier-model semantic audit; not human validation",
                "git_commit": _git_commit(),
                "command": shlex.join(sys.argv),
                "api_config_source": ".env",
                "api_backend_mode": "pending",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"packet": str(packet_path), "units": str(units_path), "items": len(included) + len(selected)}, sort_keys=True))
    return packet_path


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")


def _judge_item(item: dict[str, Any], config: Any, args: argparse.Namespace) -> dict[str, Any]:
    try:
        result = call_boundary_judge(
            item["question"],
            item["units"],
            config,
            max_attempts=args.max_attempts,
            timeout=args.request_timeout,
        )
    except Exception as exc:  # Keep one unexpected item failure resumable.
        result = {"annotation": None, "response_id": None, "usage": None, "api_backend_mode": "failed", "status": "parse_failed", "error_type": type(exc).__name__}
    annotation = result.get("annotation")
    resolution = resolve_boundary_span(item["response"], item["units"], annotation or {
        "has_complete_numeric_calculation": False,
        "unit_id": None,
        "span_text": None,
        "boundary_kind": "none",
        "confidence": "low",
        "reason_short": "API request failed",
    })
    return {
        "item_id": item["item_id"],
        "annotation": annotation,
        **resolution.as_dict(),
        "api_model": config.model,
        "api_base_host": config.base_host,
        "api_backend_mode": result.get("api_backend_mode"),
        "response_id": result.get("response_id"),
        "usage": result.get("usage"),
        "status": result.get("status", "ok" if annotation is not None else "parse_failed"),
        "error_type": result.get("error_type"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def judge_primary(args: argparse.Namespace) -> Path:
    packet_path = args.output_root / "packet_blind.jsonl"
    if not packet_path.exists():
        raise FileNotFoundError(packet_path)
    config = load_api_config(repo_root=ROOT, model_override=args.api_model, base_url_override=args.api_base_url, require_credential=not args.allow_missing_credentials)
    output = args.output_root / "judge_primary.jsonl"
    if output.exists() and not args.overwrite:
        existing_rows = [json.loads(line) for line in output.open(encoding="utf-8") if line.strip()]
        completed = {row.get("item_id") for row in existing_rows if row.get("status") == "ok" and (not args.retry_unresolved or row.get("resolver_status") == "resolved")}
        if args.retry_failed or args.retry_unresolved:
            retained = []
            retained_ids = set()
            for row in existing_rows:
                item_id = row.get("item_id")
                terminal = row.get("status") == "ok" and (not args.retry_unresolved or row.get("resolver_status") == "resolved")
                if terminal and item_id not in retained_ids:
                    retained.append(row)
                    retained_ids.add(item_id)
            if len(retained) != len(existing_rows):
                temporary = output.with_name(output.name + ".retry.tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    for row in retained:
                        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
                os.replace(temporary, output)
    else:
        completed = set()
        output.unlink(missing_ok=True)
    items = [json.loads(line) for line in packet_path.open(encoding="utf-8") if line.strip()]
    pending = [(index, item) for index, item in enumerate(items, start=1) if item["item_id"] not in completed]
    errors = 0
    finished = len(items) - len(pending)

    def record(index: int, row: dict[str, Any]) -> None:
        nonlocal errors, finished
        errors += int(row["status"] != "ok")
        finished += 1
        _append_jsonl(output, row)
        if finished % 10 == 0 or finished == len(items):
            print(f"[A2] judged {finished}/{len(items)} failed={errors}", flush=True)

    if args.concurrency <= 1:
        for index, item in pending:
            record(index, _judge_item(item, config, args))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="a2-judge") as executor:
            futures = {executor.submit(_judge_item, item, config, args): index for index, item in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())
    manifest = json.loads((args.output_root / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"api_model": config.model, "api_reasoning_effort": config.reasoning_effort, "api_base_host": config.base_host, "api_backend_mode": "fallback-capable", "api_config_source": ".env", "concurrency_requested": args.concurrency, "retry_failed": args.retry_failed, "retry_unresolved": args.retry_unresolved, "request_timeout_seconds": args.request_timeout, "judge_output": str(output), "failed_requests": errors, "git_commit": _git_commit(), "command": shlex.join(sys.argv), "judge_finished_at_utc": datetime.now(timezone.utc).isoformat()})
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reresolve(args: argparse.Namespace) -> Path:
    """Reapply the local resolver to stored API annotations without an API call."""

    packet_path = args.output_root / "packet_blind.jsonl"
    output = args.output_root / "judge_primary.jsonl"
    if not packet_path.exists() or not output.exists():
        raise FileNotFoundError("reresolve requires packet_blind.jsonl and judge_primary.jsonl")
    items = {str(item["item_id"]): item for item in (json.loads(line) for line in packet_path.open(encoding="utf-8") if line.strip())}
    rows = [json.loads(line) for line in output.open(encoding="utf-8") if line.strip()]
    temporary = output.with_name(output.name + ".reresolve.tmp")
    changed = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = items.get(str(row.get("item_id")))
            if item is not None and row.get("status") == "ok" and row.get("annotation") is not None:
                resolved = resolve_boundary_span(item["response"], item["units"], row["annotation"])
                before = row.get("resolver_status")
                row.update(resolved.as_dict())
                changed += int(before != row.get("resolver_status"))
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"rows": len(rows), "resolver_status_changed": changed, "output": str(output)}, sort_keys=True))
    return output


def replace_unresolved(args: argparse.Namespace) -> Path:
    """Replace only audit units with no locally resolvable API boundary.

    Replacements are sampled within the same benchmark/origin cell, exclude
    every current packet id, and must have an independently detected local P1
    calculation anchor.  The manifest records the old/new ids explicitly.
    """

    packet_path = args.output_root / "packet_blind.jsonl"
    units_path = args.output_root / "units.jsonl"
    judge_path = args.output_root / "judge_primary.jsonl"
    manifest_path = args.output_root / "manifest.json"
    if not packet_path.exists() or not judge_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("replace-unresolved requires packet, judge output, and manifest")
    packet = [json.loads(line) for line in packet_path.open(encoding="utf-8") if line.strip()]
    judged = [json.loads(line) for line in judge_path.open(encoding="utf-8") if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest = {str(row["item_id"]): row for row in judged}
    unresolved_ids = [
        str(item["item_id"])
        for item in packet
        if latest.get(str(item["item_id"]), {}).get("resolver_status") != "resolved"
    ]
    if not unresolved_ids:
        print(json.dumps({"replacements": 0, "packet": str(packet_path)}, sort_keys=True))
        return packet_path

    benchmarks = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    labels = _load_labels(args.labels, benchmarks)
    label_by_id = {_item_id(row, int(row["sample_index"])): row for row in labels}
    packet_ids = {str(item["item_id"]) for item in packet}
    historical_replacement_ids = {
        str(entry[key])
        for entry in manifest.get("unresolvable_unit_replacements", [])
        for key in ("old_item_id", "new_item_id")
        if entry.get(key)
    }
    excluded_ids = packet_ids | historical_replacement_ids
    unresolved_by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item_id in unresolved_ids:
        row = label_by_id.get(item_id)
        if row is None:
            raise RuntimeError(f"unresolved packet item is absent from labels: {item_id}")
        unresolved_by_cell[(str(row["benchmark"]), origin_of(str(row["model"])))].append(item_id)

    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        item_id = _item_id(row, int(row["sample_index"]))
        if item_id not in excluded_ids:
            pools[(str(row["benchmark"]), origin_of(str(row["model"])))].append(row)
    rng = np.random.default_rng(args.seed + 10_000)
    selected: dict[str, dict[str, Any]] = {}
    for cell, old_ids in sorted(unresolved_by_cell.items()):
        candidates = pools[cell]
        order = rng.permutation(len(candidates))
        chosen: list[dict[str, Any]] = []
        for index in order:
            candidate = candidates[int(index)]
            if candidate.get("p1_char_end") is not None:
                chosen.append(candidate)
            if len(chosen) == len(old_ids):
                break
        if len(chosen) != len(old_ids):
            raise RuntimeError(f"insufficient anchored replacements for cell={cell}: need={len(old_ids)} got={len(chosen)}")
        selected.update(dict(zip(old_ids, chosen)))

    source_rows = _load_source_rows(list(selected.values()))
    replacements: list[dict[str, Any]] = []
    replacement_items: dict[str, dict[str, Any]] = {}
    for old_id, row in selected.items():
        raw = source_rows[(str(row["source_file"]), int(row["source_row"]))]
        sample_index = int(row["sample_index"])
        response = completion_list(raw)[sample_index]
        marks = milestone_record(response, str(raw.get("gt", raw.get("ground_truth", ""))))
        if marks.get("p1_char_end") is None:
            raise RuntimeError(f"replacement lacks a local P1 anchor: {old_id}")
        new_id = _item_id(row, sample_index)
        units = _units(response)
        replacement_items[old_id] = {
            "item_id": new_id,
            "question": str(raw.get("question", raw.get("problem", ""))),
            "response": response,
            "units": units,
            "blind_fields_removed": ["model", "origin", "rule_p1_char_end", "source_file", "source_row", "heuristic_boundaries"],
        }
        replacements.append({
            "old_item_id": old_id,
            "new_item_id": new_id,
            "benchmark": row["benchmark"],
            "origin": origin_of(str(row["model"])),
            "local_p1_char_end": int(marks["p1_char_end"]),
        })

    updated_packet = [replacement_items.get(str(item["item_id"]), item) for item in packet]
    if len({str(item["item_id"]) for item in updated_packet}) != len(updated_packet):
        raise RuntimeError("replacement created duplicate item ids")
    packet_temp = packet_path.with_name(packet_path.name + ".replace.tmp")
    units_temp = units_path.with_name(units_path.name + ".replace.tmp")
    judge_temp = judge_path.with_name(judge_path.name + ".replace.tmp")
    with packet_temp.open("w", encoding="utf-8") as handle:
        for item in updated_packet:
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    with units_temp.open("w", encoding="utf-8") as handle:
        for item in updated_packet:
            for unit in item["units"]:
                handle.write(json.dumps({"item_id": item["item_id"], **unit}, ensure_ascii=False, sort_keys=True) + "\n")
    replacement_old = set(selected)
    with judge_temp.open("w", encoding="utf-8") as handle:
        for row in judged:
            if str(row["item_id"]) not in replacement_old:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
    os.replace(packet_temp, packet_path)
    os.replace(units_temp, units_path)
    os.replace(judge_temp, judge_path)
    manifest["unresolvable_unit_replacements"] = list(manifest.get("unresolvable_unit_replacements", [])) + replacements
    manifest["replacement_seed"] = args.seed + 10_000
    manifest["n_items"] = len(updated_packet)
    manifest["row_counts"] = {"packet_blind": len(updated_packet), "units": sum(len(item["units"]) for item in updated_packet)}
    manifest["input_hashes"].update({"packet_blind": _sha256(packet_path), "units": _sha256(units_path)})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"replacements": len(replacements), "remaining_judged": len(updated_packet) - len(replacements), "packet": str(packet_path)}, sort_keys=True))
    return packet_path


def _load_source_metadata(args: argparse.Namespace, item_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
    benchmarks = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    labels = _load_labels(args.labels, benchmarks)
    if item_ids is not None:
        labels = [row for row in labels if _item_id(row, int(row["sample_index"])) in item_ids]
    source_rows = _load_source_rows(labels)
    output: dict[str, dict[str, Any]] = {}
    for row in labels:
        sample_index = int(row["sample_index"])
        item_id = _item_id(row, sample_index)
        raw = source_rows[(str(row["source_file"]), int(row["source_row"]))]
        completions = completion_list(raw)
        response = completions[sample_index] if sample_index < len(completions) else ""
        marks = milestone_record(response, str(raw.get("gt", raw.get("ground_truth", ""))))
        rule_end = marks.get("p1_char_end")
        rule_unit_id = None
        if rule_end is not None:
            for unit in _units(response):
                if int(unit["char_start"]) < int(rule_end) <= int(unit["char_end"]):
                    rule_unit_id = str(unit["unit_id"])
                    break
        output[item_id] = {
            "item_id": item_id,
            "benchmark": row["benchmark"],
            "origin": origin_of(str(row["model"])),
            "problem_id": row["problem_id"],
            "rule_p1_char_end": rule_end,
            "rule_unit_id": rule_unit_id,
            "response": response,
        }
    return output


def summarize(args: argparse.Namespace) -> Path:
    output = args.output_root / "judge_primary.jsonl"
    if not output.exists():
        raise FileNotFoundError(output)
    judged_rows = [json.loads(line) for line in output.open(encoding="utf-8") if line.strip()]
    metadata = _load_source_metadata(args, {str(row["item_id"]) for row in judged_rows})
    rows: list[dict[str, Any]] = []
    for judged in judged_rows:
        meta = metadata.get(judged["item_id"])
        if meta is None:
            continue
        rule = meta.get("rule_p1_char_end")
        api_end = judged.get("semantic_char_end")
        api_unit_id = judged.get("resolved_unit_id")
        resolver_ok = judged.get("resolver_status") == "resolved"
        rows.append({
            "item_id": judged["item_id"],
            "benchmark": meta["benchmark"],
            "origin": meta["origin"],
            "problem_id": meta["problem_id"],
            "rule_p1_char_end": rule,
            "rule_unit_id": meta.get("rule_unit_id"),
            "api_unit_id": api_unit_id,
            "api_char_end": api_end,
            "resolver_ok": int(resolver_ok),
            "same_unit": int(resolver_ok and meta.get("rule_unit_id") is not None and api_unit_id == meta.get("rule_unit_id")),
            "same_semantic_span": int(resolver_ok and meta.get("rule_unit_id") is not None and api_unit_id == meta.get("rule_unit_id")),
            "exact_end_match": int(rule is not None and api_end is not None and int(rule) == int(api_end)),
            "abs_char_delta": abs(int(api_end) - int(rule)) if rule is not None and api_end is not None else np.nan,
            "resolver_status": judged.get("resolver_status"),
            "resolver_ambiguous": judged.get("resolver_ambiguous", False),
        })
    frame = pd.DataFrame(rows)
    resolver_path = args.output_root / "resolver.csv"
    frame.to_csv(resolver_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    for benchmark, group in frame.groupby("benchmark", sort=True):
        for origin in ("base_origin", "rl_origin"):
            sub = group[group["origin"] == origin]
            values = sub["same_unit"].to_numpy(float)
            mean, lo, hi = bootstrap_mean(values.tolist(), seed=args.seed + len(summary_rows), draws=args.bootstrap_draws) if len(values) else (float("nan"),) * 3
            summary_rows.append({"benchmark": benchmark, "origin": origin, "n": len(sub), "same_unit": mean, "same_unit_ci_lo": lo, "same_unit_ci_hi": hi, "resolver_rate": float(sub["resolver_ok"].mean()) if len(sub) else np.nan, "exact_end_match": float(sub["exact_end_match"].mean()) if len(sub) else np.nan, "median_abs_char_delta": float(sub["abs_char_delta"].median()) if sub["abs_char_delta"].notna().any() else np.nan, "resolver_failures": int((sub["resolver_status"] != "resolved").sum()), "ambiguous_resolutions": int(sub["resolver_ambiguous"].astype(bool).sum())})
        base = summary_rows[-2]["same_unit"]
        rl = summary_rows[-1]["same_unit"]
        summary_rows.append({"benchmark": benchmark, "origin": "base_minus_rl", "n": int(summary_rows[-2]["n"] + summary_rows[-1]["n"]), "same_unit": base - rl if np.isfinite(base) and np.isfinite(rl) else np.nan, "same_unit_ci_lo": np.nan, "same_unit_ci_hi": np.nan, "resolver_rate": np.nan, "exact_end_match": np.nan, "median_abs_char_delta": np.nan, "resolver_failures": np.nan, "ambiguous_resolutions": np.nan})
    summary_path = args.output_root / "agreement_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    manifest = json.loads((args.output_root / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "resolver_path": str(resolver_path),
        "agreement_summary_path": str(summary_path),
        "bootstrap_draws": args.bootstrap_draws,
        "statistical_unit": "trajectory; origin differential is primary",
        "judge_status_counts": dict(Counter(str(row.get("status")) for row in judged_rows)),
        "resolver_status_counts": dict(Counter(str(value) for value in frame["resolver_status"].tolist())),
        "api_exact_resolved_rows": int(frame["resolver_ok"].sum()),
        "summarized_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "sample":
        sample(args)
    elif args.mode == "judge-primary":
        judge_primary(args)
    elif args.mode == "reresolve":
        reresolve(args)
    elif args.mode == "replace-unresolved":
        replace_unresolved(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
