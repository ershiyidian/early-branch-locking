"""Credential-safe structured API judging and exact boundary resolution.

The API judge is deliberately blind to model origin and heuristic boundaries.
It returns a copied response span and a unit id; all character offsets are
resolved locally against the original response.
"""

from __future__ import annotations

import json
import re
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlparse


BOUNDARY_JUDGE_SYSTEM = (
    "You are annotating the first numeric calculation boundary in a math "
    "response. Locate the earliest span that evaluates an instantiated "
    "numeric relation and states its computed result. Equations need not be "
    "mathematically correct. A plan such as 'try 12 + 7' does not count. "
    "Return only the requested structured annotation. Copy span_text "
    "verbatim from one displayed unit; never invent or normalize text and "
    "never output character offsets. Keep reason_short to one short sentence."
)


class MissingAPICredential(RuntimeError):
    """Raised before any request when no supported credential is configured."""


@dataclass(frozen=True)
class APIConfig:
    api_key: str
    model: str
    base_url: str | None
    api_key_name: str
    reasoning_effort: str | None = None

    @property
    def base_host(self) -> str | None:
        if not self.base_url:
            return None
        parsed = urlparse(self.base_url)
        return parsed.netloc or parsed.path.split("/", 1)[0] or None


def _first_nonempty(names: Sequence[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def _split_model_setting(model: str | None, reasoning_effort: str | None) -> tuple[str | None, str | None]:
    """Accept either a plain model id or the project's `model effort` form."""

    if not model:
        return model, reasoning_effort
    parts = str(model).strip().rsplit(None, 1)
    if len(parts) == 2 and parts[1] in {"low", "medium", "high", "max"}:
        return parts[0], reasoning_effort or parts[1]
    return str(model).strip(), reasoning_effort


def load_api_config(
    *,
    repo_root: Path | None = None,
    model_override: str | None = None,
    base_url_override: str | None = None,
    require_credential: bool = True,
) -> APIConfig:
    """Load aliases without printing or returning secrets in diagnostics."""

    try:
        from dotenv import load_dotenv

        root = repo_root or Path(__file__).resolve().parents[1]
        for path in (root / "analysis" / "rlvr" / ".env", root / ".env"):
            if path.exists():
                load_dotenv(path, override=False)
        load_dotenv(override=False)
    except Exception:
        # dotenv is optional; process environment remains authoritative.
        pass

    key, key_name = _first_nonempty(
        ("MODEL_API_KEY", "OPENAI_API_KEY", "H3LAB_API_KEY", "BOUNDARY_JUDGE_API_KEY")
    )
    base_url, _ = _first_nonempty(
        ("MODEL_API_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "H3LAB_API_BASE_URL")
    )
    model, _ = _first_nonempty(
        ("BOUNDARY_JUDGE_MODEL", "MODEL_NAME", "OPENAI_MODEL", "H3LAB_MODEL")
    )
    reasoning_effort, _ = _first_nonempty(
        ("BOUNDARY_JUDGE_REASONING_EFFORT", "OPENAI_REASONING_EFFORT")
    )
    if model_override:
        model = model_override
    if base_url_override:
        base_url = base_url_override
    if not model:
        model = "gpt-5.6-luna" if not base_url else "boundary-judge"
    model, reasoning_effort = _split_model_setting(model, reasoning_effort)
    if not reasoning_effort and model == "gpt-5.6-luna":
        reasoning_effort = "max"
    if require_credential and not key:
        raise MissingAPICredential(
            "Missing API credential; expected one of MODEL_API_KEY, OPENAI_API_KEY, "
            "H3LAB_API_KEY, BOUNDARY_JUDGE_API_KEY"
        )
    return APIConfig(
        api_key=key or "",
        model=model,
        base_url=base_url,
        api_key_name=key_name or "",
        reasoning_effort=reasoning_effort,
    )


def boundary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "has_complete_numeric_calculation": {"type": "boolean"},
            "unit_id": {"type": ["string", "null"]},
            "span_text": {"type": ["string", "null"]},
            "boundary_kind": {
                "type": "string",
                "enum": [
                    "explicit_equation",
                    "verbal_numeric_evaluation",
                    "other_numeric_evaluation",
                    "none",
                ],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason_short": {"type": "string"},
        },
        "required": [
            "has_complete_numeric_calculation",
            "unit_id",
            "span_text",
            "boundary_kind",
            "confidence",
            "reason_short",
        ],
    }


def _parse_annotation(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("structured response is not an object")
    schema = boundary_schema()["properties"]
    output = dict(value)
    missing = [key for key in schema if key not in output]
    if missing:
        raise ValueError(f"structured response missing fields: {','.join(missing)}")
    if not isinstance(output["has_complete_numeric_calculation"], bool):
        raise ValueError("has_complete_numeric_calculation must be boolean")
    if output["unit_id"] is not None and not isinstance(output["unit_id"], str):
        raise ValueError("unit_id must be string or null")
    if output["span_text"] is not None and not isinstance(output["span_text"], str):
        raise ValueError("span_text must be string or null")
    if len(str(output.get("reason_short", ""))) > 500:
        output["reason_short"] = str(output["reason_short"])[:500]
    return output


@dataclass(frozen=True)
class BoundaryResolution:
    status: str
    semantic_char_start: int | None
    semantic_char_end: int | None
    unit_id: str | None
    span_text: str | None
    resolver_ambiguous: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolver_status": self.status,
            "semantic_char_start": self.semantic_char_start,
            "semantic_char_end": self.semantic_char_end,
            "resolved_unit_id": self.unit_id,
            "resolved_span_text": self.span_text,
            "resolver_ambiguous": self.resolver_ambiguous,
            "resolver_reason": self.reason,
        }


def resolve_boundary_span(
    response: str,
    units: Sequence[Mapping[str, Any]],
    annotation: Mapping[str, Any] | Any,
) -> BoundaryResolution:
    """Resolve a model-copied span to exact response character offsets.

    Exact matching is preferred.  A narrowly-scoped fallback accepts only a
    selected *whole unit* whose content is identical after removal of outer
    math delimiters and normalization of whitespace.  This catches a judge
    adding display delimiters or indentation without treating an edited
    mathematical expression as resolved.
    """

    annotation = _parse_annotation(annotation)
    if not annotation["has_complete_numeric_calculation"]:
        return BoundaryResolution("no_calculation", None, None, None, None, False)
    unit_id = annotation.get("unit_id")
    span_text = annotation.get("span_text")
    if not unit_id or not span_text:
        return BoundaryResolution("annotation_empty", None, None, unit_id, span_text, False)
    selected = next((dict(unit) for unit in units if str(unit.get("unit_id")) == str(unit_id)), None)
    if selected is None:
        return BoundaryResolution("resolver_failed", None, None, str(unit_id), span_text, False, "unknown_unit_id")
    unit_text = str(selected.get("text", ""))
    occurrences: list[int] = []
    cursor = 0
    while True:
        index = unit_text.find(str(span_text), cursor)
        if index < 0:
            break
        occurrences.append(index)
        cursor = index + max(1, len(str(span_text)))
    unit_start = int(selected.get("char_start", -1))
    unit_end = int(selected.get("char_end", -1))
    if unit_start < 0 or unit_end < unit_start or unit_end > len(str(response)):
        return BoundaryResolution("resolver_failed", None, None, str(unit_id), span_text, False, "unit_offsets_invalid")
    if not occurrences:
        def presentation_normal_form(text: str) -> str:
            value = str(text).strip()
            changed = True
            while changed:
                changed = False
                for left, right in (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$")):
                    if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
                        value = value[len(left):len(value) - len(right)].strip()
                        changed = True
                        break
            return re.sub(r"\s+", " ", value).strip()

        if presentation_normal_form(str(span_text)) == presentation_normal_form(unit_text):
            return BoundaryResolution(
                "resolved",
                unit_start,
                unit_end,
                str(unit_id),
                unit_text,
                False,
                "format_normalized_to_selected_unit",
            )
        return BoundaryResolution("resolver_failed", None, None, str(unit_id), span_text, False, "span_not_exact_substring")
    char_start = unit_start + occurrences[0]
    char_end = char_start + len(str(span_text))
    if str(response)[char_start:char_end] != str(span_text):
        return BoundaryResolution("resolver_failed", None, None, str(unit_id), span_text, bool(len(occurrences) > 1), "response_substring_mismatch")
    return BoundaryResolution(
        "resolved",
        char_start,
        char_end,
        str(unit_id),
        str(span_text),
        len(occurrences) > 1,
    )


def blind_prompt(question: str, units: Sequence[Mapping[str, Any]]) -> str:
    displayed = "\n".join(f"[{unit['unit_id']}] {unit['text']}" for unit in units)
    return (
        "QUESTION:\n"
        + str(question)
        + "\n\nRESPONSE UNITS:\n"
        + displayed
        + "\n\nLocate the earliest complete numeric calculation and return its unit_id. "
        "For span_text, copy one contiguous substring from that displayed unit byte-for-byte: "
        "do not add or remove any whitespace, punctuation, Markdown/LaTex delimiters, "
        "bullet prefixes, or line breaks. Do not output character indices."
    )


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    output = getattr(response, "output", None)
    if output:
        chunks: list[str] = []
        for item in output:
            for content in getattr(item, "content", None) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(str(value))
        if chunks:
            return "".join(chunks)
    return ""


def _usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, Mapping):
        return dict(usage)
    return {key: getattr(usage, key) for key in ("input_tokens", "output_tokens", "total_tokens") if hasattr(usage, key)}


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None and getattr(exc, "response", None) is not None:
        value = getattr(exc.response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code in {408, 409, 429} or (code is not None and code >= 500):
        return True
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectError"}


def _pydantic_annotation(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`").strip()
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()
    return _parse_annotation(json.loads(payload))


def call_boundary_judge(
    question: str,
    units: Sequence[Mapping[str, Any]],
    config: APIConfig,
    *,
    max_attempts: int = 6,
    schema_attempts: int = 2,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call an OpenAI or OpenAI-compatible endpoint with structured fallback."""

    if not config.api_key:
        raise MissingAPICredential("Missing API credential; configure a supported API key alias")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for API judging") from exc
    client_kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    client = OpenAI(**client_kwargs)
    prompt = blind_prompt(question, units)
    last_error: BaseException | None = None
    schema_failures = 0
    for attempt in range(max_attempts):
        try:
            if schema_failures < schema_attempts and hasattr(client, "responses") and hasattr(client.responses, "parse"):
                response_args = {
                    "model": config.model,
                    "input": [
                        {"role": "system", "content": BOUNDARY_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "text_format": BoundaryAnnotation,
                    "max_output_tokens": 1024,
                }
                if config.reasoning_effort:
                    response_args["reasoning"] = {"effort": config.reasoning_effort}
                response = client.responses.parse(**response_args)
                parsed = getattr(response, "output_parsed", None)
                annotation = _parse_annotation(parsed if parsed is not None else _response_text(response))
                return {"annotation": annotation, "response_id": getattr(response, "id", None), "usage": _usage(response), "api_backend_mode": "responses.parse"}
            if hasattr(client, "responses"):
                response_args = {
                    "model": config.model,
                    "input": [
                        {"role": "system", "content": BOUNDARY_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "text": {"format": {"type": "json_schema", "name": "boundary_annotation", "strict": True, "schema": boundary_schema()}},
                    "max_output_tokens": 1024,
                }
                if config.reasoning_effort:
                    response_args["reasoning"] = {"effort": config.reasoning_effort}
                response = client.responses.create(**response_args)
                annotation = _pydantic_annotation(_response_text(response))
                return {"annotation": annotation, "response_id": getattr(response, "id", None), "usage": _usage(response), "api_backend_mode": "responses.json_schema"}
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": BOUNDARY_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "boundary_annotation", "strict": True, "schema": boundary_schema()}},
            )
            content = response.choices[0].message.content
            annotation = _pydantic_annotation(str(content))
            return {"annotation": annotation, "response_id": getattr(response, "id", None), "usage": _usage(response), "api_backend_mode": "chat.json_schema"}
        except Exception as exc:  # network and compatibility boundaries are recorded by type only
            last_error = exc
            if type(exc).__name__ in {"ValidationError", "JSONDecodeError", "ValueError"}:
                schema_failures += 1
                if schema_failures >= schema_attempts:
                    break
            if not _retryable(exc) and schema_failures >= schema_attempts:
                break
            if attempt + 1 >= max_attempts:
                break
            delay = min(32.0, 2.0 ** attempt) + random.random()
            time.sleep(delay)
    error_type = type(last_error).__name__ if last_error is not None else "UnknownError"
    return {"annotation": None, "response_id": None, "usage": None, "api_backend_mode": "failed", "status": "parse_failed", "error_type": error_type}


try:
    from pydantic import BaseModel, Field

    class BoundaryAnnotation(BaseModel):
        has_complete_numeric_calculation: bool
        unit_id: str | None
        span_text: str | None
        boundary_kind: Literal["explicit_equation", "verbal_numeric_evaluation", "other_numeric_evaluation", "none"]
        confidence: Literal["high", "medium", "low"]
        reason_short: str = Field(max_length=500)

except ImportError:  # pragma: no cover - API mode reports the missing package
    BoundaryAnnotation = None  # type: ignore[assignment,misc]


__all__ = [
    "APIConfig",
    "BOUNDARY_JUDGE_SYSTEM",
    "BoundaryAnnotation",
    "BoundaryResolution",
    "MissingAPICredential",
    "blind_prompt",
    "boundary_schema",
    "call_boundary_judge",
    "load_api_config",
    "resolve_boundary_span",
]
