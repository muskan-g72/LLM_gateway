from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, TypeAlias, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

from app.skills import SkillDefinition


class OutputValidationError(ValueError):
    """Base error at the boundary between model text and trusted application data."""

    def __init__(self, message: str, issues: Sequence[str] = ()) -> None:
        self.issues = tuple(issues)
        details = f": {'; '.join(self.issues)}" if self.issues else ""
        super().__init__(f"{message}{details}")


class OutputParsingError(OutputValidationError):
    """The provider text is not exactly one standards-compliant JSON document."""


class OutputStructureError(OutputValidationError):
    """Parsed JSON has the wrong root, fields, types, or constrained values."""


class OutputSemanticError(OutputValidationError):
    """Typed output fails a limited deterministic task-specific check."""


class UnsupportedSkillOutputError(OutputValidationError):
    """No concrete runtime output model is registered for the skill."""


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlankText = Annotated[str, AfterValidator(_require_non_blank)]


class SummarizeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: NonBlankText
    key_points: list[NonBlankText]


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: NonBlankText
    owner: NonBlankText | None
    deadline: NonBlankText | None


class ExtractActionItemsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_items: list[ActionItem]


ValidatedOutput: TypeAlias = SummarizeOutput | ExtractActionItemsOutput

OUTPUT_MODEL_REGISTRY: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "summarize": SummarizeOutput,
        "extract_action_items": ExtractActionItemsOutput,
    }
)


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)
_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _normalize_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _meaningful_tokens(value: str) -> set[str]:
    """Return simple overlap tokens; this heuristic does not prove factual grounding."""

    return {
        token
        for token in _TOKEN_PATTERN.findall(_normalize_phrase(value))
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _reject_nonstandard_json_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant: {name}")


def _parse_json_document(raw_output: str) -> object:
    if type(raw_output) is not str:
        raise OutputParsingError("model output must be a string")

    try:
        return json.loads(
            raw_output,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise OutputParsingError(
            "model output is not one valid JSON document"
        ) from None


def _safe_error_path(location: tuple[object, ...]) -> str:
    if not location:
        return "$"
    parts = []
    for part in location:
        text = str(part)
        parts.append(text if len(text) <= 48 else f"{text[:45]}...")
    return ".".join(parts)


def _pydantic_issues(error: ValidationError) -> tuple[str, ...]:
    message_by_type = {
        "missing": "field required",
        "string_type": "expected string",
        "list_type": "expected list",
        "model_type": "expected object",
        "dict_type": "expected object",
        "extra_forbidden": "unexpected field",
    }
    issues: list[str] = []
    error_items = error.errors(include_url=False, include_input=False)

    for item in error_items[:10]:
        path = _safe_error_path(item["loc"])
        error_type = item["type"]
        if error_type == "value_error" and item["msg"].endswith(
            "must not be blank"
        ):
            message = "must not be blank"
        else:
            message = message_by_type.get(error_type, "invalid value")
        issues.append(f"{path}: {message}")

    if len(error_items) > 10:
        issues.append("$: additional validation errors omitted")
    return tuple(issues)


def _source_text(task_input: dict[str, object]) -> str:
    source = task_input.get("text")
    if type(source) is not str or not source.strip():
        raise OutputSemanticError(
            "semantic validation could not run",
            ("task_input.text: non-blank source text required",),
        )
    return source


def _validate_summarize_semantics(
    result: BaseModel,
    source_text: str,
) -> tuple[str, ...]:
    if not isinstance(result, SummarizeOutput):
        raise RuntimeError("summarize output registry is inconsistent")

    issues: list[str] = []
    seen_points: dict[str, int] = {}
    for index, point in enumerate(result.key_points):
        normalized = _normalize_phrase(point)
        if normalized in seen_points:
            issues.append(
                f"key_points.{index}: duplicates key_points.{seen_points[normalized]}"
            )
        else:
            seen_points[normalized] = index

    generated_text = " ".join([result.summary, *result.key_points])
    if not (_meaningful_tokens(source_text) & _meaningful_tokens(generated_text)):
        issues.append(
            "summary: summary and key points have no meaningful token overlap with task input"
        )
    return tuple(issues)


def _validate_action_item_semantics(
    result: BaseModel,
    source_text: str,
) -> tuple[str, ...]:
    if not isinstance(result, ExtractActionItemsOutput):
        raise RuntimeError("action-item output registry is inconsistent")

    issues: list[str] = []
    source_tokens = _meaningful_tokens(source_text)
    seen_items: dict[tuple[str, str | None, str | None], int] = {}

    for index, item in enumerate(result.action_items):
        normalized = (
            _normalize_phrase(item.task),
            _normalize_phrase(item.owner) if item.owner is not None else None,
            _normalize_phrase(item.deadline) if item.deadline is not None else None,
        )
        if normalized in seen_items:
            issues.append(
                f"action_items.{index}: duplicates action_items.{seen_items[normalized]}"
            )
        else:
            seen_items[normalized] = index

        if not (_meaningful_tokens(item.task) & source_tokens):
            issues.append(
                f"action_items.{index}.task: no meaningful token appears in task input"
            )
    return tuple(issues)


SemanticValidator = Callable[[BaseModel, str], tuple[str, ...]]
_SEMANTIC_VALIDATOR_REGISTRY: Mapping[str, SemanticValidator] = MappingProxyType(
    {
        "summarize": _validate_summarize_semantics,
        "extract_action_items": _validate_action_item_semantics,
    }
)


class OutputValidator:
    """Turn untrusted provider text into a typed result after bounded checks.

    Token overlap only catches clearly unrelated content. It cannot establish that a
    summary is complete or that an extracted action item is factually correct.
    """

    def validate(
        self,
        skill: SkillDefinition,
        task_input: dict[str, object],
        raw_output: str,
    ) -> ValidatedOutput:
        output_model = OUTPUT_MODEL_REGISTRY.get(skill.name)
        semantic_validator = _SEMANTIC_VALIDATOR_REGISTRY.get(skill.name)
        if output_model is None or semantic_validator is None:
            raise UnsupportedSkillOutputError(
                f"No output validator is registered for skill '{skill.name}'"
            )

        parsed = _parse_json_document(raw_output)
        if type(parsed) is not dict:
            raise OutputStructureError(
                "model output has an invalid root",
                ("$: expected JSON object",),
            )

        try:
            result = output_model.model_validate(parsed)
        except ValidationError as error:
            raise OutputStructureError(
                "model output does not match the required structure",
                _pydantic_issues(error),
            ) from None

        semantic_issues = semantic_validator(result, _source_text(task_input))
        if semantic_issues:
            raise OutputSemanticError(
                "model output failed deterministic semantic validation",
                semantic_issues,
            )

        return cast(ValidatedOutput, result)
