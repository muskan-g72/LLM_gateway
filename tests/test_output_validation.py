from __future__ import annotations

import json

import pytest

from app.output_validation import (
    ExtractActionItemsOutput,
    OutputParsingError,
    OutputSemanticError,
    OutputStructureError,
    OutputValidator,
    SummarizeOutput,
    UnsupportedSkillOutputError,
)
from app.skills import SkillDefinition, SkillLoader


def _skill(name: str) -> SkillDefinition:
    return SkillLoader().load(name)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _validate_summary(
    value: object,
    source: str = "The team approved the Friday release plan.",
) -> SummarizeOutput:
    result = OutputValidator().validate(
        _skill("summarize"),
        {"text": source},
        _json(value),
    )
    assert isinstance(result, SummarizeOutput)
    return result


def _validate_action_items(
    value: object,
    source: str = "Priya will submit the report by Friday.",
) -> ExtractActionItemsOutput:
    result = OutputValidator().validate(
        _skill("extract_action_items"),
        {"text": source},
        _json(value),
    )
    assert isinstance(result, ExtractActionItemsOutput)
    return result


def test_valid_summarize_json_is_accepted() -> None:
    result = _validate_summary(
        {
            "summary": "The release plan was approved.",
            "key_points": ["The release is planned for Friday."],
        }
    )

    assert result.model_dump(mode="json") == {
        "summary": "The release plan was approved.",
        "key_points": ["The release is planned for Friday."],
    }


def test_valid_action_item_json_is_accepted() -> None:
    result = _validate_action_items(
        {
            "action_items": [
                {"task": "Submit the report", "owner": "Priya", "deadline": "Friday"}
            ]
        }
    )

    assert result.model_dump(mode="json") == {
        "action_items": [
            {"task": "Submit the report", "owner": "Priya", "deadline": "Friday"}
        ]
    }


def test_malformed_json_is_a_parsing_error() -> None:
    with pytest.raises(OutputParsingError, match="one valid JSON document"):
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Release plan"},
            '{"summary": "unfinished"',
        )


def test_markdown_fenced_json_is_rejected() -> None:
    raw_output = '```json\n{"summary":"Release","key_points":[]}\n```'

    with pytest.raises(OutputParsingError):
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Release"},
            raw_output,
        )


def test_explanatory_text_around_json_is_rejected() -> None:
    raw_output = 'Here is the result: {"summary":"Release","key_points":[]}'

    with pytest.raises(OutputParsingError):
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Release"},
            raw_output,
        )


@pytest.mark.parametrize("root", [[], "text", 12, True, None])
def test_non_object_json_root_is_rejected(root: object) -> None:
    with pytest.raises(OutputStructureError) as captured:
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Release"},
            _json(root),
        )

    assert captured.value.issues == ("$: expected JSON object",)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_numeric_constants_are_rejected(constant: str) -> None:
    raw_output = f'{{"summary":{constant},"key_points":[]}}'

    with pytest.raises(OutputParsingError):
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Release"},
            raw_output,
        )


def test_non_string_raw_output_is_rejected_as_parsing_error() -> None:
    with pytest.raises(OutputParsingError, match="must be a string"):
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Release"},
            None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value", "expected_issue"),
    [
        ({"key_points": ["Release"]}, "summary: field required"),
        (
            {"summary": 42, "key_points": ["Release"]},
            "summary: expected string",
        ),
        (
            {"summary": "Release", "key_points": "Friday"},
            "key_points: expected list",
        ),
    ],
)
def test_missing_or_wrong_summarize_fields_are_rejected(
    value: object,
    expected_issue: str,
) -> None:
    with pytest.raises(OutputStructureError) as captured:
        _validate_summary(value)

    assert expected_issue in captured.value.issues


def test_unexpected_top_level_field_is_rejected() -> None:
    with pytest.raises(OutputStructureError) as captured:
        _validate_summary(
            {
                "summary": "Release approved",
                "key_points": ["Friday release"],
                "confidence": 0.9,
            }
        )

    assert "confidence: unexpected field" in captured.value.issues


def test_unexpected_nested_action_item_field_is_rejected() -> None:
    with pytest.raises(OutputStructureError) as captured:
        _validate_action_items(
            {
                "action_items": [
                    {
                        "task": "Submit report",
                        "owner": "Priya",
                        "deadline": "Friday",
                        "priority": "high",
                    }
                ]
            }
        )

    assert "action_items.0.priority: unexpected field" in captured.value.issues


@pytest.mark.parametrize(
    ("value", "expected_issue"),
    [
        (
            {"summary": "   ", "key_points": ["Release"]},
            "summary: must not be blank",
        ),
        (
            {"summary": "Release", "key_points": ["\n\t"]},
            "key_points.0: must not be blank",
        ),
    ],
)
def test_blank_summarize_text_is_rejected(
    value: object,
    expected_issue: str,
) -> None:
    with pytest.raises(OutputStructureError) as captured:
        _validate_summary(value)

    assert expected_issue in captured.value.issues


@pytest.mark.parametrize(
    ("field", "value"),
    [("task", "  "), ("owner", "\t"), ("deadline", "\n")],
)
def test_blank_action_item_text_is_rejected(field: str, value: str) -> None:
    item = {"task": "Submit report", "owner": "Priya", "deadline": "Friday"}
    item[field] = value

    with pytest.raises(OutputStructureError) as captured:
        _validate_action_items({"action_items": [item]})

    assert f"action_items.0.{field}: must not be blank" in captured.value.issues


def test_duplicate_key_points_are_rejected_after_normalization() -> None:
    with pytest.raises(OutputSemanticError) as captured:
        _validate_summary(
            {
                "summary": "The release was approved.",
                "key_points": ["Friday Release", "  friday release  "],
            }
        )

    assert "key_points.1: duplicates key_points.0" in captured.value.issues


def test_unicode_equivalent_key_points_are_duplicates() -> None:
    with pytest.raises(OutputSemanticError):
        _validate_summary(
            {
                "summary": "The café release was approved.",
                "key_points": ["Café release", "Cafe\u0301 release"],
            },
            source="The café release was approved.",
        )


def test_duplicate_action_items_are_rejected_after_normalization() -> None:
    with pytest.raises(OutputSemanticError) as captured:
        _validate_action_items(
            {
                "action_items": [
                    {
                        "task": "Submit report",
                        "owner": "Priya",
                        "deadline": "Friday",
                    },
                    {
                        "task": "  submit REPORT ",
                        "owner": "PRIYA",
                        "deadline": " friday ",
                    },
                ]
            }
        )

    assert "action_items.1: duplicates action_items.0" in captured.value.issues


def test_grounded_summarize_output_is_accepted() -> None:
    result = _validate_summary(
        {
            "summary": "Approval was given for the launch.",
            "key_points": ["Friday remains the release date."],
        }
    )

    assert result.summary == "Approval was given for the launch."


def test_empty_key_points_are_allowed_by_current_skill_schema() -> None:
    result = _validate_summary(
        {"summary": "The release was approved.", "key_points": []}
    )

    assert result.key_points == []


def test_clearly_unrelated_summarize_output_is_rejected() -> None:
    with pytest.raises(OutputSemanticError) as captured:
        _validate_summary(
            {
                "summary": "Penguins migrate across Antarctic ice.",
                "key_points": ["Ocean temperatures influence their route."],
            },
            source="Quarterly revenue increased after the product launch.",
        )

    assert "no meaningful token overlap" in str(captured.value)


def test_grounded_action_item_is_accepted() -> None:
    result = _validate_action_items(
        {
            "action_items": [
                {"task": "Submit the report", "owner": None, "deadline": None}
            ]
        }
    )

    assert result.action_items[0].task == "Submit the report"


def test_clearly_unrelated_action_item_is_rejected() -> None:
    with pytest.raises(OutputSemanticError) as captured:
        _validate_action_items(
            {
                "action_items": [
                    {"task": "Book tropical flights", "owner": None, "deadline": None}
                ]
            }
        )

    assert "action_items.0.task: no meaningful token" in str(captured.value)


def test_null_owner_and_deadline_are_accepted() -> None:
    result = _validate_action_items(
        {
            "action_items": [
                {"task": "Submit the report", "owner": None, "deadline": None}
            ]
        }
    )

    assert result.action_items[0].owner is None
    assert result.action_items[0].deadline is None


def test_returned_content_preserves_original_casing_and_wording() -> None:
    result = _validate_summary(
        {
            "summary": "  RELEASE Approval Remains  ",
            "key_points": ["  Friday TIMELINE  "],
        }
    )

    assert result.summary == "  RELEASE Approval Remains  "
    assert result.key_points == ["  Friday TIMELINE  "]


def test_unsupported_skill_fails_clearly() -> None:
    unsupported = _skill("summarize").model_copy(update={"name": "translate"})

    with pytest.raises(UnsupportedSkillOutputError, match="skill 'translate'"):
        OutputValidator().validate(
            unsupported,
            {"text": "Example"},
            '{"summary":"Example","key_points":[]}',
        )


def test_errors_do_not_include_full_raw_output() -> None:
    secret_marker = "PRIVATE_MODEL_OUTPUT_SENTINEL"
    raw_output = _json(
        {
            "summary": "Release approved",
            "key_points": ["Friday release"],
            "unexpected": secret_marker,
        }
    )

    with pytest.raises(OutputStructureError) as captured:
        OutputValidator().validate(
            _skill("summarize"),
            {"text": "Friday release"},
            raw_output,
        )

    assert secret_marker not in str(captured.value)
    assert raw_output not in str(captured.value)


def test_errors_expose_concise_nested_field_paths() -> None:
    with pytest.raises(OutputStructureError) as captured:
        _validate_action_items(
            {
                "action_items": [
                    {"task": "Submit report", "owner": None, "deadline": None},
                    {"task": 7, "owner": None, "deadline": None},
                ]
            }
        )

    assert captured.value.issues == ("action_items.1.task: expected string",)
