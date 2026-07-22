from __future__ import annotations

import inspect
import time

import pytest
from pydantic import BaseModel, ConfigDict

from app.tools import (
    MAX_TEXT_STATISTICS_CHARACTERS,
    RegisteredTool,
    ToolArgumentsInvalidError,
    ToolDefinition,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
    ToolResultInvalidError,
    ToolResultTooLargeError,
    ToolTimeoutError,
    _calculate,
    build_builtin_tool_registry,
)


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    echoed: str


def _echo_tool(
    handler,
    *,
    name: str = "echo",
    timeout_seconds: float = 1.0,
    maximum_argument_bytes: int = 1_024,
    maximum_result_bytes: int = 1_024,
) -> RegisteredTool:
    return RegisteredTool(
        ToolDefinition(
            name=name,
            description="Return deterministic test text.",
            input_model=EchoInput,
            output_model=EchoOutput,
            timeout_seconds=timeout_seconds,
            maximum_argument_bytes=maximum_argument_bytes,
            maximum_result_bytes=maximum_result_bytes,
        ),
        handler,
    )


def test_valid_tool_registration_and_lookup_succeed() -> None:
    tool = _echo_tool(lambda value: {"echoed": value.text})
    registry = ToolRegistry([tool])

    assert registry.names == ("echo",)
    assert registry.get("echo") is tool


def test_duplicate_tool_registration_fails() -> None:
    first = _echo_tool(lambda value: {"echoed": value.text})
    second = _echo_tool(lambda value: {"echoed": value.text})

    with pytest.raises(ToolRegistrationError):
        ToolRegistry([first, second])


def test_tool_registration_rejects_non_pydantic_contract_models() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolDefinition(
            name="invalid_contract",
            description="Invalid test definition.",
            input_model=dict,  # type: ignore[arg-type]
            output_model=EchoOutput,
        )


@pytest.mark.parametrize(
    "name",
    ["Upper", "has space", "path/name", "dot.name", "_private", "", "a" * 65],
)
def test_invalid_registered_tool_names_fail(name: str) -> None:
    with pytest.raises(ToolRegistrationError):
        _echo_tool(lambda value: {"echoed": value.text}, name=name)


@pytest.mark.parametrize("name", ["unknown", "../echo", "Echo", "echo.name"])
def test_unknown_or_malformed_lookup_fails_safely(name: str) -> None:
    registry = ToolRegistry([])

    with pytest.raises(ToolNotFoundError) as captured:
        registry.get(name)

    assert name not in str(captured.value)


def test_registry_metadata_is_deterministic_and_excludes_callables() -> None:
    registry = build_builtin_tool_registry()

    first = registry.metadata_for(["text_statistics", "calculator"])
    second = registry.metadata_for(["calculator", "text_statistics"])

    assert first == second
    assert [item["name"] for item in first] == ["calculator", "text_statistics"]
    assert all("handler" not in item and "callable" not in item for item in first)
    assert all("input_schema" in item and "output_schema" in item for item in first)


def test_registry_metadata_returns_stable_copies() -> None:
    registry = build_builtin_tool_registry()
    first = registry.metadata_for(["calculator"])
    first[0]["name"] = "changed"

    assert registry.metadata_for(["calculator"])[0]["name"] == "calculator"


@pytest.mark.parametrize(
    ("operation", "a", "b", "expected"),
    [
        ("add", 2, 3, 5),
        ("subtract", 7, 2, 5),
        ("multiply", 2.5, 4, 10.0),
        ("divide", 9, 2, 4.5),
    ],
)
def test_calculator_performs_supported_operations(
    operation: str,
    a: int | float,
    b: int | float,
    expected: int | float,
) -> None:
    calculator = build_builtin_tool_registry().get("calculator")

    result = calculator.run({"operation": operation, "a": a, "b": b})

    assert result.as_json_value() == {"status": "success", "data": {"result": expected}}


def test_calculator_rejects_division_by_zero_before_execution() -> None:
    calculator = build_builtin_tool_registry().get("calculator")

    with pytest.raises(ToolArgumentsInvalidError):
        calculator.validate_arguments({"operation": "divide", "a": 2, "b": 0})


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf")])
def test_calculator_rejects_boolean_and_non_finite_numbers(value: object) -> None:
    calculator = build_builtin_tool_registry().get("calculator")

    with pytest.raises(ToolArgumentsInvalidError):
        calculator.validate_arguments({"operation": "add", "a": value, "b": 1})


def test_calculator_contains_no_eval() -> None:
    source = inspect.getsource(_calculate)

    assert "eval(" not in source
    assert "exec(" not in source


def test_text_statistics_is_deterministic() -> None:
    tool = build_builtin_tool_registry().get("text_statistics")

    result = tool.run({"text": "hello world\nnext"})

    assert result.data == {"characters": 16, "words": 3, "lines": 2}


def test_text_statistics_enforces_input_length() -> None:
    tool = build_builtin_tool_registry().get("text_statistics")

    with pytest.raises(ToolArgumentsInvalidError):
        tool.validate_arguments({"text": "x" * (MAX_TEXT_STATISTICS_CHARACTERS + 1)})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"text": "hello", "extra": True},
        {"text": 7},
        ["not", "an", "object"],
    ],
)
def test_strict_argument_structure_is_enforced(arguments: object) -> None:
    tool = _echo_tool(lambda value: {"echoed": value.text})

    with pytest.raises(ToolArgumentsInvalidError):
        tool.validate_arguments(arguments)


def test_argument_size_and_nested_json_compatibility_are_enforced() -> None:
    small_tool = _echo_tool(
        lambda value: {"echoed": value.text},
        maximum_argument_bytes=20,
    )
    normal_tool = _echo_tool(lambda value: {"echoed": value.text})

    with pytest.raises(ToolArgumentsInvalidError):
        small_tool.validate_arguments({"text": "x" * 30})
    with pytest.raises(ToolArgumentsInvalidError):
        normal_tool.validate_arguments({"text": "ok", "nested": {1, 2}})


def test_validation_failure_does_not_invoke_handler() -> None:
    calls = 0

    def handler(value: BaseModel) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"echoed": "called"}

    tool = _echo_tool(handler)

    with pytest.raises(ToolArgumentsInvalidError):
        tool.run({"wrong": "value"})
    assert calls == 0


def test_tool_exception_is_sanitized() -> None:
    def handler(value: BaseModel) -> object:
        raise RuntimeError("C:\\private\\secret.txt API_KEY=secret")

    tool = _echo_tool(handler)

    with pytest.raises(ToolExecutionError) as captured:
        tool.run({"text": "hello"})

    assert "private" not in str(captured.value)
    assert "API_KEY" not in str(captured.value)


def test_tool_timeout_is_bounded_and_safe() -> None:
    def handler(value: BaseModel) -> dict[str, object]:
        time.sleep(0.05)
        return {"echoed": "late"}

    tool = _echo_tool(handler, timeout_seconds=0.001)

    with pytest.raises(ToolTimeoutError):
        tool.run({"text": "hello"})


def test_invalid_and_oversized_results_are_rejected() -> None:
    invalid = _echo_tool(lambda value: {"wrong": "shape"})
    oversized = _echo_tool(
        lambda value: {"echoed": "x" * 100},
        maximum_result_bytes=20,
    )

    with pytest.raises(ToolResultInvalidError):
        invalid.run({"text": "hello"})
    with pytest.raises(ToolResultTooLargeError):
        oversized.run({"text": "hello"})
