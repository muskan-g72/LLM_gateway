from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
DEFAULT_MAX_TOOL_ARGUMENT_BYTES = 16_384
DEFAULT_MAX_TOOL_RESULT_BYTES = 4_096
MAX_TEXT_STATISTICS_CHARACTERS = 10_000

ToolErrorCategory = Literal[
    "tool_not_allowed",
    "tool_not_found",
    "tool_arguments_invalid",
    "tool_timeout",
    "tool_execution",
    "tool_result_invalid",
    "tool_result_too_large",
    "repeated_tool_call",
    "tool_protocol",
]


class ToolError(Exception):
    """Safe tool failure containing a bounded category, never internal details."""

    def __init__(self, category: ToolErrorCategory, message: str) -> None:
        self.category = category
        super().__init__(message)


class ToolRegistrationError(ToolError):
    pass


class ToolNotFoundError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_not_found", "requested tool is not registered")


class ToolNotAllowedError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_not_allowed", "requested tool is not allowed")


class ToolArgumentsInvalidError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_arguments_invalid", "tool arguments are invalid")


class ToolTimeoutError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_timeout", "tool execution exceeded its time limit")


class ToolExecutionError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_execution", "tool execution failed")


class ToolResultInvalidError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_result_invalid", "tool result is invalid")


class ToolResultTooLargeError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_result_too_large", "tool result exceeds its size limit")


class RepeatedToolCallError(ToolError):
    def __init__(self) -> None:
        super().__init__("repeated_tool_call", "only one tool call is permitted")


class ToolProtocolError(ToolError):
    def __init__(self) -> None:
        super().__init__("tool_protocol", "model tool-call protocol is invalid")


def _ensure_json_value(
    value: object,
    active_containers: set[int] | None = None,
) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return
    if type(value) not in {dict, list}:
        raise ValueError("non-JSON value")

    active = active_containers if active_containers is not None else set()
    identity = id(value)
    if identity in active:
        raise ValueError("circular value")
    active.add(identity)
    try:
        if type(value) is list:
            for item in value:
                _ensure_json_value(item, active)
            return
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("non-string JSON key")
            _ensure_json_value(item, active)
    finally:
        active.remove(identity)


def _canonical_json(value: object) -> str:
    try:
        _ensure_json_value(value)
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (RecursionError, TypeError, ValueError, OverflowError):
        raise ValueError("value is not canonical JSON") from None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_seconds: float = 1.0
    maximum_argument_bytes: int = DEFAULT_MAX_TOOL_ARGUMENT_BYTES
    maximum_result_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES

    def __post_init__(self) -> None:
        if TOOL_NAME_PATTERN.fullmatch(self.name) is None:
            raise ToolRegistrationError(
                "tool_protocol",
                "registered tool name is invalid",
            )
        if not self.description.strip():
            raise ToolRegistrationError(
                "tool_protocol",
                "registered tool description is invalid",
            )
        if (
            not isinstance(self.input_model, type)
            or not issubclass(self.input_model, BaseModel)
            or not isinstance(self.output_model, type)
            or not issubclass(self.output_model, BaseModel)
        ):
            raise ToolRegistrationError(
                "tool_protocol",
                "registered tool models are invalid",
            )
        if self.timeout_seconds <= 0:
            raise ToolRegistrationError(
                "tool_protocol",
                "registered tool timeout must be positive",
            )
        if self.maximum_argument_bytes <= 0 or self.maximum_result_bytes <= 0:
            raise ToolRegistrationError(
                "tool_protocol",
                "registered tool size limits must be positive",
            )

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }


@dataclass(frozen=True)
class ToolExecutionResult:
    status: Literal["success"]
    data: dict[str, object]

    def as_json_value(self) -> dict[str, object]:
        return {"status": self.status, "data": deepcopy(self.data)}


ToolHandler = Callable[[BaseModel], object]


class RegisteredTool:
    """Validate, time-bound, execute, and normalize one trusted callable."""

    def __init__(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if not callable(handler):
            raise ToolRegistrationError("tool_protocol", "tool handler is invalid")
        self.definition = definition
        self._handler = handler

    def validate_arguments(self, arguments: object) -> BaseModel:
        if type(arguments) is not dict:
            raise ToolArgumentsInvalidError()
        try:
            serialized = _canonical_json(arguments)
        except ValueError:
            raise ToolArgumentsInvalidError() from None
        if len(serialized.encode("utf-8")) > self.definition.maximum_argument_bytes:
            raise ToolArgumentsInvalidError()
        try:
            return self.definition.input_model.model_validate(arguments)
        except ValidationError:
            raise ToolArgumentsInvalidError() from None

    def execute_validated(self, arguments: BaseModel) -> ToolExecutionResult:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gateway-tool")
        future = executor.submit(self._handler, arguments)
        try:
            raw_result = future.result(timeout=self.definition.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise ToolTimeoutError() from None
        except Exception:
            executor.shutdown(wait=True, cancel_futures=True)
            raise ToolExecutionError() from None
        else:
            executor.shutdown(wait=True, cancel_futures=True)

        try:
            result = self.definition.output_model.model_validate(raw_result)
        except ValidationError:
            raise ToolResultInvalidError() from None

        data = result.model_dump(mode="json")
        try:
            serialized = _canonical_json(data)
        except ValueError:
            raise ToolResultInvalidError() from None
        if len(serialized.encode("utf-8")) > self.definition.maximum_result_bytes:
            raise ToolResultTooLargeError()
        return ToolExecutionResult(status="success", data=data)

    def run(self, arguments: object) -> ToolExecutionResult:
        return self.execute_validated(self.validate_arguments(arguments))


class ToolRegistry:
    """Immutable exact-name registry for trusted application-owned tools."""

    def __init__(self, tools: tuple[RegisteredTool, ...] | list[RegisteredTool]) -> None:
        registered: dict[str, RegisteredTool] = {}
        for tool in tools:
            name = tool.definition.name
            if name in registered:
                raise ToolRegistrationError(
                    "tool_protocol",
                    "duplicate registered tool name",
                )
            registered[name] = tool
        self._tools: Mapping[str, RegisteredTool] = MappingProxyType(registered)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> RegisteredTool:
        if type(name) is not str or TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise ToolNotFoundError()
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError() from None

    def metadata_for(self, names: tuple[str, ...] | list[str]) -> tuple[dict[str, object], ...]:
        return tuple(deepcopy(self.get(name).definition.metadata()) for name in sorted(names))


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["add", "subtract", "multiply", "divide"]
    a: int | float
    b: int | float

    @field_validator("a", "b")
    @classmethod
    def number_must_be_finite_and_not_boolean(cls, value: int | float) -> int | float:
        if type(value) not in {int, float}:
            raise ValueError("expected finite number")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("expected finite number")
        return value

    @model_validator(mode="after")
    def divisor_must_not_be_zero(self) -> "CalculatorInput":
        if self.operation == "divide" and self.b == 0:
            raise ValueError("division by zero")
        return self


class CalculatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    result: int | float

    @field_validator("result")
    @classmethod
    def result_must_be_finite(cls, value: int | float) -> int | float:
        if type(value) not in {int, float}:
            raise ValueError("expected finite number")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("expected finite number")
        return value


class TextStatisticsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(max_length=MAX_TEXT_STATISTICS_CHARACTERS)


class TextStatisticsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    characters: int
    words: int
    lines: int


def _calculate(arguments: BaseModel) -> dict[str, object]:
    if not isinstance(arguments, CalculatorInput):
        raise ValueError("calculator input contract mismatch")
    if arguments.operation == "add":
        result = arguments.a + arguments.b
    elif arguments.operation == "subtract":
        result = arguments.a - arguments.b
    elif arguments.operation == "multiply":
        result = arguments.a * arguments.b
    else:
        if arguments.b == 0:
            raise ValueError("division by zero")
        result = arguments.a / arguments.b
    if type(result) is float and not math.isfinite(result):
        raise ValueError("non-finite calculator result")
    return {"result": result}


def _text_statistics(arguments: BaseModel) -> dict[str, object]:
    if not isinstance(arguments, TextStatisticsInput):
        raise ValueError("text-statistics input contract mismatch")
    text = arguments.text
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()) if text else 0,
    }


def build_builtin_tool_registry() -> ToolRegistry:
    calculator = RegisteredTool(
        ToolDefinition(
            name="calculator",
            description="Perform one bounded add, subtract, multiply, or divide operation.",
            input_model=CalculatorInput,
            output_model=CalculatorOutput,
        ),
        _calculate,
    )
    text_statistics = RegisteredTool(
        ToolDefinition(
            name="text_statistics",
            description="Count characters, whitespace-delimited words, and text lines.",
            input_model=TextStatisticsInput,
            output_model=TextStatisticsOutput,
        ),
        _text_statistics,
    )
    return ToolRegistry([calculator, text_statistics])
