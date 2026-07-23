from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from typing import Any
from uuid import UUID

import pytest

from app.config import Settings
from app.output_validation import ExtractActionItemsOutput, SummarizeOutput
from app.prompt_builder import PromptBuilder
from app.providers import (
    ProviderCompletion,
    ProviderConfigurationError,
    ProviderError,
    ProviderGateway,
    ProviderOperationalError,
)
from app.skills import SkillLoader
from app.task_executor import (
    INVALID_OUTPUT_EXCERPT_LIMIT,
    TaskExecutor,
    TaskInvalidOutputError,
    TaskProviderConfigurationError,
)
from app.output_validation import OutputValidator


ProviderEvent = ProviderCompletion | Exception


class ScriptedProviderGateway:
    def __init__(self) -> None:
        self.events: dict[str, deque[ProviderEvent]] = defaultdict(deque)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.api_key = "PROVIDER_API_KEY_SECRET"

    def queue(self, provider_name: str, *events: ProviderEvent) -> None:
        self.events[provider_name].extend(events)

    def complete_with_provider(
        self,
        provider_name: str,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        self.calls.append(
            (provider_name, [dict(message) for message in messages])
        )
        if not self.events[provider_name]:
            raise AssertionError(f"unexpected provider call: {provider_name}")
        event = self.events[provider_name].popleft()
        if isinstance(event, Exception):
            raise event
        return event


def _executor(provider: ScriptedProviderGateway) -> TaskExecutor:
    return TaskExecutor(
        SkillLoader(),
        PromptBuilder(),
        provider,  # type: ignore[arg-type]
        OutputValidator(),
    )


def _completion(
    provider: str,
    content: str,
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
) -> ProviderCompletion:
    return ProviderCompletion(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider=provider,
    )


def _summary_json(
    summary: str = "The release was approved.",
    key_points: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "key_points": key_points or ["Friday is the release date."],
        }
    )


def _prepared_summary(executor: TaskExecutor):
    return executor.prepare(
        "summarize",
        {"text": "The team approved the Friday release."},
    )


def test_valid_primary_result_succeeds_in_one_call() -> None:
    provider = ScriptedProviderGateway()
    provider.queue("groq", _completion("groq", _summary_json(), 7, 4))
    executor = _executor(provider)

    result = executor.execute(_prepared_summary(executor))

    assert isinstance(result.task_id, UUID)
    assert isinstance(result.output, SummarizeOutput)
    assert result.provider == "groq"
    assert result.attempts == 1
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 4
    assert [name for name, _ in provider.calls] == ["groq"]


def test_valid_action_items_result_is_typed() -> None:
    provider = ScriptedProviderGateway()
    raw_output = json.dumps(
        {
            "action_items": [
                {
                    "task": "Submit the report",
                    "owner": "Priya",
                    "deadline": "Friday",
                }
            ]
        }
    )
    provider.queue("groq", _completion("groq", raw_output))
    executor = _executor(provider)
    prepared = executor.prepare(
        "extract_action_items",
        {"text": "Priya will submit the report by Friday."},
    )

    result = executor.execute(prepared)

    assert isinstance(result.output, ExtractActionItemsOutput)
    assert result.output.action_items[0].owner == "Priya"


@pytest.mark.parametrize(
    "invalid_output",
    [
        '{"summary":',
        json.dumps({"summary": 42, "key_points": []}),
        json.dumps(
            {
                "summary": "Penguins cross Antarctic ice.",
                "key_points": ["Ocean temperatures affect migration."],
            }
        ),
    ],
    ids=["parsing", "structural", "semantic"],
)
def test_invalid_primary_output_receives_one_successful_repair(
    invalid_output: str,
) -> None:
    provider = ScriptedProviderGateway()
    provider.queue(
        "groq",
        _completion("groq", invalid_output, 4, 2),
        _completion("groq", _summary_json(), 6, 3),
    )
    executor = _executor(provider)

    result = executor.execute(_prepared_summary(executor))

    assert result.attempts == 2
    assert result.provider == "groq"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert [name for name, _ in provider.calls] == ["groq", "groq"]


def test_invalid_primary_and_invalid_repair_fail_without_fallback() -> None:
    provider = ScriptedProviderGateway()
    provider.queue(
        "groq",
        _completion("groq", "not json", 3, 1),
        _completion("groq", '{"still":"invalid"}', 4, 2),
    )
    executor = _executor(provider)

    with pytest.raises(TaskInvalidOutputError) as captured:
        executor.execute(_prepared_summary(executor))

    assert captured.value.attempts == 2
    assert len(captured.value.completion_usage) == 2
    assert [name for name, _ in provider.calls] == ["groq", "groq"]


def test_primary_operational_failure_uses_fallback() -> None:
    provider = ScriptedProviderGateway()
    provider.queue("groq", ProviderOperationalError("controlled"))
    provider.queue("gemini", _completion("gemini", _summary_json(), 8, 5))
    executor = _executor(provider)

    result = executor.execute(_prepared_summary(executor))

    assert result.provider == "gemini"
    assert result.attempts == 2
    assert result.usage.prompt_tokens == 8
    assert result.usage.completion_tokens == 5
    assert [name for name, _ in provider.calls] == ["groq", "gemini"]


def test_legacy_provider_error_is_treated_as_operational() -> None:
    provider = ScriptedProviderGateway()
    provider.queue("groq", ProviderError("legacy failure"))
    provider.queue("gemini", _completion("gemini", _summary_json()))
    executor = _executor(provider)

    result = executor.execute(_prepared_summary(executor))

    assert result.provider == "gemini"
    assert result.attempts == 2


def test_invalid_fallback_output_receives_fallback_repair() -> None:
    provider = ScriptedProviderGateway()
    provider.queue("groq", ProviderOperationalError("controlled"))
    provider.queue(
        "gemini",
        _completion("gemini", "not json", 6, 2),
        _completion("gemini", _summary_json(), 9, 4),
    )
    executor = _executor(provider)

    result = executor.execute(_prepared_summary(executor))

    assert result.provider == "gemini"
    assert result.attempts == 3
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 6
    assert [name for name, _ in provider.calls] == [
        "groq",
        "gemini",
        "gemini",
    ]


def test_primary_repair_operational_failure_uses_original_prompt_for_fallback() -> None:
    provider = ScriptedProviderGateway()
    provider.queue(
        "groq",
        _completion("groq", "not json", 4, 2),
        ProviderOperationalError("repair unavailable"),
    )
    provider.queue("gemini", _completion("gemini", _summary_json(), 7, 3))
    executor = _executor(provider)
    prepared = _prepared_summary(executor)

    result = executor.execute(prepared)

    assert result.attempts == 3
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 5
    assert provider.calls[2][1] == [dict(message) for message in prepared.messages]


def test_maximum_four_call_path_succeeds_and_aggregates_three_completions() -> None:
    provider = ScriptedProviderGateway()
    provider.queue(
        "groq",
        _completion("groq", "not json", 2, 1),
        ProviderOperationalError("repair unavailable"),
    )
    provider.queue(
        "gemini",
        _completion("gemini", '{"summary":7,"key_points":[]}', 3, 2),
        _completion("gemini", _summary_json(), 5, 4),
    )
    executor = _executor(provider)

    result = executor.execute(_prepared_summary(executor))

    assert result.attempts == 4
    assert len(provider.calls) == 4
    assert len(result.completion_usage) == 3
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 7


def test_maximum_four_call_path_never_makes_a_fifth_call() -> None:
    provider = ScriptedProviderGateway()
    provider.queue(
        "groq",
        _completion("groq", "not json"),
        ProviderOperationalError("repair unavailable"),
    )
    provider.queue(
        "gemini",
        _completion("gemini", "still not json"),
        _completion("gemini", '{"wrong":"shape"}'),
        _completion("gemini", _summary_json()),
    )
    executor = _executor(provider)

    with pytest.raises(TaskInvalidOutputError) as captured:
        executor.execute(_prepared_summary(executor))

    assert captured.value.attempts == 4
    assert len(provider.calls) == 4
    assert len(provider.events["gemini"]) == 1


def test_provider_configuration_error_stops_without_fallback() -> None:
    provider = ScriptedProviderGateway()
    provider.queue("groq", ProviderConfigurationError("secret details"))
    provider.queue("gemini", _completion("gemini", _summary_json()))
    executor = _executor(provider)

    with pytest.raises(TaskProviderConfigurationError) as captured:
        executor.execute(_prepared_summary(executor))

    assert captured.value.attempts == 1
    assert captured.value.completion_usage == ()
    assert [name for name, _ in provider.calls] == ["groq"]
    assert "secret details" not in str(captured.value)


def test_repair_prompt_is_safe_bounded_and_does_not_mutate_original() -> None:
    provider = ScriptedProviderGateway()
    invalid_output = json.dumps(
        {
            "summary": 7,
            "key_points": [],
            "padding": "x" * 3_000,
        }
    )
    provider.queue(
        "groq",
        _completion("groq", invalid_output),
        _completion("groq", _summary_json()),
    )
    executor = _executor(provider)
    prepared = _prepared_summary(executor)
    original_messages = copy.deepcopy(prepared.messages)

    executor.execute(prepared)

    repair_messages = provider.calls[1][1]
    system_text = "\n".join(
        message["content"]
        for message in repair_messages
        if message["role"] == "system"
    )
    repair_data_text = repair_messages[-1]["content"].split("\n", 1)[1]
    repair_data = json.loads(repair_data_text)

    assert "summary: expected string" in system_text
    assert "REQUIRED_OUTPUT_SCHEMA_JSON:" in system_text
    assert '"summary"' in system_text
    assert "Return corrected valid JSON only." in system_text
    assert "Do not wrap JSON in Markdown fences." in system_text
    assert len(repair_data["excerpt"]) == INVALID_OUTPUT_EXCERPT_LIMIT
    assert repair_data["truncated"] is True
    assert invalid_output not in system_text
    assert provider.api_key not in "\n".join(
        message["content"] for message in repair_messages
    )
    assert prepared.messages == original_messages


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "database_url": (
            "postgresql+psycopg://unused:unused@127.0.0.1:1/unused"
        ),
        "groq_api_key": "fake-groq-key",
        "groq_model": "gateway-model",
        "gemini_api_key": "fake-gemini-key",
        "gemini_model": "fallback-model",
        "provider_timeout_seconds": 1.0,
        "force_primary_fail": False,
    }
    values.update(changes)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ProviderConfigurationError),
        (400, ProviderConfigurationError),
        (429, ProviderOperationalError),
        (503, ProviderOperationalError),
    ],
)
def test_provider_http_statuses_are_classified(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    error_type: type[ProviderError],
) -> None:
    class FakeResponse:
        is_success = False

        def __init__(self) -> None:
            self.status_code = status_code

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("app.providers.httpx.post", fake_post)
    gateway = ProviderGateway(_settings())

    with pytest.raises(error_type):
        gateway.complete_with_provider(
            "groq",
            [{"role": "user", "content": "Hello"}],
        )


def test_missing_primary_configuration_fails_without_http_call() -> None:
    gateway = ProviderGateway(_settings(groq_api_key=""))

    with pytest.raises(ProviderConfigurationError):
        gateway.complete_with_provider(
            "groq",
            [{"role": "user", "content": "Hello"}],
        )
