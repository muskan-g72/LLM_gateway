from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings


class ProviderError(Exception):
    """A deliberately detail-free provider error so credentials cannot leak."""


class ProviderOperationalError(ProviderError):
    """A transient or unusable provider operation that may justify fallback."""


class ProviderConfigurationError(ProviderError):
    """A local credential, model, or request configuration problem."""


@dataclass(frozen=True)
class ProviderCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    provider: str


class ProviderGateway:
    PRIMARY_PROVIDER = "groq"
    FALLBACK_PROVIDER = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        try:
            return self.complete_with_provider(self.PRIMARY_PROVIDER, messages)
        except ProviderError:
            return self.complete_with_provider(self.FALLBACK_PROVIDER, messages)

    def complete_with_provider(
        self,
        provider_name: str,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        if provider_name == self.PRIMARY_PROVIDER:
            if self.settings.force_primary_fail:
                raise ProviderOperationalError(
                    "primary disabled by failure injection"
                )
            return self._complete_with_groq(messages)
        if provider_name == self.FALLBACK_PROVIDER:
            return self._complete_with_gemini(messages)
        raise ProviderConfigurationError("unknown provider selection")

    @staticmethod
    def _raise_for_failed_status(response: httpx.Response) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code in {408, 429} or (
            isinstance(status_code, int) and status_code >= 500
        ):
            raise ProviderOperationalError("provider request failed operationally")
        raise ProviderConfigurationError("provider request configuration was rejected")

    def _complete_with_groq(
        self,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        if not self.settings.groq_api_key:
            raise ProviderConfigurationError("primary provider is not configured")
        if not self.settings.groq_model:
            raise ProviderConfigurationError("primary model is not configured")

        try:
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.groq_model,
                    "messages": messages,
                    "stream": False,
                    "max_completion_tokens": 512,
                    "reasoning_effort": "low",
                },
                timeout=self.settings.provider_timeout_seconds,
            )
            if not response.is_success:
                self._raise_for_failed_status(response)

            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            prompt_tokens = int(payload["usage"]["prompt_tokens"])
            completion_tokens = int(payload["usage"]["completion_tokens"])
            if not isinstance(content, str) or not content.strip():
                raise ProviderOperationalError("primary provider returned no text")
            if prompt_tokens <= 0 or completion_tokens <= 0:
                raise ProviderOperationalError(
                    "primary provider returned invalid usage"
                )
        except ProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            raise ProviderOperationalError(
                "primary provider response was unusable"
            ) from None

        return ProviderCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider="groq",
        )

    def _complete_with_gemini(
        self,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        if not self.settings.gemini_api_key:
            raise ProviderConfigurationError("fallback provider is not configured")
        if not self.settings.gemini_model:
            raise ProviderConfigurationError("fallback model is not configured")

        system_parts = [
            {"text": message["content"]}
            for message in messages
            if message["role"] == "system"
        ]
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]
        body: dict[str, object] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 512},
        }
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        try:
            response = httpx.post(
                (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.settings.gemini_model}:generateContent"
                ),
                headers={
                    "x-goog-api-key": self.settings.gemini_api_key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.settings.provider_timeout_seconds,
            )
            if not response.is_success:
                self._raise_for_failed_status(response)

            payload = response.json()
            parts = payload["candidates"][0]["content"]["parts"]
            content = "".join(
                part["text"]
                for part in parts
                if isinstance(part.get("text"), str) and not part.get("thought", False)
            )
            usage = payload["usageMetadata"]
            prompt_tokens = int(usage["promptTokenCount"])
            completion_tokens = int(usage["candidatesTokenCount"])
            if not content.strip():
                raise ProviderOperationalError("fallback provider returned no text")
            if prompt_tokens <= 0 or completion_tokens <= 0:
                raise ProviderOperationalError(
                    "fallback provider returned invalid usage"
                )
        except ProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            raise ProviderOperationalError(
                "fallback provider response was unusable"
            ) from None

        return ProviderCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider="gemini",
        )
