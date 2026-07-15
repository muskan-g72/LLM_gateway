from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings


class ProviderError(Exception):
    """A deliberately detail-free provider error so credentials cannot leak."""


@dataclass(frozen=True)
class ProviderCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    provider: str


class ProviderGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> ProviderCompletion:
        try:
            if self.settings.force_primary_fail:
                raise ProviderError("primary disabled by failure injection")
            return self._complete_with_groq(messages)
        except ProviderError:
            return self._complete_with_gemini(messages)

    def _complete_with_groq(
        self,
        messages: list[dict[str, str]],
    ) -> ProviderCompletion:
        if not self.settings.groq_api_key:
            raise ProviderError("primary provider is not configured")

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
                raise ProviderError("primary provider request failed")

            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            prompt_tokens = int(payload["usage"]["prompt_tokens"])
            completion_tokens = int(payload["usage"]["completion_tokens"])
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("primary provider returned no text")
            if prompt_tokens <= 0 or completion_tokens <= 0:
                raise ProviderError("primary provider returned invalid usage")
        except ProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("primary provider response was unusable") from None

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
            raise ProviderError("fallback provider is not configured")

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
                raise ProviderError("fallback provider request failed")

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
                raise ProviderError("fallback provider returned no text")
            if prompt_tokens <= 0 or completion_tokens <= 0:
                raise ProviderError("fallback provider returned invalid usage")
        except ProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("fallback provider response was unusable") from None

        return ProviderCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider="gemini",
        )