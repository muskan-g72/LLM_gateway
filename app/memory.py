from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Protocol


PREFERENCE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_PREFERENCE_VALUE_BYTES = 4_096


class PreferenceError(ValueError):
    """Base error for safe reusable-preference validation failures."""


class InvalidPreferenceNameError(PreferenceError):
    """A preference name is outside the small public naming contract."""


class InvalidPreferenceValueError(PreferenceError):
    """A preference value is unsafe, unsupported, or too large to persist."""


class PreferenceRepository(Protocol):
    def get_preference_values(self, virtual_key_id: str) -> dict[str, str]: ...

    def upsert_preference_values(
        self,
        virtual_key_id: str,
        values: dict[str, str],
    ) -> None: ...

    def delete_preference_value(
        self,
        virtual_key_id: str,
        preference_key: str,
    ) -> bool: ...


def _validate_json_value(
    value: object,
    path: str = "$",
    active_containers: set[int] | None = None,
) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidPreferenceValueError(
                f"preference value contains a non-finite number at {path}"
            )
        return
    if type(value) not in {dict, list}:
        raise InvalidPreferenceValueError(
            f"preference value is not JSON-compatible at {path}"
        )

    active = active_containers if active_containers is not None else set()
    identity = id(value)
    if identity in active:
        raise InvalidPreferenceValueError(
            f"preference value contains a circular reference at {path}"
        )
    active.add(identity)
    try:
        if type(value) is list:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active)
            return

        for key, item in value.items():
            if type(key) is not str:
                raise InvalidPreferenceValueError(
                    f"preference value has a non-string object key at {path}"
                )
            _validate_json_value(item, f"{path}.{key}", active)
    finally:
        active.remove(identity)


def _canonical_preference_value(value: object) -> str:
    try:
        _validate_json_value(value)
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except RecursionError:
        raise InvalidPreferenceValueError(
            "preference value is nested too deeply"
        ) from None
    except (TypeError, ValueError, OverflowError):
        raise InvalidPreferenceValueError(
            "preference value could not be serialized"
        ) from None

    if len(serialized.encode("utf-8")) > MAX_PREFERENCE_VALUE_BYTES:
        raise InvalidPreferenceValueError(
            f"preference value exceeds {MAX_PREFERENCE_VALUE_BYTES} bytes"
        )
    return serialized


class PreferenceService:
    """Validate and persist explicit reusable settings, never chat history."""

    def __init__(self, repository: PreferenceRepository) -> None:
        self._repository = repository

    @staticmethod
    def validate_name(name: str) -> None:
        if type(name) is not str or PREFERENCE_NAME_PATTERN.fullmatch(name) is None:
            raise InvalidPreferenceNameError("invalid preference name")

    def get(self, virtual_key_id: str) -> dict[str, object]:
        stored = self._repository.get_preference_values(virtual_key_id)
        return {name: json.loads(value) for name, value in stored.items()}

    def put(
        self,
        virtual_key_id: str,
        preferences: dict[str, object],
    ) -> dict[str, object]:
        serialized: dict[str, str] = {}
        for name, value in preferences.items():
            self.validate_name(name)
            serialized[name] = _canonical_preference_value(value)

        self._repository.upsert_preference_values(virtual_key_id, serialized)
        return self.get(virtual_key_id)

    def delete(self, virtual_key_id: str, preference_key: str) -> bool:
        self.validate_name(preference_key)
        return self._repository.delete_preference_value(
            virtual_key_id,
            preference_key,
        )

    @staticmethod
    def merge(
        stored: dict[str, object],
        requested: dict[str, object] | None,
    ) -> dict[str, object]:
        merged = deepcopy(stored)
        if requested is not None:
            merged.update(deepcopy(requested))
        return merged
