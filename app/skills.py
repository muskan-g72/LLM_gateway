from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ToolName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]


class SkillError(Exception):
    """Base exception for safe, application-level skill failures."""


class UnknownSkillError(SkillError):
    """Raised when a requested name is not in the local allowlist."""


class SkillLoadError(SkillError):
    """Raised when a known skill file cannot be loaded safely."""


class MalformedSkillError(SkillLoadError):
    """Raised when a skill file is not valid YAML or has an invalid structure."""


class UnsafeSkillPathError(SkillLoadError):
    """Raised when a registered path resolves outside the skills directory."""


class ObjectSchema(BaseModel):
    """The small JSON-schema subset used by local skill definitions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)

    type: Literal["object"]
    properties: dict[str, dict[str, Any]] = Field(min_length=1)
    required: list[NonEmptyText] = Field(min_length=1)
    additional_properties: Literal[False] = Field(alias="additionalProperties")

    @model_validator(mode="after")
    def required_fields_must_be_declared(self) -> "ObjectSchema":
        if any(not name.strip() for name in self.properties):
            raise ValueError("property names must contain text")
        if len(self.required) != len(set(self.required)):
            raise ValueError("required fields must not contain duplicates")

        undeclared = sorted(set(self.required) - set(self.properties))
        if undeclared:
            names = ", ".join(undeclared)
            raise ValueError(f"required fields are not declared in properties: {names}")
        return self


class SkillDefinition(BaseModel):
    """Validated, declarative instructions for one registered task type."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonEmptyText
    purpose: NonEmptyText
    system_instructions: NonEmptyText
    expected_input: ObjectSchema
    output_schema: ObjectSchema
    validation_rules: list[NonEmptyText] = Field(min_length=1)
    maximum_repair_attempts: int = Field(ge=0, le=1)
    allowed_tools: list[ToolName] = Field(default_factory=list)

    @model_validator(mode="after")
    def allowed_tool_names_must_be_unique(self) -> "SkillDefinition":
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed_tools must not contain duplicates")
        return self


DEFAULT_SKILL_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "summarize": Path("summarize") / "skill.yaml",
        "extract_action_items": Path("extract_action_items") / "skill.yaml",
    }
)


class SkillRegistry:
    """Map public skill names to fixed local paths without directory scanning."""

    def __init__(self, paths: Mapping[str, str | Path] | None = None) -> None:
        configured_paths = DEFAULT_SKILL_PATHS if paths is None else paths
        self._paths: Mapping[str, Path] = MappingProxyType(
            {name: Path(path) for name, path in configured_paths.items()}
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def path_for(self, name: str) -> Path:
        try:
            return self._paths[name]
        except KeyError:
            known = ", ".join(self.names)
            raise UnknownSkillError(
                f"Unknown skill '{name}'. Known skills: {known}"
            ) from None


class SkillLoader:
    """Safely load, validate, and cache allowlisted YAML skill definitions."""

    def __init__(
        self,
        skills_root: str | Path | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        default_root = Path(__file__).resolve().parent.parent / "skills"
        self._skills_root = Path(skills_root or default_root).resolve()
        self._registry = registry or SkillRegistry()
        self._cache: dict[str, SkillDefinition] = {}
        self._cache_lock = Lock()

    @property
    def available_skills(self) -> tuple[str, ...]:
        return self._registry.names

    def load(self, name: str) -> SkillDefinition:
        with self._cache_lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached

            definition = self._load_uncached(name)
            self._cache[name] = definition
            return definition

    def _load_uncached(self, name: str) -> SkillDefinition:
        relative_path = self._registry.path_for(name)
        skill_path = (self._skills_root / relative_path).resolve()

        try:
            skill_path.relative_to(self._skills_root)
        except ValueError:
            raise UnsafeSkillPathError(
                f"Registered path for skill '{name}' leaves the skills directory"
            ) from None

        if not skill_path.is_file():
            raise SkillLoadError(f"Skill file for '{name}' does not exist")

        try:
            yaml_text = skill_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise SkillLoadError(f"Skill file for '{name}' could not be read") from None

        try:
            raw_definition = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            raise MalformedSkillError(
                f"Skill '{name}' contains malformed or unsafe YAML"
            ) from None

        try:
            json.dumps(raw_definition, allow_nan=False)
        except (TypeError, ValueError):
            raise MalformedSkillError(
                f"Skill '{name}' must contain only JSON-compatible YAML values"
            ) from None

        try:
            definition = SkillDefinition.model_validate(raw_definition)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors(include_url=False, include_input=False)
            )
            raise MalformedSkillError(
                f"Skill '{name}' has an invalid structure: {details}"
            ) from None

        if definition.name != name:
            raise MalformedSkillError(
                f"Skill file for '{name}' declares the name '{definition.name}'"
            )

        return definition
