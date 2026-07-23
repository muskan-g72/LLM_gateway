from __future__ import annotations

import json
import math
from copy import deepcopy
from types import MappingProxyType
from typing import Annotated, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from app.skills import ObjectSchema, SkillError, SkillLoader, ToolName
from app.tools import ToolError, ToolRegistry


MAX_WORKFLOW_STEPS = 8
WorkflowName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]
WorkflowSource = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^(input\.[a-z][a-z0-9_]{0,63}|"
            r"steps\.[a-z][a-z0-9_]{0,63}\.[a-z][a-z0-9_]{0,63})$"
        )
    ),
]
WorkflowSources = Annotated[tuple[WorkflowSource, ...], Field(min_length=1)]


class WorkflowError(Exception):
    """Base error for safe workflow definition and mapping failures."""


class UnknownWorkflowError(WorkflowError):
    """The requested workflow is not in the trusted registry."""


class WorkflowDefinitionError(WorkflowError):
    """A trusted workflow definition is internally inconsistent."""


class WorkflowInputError(WorkflowError):
    """Untrusted workflow input cannot satisfy a declared deterministic mapping."""


class WorkflowStepDefinition(BaseModel):
    """One fixed sequential step in a trusted workflow."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    step_id: WorkflowName
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    skill: WorkflowName
    tool: ToolName | None = None
    input_mapping: dict[WorkflowName, WorkflowSources]
    output_schema: ObjectSchema

    @model_validator(mode="after")
    def mapping_sources_must_be_unique(self) -> "WorkflowStepDefinition":
        for sources in self.input_mapping.values():
            if len(sources) != len(set(sources)):
                raise ValueError("input mapping sources must not contain duplicates")
        return self


class WorkflowDefinition(BaseModel):
    """A bounded application-owned sequence; models cannot alter this structure."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: WorkflowName
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    steps: tuple[WorkflowStepDefinition, ...] = Field(
        min_length=1,
        max_length=MAX_WORKFLOW_STEPS,
    )

    @model_validator(mode="after")
    def steps_must_be_unique_and_forward_only(self) -> "WorkflowDefinition":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow step IDs must be unique")

        available_steps: dict[str, WorkflowStepDefinition] = {}
        for step in self.steps:
            for sources in step.input_mapping.values():
                for source in sources:
                    if source.startswith("input."):
                        continue
                    _, source_step_id, source_field = source.split(".")
                    source_step = available_steps.get(source_step_id)
                    if source_step is None:
                        raise ValueError(
                            "step mappings may reference only earlier workflow steps"
                        )
                    if source_field not in source_step.output_schema.properties:
                        raise ValueError(
                            "step mapping references an undeclared output field"
                        )
            available_steps[step.step_id] = step
        return self


def _ensure_json_value(
    value: object,
    active_containers: set[int] | None = None,
) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise WorkflowInputError("workflow input contains a non-finite number")
        return
    if type(value) not in {dict, list}:
        raise WorkflowInputError("workflow input contains a non-JSON value")

    active = active_containers if active_containers is not None else set()
    identity = id(value)
    if identity in active:
        raise WorkflowInputError("workflow input contains a circular reference")
    active.add(identity)
    try:
        if type(value) is list:
            for item in value:
                _ensure_json_value(item, active)
            return
        for key, item in value.items():
            if type(key) is not str:
                raise WorkflowInputError("workflow input contains a non-string key")
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
    except WorkflowInputError:
        raise
    except (RecursionError, TypeError, ValueError, OverflowError):
        raise WorkflowInputError("workflow mapped input could not be serialized") from None


def validated_workflow_input(value: object) -> dict[str, object]:
    """Return a detached JSON-safe workflow input object."""
    if type(value) is not dict:
        raise WorkflowInputError("workflow input must be a JSON object")
    _canonical_json(value)
    return deepcopy(value)


def resolve_step_input(
    step: WorkflowStepDefinition,
    original_input: dict[str, object],
    prior_outputs: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """Resolve a fixed shallow mapping without evaluating expressions or code."""
    resolved: dict[str, object] = {}
    for target, sources in step.input_mapping.items():
        source_values: dict[str, object] = {}
        for source in sources:
            parts = source.split(".")
            if parts[0] == "input":
                source_field = parts[1]
                if source_field not in original_input:
                    raise WorkflowInputError(
                        f"workflow input is missing mapped field '{source_field}'"
                    )
                source_values[source] = deepcopy(original_input[source_field])
                continue

            source_step_id, source_field = parts[1], parts[2]
            source_output = prior_outputs.get(source_step_id)
            if source_output is None or source_field not in source_output:
                raise WorkflowDefinitionError(
                    "validated workflow output is unavailable for a later step"
                )
            source_values[source] = deepcopy(source_output[source_field])

        if len(source_values) == 1:
            resolved[target] = next(iter(source_values.values()))
        else:
            resolved[target] = _canonical_json(source_values)
    return resolved


class WorkflowRegistry:
    """Immutable registry of fully validated application-owned workflows."""

    def __init__(
        self,
        definitions: tuple[WorkflowDefinition, ...],
        skill_loader: SkillLoader,
        tool_registry: ToolRegistry,
    ) -> None:
        registered: dict[str, WorkflowDefinition] = {}
        for definition in definitions:
            if definition.id in registered:
                raise WorkflowDefinitionError("duplicate workflow ID")
            self._validate_definition(definition, skill_loader, tool_registry)
            registered[definition.id] = definition
        self._definitions: Mapping[str, WorkflowDefinition] = MappingProxyType(
            registered
        )

    @staticmethod
    def _validate_definition(
        definition: WorkflowDefinition,
        skill_loader: SkillLoader,
        tool_registry: ToolRegistry,
    ) -> None:
        for step in definition.steps:
            try:
                skill = skill_loader.load(step.skill)
            except SkillError:
                raise WorkflowDefinitionError(
                    "workflow references an unavailable skill"
                ) from None

            expected_schema = skill.output_schema.model_dump(
                mode="json",
                by_alias=True,
            )
            declared_schema = step.output_schema.model_dump(
                mode="json",
                by_alias=True,
            )
            if declared_schema != expected_schema:
                raise WorkflowDefinitionError(
                    "workflow output schema does not match its skill"
                )

            mapped_fields = set(step.input_mapping)
            declared_inputs = set(skill.expected_input.properties)
            required_inputs = set(skill.expected_input.required)
            if not required_inputs.issubset(mapped_fields):
                raise WorkflowDefinitionError(
                    "workflow mapping omits a required skill input"
                )
            if not mapped_fields.issubset(declared_inputs):
                raise WorkflowDefinitionError(
                    "workflow mapping targets an undeclared skill input"
                )

            for target, sources in step.input_mapping.items():
                target_schema = skill.expected_input.properties[target]
                if len(sources) > 1 and target_schema.get("type") != "string":
                    raise WorkflowDefinitionError(
                        "multi-source mappings require a string target"
                    )

            if step.tool is not None:
                if step.tool not in skill.allowed_tools:
                    raise WorkflowDefinitionError(
                        "workflow tool is not allowed by its skill"
                    )
                try:
                    tool_registry.get(step.tool)
                except ToolError:
                    raise WorkflowDefinitionError(
                        "workflow references an unavailable tool"
                    ) from None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def get(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._definitions[workflow_id].model_copy(deep=True)
        except KeyError:
            raise UnknownWorkflowError("unknown workflow") from None


def build_builtin_workflow_registry(
    skill_loader: SkillLoader,
    tool_registry: ToolRegistry,
) -> WorkflowRegistry:
    summarize = skill_loader.load("summarize")
    action_items = skill_loader.load("extract_action_items")
    article_processing = WorkflowDefinition(
        id="article_processing",
        name="Article processing",
        description=(
            "Summarize an article, extract its action items, and produce a "
            "statistics-assisted final report."
        ),
        steps=(
            WorkflowStepDefinition(
                step_id="summary",
                name="Summarize article",
                skill="summarize",
                input_mapping={"text": ("input.text",)},
                output_schema=summarize.output_schema,
            ),
            WorkflowStepDefinition(
                step_id="action_items",
                name="Extract action items",
                skill="extract_action_items",
                input_mapping={
                    "text": (
                        "input.text",
                        "steps.summary.summary",
                    )
                },
                output_schema=action_items.output_schema,
            ),
            WorkflowStepDefinition(
                step_id="final_report",
                name="Generate statistics-assisted report",
                skill="summarize",
                tool="text_statistics",
                input_mapping={
                    "text": (
                        "input.text",
                        "steps.summary.summary",
                        "steps.action_items.action_items",
                    )
                },
                output_schema=summarize.output_schema,
            ),
        ),
    )
    return WorkflowRegistry(
        (article_processing,),
        skill_loader,
        tool_registry,
    )
