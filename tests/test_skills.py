from __future__ import annotations

from pathlib import Path

import pytest

from app.skills import (
    MalformedSkillError,
    SkillLoadError,
    SkillLoader,
    SkillRegistry,
    UnknownSkillError,
    UnsafeSkillPathError,
)


VALID_SKILL_YAML = """
name: custom
purpose: A test-only skill definition.
system_instructions: Follow the supplied test instructions.
expected_input:
  type: object
  properties:
    text:
      type: string
  required:
    - text
  additionalProperties: false
output_schema:
  type: object
  properties:
    result:
      type: string
  required:
    - result
  additionalProperties: false
validation_rules:
  - The result must contain text.
maximum_repair_attempts: 1
""".strip()


def _custom_loader(root: Path, name: str = "custom") -> SkillLoader:
    registry = SkillRegistry({name: Path(name) / "skill.yaml"})
    return SkillLoader(root, registry)


def _write_custom_skill(
    root: Path,
    yaml_text: str = VALID_SKILL_YAML,
    name: str = "custom",
) -> Path:
    skill_path = root / name / "skill.yaml"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(yaml_text, encoding="utf-8")
    return skill_path


@pytest.mark.parametrize(
    ("name", "output_field"),
    [
        ("summarize", "summary"),
        ("extract_action_items", "action_items"),
    ],
)
def test_known_local_skill_loads_successfully(
    name: str,
    output_field: str,
) -> None:
    loader = SkillLoader()

    definition = loader.load(name)

    assert definition.name == name
    assert definition.purpose
    assert definition.system_instructions
    assert definition.maximum_repair_attempts == 1
    assert output_field in definition.output_schema.properties
    assert definition.output_schema.additional_properties is False


def test_default_registry_contains_exactly_the_two_mvp_skills() -> None:
    loader = SkillLoader()

    assert loader.available_skills == ("extract_action_items", "summarize")


def test_successfully_loaded_skill_is_cached(tmp_path: Path) -> None:
    skill_path = _write_custom_skill(tmp_path)
    loader = _custom_loader(tmp_path)

    first = loader.load("custom")
    skill_path.write_text("this: [is no longer valid", encoding="utf-8")
    second = loader.load("custom")

    assert second is first


@pytest.mark.parametrize(
    "name",
    ["unknown", "../summarize", "summarize/../extract_action_items", "C:\\skill"],
)
def test_unknown_or_path_like_skill_name_is_rejected(name: str) -> None:
    loader = SkillLoader()

    with pytest.raises(UnknownSkillError, match="Unknown skill"):
        loader.load(name)


def test_registered_path_cannot_escape_skills_root(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    outside_file = tmp_path / "outside.yaml"
    outside_file.write_text(
        VALID_SKILL_YAML.replace("name: custom", "name: escape"),
        encoding="utf-8",
    )
    registry = SkillRegistry({"escape": Path("..") / "outside.yaml"})
    loader = SkillLoader(skills_root, registry)

    with pytest.raises(UnsafeSkillPathError, match="leaves the skills directory"):
        loader.load("escape")


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    _write_custom_skill(tmp_path, "name: custom\npurpose: [unterminated")
    loader = _custom_loader(tmp_path)

    with pytest.raises(MalformedSkillError, match="malformed or unsafe YAML"):
        loader.load("custom")


def test_unsafe_yaml_tag_is_rejected_without_execution(tmp_path: Path) -> None:
    unsafe_yaml = VALID_SKILL_YAML.replace(
        "purpose: A test-only skill definition.",
        "purpose: !!python/name:builtins.str",
    )
    _write_custom_skill(tmp_path, unsafe_yaml)
    loader = _custom_loader(tmp_path)

    with pytest.raises(MalformedSkillError, match="malformed or unsafe YAML"):
        loader.load("custom")


def test_structurally_invalid_skill_is_rejected(tmp_path: Path) -> None:
    invalid_yaml = VALID_SKILL_YAML.replace(
        "maximum_repair_attempts: 1",
        "maximum_repair_attempts: unlimited",
    )
    _write_custom_skill(tmp_path, invalid_yaml)
    loader = _custom_loader(tmp_path)

    with pytest.raises(MalformedSkillError, match="invalid structure"):
        loader.load("custom")


def test_skill_file_name_must_match_registered_name(tmp_path: Path) -> None:
    _write_custom_skill(
        tmp_path,
        VALID_SKILL_YAML.replace("name: custom", "name: different"),
    )
    loader = _custom_loader(tmp_path)

    with pytest.raises(MalformedSkillError, match="declares the name 'different'"):
        loader.load("custom")


def test_missing_registered_skill_file_has_clear_error(tmp_path: Path) -> None:
    loader = _custom_loader(tmp_path)

    with pytest.raises(SkillLoadError, match="does not exist"):
        loader.load("custom")


def test_failed_load_is_not_cached(tmp_path: Path) -> None:
    skill_path = _write_custom_skill(tmp_path, "name: custom\npurpose: [unterminated")
    loader = _custom_loader(tmp_path)

    with pytest.raises(MalformedSkillError):
        loader.load("custom")

    skill_path.write_text(VALID_SKILL_YAML, encoding="utf-8")
    definition = loader.load("custom")

    assert definition.name == "custom"
