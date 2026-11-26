"""Opt-in information restrictions; importing this module changes no defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Restrictions:
    hide_task_feedback: bool = False
    disable_success_stop: bool = False
    disable_task_memory: bool = False
    disable_global_memory: bool = False
    disable_bootstrap: bool = False
    disable_privileged_grasp: bool = False
    task_language_names_only: bool = False
    restrict_file_access: bool = False
    disable_flash: bool = False
    retain_task_independent_guidance: bool = False
    restrict_observation_fields: bool = False

    @classmethod
    def restricted(cls) -> "Restrictions":
        # Preserve the original all-memory-off profile for existing manifests.
        return cls(**{f.name: True for f in fields(cls)
                      if f.name != "retain_task_independent_guidance"})

    @classmethod
    def from_mapping(cls, value: Mapping, *, restricted: bool = False) -> "Restrictions":
        if not isinstance(value, Mapping):
            raise ValueError("Restriction configuration must be a mapping")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown restriction flags: {sorted(unknown)}")
        if any(type(v) is not bool for v in value.values()):
            raise ValueError("Restriction values must be JSON booleans")
        initial = asdict(cls.restricted() if restricted else cls())
        initial.update(value)
        return cls(**initial)

    @property
    def enabled(self) -> bool:
        return any(asdict(self).values())

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


def get_restrictions(environ: Mapping[str, str] | None = None) -> Restrictions:
    """Read an aggregate profile followed by explicit independent overrides.

    JSON is used to avoid a new parser dependency. JSON config is valid YAML
    too, so the shipped restricted.yaml can be consumed by this loader.
    """
    env = os.environ if environ is None else environ
    switch = env.get("STAGE_RESTRICTED", "0").lower()
    if switch not in {"0", "1", "false", "true"}:
        raise ValueError("STAGE_RESTRICTED must be 0, 1, false or true")
    path = env.get("STAGE_RESTRICTIONS_FILE")
    value = json.loads(Path(path).read_text()) if path else {}
    return Restrictions.from_mapping(value, restricted=switch in {"1", "true"})
