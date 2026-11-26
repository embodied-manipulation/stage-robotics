# Copyright 2026 STAGE Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit public artifacts and recorded planner requests without reading private scores.

Run after an episode, for example::

    python -m research.check_public_boundary --episode /path/to/episode

The report is private. Exact evaluator keys, prohibited prior/criteria files,
and emitted paths escaping the public root fail the check. Ordinary prose and
model-authored attempted paths are review findings, not proof of delivered data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse


# Current native projection plus the research evaluator's actual field names.
# Do not blacklist generic "memory", policy reset_memory, or history fields.
ORACLE_KEYS = frozenset({
    "success", "task_progress", "robocasa_terminated", "terminated", "truncated",
    "done", "reward", "success_criteria", "success_criteria_text",
    "environment_success", "environment_result_available", "benchmark_success",
    "official_success", "official_any", "official_process_any", "official_terminal",
    "first_success_step", "final_predicate_states", "evaluator_locals",
    "physical_success", "success_source", "grasped", "grasp_detected",
    "grasp_contact", "grasp_obj", "object_names", "object_registry",
    "object_id", "object_ids", "obj_id", "obj_ids", "objectid",
})
_PRIOR_FILES = {"memory.json", "memory_reads.jsonl", "memory.md", "global_memory.md"}
_PATH_KEYS = {
    "path", "file", "filename", "directory", "cwd", "output_dir", "artifact",
    "artifacts", "run_artifacts", "files", "image_path", "video_path", "file_path",
    "workspace_roots", "uri", "image_url",
}
_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".npz", ".npy", ".mp4", ".bin", ".wav", ".mp3"}
_PATH_TOKEN = re.compile(r"(?<![\w:<])(?:file:///|/|~/|\.\.?/)[^\s<>\"'`{},;()\[\]]+")
_WORDS = re.compile(r"success|memory", re.IGNORECASE)
_PRIOR_TOKEN = re.compile(
    r"(?<![\w])(?:success_criteria\.(?:md|txt|json)|(?:global_)?memory\.(?:md|json)|memory_reads\.jsonl)\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```(?:jsonl?|JSONL?)?\s*\n(.*?)```", re.DOTALL)
_JSON_START = re.compile(r"[\[{]")


def _prior_reference(path: str) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    name = parts[-1] if parts else ""
    return (
        name in _PRIOR_FILES
        or name.startswith("success_criteria.")
        or "task-specific" in parts
        or "task-family" in parts
        or ("references" in parts and "global" in parts)
        or bool(re.fullmatch(r".+_s0_recipe\.jsonl", name))
    )


def _embedded_json(text: str):
    """Decode complete, fenced, or embedded JSON objects without executing code."""
    stripped = text.strip()
    if not stripped or stripped.startswith("data:image/"):
        return
    try:
        decoded = json.loads(stripped)
    except (ValueError, RecursionError):
        pass
    else:
        if isinstance(decoded, (dict, list, str)) and decoded != text:
            yield decoded
        return
    fenced = list(_FENCE.finditer(text))
    for match in fenced:
        yield from _embedded_json(match.group(1))
    if fenced:
        # Do not skip JSON outside a fence when the same string contains both.
        text = _FENCE.sub(lambda match: " " * len(match.group(0)), text)
    decoder = json.JSONDecoder()
    consumed = 0
    for match in _JSON_START.finditer(text):
        if match.start() < consumed:
            continue
        try:
            decoded, end = decoder.raw_decode(text, match.start())
        except (ValueError, RecursionError):
            continue
        if isinstance(decoded, (dict, list)):
            yield decoded
            consumed = end


class _Audit:
    def __init__(self, public: Path, feedback_profile: str = 'none'):
        if feedback_profile not in {'none', 'official_task_feedback'}:
            raise ValueError('Unknown feedback profile')
        self.public = public.resolve()
        self.feedback_profile = feedback_profile
        self.oracle_keys = ORACLE_KEYS - (
            {'success', 'task_progress', 'robocasa_terminated'}
            if feedback_profile == 'official_task_feedback' else set()
        )
        self.violations: list[dict[str, Any]] = []
        self.review_hits: list[dict[str, Any]] = []
        self.files: list[str] = []
        self.documents = 0
        self.nested_documents = 0
        self._seen: set[str] = set()

    def prior_reference(self, value):
        if (self.feedback_profile == 'official_task_feedback'
                and Path(value).name == 'success_criteria.md'):
            return False
        return _prior_reference(value)

    def finding(self, kind, file, location, *, review=False, **details):
        item = {"kind": kind, "file": str(file), "location": location, **details}
        identity = json.dumps(item, sort_keys=True)
        if identity not in self._seen:
            self._seen.add(identity)
            (self.review_hits if review else self.violations).append(item)

    def reference(self, value, file, location, origin, *, explicit):
        value = value.strip().rstrip(".:!")
        if not value or value in {"/", "//"} or value.startswith(("data:", "http://", "https://", "app://", "mcp://")):
            return
        if value.startswith("file://"):
            value = unquote(urlparse(value).path)
        if self.prior_reference(value):
            self.finding("prohibited_file_reference", file, location,
                         reference=value, origin=origin, review=origin in {"model_generated", "transport_metadata"})
        path = Path(value).expanduser()
        # A public relative artifact is relative to its JSON's directory; MCP
        # tool arguments and emitted model references use the public cwd.
        base = file.parent if file.is_relative_to(self.public) else self.public
        resolved = (base / path).resolve()
        if resolved.is_relative_to(self.public):
            return
        if (
            explicit
            or resolved.exists()
            or resolved.is_relative_to(self.public.parent / "private")
        ):
            self.finding("outside_public_reference", file, location,
                         reference=value, resolved=str(resolved), origin=origin,
                         review=origin in {"model_generated", "transport_metadata"})
        else:
            self.finding("unclassified_path_reference", file, location,
                         reference=value, resolved=str(resolved), origin=origin, review=True)

    def walk(self, value, file, location="$", *, origin="planner_input", key="", depth=0):
        if depth > 40:
            self.finding("nesting_limit", file, location)
            return
        if isinstance(value, dict):
            if value.get("role") == "assistant" or value.get("type") in {"function_call", "custom_tool_call"}:
                origin = "model_generated"
            # Native states.json stores the same allowed completion flag as
            # StepRecord.terminated, alongside a constant truncated=False.
            native_step = (
                self.feedback_profile == 'official_task_feedback'
                and {'step_idx', 'state', 'artifacts', 'extras'} <= value.keys()
                and isinstance(value['extras'], dict)
            )
            for child_key, child in value.items():
                child_location = f"{location}[{json.dumps(child_key)}]"
                allowed_step_alias = native_step and (
                    (child_key == 'terminated' and type(child) is bool
                     and child is value['extras'].get('success'))
                    or (child_key == 'truncated' and child is False)
                )
                if child_key in self.oracle_keys and not allowed_step_alias:
                    self.finding("oracle_key", file, child_location, key=child_key, origin=origin)
                child_origin = "transport_metadata" if child_key == "client_metadata" else origin
                self.walk(child, file, child_location, origin=child_origin, key=child_key, depth=depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.walk(child, file, f"{location}[{index}]", origin=origin, key=key, depth=depth + 1)
        elif isinstance(value, str) and not value.startswith("data:image/"):
            for match in _WORDS.finditer(value):
                self.finding("word_review", file, location, review=True,
                             word=match.group(0).lower(), origin=origin,
                             excerpt=value[max(0, match.start() - 65):match.end() + 100])
            for match in _PRIOR_TOKEN.finditer(value):
                if not self.prior_reference(match.group(0)):
                    continue
                self.finding(
                    "prohibited_file_reference", file, location,
                    reference=match.group(0), origin=origin,
                    review=origin in {"model_generated", "transport_metadata"},
                )
            is_path = key in _PATH_KEYS or key.endswith(("_path", "_filename"))
            if is_path and "\n" not in value and len(value) < 4096:
                self.reference(value, file, location, origin, explicit=True)
            else:
                for match in _PATH_TOKEN.finditer(value):
                    self.reference(match.group(0), file, location, origin, explicit=False)
            for index, nested in enumerate(_embedded_json(value)):
                self.nested_documents += 1
                self.walk(nested, file, f"{location}::<json:{index}>", origin=origin, depth=depth + 1)

    def inspect(self, file: Path, *, public: bool):
        if public and self.prior_reference(str(file.relative_to(self.public))):
            self.finding("prohibited_public_file", file, "$")
        if file.suffix.lower() in _BINARY_SUFFIXES:
            return
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeError:
            if file.suffix.lower() in {".json", ".jsonl"}:
                self.finding("unreadable_structured_file", file, "$")
            return
        except OSError as exc:
            self.finding("unreadable_file", file, "$", error=str(exc))
            return
        self.files.append(str(file))
        if file.suffix.lower() == ".json":
            records = [("$", text)]
        elif file.suffix.lower() == ".jsonl":
            records = [(f"line:{number}", line) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
        else:
            self.walk(text, file, origin="public_text" if public else "planner_input")
            return
        for location, record in records:
            try:
                parsed = json.loads(record)
            except (ValueError, RecursionError) as exc:
                self.finding("invalid_json", file, location, error=str(exc))
                continue
            self.documents += 1
            self.walk(parsed, file, location, origin="public_artifact" if public else "planner_input")


def check_public_boundary(public_dir: str | Path, model_inputs_dir: str | Path | None = None,
                          *, feedback_profile: str = 'none') -> dict[str, Any]:
    """Check completed restricted artifacts; return a report without modifying them."""
    public = Path(public_dir).resolve()
    model_inputs = Path(model_inputs_dir or public.parent / "private" / "model_inputs").resolve()
    audit = _Audit(public, feedback_profile)
    if not public.is_dir():
        audit.finding("missing_public_directory", public, "$")
    else:
        for directory, dirs, filenames in os.walk(public, followlinks=False):
            for name in [*dirs, *filenames]:
                candidate = Path(directory) / name
                if candidate.is_symlink() and not candidate.resolve().is_relative_to(public):
                    audit.finding("public_symlink_escape", candidate, "$", resolved=str(candidate.resolve()))
                elif candidate.is_file():
                    audit.inspect(candidate, public=True)
    requests = sorted(model_inputs.glob("request_*.json"))
    if not requests:
        audit.finding("missing_model_requests", model_inputs, "$")
    for request in requests:
        audit.inspect(request, public=False)
    return {
        "schema_version": "stage.public-boundary/v1",
        "passed": not audit.violations,
        "feedback_profile": feedback_profile,
        "scope": {"public_dir": str(public), "model_inputs_dir": str(model_inputs)},
        "summary": {"text_files": len(audit.files), "request_files": len(requests),
                    "json_documents": audit.documents, "nested_json_documents": audit.nested_documents,
                    "violations": len(audit.violations), "review_hits": len(audit.review_hits)},
        "violations": audit.violations,
        "review_hits": audit.review_hits,
        "checked_text_files": audit.files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path)
    parser.add_argument("--model-inputs-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    public = (args.public_dir or args.episode / "public").resolve()
    output = (args.output or args.episode / "private" / "public_boundary_report.json").resolve()
    if output.is_relative_to(public):
        parser.error("The boundary report must be written outside the public directory")
    spec_path = args.episode / 'episode.json'
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    report = check_public_boundary(public, args.model_inputs_dir or args.episode / "private" / "model_inputs",
                                   feedback_profile=spec.get('feedback_profile', 'none'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output), **report["summary"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
