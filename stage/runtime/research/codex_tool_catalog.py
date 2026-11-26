"""Derive an Astra capability catalog without changing its model or instructions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess


def restricted_tool_catalog(codex_bin, private_output_dir) -> Path:
    """Disable native patch/experimental tools using Codex's catalog override.

    The source is the exact installed binary's bundled catalog. All model
    identity, reasoning, prompt and unrelated capability fields are preserved.
    The request boundary must still check the effective advertised tool registry.
    """
    completed = subprocess.run(
        [str(codex_bin), "debug", "models", "--bundled"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    native = json.loads(completed.stdout)
    restricted = copy.deepcopy(native)
    matches = [
        model for model in restricted["models"] if model.get("slug") == "gpt-6-astra"
    ]
    if len(matches) != 1:
        raise ValueError("Native Codex catalog must contain exactly one gpt-6-astra entry")
    model = matches[0]
    model["apply_patch_tool_type"] = None
    model["experimental_supported_tools"] = []

    private = Path(private_output_dir)
    private.mkdir(parents=True, exist_ok=True)
    (private / "native_codex_model_catalog.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=2)
    )
    output = private / "restricted_codex_model_catalog.json"
    output.write_text(json.dumps(restricted, ensure_ascii=False, indent=2))
    return output
