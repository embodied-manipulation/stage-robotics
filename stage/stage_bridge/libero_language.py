"""Explicit language guidance; the accepted VLA instruction is never rewritten."""
from pathlib import Path

from subgoal_core.prompts import LIBERO_PROMPT_PROTOCOL, LEGACY_LIBERO_PROMPT_PROTOCOL

PROFILE = 'libero-benchmark-native-language/v2'
CONCISE_PROFILE = 'libero-benchmark-native-language/v3'
PROFILE_PATH = Path(__file__).resolve().parents[3] / 'profiles/libero_benchmark_native.md'

CONCISE_GUIDANCE = """
CONCISE SINGLE-OUTCOME INSTRUCTIONS
For normal and recovery Subgoals, express one observable object-level outcome
concisely. Prefer the result relation over a narration of the movements needed
to achieve it. Avoid chaining action clauses with "and"; a placement instruction
already lets the policy approach, grasp, transport and release the object.
Keep only descriptors needed to identify the actual object or destination in
the current scene. Put approach details, explanations and recovery reasoning in
observation_evidence and decision_reason rather than in the VLA prompt.
This is a general wording preference, not a fixed sentence template or word
limit. Preserve necessary identity, spatial relations and current feasibility.
Do not shorten an accepted instruction during continue or create a new Subgoal
only to rephrase it. Change the goal only when the shared transition rules and
scene evidence justify it.
"""


def guidance(prompt_protocol, profile, path=None):
    if profile in (None, 'none'):
        return ''
    if profile in (PROFILE, CONCISE_PROFILE) and prompt_protocol == LIBERO_PROMPT_PROTOCOL:
        marker = f'## Planner guidance: {PROFILE}\n\n'
    elif profile == 'libero-benchmark-native-language/v1' and prompt_protocol == LEGACY_LIBERO_PROMPT_PROTOCOL:
        marker = '## Optional Planner guidance\n\n'
    else:
        raise ValueError(f'Unsupported language/prompt combination: {profile}, {prompt_protocol}')
    text = Path(path or PROFILE_PATH).read_text()
    guidance = text.split(marker, 1)[1].split('```text\n', 1)[1].split('\n```', 1)[0]
    if profile == CONCISE_PROFILE:
        guidance = guidance.replace(
            '"pick [object] and place it in/on [destination]", ', '')
        guidance += '\n\n' + CONCISE_GUIDANCE.strip()
    return guidance
