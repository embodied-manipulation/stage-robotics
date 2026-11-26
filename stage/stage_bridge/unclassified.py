"""Hybrid classification-off control; same physical review window and tools.

There is no public decision/progress/condition tool. Instruction changes maintain
only private policy context, never invented Advance/Recover ground truth. This
is an ablation of the declaration protocol, not a claim of inert label tokens.
"""
from dataclasses import replace
from pathlib import Path
from research.common import append_json
from rpent.tools import ToolResult
from stage_bridge.review import VLA, ANALYTIC


class InstructionContext:
    def __init__(self, private, history=None):
        self.state = None
        self.history = history
        self.log = Path(private) / 'instruction_context.jsonl'

    def guarded(self, declaration):
        def invoke(**inputs):
            from research import review_window as review
            window = review._window
            if window is None or window.last_grant < 0:
                return ToolResult(data={'error': 'Observe the initial scene before motion.',
                                        'code': 'observation_required'})
            if declaration.name in VLA:
                prompt = inputs.get('prompt')
                if not isinstance(prompt, str) or not prompt.strip():
                    return ToolResult(data={'error': 'Provide a nonempty VLA instruction.'})
                previous = self.state
                changed = previous is None or previous['prompt'] != prompt
                sequence = (previous['sequence'] if previous else 0) + int(changed)
                state = {'prompt': prompt, 'active_prompt': prompt, 'sequence': sequence,
                         'subgoal_id': f'instruction{sequence:03d}',
                         'active_subgoal_id': f'instruction{sequence:03d}'}
                # These internal lifecycle events are not Planner classifications.
                lifecycle = 'start' if changed else 'continue'
                if self.history is not None:
                    self.history.accept_subgoal(lifecycle, prompt)
                from research.subgoal_images import _window as images
                if images is not None and changed:
                    images.transition('start', state, window.total)
                self.state = state
                append_json(self.log, dict(observation_action=window.total,
                    instruction_changed=changed, classification_source='absent', **state))
            return declaration(**inputs)
        return declaration.with_handler(invoke)


def install(toolkit, private):
    primitives = toolkit._primitives
    skill = getattr(primitives, '_vla_skill', None)
    history = skill if getattr(skill, 'transition_owned_history', False) else None
    context = InstructionContext(private, history)
    if history is not None:
        native = primitives.env.step
        def step(action):
            result = native(action)
            if getattr(primitives, '_active_action_tool', None) not in VLA:
                history.record_analytic_step()
            primitives._vla_desync = False
            return result
        primitives.env.step = step
    for name, declaration in list(toolkit._tools.items()):
        if name in VLA | ANALYTIC | {'finish'}:
            if name in VLA:
                schema = declaration.input_schema
                schema['properties']['prompt']['description'] = (
                    'Current local instruction, passed verbatim to the VLA. '
                    'Keep the same instruction while it remains suitable.')
                declaration = replace(declaration, _input_schema=schema,
                    description=declaration.description.replace('accepted Subgoal', 'current local instruction'))
            toolkit._tools[name] = context.guarded(declaration)
    toolkit.subgoal_review = context  # private adapter interface, not a public tool
    return toolkit


GUIDANCE = '''
Use the latest scene observations to select an immediate instruction or analytic
action. Keep a suitable VLA instruction stable while it remains useful; change
it when the scene supports a different immediate outcome. After motion, forward
the result text and review token; the framework attaches the current camera
window to the next Planner request. Inspect it before further execution.
A partially executed analytic call
is not automatically resumed: recompute targets from the current scene.
Use observed geometry for Codes and preserve completed task relationships.
Explain the intended next action and supporting observations in the ordinary
reason field. There is no required decision, progress, or condition declaration.
'''


def language_guidance(text):
    """Keep language/feedback advice while removing declaration instructions."""
    text = text.replace('observation_evidence and decision_reason', 'the ordinary reason field')
    text = text.replace('observation_evidence and\ndecision_reason', 'the ordinary reason field')
    text = text.replace('Record its visible completion criterion in completion_condition. ', '')
    text = text.replace('Put the visible expected\nresult in completion_condition. ', '')
    text = text.replace('Keep the accepted prompt stable on continue.',
                        'Keep a suitable instruction stable across observation windows.')
    text = text.replace(
        'In each subgoal_review, record the current goal ids/truth values and contacted\n'
        'object identities in observation_evidence.',
        'Use the current goal ids/truth values and contacted object identities when\n'
        'explaining the next action in the ordinary reason field.')
    text = text.replace(
        'Use subgoal_status when supplied\nto compare persistence with the actual observed movement and remaining goal;\n'
        'no count alone requires a switch. Preserve start/continue/advance/recover,\n'
        'the fixed observation cadence, visual-window rules and existing finish rules.',
        'Compare persistence with the observed movement and remaining goal; no count alone\n'
        'requires an instruction change. Preserve the observation cadence and finish rules.')
    text = text.replace(
        'Preserve the\nstandard start/continue/advance/recover decisions, fixed observation cadence and\n'
        'Subgoal visual-window rules.',
        'Preserve the fixed observation cadence and visual-window rules.')
    text = text.replace(
        'Do not shorten an accepted instruction during continue or create a new Subgoal\n'
        'only to rephrase it. Change the goal only when the shared transition rules and\n'
        'scene evidence justify it.',
        'Keep a suitable instruction stable instead of repeatedly paraphrasing it.\n'
        'Use scene evidence to decide when a different immediate outcome is needed.')
    return neutral_text(text)


# Only declaration-dependent language changes. Recovery behavior and executor
# advice stay in their original sections. Exported by the prompt-diff audit.
NEUTRAL_REPLACEMENTS = (
    ('Select an observable Subgoal and classify start.', 'Select an observable immediate outcome and instruction.'),
    ('classify continue while the current goal remains suitable; advance on completion evidence or recover on an observed blocker.',
     'keep the instruction while the current goal remains suitable; change it on completion evidence or an observed blocker.'),
    ('Advance when observations support the saved completion condition. Recover when',
     'Move to the next outcome when observations show the current one is complete. Change the instruction when'),
    ('choose\nrecover using that evidence.', 'change\nthe immediate instruction using that evidence.'),
    ('Finish only after observing and classifying the latest physical state.',
     'Finish only after observing the latest physical state.'),
    ('accepted Subgoal prompt', 'current local instruction'),
    ('accepted instruction', 'current instruction'),
    ('same Subgoal', 'same immediate outcome'),
    ('not end the Subgoal', 'not end the current instruction'),
    ('completion_condition', 'the intended visible outcome'),
    ('completion condition', 'intended visible outcome'),
    ('observation_evidence', 'the ordinary reason field'),
    ('decision_reason', 'the ordinary reason field'),
    ('transition_evidence', 'visible completion or obstacle evidence'),
    ('classifying', 'reviewing'),
    ('Classify', 'Review'),
    ('classify', 'review'),
)


def neutral_text(text):
    for old, new in NEUTRAL_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def neutral_sections(sections):
    from rpent.prompt.utils import BulletList
    result = dict(sections)
    declaration = result.pop('Subgoal_Review', '')
    from subgoal_core.observations import guidance as image_guidance
    # Keep the same checkpoint and visual-window information. Only the
    # declaration-specific paragraph is replaced by neutral decision guidance.
    checkpoint = declaration.split('\nAfter the initial images and every new motion observation delivered to GPT,')[0]
    result['Execution_Review'] = checkpoint + '\n' + GUIDANCE + image_guidance()
    result['Primitives'] = '\n'.join(line for line in result['Primitives'].split('\n')
                                     if not line.startswith('- subgoal_review:'))
    for name, value in result.items():
        if isinstance(value, str):
            result[name] = language_guidance(value)
        elif isinstance(value, BulletList):
            result[name] = BulletList([language_guidance(item) for item in value.items])
    return result
