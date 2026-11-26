"""Observation-linked STAGE decisions inside the execution Toolkit.

This module never reads evaluator predicates or object ground truth. Semantic
contradictions remain model errors; structural/stale decisions cannot move.
"""
import os
from pathlib import Path
from typing import Literal

from rpent.tools import ToolResult, tool
from research.common import append_json
from subgoal_core.protocol import (
    CONTINUATION_STATUS_PROTOCOL, continuation_status, resolve_transition,
)
import research.review_window as review

VLA = {'vla_skill', 'vla_arm', 'pi0_pick', 'pi0_doubled'}
ANALYTIC = {'move_to', 'move_delta', 'move_pose', 'rotate_wrist', 'rotate_pitch',
            'set_gripper', 'release', 'scripted_grasp', 'navigate_to', 'move_base',
            'execute_action'}


class SubgoalReview:
    def __init__(self, private, history=None, status_protocol=None):
        if status_protocol not in (None, CONTINUATION_STATUS_PROTOCOL):
            raise ValueError(f'Unsupported Subgoal status protocol: {status_protocol}')
        self.state = None
        self.status_protocol = status_protocol
        self.history = history
        self.classified_step = -1
        self.log = Path(private) / 'subgoal_decisions.jsonl'

    @tool
    def subgoal_review(
        self, observation_step: int,
        decision: Literal['start', 'continue', 'advance', 'recover'],
        progress: Literal['progressing', 'complete', 'blocked', 'uncertain'],
        target_visibility: Literal['visible', 'occluded', 'unknown', 'not_applicable'],
        observation_evidence: str, decision_reason: str,
        prompt: str | None = None, completion_condition: str | None = None,
        transition_evidence: str | None = None,
    ) -> ToolResult:
        """Classify the current Subgoal using the latest images already seen by GPT.

        Args:
            observation_step: review_window.action_count in the observed result.
            decision: start initially; continue same goal; advance after completion;
                recover on obstacle.
            progress: Observed progress of the previous/current goal.
            target_visibility: Whether completion/obstacle can currently be seen.
            observation_evidence: Brief visible evidence, not hidden task state.
            decision_reason: Brief justification of this decision.
            prompt: Local VLA instruction for start/advance/recover. Omit otherwise.
            completion_condition: Observable outcome for a new local goal.
            transition_evidence: Completion or obstacle evidence for advance/recover.
        """
        window = review._window
        if (window is None or window.last_grant < 0 or observation_step != window.total
                or observation_step != window.last_grant):
            return ToolResult(data={'error': 'Forward the latest review token AND images to GPT before classifying.',
                                    'code': 'subgoal_observation_not_delivered'})
        arguments = dict(observation_step=observation_step, decision=decision,
                         progress=progress, target_visibility=target_visibility,
                         observation_evidence=observation_evidence, decision_reason=decision_reason,
                         prompt=prompt, completion_condition=completion_condition,
                         transition_evidence=transition_evidence)
        try:
            state = resolve_transition(arguments, self.state, record_only=True,
                                       require_decision_reason=True,
                                       track_continuation=self.status_protocol is not None)
        except ValueError as exc:
            return ToolResult(data={'error': str(exc), 'code': 'subgoal_invalid_transition'})
        from research.subgoal_images import _window as images
        if images is not None:
            images.transition(decision, state, observation_step)
        history_event = (self.history.accept_subgoal(decision, state['prompt'])
                         if self.history is not None else None)
        self.state = state
        self.classified_step = observation_step
        append_json(self.log, dict(type='planner_intervention', observation_step=observation_step,
                                   review_token=window.token, vla_history=history_event, **state))
        return ToolResult(data={'subgoal': state, 'vla_history': history_event})

    def guarded(self, declaration):
        def invoke(**inputs):
            window = review._window
            if (window is None or window.last_grant < 0 or self.state is None
                    or self.classified_step != window.last_grant):
                return ToolResult(data={'error': 'Call subgoal_review on the current GPT-observed images before motion or finish.',
                                        'code': 'subgoal_review_required'})
            if declaration.name == 'finish' and self.classified_step != window.total:
                return ToolResult(data={'error': 'Observe and classify the final state before finish.',
                                        'code': 'subgoal_review_required'})
            if declaration.name in VLA:
                if inputs.get('force_reset'):
                    return ToolResult(data={'error': 'History resets belong to start/advance/recover, not VLA calls.',
                                            'code': 'subgoal_history_reset_disallowed'})
                expected = self.state['prompt']
                if inputs.get('prompt') != expected:
                    return ToolResult(data={'error': 'VLA prompt must equal the accepted Subgoal instruction.',
                                            'code': 'subgoal_prompt_mismatch', 'expected_prompt': expected})
            return declaration(**inputs)
        return declaration.with_handler(invoke)


def install(toolkit, private, *, code_as_policy, status_protocol=None, vla_enabled=True):
    if type(code_as_policy) is not bool:
        raise ValueError('code_as_policy must be an explicit boolean')
    if type(vla_enabled) is not bool or (not vla_enabled and not code_as_policy):
        raise ValueError('Codes-only requires Codes capability')
    primitives = toolkit._primitives
    skill = getattr(primitives, '_vla_skill', None)
    history = skill if vla_enabled and getattr(skill, 'transition_owned_history', False) else None
    gate = SubgoalReview(private, history=history, status_protocol=status_protocol)
    if history is not None:
        native_step = primitives.env.step

        def step(action):
            result = native_step(action)
            # VLA already records once per step in VLASkill.run().
            if getattr(primitives, '_active_action_tool', None) not in VLA:
                history.record_analytic_step()
            primitives._vla_desync = False
            return result

        primitives.env.step = step
    for name, declaration in list(toolkit._tools.items()):
        if (name in VLA and not vla_enabled) or (name in ANALYTIC and not code_as_policy):
            del toolkit._tools[name]
        elif name in VLA | ANALYTIC | {'finish'}:
            toolkit._tools[name] = gate.guarded(declaration)
    toolkit.add_tool(gate.subgoal_review)
    toolkit.subgoal_review = gate
    if not vla_enabled:
        from stage_bridge.codes_guidance import configure_tool
        toolkit._tools = {name: configure_tool(tool) for name, tool in toolkit._tools.items()}
        execute_without_vla = toolkit.execute_tool

        def reject_vla(name, inputs):
            if name in VLA:
                from research.common import write_json
                violation = dict(code='codes_only_vla_call', tool=name, variant='C')
                write_json(Path(private)/'protocol_violation.json', violation)
                return ToolResult(data={'error': 'VLA tools are unavailable in Codes-only mode.', **violation})
            return execute_without_vla(name, inputs)

        toolkit.execute_tool = reject_vla
    if status_protocol is not None:
        execute = toolkit.execute_tool

        def with_status(name, inputs):
            result = execute(name, inputs)
            if review._window is not None:
                result.data['subgoal_status'] = continuation_status(gate.state, review._window.total)
            return result

        toolkit.execute_tool = with_status
    return toolkit


def guidance(*, vla_enabled=True):
    from subgoal_core.observations import guidance as image_guidance
    from subgoal_core.prompts import guidance as protocol_guidance
    return protocol_guidance(
        visual_guidance=image_guidance(),
        checkpoint_guidance=review.GUIDANCE.replace('STAGE', 'STAGE'),
        code_as_policy=os.environ.get('STAGE_CODE_AS_POLICY') == '1',
        vla_enabled=vla_enabled,
    )
