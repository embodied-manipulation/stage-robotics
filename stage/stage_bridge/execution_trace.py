"""Record actual executor use without adding fields to the Planner protocol."""
from pathlib import Path

from research.common import append_json, write_json
from stage_bridge.review import ANALYTIC, VLA


class ExecutionTrace:
    def __init__(self, private, *, variant, classification_enabled):
        self.path = Path(private) / 'execution_segments.jsonl'
        self.variant = variant
        self.classification_enabled = classification_enabled
        self.previous_executor = None
        self.executors_by_goal = {}
        self.sequence = 0

    def record(self, toolkit, *, tool, before, after, result=None, error=None):
        executor = 'VLA' if tool in VLA else 'Codes' if tool in ANALYTIC else None
        if self.variant == 'C' and executor == 'VLA':
            write_json(self.path.parent/'protocol_violation.json',
                       dict(code='codes_only_vla_call', variant='C', tool=tool,
                            action_start=before, action_end=after))
            raise RuntimeError('Codes-only execution attempted a VLA tool')
        if executor is None:
            return
        actions = after - before
        if actions < 0:
            raise ValueError('An execution segment cannot cross an environment reset')
        state = getattr(toolkit.subgoal_review, 'state', None) or {}
        goal = state.get('subgoal_id') if self.classification_enabled else None
        switched = actions > 0 and self.previous_executor not in (None, executor)
        if actions > 0:
            self.previous_executor = executor
            if goal:
                self.executors_by_goal.setdefault(goal, set()).add(executor)
        used = sorted(self.executors_by_goal.get(goal, set()))
        data = result.data if result is not None else {}
        self.sequence += 1
        append_json(self.path, {
            'schema': 'final-execution-segment/v1', 'segment': self.sequence,
            'variant': self.variant, 'tool': tool, 'executor': executor,
            'action_start': before, 'action_end': after, 'physical_actions': actions,
            'executor_switch': switched,
            'classification_source': 'planner' if self.classification_enabled else 'absent',
            'subgoal_id': goal,
            'instruction_context_id': None if self.classification_enabled else state.get('subgoal_id'),
            'prompt': state.get('prompt'),
            'completion_condition': state.get('completion_condition') if self.classification_enabled else None,
            'declared_decision': state.get('decision') if self.classification_enabled else None,
            'declared_progress': state.get('progress') if self.classification_enabled else None,
            'assessment_subgoal_id': state.get('assessment_subgoal_id') if self.classification_enabled else None,
            'parent_subgoal_id': state.get('parent_subgoal_id') if self.classification_enabled else None,
            'observed_executors_for_subgoal': used,
            'mixed_so_far': len(used) > 1 if goal else None,
            'review_required': bool(data.get('review_required')),
            'error': error or data.get('error'),
            'outcome_source': 'not_scored_by_execution_trace',
            'vla_history': (getattr(getattr(toolkit._primitives, '_vla_skill', None),
                                    '_last_history_event', None) if executor == 'VLA' else None),
        })
