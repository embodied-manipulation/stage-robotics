"""Native LIBERO goal/contact feedback, separate from the shared Subgoal policy.

Observe the evaluator's existing calls. Reading a snapshot never calls
_check_success or _eval_predicate again and never latches completed conditions.
"""
from copy import deepcopy

PROTOCOL = 'libero-native-task-feedback/v1'
FIELDS = ('success_criteria', 'task_progress', 'grasp_feedback')


def _key(value):
    return tuple(_key(v) for v in value) if isinstance(value, (list, tuple)) else value


class NativeTaskFeedback:
    def __init__(self, env):
        self.env = env
        self.goals = deepcopy(env.parsed_problem['goal_state'])
        self.criteria = dict(source='native parsed BDDL goal_state', logic='all',
            conditions=[dict(id=f'goal_{i}', predicate=g[0], arguments=g[1:])
                        for i, g in enumerate(self.goals)])
        self.last = None
        self.active = None
        self.evaluation = 0
        original_check = env._check_success
        original_predicate = env._eval_predicate

        def predicate(state):
            value = original_predicate(state)
            if self.active is not None:
                self.active[_key(state)] = None if value is None else bool(value)
            return value

        def check():
            # Preserve exactly the native call count, order, and return value.
            self.active = {}
            try:
                value = original_check()
                rows = [dict(id=c['id'], satisfied=self.active.get(_key(g)))
                        for c, g in zip(self.criteria['conditions'], self.goals)]
                self.evaluation += 1
                self.last = dict(evaluation=self.evaluation,
                    sim_time=float(env.sim.data.time), native_success=bool(value),
                    conditions=rows)
                return value
            finally:
                self.active = None

        env._eval_predicate = predicate
        env._check_success = check

    def snapshot(self):
        current_time = float(self.env.sim.data.time)
        fresh = self.last is not None and self.last['sim_time'] == current_time
        rows = (deepcopy(self.last['conditions']) if fresh else
                [dict(id=c['id'], satisfied=None) for c in self.criteria['conditions']])
        progress = dict(protocol=PROTOCOL,
            source='cached results from native _check_success calls',
            available=fresh and all(type(r['satisfied']) is bool for r in rows),
            native_success=self.last['native_success'] if fresh else None,
            conditions=rows, satisfied_count=sum(r['satisfied'] is True for r in rows),
            total_conditions=len(rows), evaluation=self.last['evaluation'] if fresh else None)
        # This read-only contact query supplies identity, not GT object poses.
        objects, errors = [], []
        for name, obj in self.env.objects_dict.items():
            try:
                if self.env._check_grasp(self.env.robots[0].gripper, obj):
                    objects.append(name)
            except Exception as exc:
                errors.append(dict(object=name, error=type(exc).__name__))
        grasp = dict(available=not errors, objects=sorted(objects),
            queried_objects=sorted(self.env.objects_dict),
            source='native robosuite _check_grasp finger-contact predicate',
            meaning='Contact with both finger groups; not proof of stable holding or lift.')
        if errors:
            grasp['errors'] = errors
        return dict(success_criteria=deepcopy(self.criteria), task_progress=progress,
                    grasp_feedback=grasp)


def validate_observation(data, *, allow_historical=False):
    """Fail a selected full-feedback run if delivery is incomplete or stale."""
    missing = [key for key in FIELDS if key not in data]
    if missing:
        raise ValueError(f'Missing full task feedback: {missing}')
    criteria, progress, grasp = (data[key] for key in FIELDS)
    if progress.get('protocol') != PROTOCOL or progress.get('available') is not True:
        raise ValueError('Native task progress unavailable at this observation')
    expected = [row['id'] for row in criteria['conditions']]
    rows = progress['conditions']
    if not rows or expected != [row['id'] for row in rows]:
        raise ValueError('Native criterion identities do not match the progress snapshot')
    if any(type(row['satisfied']) is not bool for row in rows):
        raise ValueError('Unmeasured native condition must not be treated as false')
    if (criteria['logic'] != 'all' or
            progress['native_success'] is not all(row['satisfied'] for row in rows)):
        raise ValueError('Goal predicate conjunction differs from native evaluator')
    if progress['satisfied_count'] != sum(row['satisfied'] for row in rows):
        raise ValueError('Native satisfied count does not match its conditions')
    if grasp.get('available') is not True or not isinstance(grasp.get('objects'), list):
        raise ValueError('Native grasp contact feedback unavailable')
    action = progress.get('observation_action')
    if type(action) is not int or action < 0 or grasp.get('observation_action') != action:
        raise ValueError('Feedback is not aligned to a physical observation action')
    review = data.get('review_window', {})
    if ('action_count' in review and review['action_count'] != action
            and not (allow_historical and action < review['action_count'])):
        raise ValueError('Feedback snapshot and review action differ')


SYSTEM_GUIDANCE = '''This run supplies native task feedback at each recorded observation:
- success_criteria lists the actual BDDL goal predicates. Its logic is all:
  every listed condition must hold together for native whole-task success.
- task_progress.conditions gives the current native truth value for each
  criterion id, from the evaluator calls for that physical state. Match ids to
  success_criteria. Use observation_action to align feedback with the CURRENT
  image; earlier window images and the Subgoal anchor are historical evidence.
- grasp_feedback.objects identifies queried_objects satisfying the native
  two-finger contact test now. Unqueried fixtures/handles are outside this
  feedback. An empty list means no such contact, not necessarily no one-finger
  contact. Contact identity is not proof of a secure grasp, lift,
  transport, release, or satisfaction of a placement predicate.

Use native predicate feedback as authoritative for the predicates it measures.
In each subgoal_review, record the current goal ids/truth values and contacted
object identities in observation_evidence. Separate them from visual inferences
about approach, alignment, support, motion, or recovery feasibility.
If a relevant predicate is false while an image looks complete, it remains
unmet; inspect its criterion and current geometry. A false whole-task or local
predicate alone does not imply blocked progress and does not require recover.
Continue a suitable instruction while the scene shows a productive approach.
For a Subgoal that directly targets a listed criterion, do not declare that
criterion complete while its current native value is false. For other local
outcomes, use visible evidence and state uncertainty where necessary.

Check which requirements remain unmet and preserve satisfied relationships.
These are current-state predicates, not permanently completed milestones:
they can regress. Grasp feedback can reveal a wrong contacted object; combine
it with current images to choose a grounded correction. Do not invent a fixed
recovery sequence or a new Subgoal just because an observation arrived.
Unavailable or null feedback is unknown, never false or proof of completion.

Top-level terminated retains the native execution stop semantics: it records
whole-task success observed within the executed action chunk. task_progress
describes the final observed state; a transient success can therefore set
terminated=true even if a final-frame predicate has regressed. Honor this
termination and finish success without further motion. truncated is a budget
limit, not success. Subgoal completion never substitutes for native success.
'''
