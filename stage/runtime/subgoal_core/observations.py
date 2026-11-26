"""Planner-only rolling RGB observations plus the active Subgoal's start frame."""
import base64
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import threading

VERSION = 'subgoal-anchor-sliding-images/v1'
CAMERAS = ('agentview', 'navview', 'wrist')
LIBERO_CAMERAS = ('agentview', 'wrist')
LIBERO_STAGE_IMAGES = ('agentview_policy', 'agentview', 'wrist')
CAMERA_LABELS = {'agentview': 'external (agentview)', 'navview': 'navigation (navview)',
                 'wrist': 'wrist', 'agentview_policy': 'external in VLA preprocessing orientation (agentview_policy)'}
LABEL = 'Subgoal visual window'
_window = None


def configuration(size=4, *, cameras=CAMERAS):
    if type(size) is not int or not 1 <= size <= 15:
        raise ValueError('Planner image window must contain 1 through 15 timestamps')
    cameras = tuple(cameras)
    if cameras not in (CAMERAS, LIBERO_CAMERAS, LIBERO_STAGE_IMAGES):
        raise ValueError('Unsupported Planner camera set')
    return dict(version=VERSION, window_size=size, cameras=list(cameras),
                anchor='current_subgoal_start', deduplicate=True,
                reset_on=['start', 'advance', 'recover'], max_images=len(cameras) * (size + 1))


def without_images(value):
    """Retain call IDs, reasoning and text, but never serialized camera pixels."""
    if isinstance(value, dict):
        if value.get('type') in ('image', 'input_image', 'image_url'):
            return {'type': 'input_text' if value['type'] == 'input_image' else 'text',
                    'text': '[Camera pixels supplied in the current Subgoal visual window.]'}
        return {k: without_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_images(v) for v in value]
    if isinstance(value, str) and value.lstrip().startswith(('{', '[')) and 'image' in value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        cleaned = without_images(parsed)
        if cleaned != parsed:
            return json.dumps(cleaned, ensure_ascii=False)
    return value


class SubgoalImages:
    def __init__(self, config, private, public):
        if config != configuration(config.get('window_size'), cameras=config.get('cameras', ())):
            raise ValueError('Unsupported Subgoal visual configuration')
        self.config = config
        self.cameras = tuple(config['cameras'])
        self.private, self.public = Path(private), Path(public)
        self.frames = deque(maxlen=config['window_size'])
        self.anchor = None
        self.subgoal_id = None
        self.lock = threading.RLock()

    def save(self):
        state = dict(config=self.config, frames=list(self.frames), anchor=self.anchor,
                     subgoal_id=self.subgoal_id)
        path = self.private / 'subgoal_image_window.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)

    def restore(self, source):
        state = json.loads((Path(source) / 'private/subgoal_image_window.json').read_text())
        if state['config'] != self.config:
            raise ValueError('Cannot resume with a different Planner visual protocol')
        self.frames.extend(state['frames'])
        self.anchor, self.subgoal_id = state['anchor'], state['subgoal_id']
        self.save()

    def observe(self, result, action):
        if not result.images:
            return result
        descriptors = result.data.get('images', [])
        artifact_step = result.data.get('step')
        by_camera = {d['camera']: d for d in descriptors}
        if type(artifact_step) is not int or any(c not in by_camera for c in self.cameras):
            raise ValueError('Planner observation requires the complete configured camera set')
        frame = dict(action=action, artifact_step=artifact_step,
                     artifacts={c: by_camera[c]['artifact'] for c in self.cameras})
        with self.lock:
            # Historical reads and duplicate read-only tools do not move the queue.
            if (not self.frames or (artifact_step >= self.frames[-1]['artifact_step']
                                    and action > self.frames[-1]['action'])):
                self.frames.append(frame)
                self.save()
            result.data['planner_visual_context'] = dict(
                version=VERSION, delivery='attached to the next Planner request',
                window_size=self.config['window_size'], cameras=list(self.cameras))
        # Keep pixels out of Codex tool-history serialization, including text(result).
        result.images = []
        return result

    def transition(self, decision, state, action):
        if decision == 'continue':
            return
        with self.lock:
            if not self.frames or self.frames[-1]['action'] != action:
                raise ValueError('Subgoal start must have a captured current observation')
            self.anchor = deepcopy(self.frames[-1])
            self.subgoal_id = state['subgoal_id']
            self.frames.clear()
            self.frames.append(self.anchor)
            self.save()

    def rewrite(self, payload):
        if payload.get('previous_response_id') or payload.get('conversation'):
            raise ValueError('Subgoal image window requires client-managed history')
        cleaned = without_images(payload.get('input', []))
        with self.lock:
            frames = list(self.frames)
            if self.anchor and all(f['action'] != self.anchor['action'] for f in frames):
                frames.insert(0, self.anchor)
            content = [{'type': 'input_text', 'text':
                f'{LABEL}: active_subgoal={self.subgoal_id}; oldest to newest; '
                f"camera order: {', '.join(CAMERA_LABELS[c] for c in self.cameras)}. "
                'SUBGOAL_START is reference evidence; CURRENT is the latest state.'}]
            for frame in frames:
                tags = []
                if self.anchor and frame['action'] == self.anchor['action']:
                    tags.append('SUBGOAL_START')
                if frame is frames[-1]:
                    tags.append('CURRENT')
                for camera in self.cameras:
                    path = self.public / frame['artifacts'][camera] / f"{frame['artifact_step']:02d}.png"
                    content.extend([
                        {'type': 'input_text', 'text': f"environment_action={frame['action']}; camera={camera}; {'; '.join(tags)}"},
                        {'type': 'input_image', 'detail': 'auto', 'image_url':
                         'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()}])
            metadata = dict(image_history_policy=VERSION, window_size=self.config['window_size'],
                            cameras=list(self.cameras), retained_steps=[f['action'] for f in frames],
                            anchor_step=self.anchor['action'] if self.anchor else None,
                            active_subgoal_id=self.subgoal_id, retained_image_count=len(self.cameras) * len(frames),
                            max_images=self.config['max_images'])
        if frames:
            cleaned.append({'role': 'user', 'content': content})
        return {**payload, 'input': cleaned}, metadata


def configure(spec, private, public):
    global _window
    config = spec.get('planner_image_window')
    if config is not None:
        profiles = {'native-three-view/v1': CAMERAS, 'calibration-two-view/v1': LIBERO_CAMERAS,
                    'libero-stage-policy-plus-calibrated/v1': LIBERO_STAGE_IMAGES}
        if tuple(config.get('cameras', ())) != profiles.get(spec.get('planner_observation_profile')):
            raise ValueError('Subgoal window cameras must match the Planner observation profile')
    _window = SubgoalImages(config, private, public) if config else None


def guidance():
    if _window is None:
        return ''
    size = _window.config['window_size']
    count = len(_window.cameras)
    views = ', '.join(CAMERA_LABELS[c] for c in _window.cameras)
    view_description = (
        'Each timestamp contains three simultaneous views: left external (agentview),\n'
        'navigation (navview), and wrist, not three successive states.'
        if _window.cameras == CAMERAS else
        f'Each timestamp contains {count} simultaneous views: {views}.\n'
        'These views show the same state, not successive states.'
    )
    if _window.cameras == LIBERO_STAGE_IMAGES:
        view_description = (
            'Each timestamp contains three images from two cameras: policy-oriented external,\n'
            'calibrated external, and calibrated wrist. They show the same state.\n'
            'Use calibrated images for pixel coordinates; never back-project agentview_policy pixels.'
        )
    return f'''
PLANNER VISUAL CONTEXT
Once observations are available, each Planner decision request includes up to
the latest {size} distinct observation timestamps within the active Subgoal,
INCLUDING the current observation, PLUS the Subgoal's starting reference.
Before the first Subgoal is started, use the available initial observation;
there is no SUBGOAL_START reference yet.
{view_description}
The rolling window therefore holds up to {count * size}
images; the starting reference adds up to {count} more, for at most {count * (size + 1)} images.
If the starting observation is still in the rolling window, it appears only once.
Repeated reads of the same physical state do not consume additional window slots.
Images are ordered oldest to newest and labeled with environment_action and camera.
SUBGOAL_START marks the reference; CURRENT marks the latest state. One timestamp
may carry both labels. Compare these views to assess progress and select continue,
advance, or recover; do not treat the starting reference as the current state.
continue preserves the starting reference; newly captured observations slide
the rolling window forward.
Accepted start/advance/recover decisions establish a new starting reference and
reset the rolling window to that observation. The visual window does not limit
the lifetime of a Subgoal or set the physical-action observation interval.
Tool results carry text; the framework automatically attaches camera pixels to
the next Planner decision request. Earlier text or context summaries describe
past observations even when their images have left the window. Verify any
relationship that may have changed against CURRENT; past evidence alone is
not fresh visual confirmation. Context-compaction requests do not carry images.
'''
