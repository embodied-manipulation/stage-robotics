"""Evidence-based recovery of the two 2026-10-17 custom compositions.

The saved XML and state define each historical initial condition. Object and
fixture metadata instantiate Python task references. This is reconstructed
source, not a claim to recover the deleted file verbatim.
"""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen, FixtureType, OU
from robocasa.environments.kitchen.composite.brewing.prepare_coffee import PrepareCoffee
from robocasa.environments.kitchen.composite.arranging_condiments.categorize_condiments import CategorizeCondiments
from robosuite.utils import transform_utils as T

REFERENCE = Path(__file__).resolve().parents[2] / 'reference'


def restore_saved_state(env, arrays, index=0):
    """Restore recorded physical state without advancing simulation time."""
    for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat', 'userdata'):
        value = arrays[name][index]
        target = getattr(env.sim.data, name)
        if target.shape != value.shape:
            raise ValueError(f'Recorded {name} shape {value.shape} differs from model {target.shape}')
        target[:] = value
    env.sim.data.time = float(arrays['time'][index])
    env.sim.forward()


class RecoveredComposition(PrepareCoffee):
    STABLE_SECONDS = 1.0

    def __init__(self, *args, **kwargs):
        self.reference = REFERENCE / type(self).__name__
        self.saved_scene = json.loads((self.reference / 'scene.json').read_text())
        self.saved_xml = (self.reference / 'model.xml').read_text()
        self.saved_initial = dict(np.load(self.reference / 'initial_state.npz'))
        self._hold_since = None
        self._ever = False
        self._grasp_events = {name: False for name in self.movable_names}
        self._custom_snapshot = {}
        kwargs.update(layout_ids=[self.saved_scene['layout_id']],
                      style_ids=[self.saved_scene['style_id']], layout_and_style_ids=None)
        super().__init__(*args, **kwargs)

    def _load_model(self, *args, **kwargs):
        self._ep_meta = deepcopy(self.saved_scene['episode_metadata'])
        result = super()._load_model(*args, **kwargs)
        # Geometric task helpers use Python fixture poses as well as sim data.
        bodies = {b.get('name'): b for b in ET.fromstring(self.saved_xml).iter('body')}
        for fixture in (self.coffee_machine, self.cab, self.counter):
            node = bodies.get(fixture.root_body)
            if node is None:
                continue
            fixture.set_pos(np.fromstring(node.get('pos', '0 0 0'), sep=' '))
            if 'quat' in node.attrib:
                q = np.fromstring(node.get('quat'), sep=' ')
                fixture.set_euler(T.mat2euler(T.quat2mat(q[[1, 2, 3, 0]])))
            else:
                fixture.set_euler(np.fromstring(node.get('euler', '0 0 0'), sep=' '))
        return result

    def _initialize_sim(self, xml_string=None):
        # Add the real navigation camera to the historical physical scene.
        # No bodies, joints, geoms, state arrays or evaluator parameters change.
        saved = ET.fromstring(self.saved_xml)
        if not any(c.get('name') == 'mobilebase0_navview' for c in saved.iter('camera')):
            generated = ET.fromstring(self.model.get_xml())
            for body in generated.iter('body'):
                camera = next((c for c in body.findall('camera')
                               if c.get('name') == 'mobilebase0_navview'), None)
                if camera is not None:
                    target = next(b for b in saved.iter('body') if b.get('name') == body.get('name'))
                    target.append(deepcopy(camera))
                    break
            else:
                raise ValueError('Navigation camera missing from installed robot model')
        return super()._initialize_sim(xml_string=ET.tostring(saved, encoding='unicode'))

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref('counter', dict(id=FixtureType.COUNTER, ref=self.cab))

    def _reset_internal(self):
        super()._reset_internal()
        restore_saved_state(self, self.saved_initial)
        self._hold_since = None
        self._ever = False
        self._grasp_events = {name: False for name in self.movable_names}
        self._custom_snapshot = {}
        self.coffee_machine._turned_on = False

    def get_ep_meta(self):
        meta = Kitchen.get_ep_meta(self)
        meta['lang'] = self.saved_scene['task_goal']
        meta['custom_composition'] = deepcopy(self.saved_scene['episode_metadata']['custom_composition'])
        meta['source_recovery'] = 'saved-xml-and-state/20261021'
        return meta

    # _check_success below is copied literally from the retained original run.
    def _check_success(self):
        labels = self.current_predicates()
        a_current = bool(labels['A_current'])
        b_official = bool(PrepareCoffee._check_success(self))
        cup_supported = bool(OU.check_obj_fixture_contact(self, 'obj', self.coffee_machine))
        cup_upright = bool(OU.check_obj_upright(self, 'obj', th=15))
        released, stable = True, True
        for name in self.movable_names:
            obj = self.objects[name]
            grasped = bool(self._check_grasp(self.robots[0].gripper['right'], obj))
            self._grasp_events[name] = self._grasp_events.get(name, False) or grasped
            released &= not bool(self.check_contact(self.robots[0].gripper['right'], obj))
            velocity = np.asarray(self.sim.data.get_body_xvelp(obj.root_body))
            angular = np.asarray(self.sim.data.get_body_xvelr(obj.root_body))
            stable &= np.linalg.norm(velocity) <= 0.02 and np.linalg.norm(angular) <= 0.2
        # Deep interpenetration cannot count as a supported placement. Include
        # all movable objects and all fixtures, not just the designated blocker.
        geoms = {self.sim.model.geom_name2id(g) for n in self.movable_names
                 for g in self.objects[n].contact_geoms}
        minimum_distance = min((float(c.dist) for c in self.sim.data.contact[:self.sim.data.ncon]
                                if c.geom1 in geoms or c.geom2 in geoms), default=0.0)
        nonpenetrating = minimum_distance >= -0.003
        b_current = b_official and cup_supported and cup_upright
        all_current = bool(a_current and b_current and released and stable and nonpenetrating)
        now = float(self.sim.data.time)
        if not all_current:
            self._hold_since = None
        elif self._hold_since is None:
            self._hold_since = now
        hold_seconds = 0.0 if self._hold_since is None else now - self._hold_since
        success = bool(all_current and hold_seconds >= self.STABLE_SECONDS - 1e-8)
        self._ever |= success
        labels.update(B_official=b_official, B_current=bool(b_current),
                      cup_supported=cup_supported, cup_upright=cup_upright,
                      released=bool(released), stable=bool(stable),
                      nonpenetrating=bool(nonpenetrating), minimum_distance_m=minimum_distance,
                      all_current=all_current, hold_seconds=hold_seconds,
                      success=success, ever_success=bool(self._ever))
        labels.update({f'{k}_pick_ever': bool(v) for k,v in self._grasp_events.items()})
        self._custom_snapshot = labels
        return success


class CondimentsCoffeeDependency(RecoveredComposition):
    movable_names = ('obj1', 'obj2', 'obj')
    _xy_dist = CategorizeCondiments._xy_dist

    def current_predicates(self):
        return dict(
            bottle_in_cab=bool(OU.obj_inside_of(self, 'obj1', self.cab)),
            shaker_in_cab=bool(OU.obj_inside_of(self, 'obj2', self.cab)),
            bottle_grouped=bool(self._xy_dist('obj1', 'cab_obj1') <= 0.15),
            shaker_grouped=bool(self._xy_dist('obj2', 'cab_obj2') <= 0.15),
            A_current=bool(CategorizeCondiments._check_success(self)),
            blocker_at_dispenser=bool(self.coffee_machine.check_receptacle_placement_for_pouring(self, 'obj2')),
        )


class TeaCoffeeDependency(RecoveredComposition):
    movable_names = ('tea_mug', 'kettle', 'tray', 'obj')

    def current_predicates(self):
        tea = bool(OU.check_obj_in_receptacle(self, 'tea_mug', 'tray'))
        kettle = bool(OU.check_obj_in_receptacle(self, 'kettle', 'tray'))
        closed = bool(self.cab.is_closed(env=self))
        return dict(tea_mug_on_tray=tea, kettle_on_tray=kettle, cabinet_closed=closed,
                    A_current=tea and kettle and closed,
                    blocker_at_dispenser=bool(self.coffee_machine.check_receptacle_placement_for_pouring(self, 'tea_mug')))
