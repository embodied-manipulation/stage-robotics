"""Native LIBERO/Pi0.5 services with private evaluator and action records."""
import argparse
from collections import Counter
from pathlib import Path
import runpy
import sys
from research.common import append_json, write_json
from research.libero_bootstrap import prepare


def tracked_facade(native, spec):
    import numpy as np
    class TrackedFacade(native):
        def __init__(self,*args,**kwargs):
            self.private=Path(spec['output_dir'])/'private'
            self.actions=[]; self.states=[]; self.evaluations=[]; self.sources=Counter()
            self.source='unattributed'; self.tool='outside_motion_tool'
            self.initial=None; self.finished=False; self.success_stop=False
            super().__init__(*args,**kwargs)

        def _register_rpc(self):
            super()._register_rpc()
            self._rpc.update({'research.action_count':self.action_count,
                'env.set_action_source':self.set_action_source,'research.finalize':self.finalize})
            if spec.get('task_feedback_protocol'):
                self._rpc['env.get_task_feedback'] = self.get_task_feedback

        def get_task_feedback(self):
            from stage_bridge.libero_feedback import validate_observation
            snapshot = self._env.env.workers[self._env_idx].get_env_attr('subgoal_task_feedback')
            if snapshot is None:
                raise RuntimeError('Native environment did not install task feedback')
            for key in ('task_progress', 'grasp_feedback'):
                snapshot[key]['observation_action'] = len(self.actions)
            validate_observation(snapshot)
            if self.evaluations and snapshot['task_progress']['native_success'] != self.evaluations[-1]['official_success']:
                raise RuntimeError('Feedback differs from the last native step result')
            append_json(self.private/'task_feedback_snapshots.jsonl', snapshot)
            return snapshot

        def action_count(self):
            return dict(actions=len(self.actions),action_budget=spec['physical_action_budget'])

        def set_action_source(self,source,tool):
            self.source,self.tool=source,tool

        def reset(self):
            if self.initial is None:
                obs,info=super().reset()
                actual=int(self._env.specific_reset_id)
                if actual!=spec['resolved_reset_id']:
                    raise ValueError(f'Reset ID mismatch: {actual}')
                language=self.get_task_language()
                if language!=spec['task_language']:
                    raise ValueError(f'Unexpected task language: {language}')
                self.initial=obs
                self.states.append(np.asarray(obs['states']).copy())
                np.save(self.private/'initial_robot_state.npy',self.states[0])
                np.savez(self.private/'initial_policy_inputs.npz',states=obs['states'],
                    main_images=obs['main_images'],wrist_images=obs['wrist_images'])
                write_json(self.private/'reset_identity.json',dict(reset_id=actual,physical_resets=1,
                    task_language=language,suite=spec['suite'],task=spec['task'],seed=spec['seed'],
                    native_settling_steps=15,scene_initialization_protocol=spec['scene_initialization_protocol']))
            return self.initial,{}

        def step(self,action):
            if len(self.actions)>=spec['physical_action_budget']:
                raise RuntimeError('Physical action budget exhausted')
            obs,reward,term,trunc,info=super().step(action)
            self.actions.append(np.asarray(action).copy())
            self.states.append(np.asarray(obs['states']).copy())
            self.sources[self.source]+=1
            success=bool(np.asarray(term).any())
            self.success_stop|=success
            row=dict(action=len(self.actions),source=self.source,tool=self.tool,official_success=success)
            self.evaluations.append(row)
            append_json(self.private/'official_evaluation.jsonl',row)
            if len(self.actions)%25==0 or success or len(self.actions)==spec['physical_action_budget']:
                write_json(self.private/'progress.json',{**self.action_count(),'sources':dict(self.sources),
                    'native_success':success})
            return obs,reward,success,bool(np.asarray(trunc).any()) or len(self.actions)>=spec['physical_action_budget'],info

        def chunk_step(self,actions,*,return_all_frames=False):
            observations=[]; terminated=[]; truncated=[]
            for action in actions:
                if len(self.actions)>=spec['physical_action_budget']:break
                obs,_reward,term,trunc,_info=self.step(action)
                observations.append(obs); terminated.append(term); truncated.append(trunc)
            if not observations:raise RuntimeError('Physical action budget exhausted')
            n=len(observations)
            return (observations if return_all_frames else observations[-1],np.zeros(n),
                    np.array(terminated,dtype=bool),np.array(truncated,dtype=bool),{})

        def finalize(self):
            if self.finished:return {'recorded':True}
            before=len(self.actions)
            self.set_action_source('wait_hold','post_agent_wait')
            gripper=float(self.actions[-1][-1]) if self.actions else -1.
            for _ in range(min(spec['post_agent_wait_steps'],spec['physical_action_budget']-before)):
                if self.success_stop:break
                action=np.zeros(7); action[-1]=gripper
                self.step(action)
            np.save(self.private/'actions.npy',np.asarray(self.actions).reshape(-1,7))
            np.save(self.private/'robot_states.npy',np.asarray(self.states))
            successes=[r['action'] for r in self.evaluations if r['official_success']]
            write_json(self.private/'evaluation.json',dict(official_any_success=bool(successes),
                official_terminal_success=self.evaluations[-1]['official_success'] if self.evaluations else False,
                official_agent_terminal_success=self.evaluations[before-1]['official_success'] if before else False,
                first_success_step=successes[0] if successes else None,agent_actions=before,actions=len(self.actions),
                action_budget=spec['physical_action_budget'],post_agent_wait_steps=len(self.actions)-before,
                action_steps_by_source=dict(self.sources),success_stop_triggered=self.success_stop,
                success_stop_policy='native_chunk_boundary',feedback_source='native step terminated; no projection'))
            self.finished=True
            return {'recorded':True}
    return TrackedFacade


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component',choices=['env','policy'])
    parser.add_argument('--episode',type=Path,required=True)
    args,rest=parser.parse_known_args()
    spec=prepare(args.episode)
    root=Path(spec['backend_root'])
    target=root/('robots/libero/env_server.py' if args.component=='env' else 'rpent/robots/components/pi05_vla_server.py')
    native=runpy.run_path(str(target),run_name='libero_selected_native')
    if args.component=='env':
        if spec.get('task_feedback_protocol'):
            original_cfg = native['build_env_cfg']
            def feedback_cfg(**kwargs):
                cfg = original_cfg(**kwargs)
                cfg.init_params.task_feedback_protocol = spec['task_feedback_protocol']
                return cfg
            native['make_env'].__globals__['build_env_cfg'] = feedback_cfg
        native['main'].__globals__['LiberoEnvFacade']=tracked_facade(native['LiberoEnvFacade'],spec)
    elif spec.get('policy_rtc'):
        from stage_bridge.libero_rtc import facade_class
        native['main'].__globals__['Pi05VLAFacade'] = facade_class(native['Pi05VLAFacade'], spec)
    sys.argv=[str(target),*rest]
    native['main']()


if __name__=='__main__':main()
