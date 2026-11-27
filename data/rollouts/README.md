# STAGE & RPent-Hybrid Public Rollout Trajectories

This directory contains the canonical rollout trajectories and protocol execution metadata for all self-evaluated rows in **Table 1** of **STAGE** (*Stage by Stage: Coordinating Vision-Language-Action Policy and Code-as-Policy for Long-Horizon Manipulation*), together with the diagnostic coupled-task case studies analyzed in **Section 5.5** and the **Supplementary Material**.

Interactive rollout videos for the paper's case studies are available on the **[Project Showcase Page](https://embodied-manipulation.github.io/stage-robotics/)** (source under [`docs/`](../../docs/)).

---

## 1. Cohort Summary (Table 1 Self-Evaluated Methods)

All **680 Main-Table episodes** (5 seeds per task across every suite, using frozen VLM and VLA weights with cross-episode memory disabled) are included:

| Cohort ID | Paper Table Reference | Benchmark | Method | VLM (Effort) | Executors | Suite 1 SR (%) | Suite 2 SR (%) | **Avg SR (%)** | Episodes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `libero_pro_rpent_hybrid_pi_rlinf` | Table 1(a) | LIBERO-PRO | RPent-Hybrid* | GPT-6 Astra (low) | $\pi_{\text{RLinf}}$ + CaP | Long Task: 56.00 | Long Swap: 52.00 | **54.00** | 100 |
| `libero_pro_stage_pi_rlinf` | Table 1(a) | LIBERO-PRO | **STAGE (ours)** | GPT-6 Astra (low) | $\pi_{\text{RLinf}}$ + CaP | Long Task: **90.00** | Long Swap: 70.00 | **80.00** | 100 |
| `robocasa365_rpent_hybrid_xr1` | Table 1(b) | RoboCasa365 | RPent-Hybrid* | GPT-6 Astra (low) | XR-1 + CaP | Seen: 13.75 | Unseen: 30.00 | **21.88** | 160 |
| `robocasa365_stage_rldx1` | Table 1(b) | RoboCasa365 | **STAGE (ours)** | GPT-6 Astra (low) | RLDX-1 + CaP | Seen: 36.25 | Unseen: 35.00 | **35.63** | 160 |
| `robocasa365_stage_xr1` | Table 1(b) | RoboCasa365 | **STAGE (ours)** | GPT-6 Astra (low) | XR-1 + CaP | Seen: **76.25** | Unseen: **53.75** | **65.00** | 160 |

In addition, [`case_studies/`](case_studies/) provides the **4 coupled-task diagnostic rollouts** (`BoardWaffleDependency` and `BreakfastCoffeeComposition` at seed `1758` for both `STAGE` and `RPent-Hybrid`), kept separate from the 680-episode benchmark denominators.

> **Note on LIBERO-PRO `seed=1` recorded `action_budget` (`libero_pro_stage_pi_rlinf`)**: `84/100` STAGE episodes ran under the `2,000`-step cap, while `16` early `seed=1` episodes retained from initial $1.6\times$/$2.0\times$-horizon batches preserve their recorded `832`/`1,040` step caps (`15/16` terminated early on native task success at `172–763` steps well below their cap, and `1/16`—`long_task` task `9` seed `1`—reached `832` steps and is conservatively retained as a failure).

---

## 2. Directory Layout & Artifacts

To keep the Git repository lightweight (~32 MB for all 684 trajectories) while preserving full physical action sequences, kinematic trajectories, and coordination traces, raw multi-camera frame dumps and dense 3D point clouds are excluded.

```text
data/rollouts/
├── manifest.json                      # Global manifest, cohort statistics, and schema summary
├── episodes.csv                       # Flat 680-row summary table (Task SR, Subtask SR, VLA/Code step counts)
├── inspect_rollouts.py                # Standalone CLI verifier and episode inspector
├── libero_pro/
│   ├── rpent_hybrid_pi_rlinf/         # metadata.jsonl + trajectories/*.traj.npz (100 episodes)
│   └── stage_pi_rlinf/                # metadata.jsonl + trajectories/*.traj.npz (100 episodes)
├── robocasa365/
│   ├── rpent_hybrid_xr1/              # metadata.jsonl + trajectories/*.traj.npz (160 episodes)
│   ├── stage_rldx1/                   # metadata.jsonl + trajectories/*.traj.npz (160 episodes)
│   └── stage_xr1/                     # metadata.jsonl + trajectories/*.traj.npz (160 episodes)
└── case_studies/                      # Coupled-goal diagnostic rollouts (4 episodes)
```

### Trajectory Archive (`*.traj.npz`)

Each episode's compressed NumPy archive (`np.load(path, allow_pickle=False)`) stores:

- **RoboCasa365 (`480 + 4` episodes)**:
  - `actions` (`float32`, shape `[T, 12]`): Exact physical action vector executed at every 20 Hz step.
  - `initial_env_state` / `final_env_state` (`float64`, shape `[D]`) and `nq`, `nv` (`int32`): Initial and terminal MuJoCo simulator state vectors for deterministic environment reset and replay.
  - `checkpoint_steps` (`int32`, `[K]`), `checkpoint_eef_pos` (`float32`, `[K, 3]`), `checkpoint_eef_quat` (`float32`, `[K, 4]`), `checkpoint_gripper_qpos` (`float32`, `[K, G]`), `checkpoint_base_pos` (`float32`, `[K, 3]`), `checkpoint_base_quat` (`float32`, `[K, 4]`): Observation-checkpoint robot poses from `public/states.json`.
- **LIBERO-PRO (`200` episodes)**:
  - `actions` (`float32`, shape `[T, 7]`): Panda `OSC_POSE` 7D delta actions (`xyz`, axis-angle, gripper) at 20 Hz.
  - `robot_states` (`float32`, shape `[T+1, 8]`): Step-by-step robot state trajectory (`eef_pos`, `eef_quat`, `gripper`).
  - `checkpoint_steps`, `checkpoint_eef_pos`, `checkpoint_eef_quat`, `checkpoint_gripper_qpos`: Observation-checkpoint poses.

### Episode Metadata (`metadata.jsonl`)

Each line in `metadata.jsonl` is a self-contained JSON object with:

- **Identity & Configuration**: `episode_id`, `cohort`, `benchmark`, `suite`, `method`, `vlm`, `vla`, `executors`, `task`, `task_language`, `seed`, `split`, `review_window_actions`, `official_horizon`, `action_budget`.
- **Native Evaluation & Accounting**: `task_success` (benchmark-native evaluator), `first_success_step`, `subtask_sr` (native subtask-condition attainment $P_i$), `executed_actions`, `vla_actions`, `code_actions`.
- **`stage_decisions`** *(STAGE cohorts)*: Ordered list of structured planner reviews (`start`, `continue`, `advance`, `recover`, `finish`) with `observation_step`, `subgoal_id`, `parent_subgoal_id`, active stage `instruction`, observable `completion_condition`, and visual evidence summary.
- **`tool_calls`**: Ordered list of VLA (`vla_skill`, `rldx_skill`, `pi0_doubled`, etc.) and Code-as-Policy (`move_to`, `move_delta`, `move_pose`, `navigate_to`, `set_gripper`, etc.) invocations with exact `[action_start, action_end]` step intervals and sanitized arguments.

---

## 3. Quick Verification & Loading

Verify all 684 trajectory archives and reproduce every self-evaluated row in Table 1:

```bash
python data/rollouts/inspect_rollouts.py
```

Inspect a single rollout's stage transitions, executor handoffs, and array shapes:

```bash
python data/rollouts/inspect_rollouts.py \
  --episode robocasa365_stage_xr1__composite_unseen__ArrangeTea__s1761
```

Load trajectories directly in Python:

```python
import json
import numpy as np

with open("data/rollouts/robocasa365/stage_xr1/metadata.jsonl") as f:
    ep = json.loads(next(f))

traj = np.load(ep["trajectory_npz"], allow_pickle=False)
actions = traj["actions"]              # (T, 12) float32
init_qpos_qvel = traj["initial_env_state"]
print(ep["episode_id"], ep["task_success"], actions.shape)
```
