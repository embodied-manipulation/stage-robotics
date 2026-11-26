# STAGE Implementation Guide

This directory contains the online STAGE coordination framework for **RoboCasa365** (XR-1 and RLDX-1) and **LIBERO-PRO** (RLinf $\pi_{0.5}$).

## Decisions and Execution

The VLM planner observes the scene, submits `subgoal_review`, and selects subsequent executor tools. The framework validates and stores the accepted stage state. The review schema has no executor-selection field: `vla_skill` / `vla_arm` (RoboCasa365) and `pi0_pick` (LIBERO-PRO) invoke a VLA policy, while `move_to`, `move_delta`, `set_gripper`, `navigate_to`, and related primitives perform Code-as-Policy analytic control.

| Review Decision | Stage Identity | Instruction and Condition | Policy Context (XR-1 / RLDX-1) |
| :--- | :--- | :--- | :--- |
| `start` | First stage ID | Required | Initialize from current observation |
| `continue` | Preserve active ID | Preserve exact text | Preserve and update measured frames |
| `advance` | New stage ID | Required | Initialize from current observation |
| `recover` | New stage ID (records parent) | Required | Initialize from current observation |

- `progress` assesses the previously active stage before the next action. The formal review path records semantic contradictions under `record_only` while strictly enforcing structural invariants (such as exact instruction preservation on `continue`).
- An executor switch or observation checkpoint alone never creates a new stage.
- Execution composition is derived from actual physical actions recorded by [`stage_bridge/execution_trace.py`](stage_bridge/execution_trace.py). Planner-generated code that invokes a VLA still counts as VLA execution.

## Observation Window and Selective Context Retention

[`runtime/subgoal_core/cadence.py`](runtime/subgoal_core/cadence.py) enforces a shared physical-action window across both executors (default ceiling: **40 physical action steps**; native completion, truncation, or shorter tool calls may return earlier). At the limit, unfinished analytic motion returns for planner review without automatically replaying the remaining script.

- **Planner Visual Window**: [`runtime/subgoal_core/observations.py`](runtime/subgoal_core/observations.py) retains the stage-start anchor observation plus up to four recent review observations, resetting visual history upon stage transitions while keeping textual history across the episode.
- **RoboCasa365 (XR-1 / RLDX-1)**: [`runtime/research/transition_history.py`](runtime/research/transition_history.py) initializes finite measured sensor history on `start` / `advance` / `recover` and retains it on `continue`. Analytic motion after VLA has executed in the stage appends measured frames; analytic motion before the stage's first VLA call causes that first call to reseed from the post-code observation.
- **LIBERO-PRO (RLinf $\pi_{0.5}$)**: [`stage_bridge/libero_rtc.py`](stage_bridge/libero_rtc.py) maintains normalized model-space Real-Time Chunking (RTC) action tails (`delay_steps=0`). New stages clear tails, `continue` preserves them across consecutive VLA calls, and intervening analytic motion invalidates the unexecuted tail before the next VLA prediction.

## Module Layout

| Directory | Responsibility |
| :--- | :--- |
| `runtime/subgoal_core` | Stage state machine, shared action cadence, visual window, shared prompt protocol |
| `runtime/research` | Evaluation runners, policy RPC servers, transition-owned sensor history, metrics |
| `stage_bridge` | Stage review gate, capability modes (`V`/`H`/`C`/`H_unclassified`), execution trace, LIBERO & RTC integration |
| `backends/cap` | Reused Code-as-Policy primitives, MCP server/tool dispatch, and LIBERO-PRO compatibility layer |
| `transport/stage_transport` | Stateless provider bridge, request pacing, and budget guards |
| `analysis` | Offline declaration auditing and execution trace summaries |
| `configs` / `profiles` | Configuration templates and benchmark language profiles |
| `examples` / `tests` | Dependency-free CPU walkthrough and unit/integration fixtures |

## Running Locally

Import paths for the STAGE runtime are `stage`, `stage/runtime`, `stage/backends/cap`, and `stage/transport`:

```bash
# 1. Dependency-free walkthrough
python stage/examples/protocol_walkthrough.py

# 2. CPU integration tests (from stage/)
cd stage
PYTHONPATH=.:runtime:backends/cap:transport:tests \
python -m unittest \
  test_review_window test_subgoal_review test_subgoal_images \
  test_transition_history test_codes_only

# 3. Manifest preparation CLI
python stage/prepare.py --help
```

## Dependencies & Licensing

Reused Code-as-Policy primitives and MCP infrastructure under `backends/cap/` retain their original **Apache 2.0 License** ([`backends/cap/LICENSE`](backends/cap/LICENSE)). Included LIBERO-PRO compatibility sources retain their **MIT License**. Benchmark assets, checkpoints, credentials, and rollout outputs remain external.
