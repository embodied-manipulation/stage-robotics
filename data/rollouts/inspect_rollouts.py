#!/usr/bin/env python3
# Copyright 2026 STAGE Authors. Apache-2.0 License.
"""Verify and inspect the curated STAGE & RPent-Hybrid public rollout dataset.

Usage examples:
    python data/rollouts/inspect_rollouts.py
    python data/rollouts/inspect_rollouts.py --cohort robocasa365_stage_xr1 --limit 5
    python data/rollouts/inspect_rollouts.py --episode robocasa365_stage_xr1__composite_unseen__ArrangeTea__s1761
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROLLOUT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROLLOUT_ROOT.parent.parent


def load_manifest() -> dict:
    return json.loads((ROLLOUT_ROOT / "manifest.json").read_text(encoding="utf-8"))


def iter_records(cohort_filter: str | None = None, include_cases: bool = False):
    manifest = load_manifest()
    for c in manifest["cohorts"]:
        if cohort_filter and c["cohort_id"] != cohort_filter:
            continue
        meta_path = REPO_ROOT / c["metadata_jsonl"]
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    if include_cases:
        cs_path = REPO_ROOT / manifest["coupled_case_studies"]["metadata_jsonl"]
        with cs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def print_summary() -> None:
    manifest = load_manifest()
    print("=" * 94)
    print(f"{manifest['title']}")
    print("=" * 94)
    header = f"{'Benchmark':<13} {'Method':<16} {'Executors':<16} {'Suite A':<18} {'Suite B':<18} {'Avg SR':>8}"
    print(header)
    print("-" * 94)
    for c in manifest["cohorts"]:
        suites = list(c["suites"].items())
        (s1, v1), (s2, v2) = suites[0], suites[1]
        c1 = f"{s1}: {v1['task_sr_pct']:5.2f}%"
        c2 = f"{s2}: {v2['task_sr_pct']:5.2f}%"
        print(
            f"{c['benchmark']:<13} {c['method']:<16} {c['executors']:<16} "
            f"{c1:<18} {c2:<18} {c['avg_task_sr_pct']:>7.2f}%"
        )
    print("-" * 94)

    # Verify all .traj.npz files exist
    missing = 0
    total = 0
    for rec in iter_records(include_cases=True):
        total += 1
        npz_path = REPO_ROOT / rec["trajectory_npz"]
        if not npz_path.is_file():
            missing += 1
    print(
        f"Verified {total - missing}/{total} trajectory .traj.npz archives "
        f"({manifest['main_table_episode_count']} main-table + "
        f"{manifest['coupled_case_study_episode_count']} coupled diagnostic cases)."
    )
    if missing:
        raise SystemExit(f"ERROR: {missing} trajectory files missing!")


def inspect_episode(episode_id: str) -> None:
    target = None
    for rec in iter_records(include_cases=True):
        if rec["episode_id"] == episode_id or episode_id in rec["episode_id"]:
            target = rec
            break
    if target is None:
        raise SystemExit(f"No episode matching {episode_id!r}")

    npz_path = REPO_ROOT / target["trajectory_npz"]
    print(f"Episode ID      : {target['episode_id']}")
    print(f"Benchmark/Suite : {target['benchmark']} / {target['suite']}")
    print(f"Method          : {target['method']} ({target['vlm']} | {target['executors']})")
    print(f"Task (seed)     : {target['task']} (seed={target['seed']})")
    if target.get("task_language"):
        print(f"Instruction     : {target['task_language']}")
    print(
        f"Outcome         : task_success={target['task_success']} "
        f"(first_success_step={target['first_success_step']}, subtask_sr={target['subtask_sr']})"
    )
    print(
        f"Executed Steps  : {target['executed_actions']} / budget={target['action_budget']} "
        f"(VLA={target['vla_actions']}, Code={target['code_actions']})"
    )

    with np.load(npz_path, allow_pickle=False) as z:
        shapes = {k: (z[k].shape, str(z[k].dtype)) for k in z.files}
    print(f"Trajectory NPZ  : {target['trajectory_npz']} -> {shapes}")

    decs = [d for d in target.get("stage_decisions", []) if d.get("decision") in ("start", "advance", "recover", "finish")]
    if decs:
        print(f"\nSTAGE Protocol Transitions ({len(decs)} stage boundaries, {len(target['stage_decisions'])} total reviews):")
        for d in decs[:15]:
            print(
                f"  [step {str(d.get('observation_step', '?')):>4}] {str(d.get('decision', '')).upper():<8} "
                f"{str(d.get('subgoal_id', '')):<6} | {str(d.get('instruction', ''))[:72]}"
            )
        if len(decs) > 15:
            print(f"  ... ({len(decs) - 15} more transitions)")

    phys_calls = [c for c in target.get("tool_calls", []) if c.get("physical_actions", 0) > 0]
    print(f"\nPhysical Executor Calls ({len(phys_calls)} calls):")
    for c in phys_calls[:15]:
        arg_preview = ""
        if isinstance(c.get("arguments"), dict) and "prompt" in c["arguments"]:
            arg_preview = f"prompt={c['arguments']['prompt']!r}"
        elif c.get("arguments"):
            arg_preview = json.dumps(c["arguments"], ensure_ascii=False)[:65]
        print(
            f"  [{c['action_start']:>4}..{c['action_end']:<4}] ({c['physical_actions']:>3} steps) "
            f"{c['executor'].upper():<4} {c['tool']:<16} {arg_preview}"
        )
    if len(phys_calls) > 15:
        print(f"  ... ({len(phys_calls) - 15} more physical calls)")


def list_cohort(cohort_id: str, limit: int) -> None:
    by_suite = defaultdict(list)
    for rec in iter_records(cohort_filter=cohort_id, include_cases=True):
        by_suite[rec["suite"]].append(rec)
    if not by_suite:
        raise SystemExit(f"Unknown cohort {cohort_id!r}")
    for suite, eps in by_suite.items():
        print(f"\n=== {cohort_id} / {suite} ({len(eps)} episodes) ===")
        for rec in eps[:limit]:
            mark = "PASS" if rec["task_success"] else "FAIL"
            print(
                f"  [{mark}] {rec['episode_id']:<68} steps={rec['executed_actions']:>4} "
                f"(VLA={rec['vla_actions']:>4}, Code={rec['code_actions']:>4})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", help="Cohort ID to list sample episodes from")
    parser.add_argument("--episode", help="Episode ID (or substring) to inspect in detail")
    parser.add_argument("--limit", type=int, default=10, help="Number of episodes per suite to display with --cohort")
    args = parser.parse_args()

    if args.episode:
        inspect_episode(args.episode)
    elif args.cohort:
        list_cohort(args.cohort, args.limit)
    else:
        print_summary()


if __name__ == "__main__":
    main()
