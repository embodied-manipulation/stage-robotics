"""Exercise the exported stage resolver with synthetic observations on CPU."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from subgoal_core.protocol import resolve_transition


def review(decision, step, progress, evidence, *, prompt=None, condition=None):
    args = dict(decision=decision, observation_step=step, progress=progress,
                target_visibility="visible", observation_evidence=evidence,
                decision_reason="Select the next action from current scene evidence.")
    if decision != "continue":
        args.update(prompt=prompt, completion_condition=condition)
    if decision in {"advance", "recover"}:
        args["transition_evidence"] = evidence
    return args


def main():
    requests = [
        review("start", 0, "progressing", "The cup is on the counter.",
               prompt="Place the cup on the tray.", condition="The cup rests on the tray."),
        review("continue", 40, "progressing", "The gripper is approaching the cup."),
        review("recover", 80, "blocked", "The approach is blocked by the bowl.",
               prompt="Move the bowl clear of the cup.", condition="The cup approach is clear."),
        review("advance", 120, "complete", "The bowl is clear of the cup approach.",
               prompt="Place the cup on the tray.", condition="The cup rests on the tray."),
    ]
    state = None
    states = []
    for request in requests:
        state = resolve_transition(request, state, record_only=True,
                                   require_decision_reason=True)
        states.append(state)
        print(json.dumps({key: state[key] for key in (
            "decision", "subgoal_id", "assessment_subgoal_id", "parent_subgoal_id",
            "prompt", "completion_condition")}, indent=2))
    assert states[0]["subgoal_id"] == states[1]["subgoal_id"]
    assert states[0]["prompt"] == states[1]["prompt"]
    assert states[2]["parent_subgoal_id"] == states[1]["subgoal_id"]
    assert states[3]["subgoal_id"] != states[2]["subgoal_id"]
    print("Synthetic walkthrough passed; declarations are not native task outcomes.")


if __name__ == "__main__":
    main()
