# LIBERO benchmark-native Subgoal language candidate

Historical v1 status: one subsequently authorized LongTask task00 / seed 0 pilot completed
with this candidate on 2026-10-24: native whole-task result 0/1. This establishes
runtime feasibility, not a benefit caused by the language profile. The confirmed
20-episode batch remains paused. This document does not authorize more jobs,
model calls or comparison episodes. The Planner-guidance block below is unchanged.

Target artifact: `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`. This is the selected
SFT checkpoint; an RL-trained pi_RL policy is a separate artifact. Its public
Hub page currently has no model card with a complete training-prompt inventory:
https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT

The official RLinf interface consumes the environment task description as the
VLA prompt:
https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html#observation-and-action

## Optional Planner guidance

```text
LIBERO POLICY LANGUAGE
Describe the current Subgoal as a short LIBERO-style task outcome. Prefer the
native task's object names and result relations. Typical constructions are
"put [object] in/on [destination]", "pick [object] and place it in/on
[destination]", "open/close [drawer or door]", and "turn on/off [appliance]".

Choose one outcome that the policy can pursue continuously. A placement outcome
can include approach, grasp, lift, transport and placement within that one skill.
Keep separately verifiable task requirements separate. Preserve the live task's
object identities, destination and relations, including changed objects in PRO.
Ground scene-relative disambiguation in the current images.

Use the accepted instruction verbatim while continuing that Subgoal. Keep camera
references, finger/wrist commands, trajectories, alternatives and explanatory
reasoning in observation_evidence and decision_reason. Put the visible expected
result in completion_condition. Only prompt is forwarded to the VLA.

Use available benchmark-language examples as linguistic templates. Choose the
next goal from the current scene and unmet task requirements. Preserve the
standard start/continue/advance/recover decisions, fixed observation cadence and
Subgoal visual-window rules.
```

This profile does not add task solutions, successful seeds, prior trajectories
or task-specific recovery recipes. It does not rewrite a Planner instruction in
the backend or change a task to match a familiar training label.

## What the evidence supports

`libero_standard_instruction_reference.json` contains 130 standard-suite BDDL
instruction labels (Spatial/Object/Goal/Long/90). They are benchmark references,
not a verified dump of this checkpoint's training corpus. An exact label match
establishes benchmark-language membership only. A composed phrase is a
language-style match, not proof of in-distribution training exposure. Even a
verified training phrase does not establish that the current image, robot state
and instruction jointly lie in the training distribution.

## Comparative question

Hypothesis: language closer to the policy's training task style reduces failures
caused by unhelpful micro-action instructions or paraphrases. The competing
explanation is that changed Subgoal scope, rather than wording, causes any gain.

For a wording comparison, use the same task/seed/reset, checkpoint, action budget,
feedback, classification, cadence and visual window. Compare the baseline with
this language profile; report native whole-task success, prompt compliance,
decision counts and recovery cost. Each task/seed episode is one comparison unit;
many Subgoals within it are process measurements, not independent success trials.

If the Planner also changes Subgoal granularity, label the result as a combined
language-and-scope intervention. A later scope control can separate the effects.
An extra paired 20-episode control is a new batch, not part of the currently
confirmed 20-episode pilot. No extra experiment is launched here.

## Planner guidance: libero-benchmark-native-language/v2

```text
LIBERO POLICY LANGUAGE AND FEASIBILITY
Use short object-and-goal-relation instructions for both normal execution and
recovery. The local standard LIBERO instruction reference supplies language
patterns, not task solutions or a verified inventory of this checkpoint's
training prompts. Useful patterns are "put [object] in/on [destination]",
"pick [object] and place it in/on [destination]", "open/close [drawer or door]",
and "turn on/off [appliance]".

Preserve the actual objects, target and required relations in the live PRO task
and current scene. Keep descriptors needed to distinguish targets; omit
irrelevant adjectives. Never substitute a standard-task object or destination
to match a familiar sentence: a requested wine bottle must remain the wine
bottle, not become a black bowl from a reference example. Choose the task from
the live requirements, and use reference patterns only to express that goal.

Before accepting a goal, check its current prerequisites and feasibility from
images and available feedback: target identity and visibility, support/holding,
access and relevant clearance. If a prerequisite is blocked, select a feasible
object-level outcome that resolves it and preserves achieved relationships.
Do not issue an inaccessible final goal merely because its wording is familiar.
An adjacent object is not by itself evidence of an obstruction. When feasibility
is uncertain, state what is uncertain and use available observations to resolve
it rather than inventing a blocking condition or an unsupported recovery skill.

A coherent placement can include approach, grasp, transport and release. Let
Pi0.5 choose the movements within that outcome. Avoid standalone finger closure,
wrist rotation, gripper retreat or under-handle lifting as recovery Subgoals:
these are not established independent skills for this checkpoint. Geometric
reasoning in the shared recovery guidance belongs in observation_evidence and
decision_reason; express the next VLA instruction as a feasible task outcome.
Record its visible completion criterion in completion_condition. Only prompt
is passed to the VLA, verbatim. If no executable next outcome is supported by
the current evidence, report that capability/feasibility uncertainty rather
than pretending an arbitrary motion instruction is a known skill.

Keep the accepted prompt stable on continue. Rewording the same ineffective
attempt is not a different recovery strategy. Use subgoal_status when supplied
to compare persistence with the actual observed movement and remaining goal;
no count alone requires a switch. Preserve start/continue/advance/recover,
the fixed observation cadence, visual-window rules and existing finish rules.
```

V2 is the user's 2026-10-25 language-and-feasibility adaptation of the shared
XR-1 protocol. The first guidance block remains the historical v1 text. Fresh
manifests select v2 explicitly; historical manifests and results keep their
recorded profile. Only the compact guidance is delivered to the Planner; the
130-label JSON remains a benchmark-language reference, not a task recipe bank.
No backend substitution, policy-state modification or rollout is implied.
