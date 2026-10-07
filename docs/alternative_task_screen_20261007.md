# Alternative RoboCasa task screen (2026-10-07)

## Data and method

`scripts/rank_alternative_tasks.py` reads episode metadata from the cached
DAVIAN-Robotics/robocasa-H50 snapshot `7e9802803a02be938683ea3f14e56e877d1c1ef7`
and Keh0t0/dreamgen_robocasa_p3_init0_v2_droid snapshot
`686cd91d4502b0606e473b07adb73c920b05d4e3`. It counts H50 human
episodes whose *complete task instruction* exactly matches each of the 24
DreamGen instructions. The full table is `records/alternative_task_h50_screen.csv`.

The DreamGen snapshot contains 60 episodes per task, but does **not** have a
task-success label. `next.done` denotes an episode boundary. Those 60 episodes
are candidates for video review, **not** 60 successes. Existing OpenDrawer
labels (21 visually positive, 33 negative, 6 uncertain) are an assistant
single-camera visual proxy, not simulator success.

DreamGen is 16 fps and its actions/states come from a droid IDM; H50 is 20 fps.
Any joint training requires temporal conversion plus checks of action/state
meaning. Prompt overlap alone is not training compatibility.

Source cards: [H50](https://huggingface.co/datasets/DAVIAN-Robotics/robocasa-H50),
[DreamGen](https://huggingface.co/datasets/Keh0t0/dreamgen_robocasa_p3_init0_v2_droid),
and [current RoboCasa atomic task definitions](https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html).

## Human-first shortlist

| Task | H50 exact human | MG30 exact real | DreamGen clips | Why inspect next |
| --- | ---: | ---: | ---: | --- |
| StartCoffeeMachine (older name: CoffeePressButton) | 54 | 30 | 60 | Single button press; coffee dispensing may give a visible outcome. |
| TurnOnSinkFaucet | 54 | 30 | 60 | Single lever task; water flow may be visible. |
| TurnOffSinkFaucet | 54 | 30 | 60 | Same family with a contrasting initial state and visual water stop. |
| TurnOnMicrowave | 54 | 30 | 60 | Single button press; video may not show the small state indicator clearly. |
| TurnOffMicrowave | 54 | 30 | 60 | Same caveat as TurnOnMicrowave. |

OpenDrawer has only 28 exact H50 episodes and 60 DreamGen clips. Other tasks
with 54 human episodes include CloseDoubleDoor, CoffeeSetupMug, and
CoffeeServeMug, but their success requires multi-step door or pick/place motion.
The one-step tasks above are a *mechanical-complexity hypothesis*, not measured
policy success or verified DreamGen success. The cached trajectory-variation
scan also puts these button/faucet tasks among the least variable of the 24,
but that statistic does not prove they are easy.

## Audit of the left/right concern

The H50 counts above are **exact complete-language matches**, not totals for a
task family. I enumerated every H50 and DreamGen task string in the relevant
families. H50 has 28 `open the left drawer` and 26 `open the right drawer`;
the DreamGen snapshot has 60 left and no right. This reproduces the split that
made OpenDrawer unsuitable for a simple pooled comparison.

For the candidates above, H50 has 54 episodes for **each specific complete
instruction**, and DreamGen has 60 for the same complete instruction. In H50,
`turn on the sink faucet` and `turn off the sink faucet` are 54 **separate**
episodes each; `press the start button on the microwave` and `press the stop
button on the microwave` are 54 **separate** episodes each. The coffee button
instruction has 54 episodes and no opposite-side coffee-button instruction.

I also inspected the pinned RoboCasa task classes. `TurnOnSinkFaucet` and
`TurnOffSinkFaucet` each hardcode one behavior and one success predicate;
`TurnOnMicrowave` and `TurnOffMicrowave` each hardcode the start or stop button;
`StartCoffeeMachine` has one button goal and one success predicate. These
classes have no task-level left/right branch. By contrast, `TurnSinkSpout`
explicitly chooses a left/right target and H50 has 19 left plus 35 right, so
exclude it from the first screen. RoboCasa's current atomic task table also
lists OpenDrawer and TurnSinkSpout as left/right tasks and the shortlisted
button/faucet tasks separately.

This establishes **same stated goal**, not identical physical initial states.
The cached LeRobot episode metadata records the task string, length, and video
offsets, but no per-episode kitchen layout, fixture instance, button placement,
or simulator reset state. Sink tasks also randomize distractors. Before a paired
rollout, pin the simulator version, task class, layout/fixture configuration,
seeds, and camera mapping, and inspect representative human and synthetic
videos. We should not call these episodes `zero distinction` beyond their
task-level objective.

The cached DreamGen media contains only the left camera locally. A valid
synthetic success audit for the finalists still needs all three views or the
original four-view tile; no synthetic success counts are claimed here.

## Next decision gate

1. Review all 60 clips for CoffeePressButton and TurnOnSinkFaucet, then
   TurnOffSinkFaucet or TurnOnMicrowave if needed. Inspect all three cameras.
   Label `success`, `failure`, or `uncertain` using visible task completion;
   track action/contact plausibility separately.
2. Report success/60 and an uncertainty interval; never treat uncertain clips
   as confirmed success.
3. Prefer a task with many confirmed successful synthetic clips, a clearly
   observable outcome, exact human prompt coverage, and non-ceiling untouched
   human300 rollout success on fixed seeds. Only then prepare a new LoRA run.
