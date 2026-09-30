# Methodology

FoxMind is trained as a sequence of measured experiments rather than one uninterrupted run.

## Core ideas
- **Active-time accounting:** sleep/idle gaps are not counted as training time.
- **Mixed objectives:** LM, general SFT, MC classification, candidate ranking, math rationale SFT, answer-only SFT, reasoning drills, and exact-output instruction tasks.
- **Adaptive source weighting:** weaker benchmark sources receive more sampling weight.
- **Frequent evaluation:** held-out metrics are checked repeatedly during runs.
- **Anti-regression gates:** a checkpoint must preserve important baseline abilities before it can replace the current best.
- **Best-checkpoint export:** the final model can come from an earlier point in training if later steps regress.

## Why V16 was selected
V16's best checkpoint occurred at about **3.67 active hours** inside a 6-hour run. Later weights were weaker overall, so the system retained the earlier checkpoint.

## Why V17 was rejected
V17 specialized heavily in math/reasoning. It improved ranking and general loss, but did not improve math/reasoning accuracy enough and reduced MC accuracy, so the selector kept V16.

This behavior is intentional and is part of the project's main contribution: more training is not automatically treated as better training.
