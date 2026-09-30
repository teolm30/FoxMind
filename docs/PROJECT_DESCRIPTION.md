# Project Description

## What I built
FoxMind 49M is a compact language model with exactly **49,430,016 parameters**. It was built to test how much useful reasoning, benchmark accuracy, and instruction following can be achieved under a strict sub-50M parameter budget on consumer hardware.

## Problem
Modern language-model progress often depends on rapidly increasing model size and compute. That makes experimentation difficult for independent developers. FoxMind explores a different direction: improve a small model through better data, adaptive training, held-out evaluation, and strict checkpoint selection.

## How it works
The training pipeline mixes language modeling, multiple-choice supervision, answer ranking, math reasoning, final-answer supervision, deterministic reasoning drills, and instruction-following data.

During a run, FoxMind:
1. tracks active training time;
2. saves resumable checkpoints;
3. evaluates on held-out examples;
4. adapts source weights toward weaker benchmark areas;
5. rejects regressions;
6. exports the strongest accepted checkpoint instead of blindly using the final weights.

The current verified best model is the selected checkpoint from **V16**. A later V17 math-specialist run improved some metrics but was rejected overall, demonstrating the anti-regression system working as intended.

## Current verified V16 selected metrics
- Multiple-choice accuracy: **47.92%**
- Ranking accuracy: **40.97%**
- Reasoning accuracy: **12.50%**
- Exact instruction accuracy: **88.89%**
- Arithmetic accuracy: **6.67%**
- Math accuracy: **6.25%**
- Must-pass checks: **9/10**

The project intentionally reports weaker results rather than hiding them. Free-form math is still the largest limitation.

## Hardware
Training was performed locally with an AMD Radeon RX 7900 XTX (24 GB VRAM), 48 GB system RAM, Python, PyTorch, Hugging Face Transformers, and DirectML.
