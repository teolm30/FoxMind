# Demo Video Script (2–5 minutes)

## 0:00–0:20 — Intro
"This is FoxMind 49M, a language model with 49,430,016 parameters. I built it to test how far a very small model can be pushed using targeted training and strict evaluation instead of parameter scale."

Show the repository and parameter count.

## 0:20–0:55 — Architecture and training
Show `src/model.py`, then `src/train_v16_final_competition.py`.

Explain:
- under 50M parameters;
- trained locally;
- multiple objectives;
- active-time tracking;
- automatic checkpoint selection.

## 0:55–1:30 — Results
Open `results/v16_run.json` and show the selected metrics.

Emphasize that the selected checkpoint happened around 3.67h, before the 6h endpoint.

## 1:30–2:15 — Live model
Run:
```bash
python src/run_model.py --prompt "What is 17 - 9?"
```

Then show 2–4 more unseen questions covering reasoning, strict output, and multiple choice.

Do not cut out failures; showing real limitations makes the methodology credible.

## 2:15–2:45 — What is innovative
Explain adaptive source weighting, anti-regression gates, and why V17 was rejected even after four more hours of specialization.

## 2:45–3:00 — Close
"The goal of FoxMind is efficiency: making a small model more capable through better training decisions instead of just making it larger."
