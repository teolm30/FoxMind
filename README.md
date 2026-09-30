# FoxMind 49M

**How much intelligence can you squeeze into 49 million parameters?**

FoxMind 49M is a compact language-model project by **teolm30 / Fox AI**. The model has **49,430,016 parameters** and was trained locally on consumer hardware with a pipeline built around held-out evaluation, adaptive training, resumable checkpoints, and automatic anti-regression selection.

> Current verified best checkpoint: **V16 selected checkpoint**.  
> V17 was a reasoning/math specialization experiment, but its selector correctly kept V16 because the later weights did not improve the overall held-out score.

## Why FoxMind 49M?

Large language models usually improve by scaling parameter count and compute. FoxMind explores the opposite constraint: keep the model below 50M parameters and use better training strategy, measurement, and checkpoint selection to extract as much capability as possible.

The pipeline combines:
- language-model retention;
- multiple-choice supervision;
- candidate ranking;
- human-written math reasoning;
- final-answer math supervision;
- deterministic arithmetic/reasoning drills;
- instruction-following data;
- adaptive benchmark-source weighting;
- repeated held-out evaluation;
- automatic best-checkpoint preservation.

## Current verified results

These are the saved **V16 selected-checkpoint** metrics from `results/v16_run.json`.

| Metric | Result |
|---|---:|
| Parameters | **49,430,016** |
| Multiple-choice accuracy | **47.92%** |
| Ranking accuracy | **40.97%** |
| Reasoning accuracy | **12.50%** |
| Exact instruction accuracy | **88.89%** |
| Arithmetic accuracy | **6.67%** |
| Math accuracy | **6.25%** |
| Must-pass checks | **9 / 10** |
| V16 tokens processed | **125,663,232** |
| V16 active training time | **21,601 s (~6 h)** |
| Selected checkpoint time | **13,200 s (~3.67 h)** |

The weaker math/arithmetic numbers are intentionally reported rather than hidden; they are the main remaining limitation.

## Hardware

Primary training machine:
- AMD Radeon RX 7900 XTX, 24 GB VRAM
- 48 GB DDR4 system RAM
- Windows
- PyTorch + DirectML

V16 averaged about **5.8k processed tokens/s** across the whole run. Throughput varies significantly by training objective.

## Repository layout

```text
FoxMind/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ model.py
│  ├─ train_v16_final_competition.py
│  ├─ train_v17_reasoning_math.py
│  ├─ run_model.py
│  └─ evaluate.py
├─ data_prep/
│  ├─ prepare_v6_judge_data.py
│  ├─ prepare_v9_generalization.py
│  └─ prepare_v10_reasoning.py
├─ results/
│  ├─ v16_run.json
│  └─ v17_run.json
├─ docs/
│  ├─ PROJECT_DESCRIPTION.md
│  ├─ BUILT_WITH.md
│  ├─ DATASETS.md
│  ├─ METHODOLOGY.md
│  ├─ DEMO_SCRIPT.md
│  └─ DEVPOST_CHECKLIST.md
└─ screenshots/
   └─ README.md
```

The original `foxmind.html` in this repository is an experimental local-AI interface prototype and is separate from the training/evaluation pipeline.

## Prerequisites

Recommended environment for reproducing the original training setup:
- Windows 10/11
- Python 3.11
- a DirectML-compatible GPU
- enough free storage for downloaded datasets and generated token files
- at least 24 GB VRAM for the same batch sizes used in the saved runs

The core model architecture itself is standard Hugging Face Transformers code and can be adapted to CUDA/CPU, but the competition training scripts use `torch_directml` directly.

## Setup

```bash
git clone https://github.com/teolm30/FoxMind.git
cd FoxMind

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare datasets

The public preparation scripts use Hugging Face Datasets and generate the local files expected by the training pipeline.

```bash
python data_prep/prepare_v6_judge_data.py
python data_prep/prepare_v9_generalization.py
python data_prep/prepare_v10_reasoning.py
```

See `docs/DATASETS.md` for the source list and split policy.

Some earlier local corpora used by the full historical training chain are not committed because the generated binary token files are large. The preparation/source scripts and manifests document the relevant data pipeline.

## Model architecture

`src/model.py` defines the 49,430,016-parameter Llama-style architecture:

- vocabulary size: 8,192
- hidden size: 512
- intermediate size: 1,536
- layers: 15
- attention heads: 8
- key/value heads: 2
- maximum positions: 2,048
- tied input/output embeddings

Run:

```bash
python src/model.py
```

or import `build_model()` to instantiate it.

## Running the trained model

The final V16 model directory contains a ~198 MB `model.safetensors` file, which is larger than GitHub's normal file limit and therefore is not stored directly in this source repository.

Place the exported Hugging Face model folder at:

```text
checkpoints/v16_final_competition/final_hf/
```

It should contain:
- `config.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

Then run:

```bash
python src/run_model.py
```

## Training

V16 is the current verified best competition training recipe:

```bash
python src/train_v16_final_competition.py
```

The trainer:
1. loads the previous selected checkpoint;
2. mixes multiple objectives;
3. evaluates at fixed active-time intervals;
4. adapts source weights from held-out errors;
5. saves resumable checkpoints;
6. uses anti-regression gates;
7. exports the strongest accepted checkpoint instead of blindly taking the final weights.

V17 is included because it demonstrates the anti-regression approach: a specialist run can improve some metrics while still being rejected overall.

## Evaluation

The saved run metadata is in `results/`. The training scripts keep training examples separate from their local held-out sets, and their policy excludes official benchmark validation/test examples from gradient training.

To inspect the final saved metrics:

```bash
python -c "import json; print(json.dumps(json.load(open('results/v16_run.json'))['selected_metrics'], indent=2))"
```

## Data / benchmark sources

The project uses or prepares data from datasets including:
- GSM8K
- MATH
- SVAMP
- HellaSwag
- ARC Easy / ARC Challenge
- PIQA
- WinoGrande
- OpenBookQA
- CommonsenseQA
- SciQ
- BoolQ
- WikiText-103
- additional general/instruction corpora used by earlier training stages

See `docs/DATASETS.md` for details.

## Competition submission material

Devpost-ready text and the video plan are included in `docs/`:
- project description;
- complete Built With list;
- methodology;
- dataset disclosure;
- demo-video script;
- six-item submission checklist.

## Reproducibility note

This repository publishes the source, data-preparation logic, saved evaluation metadata, and final training recipes. Large generated token binaries, optimizer checkpoints, and the ~198 MB model weight file are intentionally excluded from normal Git history.

## Limitations

FoxMind 49M is an experimental small model, not a production assistant. Its strongest areas in the current evaluation are exact-format instruction following and multiple-choice behavior. Free-form math, arithmetic, and broader reasoning remain substantially weaker.

## Author

**teolm30 / Fox AI**

Built for the Global Innovation Build Challenge V2.
