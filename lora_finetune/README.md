# LoRA Fine-tuning Pipeline — Code Review Quality Scorer

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` using LoRA/PEFT to automatically score and classify code review comments by severity. Training data is generated synthetically using GPT-4o-mini.

Companion project to [`pr_review_agent`](https://github.com/rajuclh-ai/ai-systems-engineering/tree/main/pr_review_agent) — that agent generates the comments; this model scores them.

---

## What It Does

| Input | Output |
|---|---|
| `"Variable name 'x' is not descriptive"` | `{ "score": 1, "label": "nitpick", "severity": "low" }` |
| `"SQL query passes user input without sanitization"` | `{ "score": 5, "label": "blocking", "severity": "critical" }` |
| `"This loop is O(n²), consider using a dict"` | `{ "score": 3, "label": "suggestion", "severity": "medium" }` |

**Score schema:**

| Score | Label | Severity | Meaning |
|---|---|---|---|
| 1 | nitpick | low | Style, naming, minor preference |
| 2 | suggestion | low | Improvement but not required |
| 3 | suggestion | medium | Noticeable issue, should fix |
| 4 | warning | high | Bug risk, logic issue |
| 5 | blocking | critical | Security, correctness, must fix |

---

## Why LoRA?

Training all 1.5 billion parameters of Qwen2.5-1.5B requires significant GPU RAM. LoRA (Low-Rank Adaptation) freezes the base model and inserts small trainable adapter matrices — only **~0.14% of total parameters** are trained. Same task-specific result, fraction of the compute.

```
Base model (frozen, 1.5B params)
    +
LoRA adapters (trainable, ~2.2M params)   ←  only these update during training
    =
Fine-tuned model that scores code review comments
```

---

## Two Implementations

This project follows the same pattern as [`bert_sentiment`](../bert_sentiment) — a notebook for learning and a production `src/` layer for portfolio:

| | Notebook | Production (`src/`) |
|---|---|---|
| **Purpose** | Step-by-step learning, experimentation | Modular, testable, CLI-driven |
| **Audience** | You learning the concepts | Hiring managers reading the code |
| **Run via** | Jupyter cell by cell | `python cli.py train` |
| **Tests** | None | pytest suite |

---

## Architecture

```
lora_finetune/
├── notebook/
│   └── lora_finetune.ipynb        # End-to-end walkthrough: data → train → eval → infer
│                                  # Self-contained, no src/ imports
├── src/
│   ├── config.py                  # Pydantic-settings — reads .env + lora_config.yaml
│   ├── prepare_data.py            # Generate labeled examples via GPT-4o-mini
│   ├── train.py                   # LoRA fine-tuning loop (SFTTrainer + PEFT)
│   ├── evaluate.py                # Accuracy, F1, base vs fine-tuned comparison table
│   └── infer.py                   # Load adapter + score a single comment
├── configs/
│   └── lora_config.yaml           # All hyperparams — reproduce any run from this file
├── data/
│   ├── train.jsonl                # Generated training examples (gitignored)
│   └── test.jsonl                 # Held-out evaluation set (gitignored)
├── adapter/                       # Saved LoRA adapter weights (gitignored)
├── cli.py                         # Unified CLI entry point
├── tests/
│   ├── test_prepare_data.py
│   ├── test_evaluate.py
│   └── test_infer.py
├── requirements.txt
└── .env.example
```

---

## Pipeline Flow

```
Step 1: prepare_data.py
  └─ Calls GPT-4o-mini across 6 categories to generate labeled (comment, score, label) pairs
  └─ Splits 85/15 → data/train.jsonl and data/test.jsonl

Step 2: train.py
  └─ Loads Qwen/Qwen2.5-1.5B-Instruct from HuggingFace
  └─ Applies LoRA config (rank=16, alpha=32, target: q_proj, v_proj)
  └─ Formats examples using Qwen ChatML template
  └─ Trains with SFTTrainer (trl 0.29+)
  └─ Saves adapter weights to ./adapter/

Step 3: evaluate.py
  └─ Loads base model and fine-tuned model sequentially
  └─ Reports accuracy + F1 per label class for both
  └─ Prints side-by-side comparison table

Step 4: infer.py
  └─ Loads base model + adapter from ./adapter/
  └─ Accepts a comment string, returns structured JSON
```

---

## Prompt Template (Qwen ChatML Format)

```
<|im_start|>system
You are a code review classifier. Given a code review comment, output a JSON object
with: score (integer 1-5), label (nitpick | suggestion | warning | blocking),
and severity (low | medium | high | critical). Output only the JSON object, nothing else.
<|im_end|>
<|im_start|>user
{comment}
<|im_end|>
<|im_start|>assistant
{"score": 4, "label": "warning", "severity": "high"}
<|im_end|>
```

Training examples include the assistant answer. Inference prompts stop before `assistant` and let the model generate.

---

## Requirements

```
# Core ML
torch>=2.1.0
transformers>=4.57.0
peft>=0.17.0
trl>=0.29.0
datasets>=2.18.0
accelerate>=0.28.0

# Data generation
openai>=1.0.0

# Config & CLI
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
click>=8.0.0
pyyaml>=6.0

# Testing
pytest>=7.0.0
pytest-mock>=3.0.0
```

---

## Environment Variables

```bash
# .env
OPENAI_API_KEY=...    # For synthetic data generation
HF_TOKEN=...          # HuggingFace token — free at huggingface.co/settings/tokens
```

> No license acceptance required for `Qwen/Qwen2.5-1.5B-Instruct`. To switch to Llama 3.2 3B, accept the license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct and update `model.name` in `configs/lora_config.yaml`.

---

## LoRA Hyperparameters (`configs/lora_config.yaml`)

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  max_length: 256

lora:
  r: 16                     # Rank — controls adapter expressiveness
  lora_alpha: 32            # Scaling factor (typically 2x rank)
  target_modules:           # Attention layers to adapt
    - q_proj
    - v_proj
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 0.0002     # use explicit float — 2e-4 parses as string in PyYAML
  warmup_ratio: 0.03
  weight_decay: 0.001
  logging_steps: 10
  save_steps: 50
  output_dir: ./adapter
```

---

## CLI Usage

```bash
# Step 1 — generate training data (~30s, ~$0.01)
python cli.py prepare

# Step 2 — fine-tune (~2 min on M3 Max)
python cli.py train --config configs/lora_config.yaml

# Step 3 — evaluate base vs fine-tuned
python cli.py eval --adapter ./adapter

# Step 4 — score a single comment
python cli.py infer "This SQL query passes user input without sanitization"
# {"score": 4, "label": "warning", "severity": "high"}

# Run tests (no model loading required)
pytest tests/ -v
```

---

## Hardware Notes

| Hardware | Works? | Notes |
|---|---|---|
| **Mac Apple Silicon (M1/M2/M3)** | Yes | MPS backend. 1.5B model uses ~3GB RAM. Training ~2 min |
| Mac Intel | Slow | CPU only — training will take 30–60 min |
| Google Colab T4 (free) | Yes | CUDA, faster iteration |
| GPU (CUDA) | Best | Can use larger models (Llama 3.2 3B, Mistral 7B) |

---

## Actual Results (M3 Max, Qwen2.5-1.5B, 306 train examples)

| Metric | Base Model | Fine-tuned |
|---|---|---|
| Training time | — | ~2 min |
| Trainable params | — | 2.2M / 1.54B (0.14%) |
| Training loss | — | 2.67 → 0.31 |
| Token accuracy (train) | — | 91.4% |
| Label accuracy (test) | 50.0% | 54.8% |

**On label accuracy:** The improvement is modest because the dataset is small (~300 examples). The training loss curve (2.67 → 0.31) and token accuracy (91.4%) confirm the model is learning the schema. To improve test accuracy: increase `--n-train` to 1000+, bump epochs to 5, or switch to Llama 3.2 3B.

---

## Setup

```bash
# 1. Install dependencies
cd ml-fundamentals/lora_finetune
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in OPENAI_API_KEY and HF_TOKEN

# 3. Run the pipeline
python cli.py prepare          # generate data
python cli.py train            # fine-tune
python cli.py eval             # compare base vs fine-tuned
python cli.py infer "comment"  # score a comment
```

---

## Portfolio Context

This project demonstrates the full ML lifecycle with modern techniques:

- **Synthetic data generation** — using an LLM (GPT-4o-mini) to create labeled training data
- **Parameter-efficient fine-tuning** — LoRA adapters (0.14% of params), not full fine-tuning
- **Instruction tuning format** — ChatML prompt template, the standard for generative fine-tuning
- **Reproducible config** — all hyperparams in `lora_config.yaml`, any run reproducible from file
- **Base vs fine-tuned comparison** — quantified, not just "it works"
- **Production structure** — `src/`, CLI, pytest, pydantic-settings

Complements [`bert_sentiment`](../bert_sentiment) (DistilBERT encoder, full fine-tuning, classification) by showing decoder/generative fine-tuning with PEFT — a distinct and more current skill set.
