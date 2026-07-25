# BERT Sentiment Analysis

Fine-tuning a pre-trained Transformer model (DistilBERT) to classify movie reviews as **Positive** or **Negative**.

---

## Two Implementations

| | Notebook | Production (`src/`) |
|---|---|---|
| **Location** | `notebook/bert_sentiment.ipynb` | `src/` + `cli.py` |
| **Purpose** | Learn and experiment step-by-step | Inference on new text, ready to deploy |
| **Runs** | Jupyter | `python3 cli.py predict "..."` |
| **Includes** | Training, evaluation, visualisation | Predictor class, pipeline, CLI, pytest tests |
| **When to use** | Understand the model or retrain | Run predictions without touching the notebook |

---

## What This Project Does

Given a movie review like:
> *"This film was a complete masterpiece. I was blown away by the performances."*

The model predicts:
```
Sentiment : POSITIVE
Confidence: 97.4%
```

---

## Why This Is Interesting

Traditional approaches count words like "good" or "bad" — they miss context.

> *"The movie was not bad at all"* → a word counter sees "bad" and wrongly predicts Negative.

DistilBERT reads the **full sentence in both directions**, understanding context, negation, and nuance — just like a human would.

---

## Dataset

| | Detail |
|---|---|
| Source | [IMDb Movie Reviews](https://huggingface.co/datasets/imdb) via Hugging Face |
| Total size | 50,000 reviews |
| Used for training | 2,000 |
| Used for testing | 500 |
| Classes | Positive / Negative (balanced 50/50) |

---

## Model

| | Detail |
|---|---|
| Base model | `distilbert-base-uncased` |
| Approach | Fine-tuning (not training from scratch) |
| Parameters | ~67 million |
| Training epochs | 3 |

**DistilBERT** is a compressed version of BERT — 40% smaller, 60% faster, retains 97% of performance. It was pre-trained on 3.3 billion words and already understands English deeply. Fine-tuning teaches it one specific new skill: sentiment classification.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 89% |
| F1 Score (Negative) | 0.88 |
| F1 Score (Positive) | 0.89 |

Achieved **89% accuracy using only 2,000 training examples** — demonstrating the power of transfer learning.

---

## Project Structure

```
bert_sentiment/
├── notebook/
│   └── bert_sentiment.ipynb     # End-to-end training notebook (10 steps)
├── src/
│   ├── config.py                # Pydantic-settings config (model dir, device, batch size)
│   ├── predictor.py             # Predictor class — loads model, predict / predict_batch
│   └── pipeline.py             # Public API — lazy singleton, predict_sentiment, batch_predict
├── tests/
│   ├── conftest.py              # Shared fixtures (mock model, tokenizer, predictor)
│   ├── test_predictor.py        # Unit tests for Predictor class
│   └── test_pipeline.py        # Unit tests for pipeline functions
├── cli.py                       # Click CLI — predict, batch, info commands
├── requirements.txt
├── bert_sentiment_model/        # Saved fine-tuned model (safetensors)
└── training_curves.png          # Loss and accuracy plots
```

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Predict sentiment for a single review**
```bash
python3 cli.py predict "This movie was absolutely fantastic!"
```
```
Text      : This movie was absolutely fantastic!
Sentiment : POSITIVE
Confidence: 98.2%
```

**3. Batch predict from a file** (one review per line)
```bash
python3 cli.py batch --file reviews.txt
```

**4. Show model info**
```bash
python3 cli.py info
```

**5. Run the notebook** (training from scratch)
```bash
jupyter notebook notebook/bert_sentiment.ipynb
```

**6. Run tests**
```bash
python3 -m pytest tests/ -v
```

---

## How It Works

```
Movie review text
      ↓
Tokenizer         — splits text into tokens, converts to numbers
      ↓
DistilBERT        — 6 transformer layers understand the full context
      ↓
[CLS] embedding   — single vector summarising the whole review
      ↓
Classifier head   — maps vector → Positive or Negative
      ↓
Label + Confidence score
```

---

## How This Differs from the LoRA Project

This repo contains **two fine-tuning projects on purpose** — `bert_sentiment` and [`lora_finetune`](../lora_finetune). They look similar ("fine-tune a pretrained model") but sit on opposite ends of the fine-tuning map. Understanding *why you'd pick each* is the real skill.

> **One-line difference:** `bert_sentiment` is **full fine-tuning** of a small **encoder** for **classification**; `lora_finetune` is **LoRA/PEFT** on a large **decoder** for **structured generation**.

| Dimension | `bert_sentiment` (this project) | [`lora_finetune`](../lora_finetune) |
|---|---|---|
| Base model | DistilBERT (~67M params) | Qwen2.5-1.5B (~1.5B params) |
| Architecture | **Encoder-only** (BERT family) | **Decoder-only** (GPT/Qwen family) |
| Task type | **Classification** — pick 1 of N labels | **Generative** — produce JSON text |
| Fine-tuning method | **Full fine-tuning** (all weights update) | **LoRA / PEFT** (~0.14% of weights) |
| How it outputs | Classification head → softmax → label + confidence | Generates tokens → parse JSON |
| Loss function | Cross-entropy over 2 classes | Cross-entropy over next-token |
| Training data | **Real** — IMDb (2,000 train) | **Synthetic** — generated by GPT-4o-mini |
| Output artifact | Full model copy (`bert_sentiment_model/`) | Small adapter file (`adapter/`) |
| Result | 89% acc, 0.88 F1 | 53% → 64.5% acc (base vs tuned) |

**The three differences that matter:**

1. **Encoder vs decoder.** DistilBERT *reads* the whole input and a native classification head maps it to a label — one forward pass, one softmax. Qwen *generates* the answer token by token, so it classifies as a side effect of writing JSON. BERT was built for classification; Qwen does it by generating.
2. **Full FT vs LoRA.** Full fine-tuning all 67M of DistilBERT is cheap, so this project updates every weight and saves a complete model. Full fine-tuning 1.5B Qwen on a laptop is impractical, so `lora_finetune` freezes the base and trains a tiny adapter. **Same idea, chosen to fit the constraint.**
3. **Real vs synthetic data.** This project uses a benchmark dataset (IMDb); `lora_finetune` bootstraps its own dataset with a larger LLM when no labeled data exists.

**Why the evaluation framing differs:** DistilBERT has no sentiment head before training, so there's no meaningful "before" — you just report the final 89%. Qwen can already attempt the task untrained, so the honest measure is the *lift* the adapter adds (53% → 64.5%), which is why that project evaluates base vs fine-tuned.

---

## Tech Stack

- [PyTorch](https://pytorch.org/) — neural network training and inference
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — DistilBERT model and tokenizer
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) — IMDb dataset
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics (notebook only)
- [Matplotlib](https://matplotlib.org/) — training curve visualisation (notebook only)
- [Click](https://click.palletsprojects.com/) — CLI
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — configuration

---

## Key Concepts Demonstrated

- **Transfer learning** — reusing a pre-trained model instead of training from scratch
- **Fine-tuning** — adapting a general model to a specific task with minimal data
- **Tokenization** — converting raw text into numerical input a model can process
- **F1 Score** — balanced evaluation metric for classification tasks
- **Production tradeoffs** — batching, latency, model versioning, scaling
