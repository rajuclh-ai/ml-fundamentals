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
