# BERT Sentiment Analysis

Fine-tuning a pre-trained Transformer model (DistilBERT) to classify movie reviews as **Positive** or **Negative**.

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

## Project Structure

```
bert_sentiment/
├── bert_sentiment.ipynb     # Main notebook — run this
├── bert_sentiment_model/    # Saved model after training (generated)
├── training_curves.png      # Loss and accuracy plots (generated)
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install torch transformers datasets scikit-learn matplotlib accelerate
```

**2. Open the notebook**
```bash
jupyter notebook bert_sentiment.ipynb
```

**3. Run all cells top to bottom**

No API key needed. The IMDb dataset and DistilBERT model are downloaded automatically from Hugging Face (free).

---

## Notebook Steps

| Step | What it does |
|---|---|
| 1 | Load IMDb dataset |
| 2 | Tokenize reviews into BERT token IDs |
| 3 | Load pre-trained DistilBERT + classification head |
| 4 | Define accuracy and F1 evaluation metrics |
| 5 | Configure training (epochs, batch size, learning rate) |
| 6 | Fine-tune the model for 3 epochs |
| 7 | Evaluate — accuracy, F1, classification report, training curves |
| 8 | Save the fine-tuned model to disk |
| 9 | Run inference on custom reviews |
| 10 | Production considerations for serving at scale |

---

## Tech Stack

- [PyTorch](https://pytorch.org/) — neural network training
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — DistilBERT model and tokenizer
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) — IMDb dataset
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics
- [Matplotlib](https://matplotlib.org/) — training curve visualisation

---

## Key Concepts Demonstrated

- **Transfer learning** — reusing a pre-trained model instead of training from scratch
- **Fine-tuning** — adapting a general model to a specific task with minimal data
- **Tokenization** — converting raw text into numerical input a model can process
- **F1 Score** — balanced evaluation metric for classification tasks
- **Production tradeoffs** — batching, latency, model versioning, scaling
