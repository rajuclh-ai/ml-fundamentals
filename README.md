# ml-fundamentals

Hands-on machine learning projects covering core deep learning concepts — fine-tuning, transfer learning, and NLP classification.

Each project ships with both a **learning notebook** and a **production implementation** (modular `src/`, CLI, pytest tests).

Part of the [rajuclh-ai](https://github.com/rajuclh-ai) portfolio.

---

## Projects

### [bert_sentiment](bert_sentiment/) — BERT Sentiment Analysis

Fine-tuning DistilBERT on IMDb movie reviews for binary sentiment classification.

| | |
|---|---|
| Model | `distilbert-base-uncased` (~67M parameters) |
| Dataset | IMDb — 2,000 train / 500 test |
| Accuracy | **89%** |
| F1 Score | 0.88 – 0.89 |
| Approach | Transfer learning + fine-tuning |

**Stack:** PyTorch · HuggingFace Transformers · HuggingFace Datasets · Click · Pydantic Settings

```
bert_sentiment/
├── notebook/bert_sentiment.ipynb   # 10-step end-to-end training notebook
├── src/                            # Production layer
│   ├── config.py                   # Pydantic-settings config
│   ├── predictor.py                # Predictor class — load model, run inference
│   └── pipeline.py                 # Public API — lazy singleton, batch support
├── tests/                          # pytest suite (all externals mocked)
├── cli.py                          # CLI: predict / batch / info
└── bert_sentiment_model/           # Saved fine-tuned model (safetensors)
```

```bash
python3 cli.py predict "This film was a masterpiece."
python3 cli.py batch --file reviews.txt
python3 -m pytest tests/ -v
```

---

## Related Repositories

- [ai-systems-engineering](https://github.com/rajuclh-ai/ai-systems-engineering) — production-grade AI systems: RAG pipelines, multi-agent PR review, LLM evaluation
