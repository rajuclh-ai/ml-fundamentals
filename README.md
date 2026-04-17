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

### [lora_finetune](lora_finetune/) — LoRA Fine-tuning: Code Review Quality Scorer

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` with LoRA/PEFT to automatically score and classify code review comments by severity. Training data is generated synthetically using GPT-4o-mini.

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (~1.5B parameters) |
| Trainable params | ~2.2M — **0.14% of total** (LoRA rank=16) |
| Dataset | ~300 synthetic examples via GPT-4o-mini |
| Training loss | 2.67 → 0.31 |
| Token accuracy | 91.4% (train) |
| Approach | Parameter-efficient fine-tuning (PEFT/LoRA) |

**Stack:** PyTorch · HuggingFace Transformers · PEFT · TRL (SFTTrainer) · OpenAI · Click · Pydantic Settings

```
lora_finetune/
├── notebook/lora_finetune.ipynb   # End-to-end walkthrough: data → train → eval → infer
├── src/
│   ├── config.py                  # Pydantic-settings — .env + lora_config.yaml
│   ├── prepare_data.py            # Synthetic data generation via GPT-4o-mini
│   ├── train.py                   # LoRA fine-tuning loop (SFTTrainer + PEFT)
│   ├── evaluate.py                # Base vs fine-tuned accuracy/F1 comparison
│   └── infer.py                   # Load adapter, score a single comment
├── configs/lora_config.yaml       # All hyperparams — any run reproducible from this file
├── tests/                         # pytest suite — 17 tests, all passing
└── cli.py                         # CLI: prepare / train / eval / infer
```

```bash
python cli.py prepare                                           # generate training data
python cli.py train --config configs/lora_config.yaml          # fine-tune
python cli.py eval --adapter ./adapter                          # base vs fine-tuned
python cli.py infer "SQL query passes user input unsanitized"   # score a comment
python3 -m pytest tests/ -v                                     # run tests
```

---

## Related Repositories

- [ai-systems-engineering](https://github.com/rajuclh-ai/ai-systems-engineering) — production-grade AI systems: RAG pipelines, multi-agent PR review, LLM evaluation
