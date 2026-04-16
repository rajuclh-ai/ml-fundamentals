"""Sentiment predictor — loads saved DistilBERT model and runs inference."""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    text: str
    label: str
    confidence: float
    label_id: int


class Predictor:
    def __init__(self, model_dir: str | None = None) -> None:
        path = str(model_dir or settings.model_dir)
        logger.info(f"Loading tokenizer and model from {path}")
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._model.to(settings.device)
        self._model.eval()
        self._id2label: dict[int, str] = self._model.config.id2label
        logger.info(f"Model ready — labels: {self._id2label}")

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(settings.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits

        probs = torch.softmax(logits, dim=-1)
        label_ids = torch.argmax(probs, dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()

        return [
            Prediction(
                text=text,
                label=self._id2label[label_id].upper(),
                confidence=round(confidence, 4),
                label_id=label_id,
            )
            for text, label_id, confidence in zip(texts, label_ids, confidences)
        ]
