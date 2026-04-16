"""Pipeline — lazy-init predictor singleton and public API."""

import logging
from pathlib import Path

from .predictor import Prediction, Predictor

logger = logging.getLogger(__name__)

_predictor: Predictor | None = None


def _get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def predict_sentiment(text: str) -> Prediction:
    return _get_predictor().predict(text)


def batch_predict(texts: list[str]) -> list[Prediction]:
    return _get_predictor().predict_batch(texts)


def predict_from_file(filepath: Path) -> list[Prediction]:
    lines = [line.strip() for line in filepath.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(f"Predicting {len(lines)} reviews from {filepath}")
    return batch_predict(lines)
