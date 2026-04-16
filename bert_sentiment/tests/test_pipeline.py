"""Tests for src/pipeline.py"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.predictor import Prediction


POSITIVE = Prediction(text="Great!", label="POSITIVE", confidence=0.97, label_id=1)
NEGATIVE = Prediction(text="Awful!", label="NEGATIVE", confidence=0.93, label_id=0)


# ---------------------------------------------------------------------------
# predict_sentiment
# ---------------------------------------------------------------------------

def test_predict_sentiment_returns_prediction():
    with patch("src.pipeline._get_predictor") as mock_get:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = POSITIVE
        mock_get.return_value = mock_predictor

        from src.pipeline import predict_sentiment
        result = predict_sentiment("Great film!")

    assert result.label == "POSITIVE"
    assert result.confidence == 0.97
    mock_predictor.predict.assert_called_once_with("Great film!")


# ---------------------------------------------------------------------------
# batch_predict
# ---------------------------------------------------------------------------

def test_batch_predict_returns_list():
    with patch("src.pipeline._get_predictor") as mock_get:
        mock_predictor = MagicMock()
        mock_predictor.predict_batch.return_value = [POSITIVE, NEGATIVE]
        mock_get.return_value = mock_predictor

        from src.pipeline import batch_predict
        results = batch_predict(["Great!", "Awful!"])

    assert len(results) == 2
    assert results[0].label == "POSITIVE"
    assert results[1].label == "NEGATIVE"


# ---------------------------------------------------------------------------
# predict_from_file
# ---------------------------------------------------------------------------

def test_predict_from_file_reads_lines(tmp_path):
    reviews_file = tmp_path / "reviews.txt"
    reviews_file.write_text("Great movie!\nTerrible film.\n\nNot bad.\n", encoding="utf-8")

    expected = [
        Prediction(text="Great movie!", label="POSITIVE", confidence=0.95, label_id=1),
        Prediction(text="Terrible film.", label="NEGATIVE", confidence=0.91, label_id=0),
        Prediction(text="Not bad.", label="POSITIVE", confidence=0.72, label_id=1),
    ]

    with patch("src.pipeline._get_predictor") as mock_get:
        mock_predictor = MagicMock()
        mock_predictor.predict_batch.return_value = expected
        mock_get.return_value = mock_predictor

        from src.pipeline import predict_from_file
        results = predict_from_file(reviews_file)

    assert len(results) == 3
    mock_predictor.predict_batch.assert_called_once_with(
        ["Great movie!", "Terrible film.", "Not bad."]
    )


def test_predict_from_file_skips_blank_lines(tmp_path):
    reviews_file = tmp_path / "reviews.txt"
    reviews_file.write_text("\n\nOnly one real line.\n\n", encoding="utf-8")

    with patch("src.pipeline._get_predictor") as mock_get:
        mock_predictor = MagicMock()
        mock_predictor.predict_batch.return_value = [POSITIVE]
        mock_get.return_value = mock_predictor

        from src.pipeline import predict_from_file
        predict_from_file(reviews_file)

    mock_predictor.predict_batch.assert_called_once_with(["Only one real line."])
