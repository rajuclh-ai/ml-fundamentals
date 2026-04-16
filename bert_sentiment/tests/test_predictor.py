"""Tests for src/predictor.py"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from src.predictor import Prediction, Predictor


# ---------------------------------------------------------------------------
# Prediction dataclass
# ---------------------------------------------------------------------------

def test_prediction_fields():
    p = Prediction(text="Great film!", label="POSITIVE", confidence=0.97, label_id=1)
    assert p.text == "Great film!"
    assert p.label == "POSITIVE"
    assert p.confidence == 0.97
    assert p.label_id == 1


# ---------------------------------------------------------------------------
# Predictor.predict
# ---------------------------------------------------------------------------

def test_predict_returns_positive(mock_predictor):
    result = mock_predictor.predict("This movie was amazing!")
    assert result.label in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= result.confidence <= 1.0
    assert result.text == "This movie was amazing!"


def test_predict_batch_returns_correct_count(mock_predictor, mock_model):
    batch_logits = torch.tensor([[0.1, 2.5], [3.0, 0.2], [0.5, 1.8]])
    output = MagicMock()
    output.logits = batch_logits
    mock_model.return_value = output

    texts = ["Great!", "Awful!", "OK film."]
    results = mock_predictor.predict_batch(texts)

    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)
    assert [r.text for r in results] == texts


def test_predict_batch_label_matches_argmax(mock_predictor, mock_model):
    output = MagicMock()
    output.logits = torch.tensor([[3.0, 0.1]])
    mock_model.return_value = output

    results = mock_predictor.predict_batch(["Boring film."])
    assert results[0].label == "NEGATIVE"
    assert results[0].label_id == 0


def test_predict_confidence_between_0_and_1(mock_predictor):
    result = mock_predictor.predict("Some review text.")
    assert 0.0 <= result.confidence <= 1.0


def test_predict_batch_confidence_rounded(mock_predictor, mock_model):
    output = MagicMock()
    output.logits = torch.tensor([[0.1, 2.5]])
    mock_model.return_value = output

    results = mock_predictor.predict_batch(["test"])
    assert len(str(results[0].confidence).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# Predictor.__init__ — loading
# ---------------------------------------------------------------------------

def test_predictor_init_loads_from_model_dir(tmp_path):
    with patch("src.predictor.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("src.predictor.AutoModelForSequenceClassification.from_pretrained") as mock_mod:

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_mod.return_value = mock_model

        p = Predictor(model_dir=str(tmp_path))

        mock_tok.assert_called_once_with(str(tmp_path))
        mock_mod.assert_called_once_with(str(tmp_path))
