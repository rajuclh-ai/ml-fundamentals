"""Shared pytest fixtures."""

import pytest
from unittest.mock import MagicMock


SAMPLE_TEXTS = [
    "This movie was absolutely fantastic, I loved every moment.",
    "Terrible film. Boring plot, bad acting, complete waste of time.",
    "Not bad at all, actually enjoyed it more than expected.",
]


@pytest.fixture
def mock_model():
    import torch

    model = MagicMock()
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.model_type = "distilbert"

    logits = torch.tensor([[0.1, 2.5]])
    output = MagicMock()
    output.logits = logits
    model.return_value = output
    return model


@pytest.fixture
def mock_tokenizer():
    import torch

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    return tokenizer


@pytest.fixture
def mock_predictor(mock_model, mock_tokenizer):
    from src.predictor import Predictor

    p = Predictor.__new__(Predictor)
    p._model = mock_model
    p._tokenizer = mock_tokenizer
    p._id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    return p
