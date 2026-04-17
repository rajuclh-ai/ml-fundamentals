import json
from unittest.mock import MagicMock, patch

import torch

from src.infer import CLASSIFIER_INSTRUCTION, format_prompt, predict


def test_format_prompt_contains_comment():
    comment = "Missing null check on user input"
    prompt = format_prompt(comment)
    assert comment in prompt


def test_format_prompt_contains_instruction():
    prompt = format_prompt("any comment")
    assert CLASSIFIER_INSTRUCTION in prompt


def test_format_prompt_ends_at_assistant_header():
    prompt = format_prompt("any comment")
    assert prompt.endswith("<|im_start|>assistant\n")


def _make_mock_tokenizer(decoded_output: str):
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    encoded = MagicMock()
    encoded.to.return_value = encoded
    encoded.__getitem__ = MagicMock(return_value=MagicMock())
    input_ids = MagicMock()
    input_ids.shape = (1, 10)
    encoded.input_ids = input_ids
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = decoded_output
    return tokenizer


def test_predict_parses_valid_json():
    model = MagicMock()
    model.generate.return_value = torch.zeros(1, 15, dtype=torch.long)
    tokenizer = _make_mock_tokenizer('{"score": 4, "label": "warning", "severity": "high"}')

    result = predict(model, tokenizer, "cpu", "No error handling on DB call")

    assert result is not None
    assert result["label"] == "warning"
    assert result["score"] == 4


def test_predict_returns_none_on_no_json():
    model = MagicMock()
    model.generate.return_value = torch.zeros(1, 15, dtype=torch.long)
    tokenizer = _make_mock_tokenizer("I cannot classify this comment.")

    result = predict(model, tokenizer, "cpu", "some comment")

    assert result is None


def test_predict_returns_none_on_malformed_json():
    model = MagicMock()
    model.generate.return_value = torch.zeros(1, 15, dtype=torch.long)
    tokenizer = _make_mock_tokenizer("{score: 4, label: warning}")  # invalid JSON

    result = predict(model, tokenizer, "cpu", "some comment")

    assert result is None
