import json
from unittest.mock import MagicMock

import pytest

from src.prepare_data import generate_batch, load_dataset, save_dataset


def test_save_and_load_roundtrip(tmp_path):
    examples = [
        {"comment": "Missing error handling", "score": 4, "label": "warning", "severity": "high"},
        {"comment": "Poor variable name", "score": 1, "label": "nitpick", "severity": "low"},
    ]
    path = str(tmp_path / "test.jsonl")
    save_dataset(examples, path)
    loaded = load_dataset(path)
    assert loaded == examples


def test_save_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "data.jsonl")
    save_dataset([{"comment": "x", "score": 1, "label": "nitpick", "severity": "low"}], path)
    assert load_dataset(path)[0]["comment"] == "x"


def test_generate_batch_parses_response():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        {
            "examples": [
                {
                    "comment": "SQL query passes user input without sanitization",
                    "score": 5,
                    "label": "blocking",
                    "severity": "critical",
                }
            ]
        }
    )

    result = generate_batch(mock_client, "security", 1)

    assert len(result) == 1
    assert result[0]["label"] == "blocking"
    assert result[0]["score"] == 5


def test_generate_batch_handles_empty_response():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        {"examples": []}
    )

    result = generate_batch(mock_client, "style", 1)
    assert result == []


def test_load_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"comment": "a", "score": 1, "label": "nitpick", "severity": "low"}\n\n'
        '{"comment": "b", "score": 5, "label": "blocking", "severity": "critical"}\n'
    )
    result = load_dataset(str(path))
    assert len(result) == 2
