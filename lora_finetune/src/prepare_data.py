import json
import random
from pathlib import Path

from openai import OpenAI

from .config import get_settings

SYSTEM_PROMPT = """You are a dataset generator for training an ML classifier on code review comments.

Generate realistic, diverse code review comments with quality scores.

Return a JSON object with key "examples" containing an array. Each item must have:
- "comment": the code review comment (string)
- "score": integer 1-5
- "label": one of nitpick | suggestion | warning | blocking
- "severity": one of low | medium | high | critical

Score mapping (strictly follow this):
  1 = nitpick,    low      — style, naming, minor formatting preference
  2 = suggestion, low      — improvement but not strictly required
  3 = suggestion, medium   — noticeable issue, team should fix soon
  4 = warning,    high     — bug risk or logic issue, fix before merge
  5 = blocking,   critical — security flaw, data loss, correctness bug, must fix
"""

CATEGORIES = [
    "security vulnerabilities (SQL injection, XSS, hardcoded secrets, auth bypass)",
    "performance issues (O(n²) nested loops, N+1 database queries, memory leaks)",
    "error handling (missing try/except, unhandled None, no input validation)",
    "logic bugs (off-by-one errors, wrong boolean conditions, silent data loss)",
    "code style (poor variable names, magic numbers, excessive nesting)",
    "documentation (missing docstrings, misleading comments, undocumented params)",
]


def generate_batch(client: OpenAI, category: str, n: int) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Generate {n} varied code review comments specifically about: {category}. Include a mix of scores 1-5.",
            },
        ],
        temperature=0.8,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("examples", [])


def generate_dataset(n_train: int = 500, n_test: int = 100) -> tuple[list, list]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    total = n_train + n_test
    per_category = max(2, (total // len(CATEGORIES)) + 2)

    all_examples = []
    for category in CATEGORIES:
        print(f"  Generating: {category[:50]}...")
        batch = generate_batch(client, category, per_category)
        all_examples.extend(batch)

    random.shuffle(all_examples)
    total_generated = len(all_examples)
    split = int(total_generated * (n_train / (n_train + n_test)))
    train = all_examples[:split]
    test = all_examples[split:]
    print(f"Total generated: {total_generated} → train={len(train)}, test={len(test)}")
    return train, test


def save_dataset(examples: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
