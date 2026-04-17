import json

import click

from src.evaluate import evaluate
from src.prepare_data import generate_dataset, save_dataset
from src.train import train
from src.infer import score_comment


@click.group()
def cli():
    """LoRA Fine-tuning Pipeline — Code Review Quality Scorer."""


@cli.command()
@click.option("--n-train", default=500, show_default=True, help="Training examples to generate")
@click.option("--n-test", default=100, show_default=True, help="Test examples to generate")
def prepare(n_train, n_test):
    """Generate labeled training data via GPT-4o-mini."""
    click.echo(f"Generating {n_train} train + {n_test} test examples...")
    train_data, test_data = generate_dataset(n_train, n_test)
    save_dataset(train_data, "data/train.jsonl")
    save_dataset(test_data, "data/test.jsonl")
    click.echo("Done.")


@cli.command("train")
@click.option("--config", default="configs/lora_config.yaml", show_default=True)
@click.option("--data", default="data/train.jsonl", show_default=True)
def train_cmd(config, data):
    """Fine-tune Llama 3.2 3B with LoRA."""
    train(config_path=config, train_path=data)


@cli.command("eval")
@click.option("--adapter", default="./adapter", show_default=True)
@click.option("--data", default="data/test.jsonl", show_default=True)
@click.option("--config", default="configs/lora_config.yaml", show_default=True)
def eval_cmd(adapter, data, config):
    """Evaluate base vs fine-tuned model on test set."""
    evaluate(adapter_path=adapter, test_path=data, config_path=config)


@cli.command()
@click.argument("comment")
@click.option("--adapter", default="./adapter", show_default=True)
@click.option("--config", default="configs/lora_config.yaml", show_default=True)
def infer(comment, adapter, config):
    """Score a single code review comment.

    Example:

        python cli.py infer "This SQL query is vulnerable to injection"
    """
    result = score_comment(comment, adapter_path=adapter, config_path=config)
    if result:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Could not parse a valid response from the model.", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
