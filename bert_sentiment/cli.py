"""CLI entry point for BERT Sentiment Analysis.

Usage:
    python cli.py predict "This movie was amazing!"
    python cli.py batch --file reviews.txt
    python cli.py info
"""

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
def cli() -> None:
    """BERT Sentiment Analysis — classify text as POSITIVE or NEGATIVE."""


@cli.command()
@click.argument("text")
def predict(text: str) -> None:
    """Predict sentiment for a single text input."""
    from src.pipeline import predict_sentiment

    result = predict_sentiment(text)
    click.echo(f"\nText      : {result.text}")
    click.echo(f"Sentiment : {result.label}")
    click.echo(f"Confidence: {result.confidence * 100:.1f}%\n")


@cli.command()
@click.option("--file", "-f", required=True, type=click.Path(exists=True, path_type=Path), help="File with one review per line")
def batch(file: Path) -> None:
    """Predict sentiment for each line in a file."""
    from src.pipeline import predict_from_file

    results = predict_from_file(file)
    click.echo(f"\n{'#':<5} {'Sentiment':<12} {'Confidence':<12} Text")
    click.echo("-" * 70)
    for i, r in enumerate(results, 1):
        preview = r.text[:50] + ("..." if len(r.text) > 50 else "")
        click.echo(f"{i:<5} {r.label:<12} {r.confidence * 100:>6.1f}%     {preview}")
    click.echo(f"\nTotal: {len(results)} reviews\n")


@cli.command()
def info() -> None:
    """Show model details: architecture, labels, device."""
    from src.config import settings
    from src.predictor import Predictor

    p = Predictor()
    click.echo(f"\nModel directory : {settings.model_dir}")
    click.echo(f"Architecture    : {p._model.config.model_type}")
    click.echo(f"Labels          : {p._id2label}")
    click.echo(f"Device          : {settings.device}")
    click.echo(f"Max length      : {settings.max_length} tokens\n")


if __name__ == "__main__":
    cli()
