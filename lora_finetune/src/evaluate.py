import gc
from collections import defaultdict

import torch

from .config import get_settings, load_pipeline_config
from .infer import load_model, predict
from .prepare_data import load_dataset


def run_eval(model, tokenizer, device, examples, max_length):
    results = []
    for ex in examples:
        pred = predict(model, tokenizer, device, ex["comment"], max_length)
        results.append({"true": {"score": ex["score"], "label": ex["label"]}, "pred": pred})
    return results


def compute_metrics(results: list) -> dict:
    by_label: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    correct = 0

    for r in results:
        true_label = r["true"]["label"]
        pred_label = r["pred"].get("label") if r["pred"] else None

        if pred_label == true_label:
            correct += 1
            by_label[true_label]["tp"] += 1
        else:
            by_label[true_label]["fn"] += 1
            if pred_label:
                by_label[pred_label]["fp"] += 1

    accuracy = correct / len(results) * 100 if results else 0.0
    return {"accuracy": accuracy, "by_label": dict(by_label)}


def print_report(label: str, metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Label Accuracy: {metrics['accuracy']:.1f}%")
    print(f"{'='*50}")
    print(f"  {'Label':<12} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print(f"  {'-'*38}")

    for lbl, counts in sorted(metrics["by_label"].items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {lbl:<12} {prec:>10.2f} {rec:>8.2f} {f1:>6.2f}")


def evaluate(
    adapter_path: str = "./adapter",
    test_path: str = "data/test.jsonl",
    config_path: str = "configs/lora_config.yaml",
) -> None:
    cfg = load_pipeline_config(config_path)
    settings = get_settings()
    examples = load_dataset(test_path)
    print(f"Evaluating on {len(examples)} test examples\n")

    for label, use_adapter in [("BASE MODEL", None), ("FINE-TUNED MODEL", adapter_path)]:
        print(f"Loading {label}...")
        model, tokenizer, device = load_model(cfg, settings, adapter_path=use_adapter)
        results = run_eval(model, tokenizer, device, examples, cfg.model.max_length)
        metrics = compute_metrics(results)
        print_report(label, metrics)

        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
