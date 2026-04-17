import json

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import PipelineConfig, get_settings, load_pipeline_config
from .infer import CLASSIFIER_INSTRUCTION, get_device
from .prepare_data import load_dataset


def format_training_example(example: dict) -> str:
    """Full prompt including the assistant answer — used only for training."""
    response = json.dumps(
        {
            "score": example["score"],
            "label": example["label"],
            "severity": example["severity"],
        }
    )
    return (
        "<|im_start|>system\n"
        f"{CLASSIFIER_INSTRUCTION}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{example['comment']}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{response}<|im_end|>"
    )


def build_model(cfg: PipelineConfig, settings):
    device = get_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, token=settings.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=dtype,
        device_map={"": device},
        token=settings.hf_token,
    )
    return model, tokenizer


def apply_lora(model, lora_cfg):
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def train(
    config_path: str = "configs/lora_config.yaml",
    train_path: str = "data/train.jsonl",
) -> None:
    cfg = load_pipeline_config(config_path)
    settings = get_settings()

    print("Loading dataset...")
    raw = load_dataset(train_path)
    dataset = Dataset.from_list([{"text": format_training_example(ex)} for ex in raw])
    print(f"Training on {len(dataset)} examples")

    print("Loading model...")
    model, tokenizer = build_model(cfg, settings)
    model = apply_lora(model, cfg.lora)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        report_to="none",
        max_length=cfg.model.max_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)
    print(f"Adapter saved → {cfg.training.output_dir}")
