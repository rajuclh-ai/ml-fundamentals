from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    openai_api_key: str
    hf_token: str

    model_config = {"env_file": ".env", "extra": "ignore"}


@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_length: int = 256


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    output_dir: str = "./adapter"


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_pipeline_config(config_path: str = "configs/lora_config.yaml") -> PipelineConfig:
    path = Path(config_path)
    if not path.exists():
        return PipelineConfig()

    with open(path) as f:
        data = yaml.safe_load(f)

    cfg = PipelineConfig()
    for section, cls_obj in [("model", cfg.model), ("lora", cfg.lora), ("training", cfg.training)]:
        if section in data:
            for k, v in data[section].items():
                setattr(cls_obj, k, v)

    return cfg


def get_settings() -> AppSettings:
    return AppSettings()
