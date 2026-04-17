import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PipelineConfig, get_settings, load_pipeline_config

CLASSIFIER_INSTRUCTION = (
    "You are a code review classifier. Given a code review comment, "
    "output a JSON object with: score (integer 1-5), "
    "label (nitpick | suggestion | warning | blocking), "
    "and severity (low | medium | high | critical). "
    "Output only the JSON object, nothing else."
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    cfg: PipelineConfig,
    settings,
    adapter_path: str | None = None,
):
    device = get_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, token=settings.hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=dtype,
        device_map={"": device},
        token=settings.hf_token,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer, device


def format_prompt(comment: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{CLASSIFIER_INSTRUCTION}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{comment}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def predict(model, tokenizer, device: str, comment: str, max_length: int = 256) -> dict | None:
    prompt = format_prompt(comment)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    match = re.search(r"\{[^}]+\}", generated)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def score_comment(
    comment: str,
    adapter_path: str = "./adapter",
    config_path: str = "configs/lora_config.yaml",
) -> dict | None:
    cfg = load_pipeline_config(config_path)
    settings = get_settings()
    model, tokenizer, device = load_model(cfg, settings, adapter_path=adapter_path)
    return predict(model, tokenizer, device, comment, cfg.model.max_length)
