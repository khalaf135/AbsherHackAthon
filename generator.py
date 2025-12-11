from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    # Only needed if you use LoRA / PEFT adapters
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore


@dataclass
class GeneratorConfig:
    base_model_id: str = "dphn/Dolphin3.0-Llama3.1-8B"
    adapter_dir: Optional[str] = None
    load_in_4bit: bool = True
    max_new_tokens: int = 40
    temperature: float = 0.7
    top_p: float = 0.9


class MaliciousGenerator:
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        bnb_config = None
        if self.config.load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        if self.config.adapter_dir is not None:
            if PeftModel is None:
                raise ImportError(
                    "peft is required to load adapter weights. "
                    "Install with `pip install peft` or set adapter_dir=None."
                )
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.adapter_dir,
            )

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if prompt in text:
            after_prompt = text.split(prompt, 1)[1].strip()
            return after_prompt or text.strip()
        return text.strip()