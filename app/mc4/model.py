"""Local text LLM on the AMD GPU (ROCm exposes it as torch.cuda). Loaded once,
by the resident server only."""
from __future__ import annotations

import os

from . import paths

MAX_INPUT_TOKENS = 6000
# Tests only (CI runs a tiny model on CPU to exercise this code path). Never set in the image.
ALLOW_CPU = os.environ.get("MC4_ALLOW_CPU") == "1"


class LocalLLM:
    def __init__(self, model_dir: str = paths.MODEL_DIR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif ALLOW_CPU:
            self.device = "cpu"
        else:
            raise RuntimeError("no GPU visible to torch (ROCm)")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=dtype, local_files_only=True, attn_implementation="sdpa",
                device_map=self.device,
            ).eval()
        except Exception as e:  # noqa: BLE001 - older/odd accelerate setups
            paths.log(f"device_map load failed ({type(e).__name__}: {e}); loading on CPU then moving")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=dtype, local_files_only=True, attn_implementation="sdpa",
            ).to(self.device).eval()
        self.chat("", "Say OK.", max_new_tokens=2)          # warm up kernels before the clock starts

    def chat(self, system: str, text: str, max_new_tokens: int = 120) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True, enable_thinking=False)
        ids = enc["input_ids"][:, -MAX_INPUT_TOKENS:].to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                                      max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
