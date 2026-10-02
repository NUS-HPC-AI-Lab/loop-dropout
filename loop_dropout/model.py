"""Load the pinned Ouro model with four loops and a final-loop readout."""

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .settings import MODEL_DIR, NUM_LOOPS


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")


def load_tokenizer(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def configure_ouro(model):
    """Use every backbone loop and fix padded decoding in the pinned KV cache."""
    if model.config.model_type != "ouro":
        raise ValueError("This implementation supports Ouro-1.4B only.")
    model.config.total_ut_steps = NUM_LOOPS
    model.model.total_ut_steps = NUM_LOOPS
    for name in ("early_exit_step", "early_exit_threshold"):
        setattr(model, name, None)
        setattr(model.config, name, None)

    cache_class = getattr(sys.modules[type(model).__module__], "UniversalTransformerCache", None)
    if cache_class is None:
        raise ValueError("Ouro's UniversalTransformerCache is missing.")
    if "get_mask_sizes" not in cache_class.__dict__:
        # Cache masks must include both the existing KV prefix and the new query.
        def get_mask_sizes(self, cache_position, layer_idx=0):
            return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

        cache_class.get_mask_sizes = get_mask_sizes
    return model.eval()


def load_model(model_dir=MODEL_DIR, device="cuda", dtype="bfloat16"):
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"No model at {model_dir}; run python -m loop_dropout.prepare first.")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device.")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU checks can use --device cpu --dtype float32.")
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("dtype must be bfloat16 or float32.")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    if config.model_type != "ouro":
        raise ValueError("Expected an Ouro-1.4B model directory.")
    config.total_ut_steps = NUM_LOOPS
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, local_files_only=True,
        dtype=getattr(torch, dtype), attn_implementation="sdpa",
    )
    return configure_ouro(model.to(target))
