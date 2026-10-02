"""Load an adapter and generate an answer with all four loops active."""

import argparse
from pathlib import Path

import torch
from transformers import GenerationConfig

from .adapter import LoopDropoutAdapter
from .model import add_model_arguments, load_model, load_tokenizer
from .settings import EVAL_PROMPT, OUTPUT_DIR, STOP_STRINGS


def generate_completions(model, tokenizer, prompts, max_new_tokens=512):
    if not prompts or max_new_tokens < 1:
        raise ValueError("Provide at least one prompt and a positive generation limit.")
    device = next(model.parameters()).device
    previous_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        encoded = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    finally:
        tokenizer.padding_side = previous_side
    encoded = {name: value.to(device) for name, value in encoded.items() if name in ("input_ids", "attention_mask")}
    context_length = getattr(model.config, "max_position_embeddings", None)
    if context_length and encoded["input_ids"].shape[1] + max_new_tokens > context_length:
        raise ValueError("Prompt plus generation limit exceeds the model's context length.")
    settings = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        repetition_penalty=1.0, top_p=1.0, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **encoded, generation_config=settings, stop_strings=STOP_STRINGS,
            tokenizer=tokenizer, use_weighted_exit=False,
        )
    continuations = output[:, encoded["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(continuations, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [text[:min([text.find(stop) for stop in STOP_STRINGS if stop in text] + [len(text)])] for text in texts]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--adapter", default=str(Path(OUTPUT_DIR) / "adapter"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, args.device, args.dtype)
    LoopDropoutAdapter.load(model, args.adapter)
    prompt = EVAL_PROMPT.format(instruction=args.prompt)
    print(generate_completions(model, tokenizer, [prompt], args.max_new_tokens)[0])


if __name__ == "__main__":
    main()
