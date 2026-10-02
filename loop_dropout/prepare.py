"""Download Ouro-1.4B and prepare MetaMath, GSM8K and MATH-500."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from .data import prepare_data
from .model import load_tokenizer
from .settings import DATA_DIR, DATA_REVISIONS, MODEL_DIR, MODEL_ID, MODEL_REVISION


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--length-limit", type=int, default=512)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.train_size < 1 or args.length_limit < 2:
        raise ValueError("train-size must be positive and length-limit must be at least 2.")
    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.model", "*.txt"],
    )
    tokenizer = load_tokenizer(args.model_dir)
    datasets = {}
    for name in DATA_REVISIONS:
        extra = {"name": "main"} if name == "openai/gsm8k" else {}
        datasets[name] = load_dataset(
            name, revision=DATA_REVISIONS[name],
            split="train" if name == "meta-math/MetaMathQA" else "test", **extra,
        )
    summary = prepare_data(
        Path(args.data_dir), tokenizer, datasets["meta-math/MetaMathQA"],
        datasets["openai/gsm8k"], datasets["HuggingFaceH4/MATH-500"], args.train_size, args.length_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
