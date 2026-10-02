"""Train a shared Loop Dropout adapter on the prepared MetaMath examples."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .adapter import AdapterConfig, LoopDropoutAdapter
from .data import collate, load_training_data, write_json
from .model import add_model_arguments, load_model, load_tokenizer
from .settings import DATA_DIR, NUM_LOOPS, OUTPUT_DIR


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-4
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 101
    log_every: int = 10
    max_steps: int = 0

    def __post_init__(self):
        for name in ("epochs", "batch_size", "gradient_accumulation", "max_length", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not 0 <= self.warmup_ratio <= 1 or self.max_steps < 0:
            raise ValueError("Invalid warmup_ratio or max_steps.")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("weight_decay must be nonnegative and max_grad_norm must be positive.")


def final_loop_loss(model, batch):
    # Ouro's built-in labels loss mixes loop readouts; supervise the final loop directly.
    _, states, _ = model.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
    )
    if len(states) != NUM_LOOPS:
        raise ValueError("Expected four Ouro loop readouts.")
    targets = batch["labels"][:, 1:]
    active = targets != -100
    if not active.any():
        raise ValueError("This batch has no supervised answer tokens.")
    logits = model.lm_head(states[-1][:, :-1][active]).float()
    return F.cross_entropy(logits, targets[active])


def train(model, tokenizer, examples, output_dir, adapter_config=None, config=None):
    config = config or TrainConfig()
    if not examples:
        raise ValueError("The training set is empty.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    adapter = LoopDropoutAdapter(model, adapter_config)
    parameters = list(adapter.parameters())
    device = next(model.parameters()).device
    order_generator = torch.Generator().manual_seed(config.seed)
    mask_generator = torch.Generator().manual_seed(config.seed + 1)
    total_steps = config.max_steps or math.ceil(
        len(examples) / (config.batch_size * config.gradient_accumulation)
    ) * config.epochs
    warmup_steps = max(1, int(config.warmup_ratio * total_steps))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    order, offset = torch.randperm(len(examples), generator=order_generator).tolist(), 0
    write_json(output_dir / "training_config.json", {
        **asdict(config), "adapter": asdict(adapter.config), "num_examples": len(examples),
        "optimizer_steps": total_steps, "device": str(device), "dtype": str(next(model.parameters()).dtype),
    })
    start = time.monotonic()
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            learning_rate = optimizer.param_groups[0]["lr"]
            for _ in range(config.gradient_accumulation):
                if offset >= len(order):
                    order, offset = torch.randperm(len(examples), generator=order_generator).tolist(), 0
                selected = [examples[i] for i in order[offset:offset + config.batch_size]]
                offset += config.batch_size
                batch = collate(selected, tokenizer.pad_token_id, device)
                adapter.train_batch(len(selected), mask_generator)
                loss = final_loop_loss(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss.")
                (loss / config.gradient_accumulation).backward()
                loss_sum += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            record = {"step": step + 1, "loss": loss_sum / config.gradient_accumulation,
                      "learning_rate": learning_rate, "seconds": round(time.monotonic() - start, 2)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (step + 1) % config.log_every == 0 or step + 1 == total_steps:
                print(json.dumps(record), flush=True)
    adapter.eval()
    adapter.save(output_dir / "adapter")
    return adapter


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    for name, value in asdict(TrainConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__})
    adapter_config = AdapterConfig(rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    tokenizer = load_tokenizer(args.model_dir)
    examples, dropped = load_training_data(args.data_dir, tokenizer, config.max_length)
    print(f"Training examples: {len(examples)}; dropped for length: {dropped}", flush=True)
    model = load_model(args.model_dir, args.device, args.dtype)
    train(model, tokenizer, examples, args.output_dir, adapter_config, config)


if __name__ == "__main__":
    main()
