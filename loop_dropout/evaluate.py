"""Evaluate a saved adapter on GSM8K and MATH-500 using MetaMath scoring."""

import argparse
import json
from pathlib import Path

from .adapter import LoopDropoutAdapter
from .data import read_jsonl, write_json
from .generate import generate_completions
from .model import add_model_arguments, load_model, load_tokenizer
from .settings import DATA_DIR, MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, OUTPUT_DIR
from .vendor.metamath.scoring import extract_answer_number, process_results


def score(task, completion, answer):
    if task == "gsm8k":
        prediction = extract_answer_number(completion)
        return prediction is not None and float(prediction) == float(answer)
    if task == "math500":
        return bool(process_results("", completion, str(answer)))
    raise ValueError(f"Unknown evaluation task: {task}")


def evaluate(model, tokenizer, data_dir, output_dir, tasks=("gsm8k", "math500"), batch_size=16,
             limit=None, max_new_tokens=None):
    if batch_size < 1 or (limit is not None and limit < 1) or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError("Batch size and any supplied limits must be positive.")
    if not tasks or any(task not in MAX_NEW_TOKENS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("Choose gsm8k and/or math500 without duplicates.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "tasks": {}}
    for task in tasks:
        all_rows = read_jsonl(Path(data_dir) / f"{task}.jsonl")
        rows = all_rows[:limit] if limit is not None else all_rows
        if not rows:
            raise ValueError(f"The {task} dataset is empty.")
        cap = max_new_tokens or MAX_NEW_TOKENS[task]
        correct = 0
        with (output_dir / f"{task}.jsonl").open("w", encoding="utf-8") as stream:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                completions = generate_completions(model, tokenizer, [row["prompt"] for row in batch], cap)
                for row, completion in zip(batch, completions, strict=True):
                    is_correct = score(task, completion, row["answer"])
                    correct += int(is_correct)
                    stream.write(json.dumps({**row, "completion": completion, "correct": is_correct}, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{task}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
        summary["tasks"][task] = {
            "num_examples": len(rows), "dataset_size": len(all_rows), "correct": correct,
            "accuracy_percent": 100.0 * correct / len(rows), "max_new_tokens": cap, "batch_size": batch_size,
        }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--adapter", default=str(Path(OUTPUT_DIR) / "adapter"))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=str(Path(OUTPUT_DIR) / "evaluation"))
    parser.add_argument("--tasks", nargs="+", choices=tuple(MAX_NEW_TOKENS), default=list(MAX_NEW_TOKENS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, help="Default: GSM8K 512, MATH-500 2048.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, args.device, args.dtype)
    LoopDropoutAdapter.load(model, args.adapter)
    summary = evaluate(model, tokenizer, args.data_dir, args.output_dir, args.tasks,
                       args.batch_size, args.limit, args.max_new_tokens)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
