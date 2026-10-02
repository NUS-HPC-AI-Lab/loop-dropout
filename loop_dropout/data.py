"""MetaMath training examples and answer-only language-model supervision."""

import json
from pathlib import Path

import torch

from .settings import DATA_REVISIONS, EVAL_PROMPT, MODEL_ID, MODEL_REVISION, TRAIN_PROMPT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_training_rows(rows, tokenizer, train_size=100000, length_limit=512):
    if train_size < 1 or length_limit < 2:
        raise ValueError("train_size must be positive and length_limit must be at least 2.")
    selected = []
    for index, row in enumerate(rows):
        if "GSM" not in row["type"]:
            continue
        prompt = TRAIN_PROMPT.format(instruction=row["query"])
        # Preserve the training recipe's strict length filter and source order.
        if len(tokenizer(prompt + " " + row["response"])["input_ids"]) >= length_limit:
            continue
        selected.append({"id": index, "prompt": prompt, "response": row["response"]})
        if len(selected) == train_size:
            return selected
    raise ValueError(f"Only {len(selected)} eligible training examples; requested {train_size}.")


def prepare_data(directory, tokenizer, metamath, gsm8k, math500, train_size=100000, length_limit=512):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train = select_training_rows(metamath, tokenizer, train_size, length_limit)
    evaluation = {
        "gsm8k": [
            {"id": i, "prompt": EVAL_PROMPT.format(instruction=row["question"]),
             "answer": int(row["answer"].split("#### ")[1].replace(",", ""))}
            for i, row in enumerate(gsm8k)
        ],
        "math500": [
            {"id": i, "prompt": EVAL_PROMPT.format(instruction=row["problem"]), "answer": row["answer"]}
            for i, row in enumerate(math500)
        ],
    }
    if any(not rows for rows in evaluation.values()):
        raise ValueError("Evaluation datasets must not be empty.")
    write_jsonl(directory / "train.jsonl", train)
    for name, rows in evaluation.items():
        write_jsonl(directory / f"{name}.jsonl", rows)
    metadata = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_revisions": DATA_REVISIONS, "train_size": len(train), "length_limit": length_limit,
        "evaluation_sizes": {name: len(rows) for name, rows in evaluation.items()},
    }
    write_json(directory / "metadata.json", metadata)
    return metadata


def encode_example(tokenizer, row, max_length=1024):
    prompt, answer = row["prompt"], row["response"]
    trailing = len(prompt) - len(prompt.rstrip())
    if trailing:
        answer, prompt = prompt[-trailing:] + answer, prompt[:-trailing]
    ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    boundary = len(prompt_ids)
    # A BPE token can span the prompt/answer boundary.
    while boundary and ids[:boundary] != prompt_ids[:boundary]:
        boundary -= 1
    ids.append(tokenizer.eos_token_id)
    labels = [-100] * boundary + ids[boundary:]
    if len(ids) > max_length or not any(value != -100 for value in labels[1:]):
        return None
    return ids, labels


def load_training_data(directory, tokenizer, max_length=1024):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["model_id"], metadata["model_revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Prepared data uses a different model or tokenizer revision.")
    rows = read_jsonl(directory / "train.jsonl")
    examples = [example for row in rows if (example := encode_example(tokenizer, row, max_length)) is not None]
    if not examples:
        raise ValueError("No training examples fit max_length.")
    return examples, len(rows) - len(examples)


def collate(examples, pad_token_id, device):
    width = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, (ids, target) in enumerate(examples):
        input_ids[index, :len(ids)] = torch.tensor(ids)
        attention_mask[index, :len(ids)] = 1
        labels[index, :len(ids)] = torch.tensor(target)
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "labels": labels.to(device)}
