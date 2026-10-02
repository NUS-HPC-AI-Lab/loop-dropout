# Loop Dropout: Regularizing Shared Updates in Looped Language Models

Official implementation of **Loop Dropout**.

> 📄 [Read the paper on arXiv](https://arxiv.org/abs/2609.34218)

Looped language models increase computational depth by repeatedly applying the same transformer block. Adapting these models requires a shared update that remains effective as hidden states evolve across loops. Our analysis reveals a pronounced late-loop bias in standard low-rank adaptation (LoRA): the shared update is more effective at later loop positions.

We propose **Loop Dropout**, a regularizer for shared updates in looped language models. During training, it randomly masks complete applications of the shared adapter independently for each example and loop, and rescales retained updates by their inverse survival probability. This trains the adapter under varying application patterns while preserving its expected update strength. Every backbone loop remains active, and inference follows standard LoRA with the adapter applied at all loops, adding no trainable parameters or inference computation.

Experiments in the paper demonstrate improved mathematical reasoning across model sizes, adapter ranks, and training recipes, together with stronger early-loop adaptation and generalization to deeper recurrence.

## Installation

Use Python 3.12 and a CUDA GPU for training.

```bash
git clone https://github.com/NUS-HPC-AI-Lab/loop-dropout.git
cd loop-dropout
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

The commands below use Ouro-1.4B and the default configuration. Run them from the repository root.

### Prepare the model and data

Download the model and datasets from pinned public Hugging Face revisions, and prepare 100,000 training examples from MetaMath.

```bash
python -m loop_dropout.prepare \
  --model-dir models/Ouro-1.4B --data-dir data \
  --train-size 100000 --length-limit 512
```

### Train

Train a shared adapter with Loop Dropout. The default recipe uses four loops, all attention and MLP projections, an effective batch size of 32, answer-only loss, AdamW, and cosine learning-rate decay.

```bash
python -m loop_dropout.train \
  --model-dir models/Ouro-1.4B --data-dir data \
  --output-dir outputs/loop-dropout \
  --rank 16 --alpha 32 --dropout 0.5 --learning-rate 1e-4 \
  --epochs 1 --batch-size 8 --gradient-accumulation 4 --max-length 1024 \
  --warmup-ratio 0.03 --weight-decay 0 --max-grad-norm 1 \
  --seed 101 --log-every 10 --max-steps 0 --device cuda --dtype bfloat16
```

The adapter weights and configuration are saved to `outputs/loop-dropout/adapter/`. Use a new output directory for each run.

### Inference

Load the saved adapter for text generation:

```bash
python -m loop_dropout.generate \
  --model-dir models/Ouro-1.4B --adapter outputs/loop-dropout/adapter \
  --prompt "A box has 12 red balls and 8 blue balls. How many balls are there?" \
  --max-new-tokens 512 --device cuda --dtype bfloat16
```

## Evaluation

Evaluate the saved adapter using greedy decoding on the prepared benchmarks:

```bash
python -m loop_dropout.evaluate \
  --model-dir models/Ouro-1.4B --adapter outputs/loop-dropout/adapter \
  --data-dir data --output-dir outputs/loop-dropout/evaluation \
  --batch-size 16 --device cuda --dtype bfloat16
```

Results are saved under `outputs/loop-dropout/evaluation/`, including per-example predictions in JSONL files and aggregate metrics in `summary.json`.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{zhu2026loopdropout,
  title         = {Loop Dropout: Regularizing Shared Updates in Looped Language Models},
  author        = {Zhu, Zirui and Xu, Hailun and Zhao, Xuanlei and Liu, Yong and Ren, Yingxuan and Sarkar, Kanchan and Xu, Kun and You, Yang},
  year          = {2026},
  eprint        = {2609.34218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.34218}
}
```

## Acknowledgments

This implementation builds on Ouro and uses the [MetaMath scoring utilities](loop_dropout/vendor/metamath/SOURCE.md).
