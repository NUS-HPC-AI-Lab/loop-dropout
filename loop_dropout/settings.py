"""Model, data and prompt defaults shared by every command."""

MODEL_ID = "ByteDance/Ouro-1.4B"
MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
NUM_LOOPS = 4
MODEL_DIR = "models/Ouro-1.4B"
DATA_DIR = "data"
OUTPUT_DIR = "outputs/loop-dropout"

DATA_REVISIONS = {
    "meta-math/MetaMathQA": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
}
TRAIN_PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
EVAL_PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)
STOP_STRINGS = [
    "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT",
    "Instruction:", "Instruction", "Response:", "Response",
]
MAX_NEW_TOKENS = {"gsm8k": 512, "math500": 2048}
