"""Shared LoRA with one rescaled Bernoulli mask per example and loop."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from .settings import MODEL_ID, MODEL_REVISION, NUM_LOOPS

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.5
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    num_loops: int = NUM_LOOPS

    def __post_init__(self):
        if self.rank < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("rank and alpha must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if (self.model_id, self.model_revision, self.num_loops) != (MODEL_ID, MODEL_REVISION, NUM_LOOPS):
            raise ValueError("The adapter must use the pinned Ouro-1.4B model with four loops.")


def sample_mask(batch_size, dropout, generator=None):
    """Return [loop, example] masks with E[mask] = 1."""
    if batch_size < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid batch size or dropout probability.")
    keep = (torch.rand(NUM_LOOPS, batch_size, generator=generator) >= dropout).float()
    return keep / (1.0 - dropout)


class LoopContext:
    def __init__(self):
        self.loop = 0
        self.mask = None

    def layer_hook(self, module, args, kwargs):
        # Ouro passes this keyword to every decoder layer at every loop.
        self.loop = int(kwargs["current_ut"])


class LoRALinear(nn.Module):
    def __init__(self, base, context, config):
        super().__init__()
        self.base = base
        self.context = context
        self.scale = config.alpha / config.rank
        self.lora_A = nn.Parameter(torch.empty(
            config.rank, base.in_features, device=base.weight.device, dtype=torch.float32,
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, config.rank, device=base.weight.device, dtype=torch.float32,
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        delta = delta * self.scale
        mask = self.context.mask
        if mask is not None:
            if mask.shape[1] != x.shape[0]:
                raise ValueError("Loop mask and input batch sizes differ.")
            gate = mask[self.context.loop].to(device=x.device, dtype=x.dtype)
            delta = delta * gate.reshape(-1, *([1] * (x.ndim - 1)))
        return base_output + delta


class LoopDropoutAdapter:
    def __init__(self, model, config=None):
        self.config = config or AdapterConfig()
        self.model = model
        self.context = LoopContext()
        self.modules = {}
        self._hooks = []
        if any(isinstance(module, LoRALinear) for module in model.modules()):
            raise ValueError("This model already has an adapter.")
        if model.config.model_type != "ouro" or model.model.total_ut_steps != NUM_LOOPS:
            raise ValueError("Expected Ouro configured for four loops.")
        model.requires_grad_(False)
        for index, layer in enumerate(model.model.layers):
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name)
                for name, child in list(parent.named_children()):
                    if name in TARGETS and isinstance(child, nn.Linear):
                        module = LoRALinear(child, self.context, self.config)
                        setattr(parent, name, module)
                        self.modules[f"layers.{index}.{parent_name}.{name}"] = module
            self._hooks.append(layer.register_forward_pre_hook(self.context.layer_hook, with_kwargs=True))
        if len(self.modules) != len(model.model.layers) * len(TARGETS):
            raise ValueError("Ouro must have all seven target projections in every layer.")
        self.eval()

    def parameters(self):
        for module in self.modules.values():
            yield module.lora_A
            yield module.lora_B

    def train_batch(self, batch_size, generator=None):
        # Keep the frozen backbone in eval mode; only the LoRA masks are stochastic.
        self.model.eval()
        self.context.mask = sample_mask(batch_size, self.config.dropout, generator)

    def eval(self):
        self.model.eval()
        self.context.mask = None
        return self

    def state_dict(self):
        return {
            f"{name}.{key}": getattr(module, key).detach().cpu().contiguous()
            for name, module in self.modules.items() for key in ("lora_A", "lora_B")
        }

    def load_state_dict(self, state):
        expected = {
            f"{name}.{key}": getattr(module, key)
            for name, module in self.modules.items() for key in ("lora_A", "lora_B")
        }
        if state.keys() != expected.keys():
            raise ValueError("Adapter tensor names do not match the model.")
        for name, parameter in expected.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f"Adapter tensor has the wrong shape: {name}")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(directory / "adapter_model.safetensors"))
        (directory / "adapter_config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, model, directory):
        directory = Path(directory)
        config = AdapterConfig(**json.loads((directory / "adapter_config.json").read_text(encoding="utf-8")))
        state = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
        adapter = cls(model, config)
        adapter.load_state_dict(state)
        return adapter.eval()
