from __future__ import annotations
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        #added eps to prevent division from zero
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()

        return x * rms * self.weight.to(x.dtype)

class Embedding(nn.Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self._scale = math.sqrt(config.d_model)

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * self._scale

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

class SwiGLUFFN(nn.Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        d = config.d_model
        h = config.d_ff

        self.w_gate = nn.Linear(d, h, bias=False) #gatting projection
        self.w_up = nn.Linear(d, h, bias=False) #content project
        self.w_down = nn.Linear(h, d, bias=False) #output projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))

        up = self.w_up(x)

        hidden = gate * up
        hidden = self.dropout(hidden)

        return self.w_down(hidden)

class GELUFFN(nn.Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        d = config.d_model
        h = config.d_ff

        self.w1 = nn.Linear(d, h)
        self.w2 = nn.Linear(h, d)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))

def build_ffn(config:ModelConfig) -> nn.Module:
        if config.activation == "swiglu":
            return SwiGLUFFN(config)
        elif config.activation == "gelu":
            return GELUFFN(config)
        else:
            raise ValueError(f"Unknown activation: {config.activation!r}. "
                            f"Choose 'swiglu' or 'gelu'.")