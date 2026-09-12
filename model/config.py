from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import json
import math

@dataclass
class ModelConfig:
    #vocab
    vocab_size : int = 32_000
    pad_token_id : int = 0 
    bos_token_id : int = 1
    eos_token_id : int = 2

    #architecture dimentions 
    d_model : int = 512
    n_heads : int = 8
    n_layers : int = 8
    d_ff : int = 2048
    max_seq_len : int = 2048

    n_kv_heads : int = 8

    dropout : float = 0.0

    # numerical precision
    #float32 for development and float16 for GPU inference
    dtype : Literal["float32", "float16", "bfloat16"] = "float32"

    activation: Literal["swiglu", "gelu"] = "swiglu"

    #weight typing: if ture then LM heads share wieghts with input embedding matrix 
    #saves ~ vocab_size * d_model params
    tie_weights : bool = True

    #RMSnorm
    norm_eps : float = 1e-5 

    def __post_init__(self):
        #derived dimenstions validations 
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_model % self.n_kv_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_heads % self.n_kv_heads !=0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"               
            )

    #deriving the pripertiess
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def n_params_approx(self) -> int:
        embed = self.vocab_size * self.d_model

        attn = self.d_model * (self.d_model + 2*self.n_kv_heads * self.d_head)
        attn += self.d_model * self.d_model

        if self.activation == "swiglu":
            ff = 3 * self.d_model * self.d_ff

        else:
            ff = 2 * self.d_model * self.d_ff

        norms = 2 * self.d_model
        block = attn + ff + norms

        lm_head = 0 if self.tie_weights else self.d_model * self.vocab_size

        return embed + self.n_layers * block + self.d_model + lm_head
    
    #factory methods 
    @classmethod
    def tiny(cls) -> "ModelConfig":
        #model config for unit test and CI
        return cls(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_kv_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
            dtype="float32",           
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        "25M model good on Laptop GPU"

        return cls(
            vocab_size=32_000,
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_len=2048,
        )

    @classmethod
    def from_json(cls, path:str) -> "ModelConfig":
        #load config from json file

        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    def to_json(self, path:str) -> None:
        import dataclasses
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


    def __repr__(self) -> str:
        params = self.n_params_approx

        if params >= 1_000_000_000:
            size = f"{params/1e9:.1f}B"

        elif params >= 1_000_000:
            size = f"{params/1e6:.1f}M"

        else:
            size = f"{params/1e3:.1f}K"

        return (
            f"ModelConfig("
            f"layers={self.n_layers}, "
            f"d_model={self.d_model}, "
            f"heads={self.n_heads}/{self.n_kv_heads}, "
            f"d_ff={self.d_ff}, "
            f"vocab={self.vocab_size}, "
            f"~{size} params)"
        )