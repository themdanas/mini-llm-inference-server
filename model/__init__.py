"""
model/
──────
Public API for the model package.

"""
from model.config import ModelConfig
from model.transformer import Transformer, TransformerBlock
from model.attention import MultiHeadAttention, KVCache, precompute_rope_freq
from model.layers import RMSNorm, Embedding, SwiGLUFFN, GELUFFN, build_ffn
from model.weights import load_weights, save_weights

__all__ = [
    "ModelConfig",
    "Transformer",
    "TransformerBlock",
    "MultiHeadAttention",
    "KVCache",
    "precompute_rope_freq",
    "RMSNorm",
    "Embedding",
    "SwiGLUFFN",
    "GELUFFN",
    "build_ffn",
    "load_weights",
    "save_weights",
]