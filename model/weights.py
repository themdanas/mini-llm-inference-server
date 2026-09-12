from __future__ import annotations
import torch
import json
from pathlib import Path
from typing import Dict, Optional
from model.config import ModelConfig
from model.transformer import Transformer


#Key MAping table
# HuggingFace LLaMA weight keys → our key names
# If you load a real LLaMA / Mistral checkpoint from HF, apply this map.
HF_LLAMA_KEY_MAP = {
    "model.embed_tokens.weight":          "embedding.embedding.weight",
    "model.norm.weight":                  "norm.weight",
    "lm_head.weight":                     "lm_head.weight",
}
# Per-layer patterns (replace {i} with actual layer index)
HF_LLAMA_LAYER_MAP = {
    "model.layers.{i}.input_layernorm.weight":          "blocks.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blocks.{i}.ffn_norm.weight",
    "model.layers.{i}.self_attn.q_proj.weight":         "blocks.{i}.attn.w_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight":         "blocks.{i}.attn.w_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight":         "blocks.{i}.attn.w_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight":         "blocks.{i}.attn.w_o.weight",
    "model.layers.{i}.mlp.gate_proj.weight":            "blocks.{i}.ffn.w_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight":              "blocks.{i}.ffn.w_up.weight",
    "model.layers.{i}.mlp.down_proj.weight":            "blocks.{i}.ffn.w_down.weight",
}


def remap_hf_llama_keys(
    state_dict: Dict[str, torch.Tensor],
    n_layers: int,
) -> Dict[str, torch.Tensor]:
    """
    Remap HuggingFace LLaMA weight keys to our naming convention.
    Returns a new state dict with our key names.
    Unmapped keys are passed through unchanged (with a warning).
    """
    new_sd: Dict[str, torch.Tensor] = {}
    used = set()

    # Top-level keys
    for hf_key, our_key in HF_LLAMA_KEY_MAP.items():
        if hf_key in state_dict:
            new_sd[our_key] = state_dict[hf_key]
            used.add(hf_key)

    # Per-layer keys
    for i in range(n_layers):
        for hf_pattern, our_pattern in HF_LLAMA_LAYER_MAP.items():
            hf_key = hf_pattern.format(i=i)
            our_key = our_pattern.format(i=i)
            if hf_key in state_dict:
                new_sd[our_key] = state_dict[hf_key]
                used.add(hf_key)

    # Warn about any unmapped keys
    for key in state_dict:
        if key not in used:
            print(f"[weights.py] WARNING: unmapped key skipped: {key!r}")

    return new_sd


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_safetensors(path: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load a .safetensors file.  Requires: pip install safetensors

    safetensors advantages over torch.load():
      - No pickle → no arbitrary code execution risk
      - Memory-mapped by default → loads only the tensors you actually access
      - Supports lazy loading (load individual tensors without reading the whole file)
      - Faster than pickle for large files
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError(
            "safetensors is not installed. Run: pip install safetensors"
        )
    return load_file(str(path))


def load_pytorch_bin(path: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load a .bin or .pt PyTorch checkpoint.

    SECURITY WARNING: only load files from trusted sources.
    torch.load() uses pickle which can execute arbitrary Python.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)


def load_sharded(directory: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load a model sharded across multiple safetensors files.
    Reads the shard index (pytorch_model.bin.index.json or model.safetensors.index.json)
    and merges all shards into a single state dict.

    Used for large models (>10GB) that HF splits into multiple files.
    """
    directory = Path(directory)
    merged: Dict[str, torch.Tensor] = {}

    # Try safetensors index first, fall back to pytorch bin index
    for index_file in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        index_path = directory / index_file
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)

            shard_files = set(index["weight_map"].values())
            for shard_file in sorted(shard_files):
                shard_path = directory / shard_file
                print(f"[weights.py] Loading shard: {shard_file}")
                if shard_file.endswith(".safetensors"):
                    shard = load_safetensors(shard_path)
                else:
                    shard = load_pytorch_bin(shard_path)
                merged.update(shard)
            return merged

    raise FileNotFoundError(
        f"No shard index found in {directory}. "
        "Expected model.safetensors.index.json or pytorch_model.bin.index.json"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main load function
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(
    model: Transformer,
    path: str | Path,
    key_format: str = "native",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Transformer:
    """
    Load pretrained weights into a Transformer model.

    Args:
        model:      Transformer instance (already initialised with the right config)
        path:       path to a .safetensors file, .bin file, or directory of shards
        key_format: "native"   — keys already match our naming convention
                    "hf_llama" — HuggingFace LLaMA/Mistral/Gemma naming
        dtype:      cast all weights to this dtype after loading
                    (e.g. torch.float16 for inference; None = keep source dtype)
        device:     move model to this device after loading
        strict:     if True, raise error on missing/unexpected keys

    Returns:
        The model with weights loaded (same object as input, modified in-place).

    DTYPE CASTING STRATEGY:
    Load weights in float32 (most checkpoints are stored in fp32 or fp16).
    Then cast to the inference dtype.  This avoids accumulating rounding
    errors by doing the cast once cleanly rather than at every op.
    """
    path = Path(path)

    # ── Load raw state dict ───────────────────────────────────────────────
    if path.is_dir():
        print(f"[weights.py] Loading sharded checkpoint from {path}")
        state_dict = load_sharded(path)
    elif path.suffix == ".safetensors":
        print(f"[weights.py] Loading safetensors from {path}")
        state_dict = load_safetensors(path)
    elif path.suffix in (".bin", ".pt", ".pth"):
        print(f"[weights.py] Loading PyTorch checkpoint from {path}")
        state_dict = load_pytorch_bin(path)
    else:
        raise ValueError(f"Unrecognised weight file extension: {path.suffix}")

    # ── Remap keys if needed ──────────────────────────────────────────────
    if key_format == "hf_llama":
        print("[weights.py] Remapping HuggingFace LLaMA keys")
        state_dict = remap_hf_llama_keys(state_dict, model.config.n_layers)
    elif key_format != "native":
        raise ValueError(f"Unknown key_format: {key_format!r}. "
                         f"Use 'native' or 'hf_llama'.")

    # ── Cast dtype if requested ───────────────────────────────────────────
    if dtype is not None:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    # ── Load into model ───────────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[weights.py] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[weights.py] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    # ── Re-apply weight tying after load ──────────────────────────────────
    # Loading a state dict can break weight tying if both keys are present
    # as separate tensors.  Re-tie explicitly to be safe.
    if model.config.tie_weights:
        model.lm_head.weight = model.embedding.weight
        print("[weights.py] Weight tying applied (lm_head ← embedding)")

    # ── Move to device ────────────────────────────────────────────────────
    if device is not None:
        model = model.to(device)
        print(f"[weights.py] Model moved to {device}")

    total_params = model.n_params()
    print(f"[weights.py] Loaded {total_params/1e6:.1f}M parameters. ✓")

    return model


def save_weights(model: Transformer, path: str | Path) -> None:
    """
    Save model weights to a .safetensors file.
    Only saves parameters (no optimizer state, no buffers like RoPE tables).
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path))
    print(f"[weights.py] Saved {len(sd)} tensors to {path}")