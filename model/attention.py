from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple

from model.config import ModelConfig


#here implementing the component of attention

#rope : rotatory positional embedding 

def precompute_rope_freq(
        d_head : int,
        max_seq_len : int,
        base : float = 10_000.0,
        device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    #compute 0_i, for i=0,2,4....d_head-2 so the shape becomes this
    half = d_head // 2

    #
    exponent = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)

    # θ_i = base^(-2i/d_head)
    freq = 1.0 / (base ** (exponent/d_head))

    #building the postions indices 0,1,2..max_sqe_len
    postions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    angles =torch.outer(postions, freq)

    angles = torch.cat([angles, angles], dim=-1)

    return angles.cos(),angles.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat([-x2,x1],dim=-1)

def apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    #applying rope to query and key 
    # cos_pos = cos[position_ids]
    # sin_pos = sin[position_ids]

    # Cast cos/sin to match q.dtype so fp16/bf16 tensors are not upcast to fp32
    cos_pos = cos[position_ids].unsqueeze(1).to(dtype=q.dtype, device=q.device)
    sin_pos = sin[position_ids].unsqueeze(1).to(dtype=q.dtype, device=q.device) 

    #applying the rotation 
    q_rot = q * cos_pos + rotate_half(q) * sin_pos
    k_rot = k * cos_pos + rotate_half(k) * sin_pos

    return q_rot, k_rot


class KVCache:

    def __init__(
        self,
        n_kv_heads: int,
        d_head: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        batch_size: int = 1,        
    ):
        self.n_kv_head = n_kv_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self._len = 0

        self.k_cache = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, d_head,
            dtype=dtype,
            device=device,
        )

        self.v_cache = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, d_head,
            dtype=dtype,
            device=device,
        )
    
    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B_new, _, T_new, _ = k_new.shape
        end = self._len + T_new

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: tried to store {end} tokens "
                f"but max_seq_len={self.max_seq_len}"
            )
        
        if B_new > self.k_cache.shape[0]:
            self.k_cache = self.k_cache.expand(B_new, -1, -1, -1).contiguous()
            self.v_cache = self.v_cache.expand(B_new, -1, -1, -1).contiguous()

        self.k_cache[:B_new, :, self._len:end, :] = k_new
        self.v_cache[:B_new, :, self._len:end, :] = v_new

        self._len = end

        return (
            self.k_cache[:B_new, :, :self._len, :],
            self.v_cache[:B_new, :, :self._len, :],
        )

    @property
    def seq_len(self) -> int:
        return self._len

    def reset(self) -> None:
        self._len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.kv_group = config.kv_group_size
        self.dropout_p = config.dropout

        self.w_q = nn.Linear(
            config.d_model,
            config.n_heads * config.d_head,
            bias=False,
        )

        self.w_k = nn.Linear(
            config.d_model,
            config.n_kv_heads * config.d_head,
            bias=False,
        )

        self.w_v = nn.Linear(
            config.d_model,
            config.n_kv_heads * config.d_head,
            bias=False,
        )

        self.w_o = nn.Linear(
            config.n_heads * config.d_head,
            config.d_model,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, positions_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.kv_group > 1:
            k = k.repeat_interleave(self.kv_group, dim=1)
            v = v.repeat_interleave(self.kv_group, dim=1)

        S = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores + attn_mask
        elif T > 1:
            causal_mask = torch.triu(
                torch.full((T, S), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=S - T + 1,
            )
            scores = scores + causal_mask

        weights = F.softmax(scores, dim=-1)

        if self.training:
            weights = self.attn_dropout(weights)

        attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

        return self.w_o(attn_out)
