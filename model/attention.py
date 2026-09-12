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
    cos_pos = cos[position_ids]
    sin_pos = sin[position_ids]

    cos_pos = cos_pos.unsqueeze(1)
    sin_pos = sin_pos.unsqueeze(1)

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
    ):
        self.n_kv_head = n_kv_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self._len = 0

        self.k_cache = torch.zeros(
            1, n_kv_heads, max_seq_len, d_head,
            dtype=dtype,
            device=device,
        )

        self.v_cache = torch.zeros(
            1, n_kv_heads, max_seq_len, d_head,
            dtype=dtype,
            device=device,
        )
    
    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        T_new = k_new.shape[2]
        end = self._len + T_new

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: tried to store {end} tokens "
                f"but max_seq_len={self.max_seq_len}"
            )

        self.k_cache[:, :, self._len:end, :] = k_new
        self.v_cache[:, :, self._len:end, :] = v_new

        self._len = end

        return (
            self.k_cache[:, :, :self._len, :],
            self.v_cache[:, :, :self._len, :],
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
        x,
        cos,
        sin,
        positions_ids,
        kv_cache=None,
        attn_mask=None,
    ):
        B, T, _ = x.shape

        # -------------------------
        # 1. Project Q, K, V
        # -------------------------
        q = self.w_q(x).view(
            B,
            T,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)

        k = self.w_k(x).view(
            B,
            T,
            self.n_kv_heads,
            self.d_head,
        ).transpose(1, 2)

        v = self.w_v(x).view(
            B,
            T,
            self.n_kv_heads,
            self.d_head,
        ).transpose(1, 2)

        # -------------------------
        # 2. Apply RoPE
        # -------------------------
        q, k = apply_rope(
            q,
            k,
            cos,
            sin,
            positions_ids,
        )

        # -------------------------
        # 3. KV Cache
        # -------------------------
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # -------------------------
        # 4. Grouped Query Attention
        # -------------------------
        if self.kv_group > 1:
            k = k.repeat_interleave(
                self.kv_group,
                dim=1,
            )

            v = v.repeat_interleave(
                self.kv_group,
                dim=1,
            )

        # -------------------------
        # 5. Attention scores
        # -------------------------
        S = k.shape[2]

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.d_head)

        # -------------------------
        # 6. Causal mask
        # -------------------------
        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            causal_mask = torch.triu(
                torch.full(
                    (T, S),
                    float("-inf"),
                    device=q.device,
                    dtype=q.dtype,
                ),
                diagonal=S - T + 1,
            )

            scores = scores + causal_mask

        # -------------------------
        # 7. Softmax
        # -------------------------
        weights = F.softmax(
            scores,
            dim=-1,
        )

        # -------------------------
        # 8. Dropout
        # -------------------------
        if self.training:
            weights = self.attn_dropout(weights)

        # -------------------------
        # 9. Weighted values
        # -------------------------
        attn_out = torch.matmul(
            weights,
            v,
        )

        # -------------------------
        # 10. Merge heads
        # -------------------------
        attn_out = (
            attn_out
            .transpose(1, 2)
            .contiguous()
            .view(
                B,
                T,
                self.n_heads * self.d_head,
            )
        )

        # -------------------------
        # 11. Output projection
        # -------------------------
        return self.w_o(attn_out)

"""
class MultiHeadAttention(nn.Module):

    def __init__(self, config: ModelConfig):
            super().__init__()
            self.n_heads = config.n_heads
            self.n_kv_heads = config.n_kv_heads
            self.d_head = config.d_head
            self.d_model = config.d_model
            self.kv_group = config.kv_group_size
            self.dropout_p = config.dropout

            #projections matrix
            self.w_q = nn.Linear(config.d_model, config.n_heads*config.d_head, bias=False)

            self.w_k = nn.Linear(config.d_model, config.n_kv_heads*config.d_head, bias=False)

            self.w_v = nn.Linear(config.d_model, config.n_kv_heads*config.d_head, bias=False)

            self.w_o = nn.Linear(config.n_heads*config.d_head, config.d_model, bias=False)

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

                #Projecting input into q, k, v.
                q = self.w_q(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
                k = self.w_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
                v = self.w_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
                    # q: (B, H,  T, D)
                    # k: (B, Hk, T, D)
                    # v: (B, Hk, T, D)

                #applying rope to q and k
                q, k = apply_rope(q, k, cos, sin, positions_ids)

                #updating the KV Cache 
                if kv_cache is not None:
                        k, v = kv_cache.update(k, v)

                # GQA: expand K/V to match Q head count 
                # If n_kv_heads < n_heads, each K/V head is repeated kv_group times.
                    # torch.repeat_interleave on dim=1 duplicates heads contiguously.
                if self.kv_group > 1:
                        k = k.repeat_interleave(self.kv_group, dim=1)
                        v = v.repeat_interleave(self.kv_group, dim=1)

                #scaled dot product attention
                # pytorch 2.0 has fused SDPA that automattically chose Flashattention when available 
                # attn scores = Q @ K^T / sqrt(D)
                        # Attention
                S = k.shape[2]

                causal_mask = torch.triu(
                    torch.full(
                        (T, S),
                        float("-inf"),
                        device=q.device,
                        dtype=q.dtype,
                    ),
                    diagonal=S - T + 1,
                )

                if attn_mask is not None:
                    causal_mask = causal_mask + attn_mask

                scores = torch.matmul(
                    q,
                    k.transpose(-2, -1),
                ) / math.sqrt(self.d_head)

                scores = scores + causal_mask

                weights = F.softmax(scores, dim=-1)

                if self.training:
                    weights = self.attn_dropout(weights)

                attn_out = torch.matmul(weights, v)

                
                #for test calculation of attention scores, 
                # we are using the manual implementation of attention, 
                # but in production we will use the fused implementation of attention.


                if hasattr(F, "scaled_dot_product_attention"):
                        #using pytorch 2.0 fused kernal flashattention. on cuda
                        is_causal = attn_mask is None
                        print("T:", T)
                        print("attn_mask:", attn_mask)
                        print("is_causal:", is_causal)
                        attn_out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=self.dropout_p if self.training else 0.0,
                            is_causal=is_causal
                        )
                else:
                        #manual fallback for older pytoch versions 
                        attn_out = self._manual_attention(q, k, v, attn_mask, T)

                        #attn_out (B, H, T, D)

                #merging heads and project
                #concentrating all head outputs along the feature dimentions 
                #contigious ensures that memory layout is correct for the linear layer

                attn_out = attn_out.transpose(1,2).contiguous().view(B, T, self.n_heads * self.d_head)
                #attn_out (B, T, d_model)
                
                return self.w_o(attn_out)

                


    def _manual_attention(
                  self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  T: int
    ) -> torch.Tensor:
            S = k.shape[2]
            scale = 1.0/ math.sqrt(self.d_head)

            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attn_mask is not None:
                    scores = scores + attn_mask

            else:
                    causal = torch.triu(
                    torch.full((T, S), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=S - T + 1,   # offset accounts for KV cache
                )

                    scores = scores + causal

            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)


            return torch.matmul(weights, v)

"""