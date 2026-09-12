from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Literal, Optional

from model.config import ModelConfig
from model.attention import KVCache, MultiHeadAttention, precompute_rope_freq
from model.layers import RMSNorm, Embedding, SwiGLUFFN, GELUFFN, build_ffn


#transformer block
class TransformerBlock(nn.Module):
    #contains all the components of a transformer 

    def __init__(self, config: ModelConfig,layers_idx: int):
        super().__init__()
        self.layers_idx = layers_idx

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        self.attn = MultiHeadAttention(config)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        self.fnn = build_ffn(config)

    def forward(
            self,
            x: torch.Tensor,
            sin: torch.Tensor,
            cos: torch.Tensor,
            positional_ids: torch.Tensor,
            kv_cache: Optional[KVCache] = None,
            attn_mask: Optional[torch.Tensor] = None,

    ) -> torch.Tensor:
        # attn-sub layer
        #pre-norms

        residuals = x
        x = self.attn_norm(x)
        x = self.attn(x, cos, sin, positional_ids, kv_cache, attn_mask)
        x = residuals + x

        #ffn -sub layer
        residuals = x
        x = self.ffn_norm(x)
        x = self.fnn(x)
        x = residuals + x

        return x


class Transformer(nn.Module):
    #complete autoregresive language model

    def __init__(self,config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = Embedding(config)

        #transformer block
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layers_idx=i)
            for i in range(config.n_layers)
        ])

        #final normalization
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        #LM Head : d_model -> vocab size
        #No bias
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False)

        #weight tying 
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        #pre-compute rope tables 
        #compute once at init and registers as non-param buffer 
        cos, sin = precompute_rope_freq(
            d_head=config.d_head,
            max_seq_len=config.max_seq_len
        )

        self.register_buffer("rope_cos",cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        #weight intilization
        self._init_weights()


    def _init_weights(self) -> None:

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        #scaling the residual projection by 1/sqrt(n_layers)
        scale = (self.config.n_layers ** -0.5)
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0,std =0.02 *scale)
            if hasattr(block.fnn, 'w_down'):
                nn.init.normal_(block.fnn.w_down.weight, mean=0.0, std=0.02 * scale)
            elif hasattr(block.fnn, 'w2'):
                nn.init.normal_(block.fnn.w2.weight, mean=0.0, std=0.02 * scale)


    #kv cache management
    def build_kv_cache(
            self,
            batch_size: int = 1,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.dtype] = None
    ) -> List[KVCache]:

        if dtype is None:
            dtype_map = {
                "float32" : torch.float32,
                "float16" : torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map[self.config.dtype]

        if device is None:
            device = next(self.parameters()).device

        return [
            KVCache(
                n_kv_heads=self.config.n_kv_heads,
                d_head=self.config.d_head,
                max_seq_len=self.config.max_seq_len,
                dtype=dtype,
                device=device
            )

            for _ in range(self.config.n_layers)
        ]

    #forward pass
    def forward(
            self,
            token_ids : torch.Tensor,
            kv_caches : Optional[List[KVCache]] = None,
            positional_ids : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  
        B, T = token_ids.shape
        device = token_ids.device

        #build positional ids
        if positional_ids is None:
            if kv_caches is not None:
                offset = kv_caches[0].seq_len
            else:
                offset = 0


            positional_ids = torch.arange(offset, offset + T, device=device).unsqueeze(0)
            positional_ids = positional_ids.expand(B, -1)


        x = self.embedding(token_ids)

        attn_mask = None

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(
                x,
                cos = self.rope_cos,
                sin = self.rope_sin,
                positional_ids = positional_ids,
                kv_cache = cache,
                attn_mask = attn_mask
            )

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids : torch.Tensor,
        max_new_tokens : int,
        temprature : float = 1.0,
        top_k : Optional[int] = None,
        eos_token_id : Optional[int] = None
    ) -> torch.Tensor:

        self.eval()
        device = prompt_ids.device
        generated = prompt_ids.clone()

        # allocate KV Cache for this generation
        kv_caches = self.build_kv_cache(batch_size=1, device=device)

        #prefill : process the entire prompt in one pass
        logits = self.forward(generated, kv_caches=kv_caches)

        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            #applying temprature

            if temprature !=0:
                next_logits = next_logits/temprature

            #top-k filtering 
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            #sample (or greedy if temprature is very low)
            probs = torch.softmax(next_logits, dim=-1)
            if temprature < 1e-6:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            #next token (1,1)

            generated = torch.cat([generated, next_token], dim=1)

            #checking for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            logits = self.forward(next_token, kv_caches=kv_caches)
            next_logits = logits[:,-1,:]

        return generated

    def n_params(self, exxclude_embeddings: bool = False) -> int:
        """Counts trainable parameters"""

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exxclude_embeddings:
            params -= self.embedding.embedding.weight.numel()

        return params

    def __repr__(self) -> str:
        n = self.n_params()
        unit = "M" if n>= 1e6 else "K"
        val = n / 1e6 if n>= 1e6 else n/1e3
        return (
            f"Transformer({self.config.n_layers}),"
            f"d={self.config.d_model},"
            f"h={self.config.n_heads},"
            f"~{val:.1f}{unit}params"
        )

