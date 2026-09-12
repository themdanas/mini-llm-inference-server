from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from sampler.strategies import(
    apply_temperature,
    apply_repetation_penalty,
    apply_top_k,
    apply_top_p,
    apply_min_p,
    greedy_sample,
    multinomial_sample,
)

#sample config
@dataclass
class SamplerConfig:

    greedy: bool = False

    temperature: float = 1.0

    top_k: int = 0
    top_p: float = 0.0
    min_p: float = 0.0
    rep_penalty: float = 1.0

    max_new_tokens: int = 256
    min_new_tokens: int = 0
    stop_token_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temprature must be > 0, got {self.temperature}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got{self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be (0,1], got{self.top_k}")
        if self.min_p < 0 or self.min_p >= 1.0: 
            raise ValueError(f"min_p must be in [0,1) got{self.min_p}")
        if self.rep_penalty < 1.0:
            raise ValueError(f"rep_penalty must be >= 1.0, got {self.rep_penalty}")

    #Named presets 
    @classmethod
    def greedy_config(cls) -> "SamplerConfig":
        #deterministic greedy decoding. Best for evaluating/testing.
        return cls(greedy=True, temperature=1.0)

    @classmethod
    def creative(cls) -> "SamplerConfig":
        #High diversity for creating writing tasks
        return cls(temperature=0.9, top_p=0.95, reg_penalty=1.1, max_new_tokens=512)

    @classmethod
    def balanced(cls) -> "SamplerConfig":
        #Good balance of quality and diversity for general chat
        return cls(temperature=0.8, top_k=40, top_p=0.9, rep_penalty=1.1)

    @classmethod
    def conservative(cls) -> "SamplerConfig":
        return cls(temperature=0.6, top_K=50, rep_penalty=1.2)


#LogitsProcessor
class LogitsProcessor:
    #composablr logit processing pipeline

    def __init__(self, config: SamplerConfig):
        self.config = config

    def __call__(self, 
                 logits: torch.Tensor,
                 generated_ids: Optional[List[int]] = None,
                 ) -> torch.Tensor:
        #Process logits and return the next token ID.
        if logits.dim()==1:
            logits = logits.unsqueeze(0)
            squeez = True
        else:
            squeez = False

        #Greedy first path
        if self.config.greedy:
            result = greedy_sample(logits)
            return result.unsqueez(0) if squeez else result

        #1-> repetition penalty
        if self.config.rep_penalty != 1.0 and generated_ids:
            logits = apply_repetation_penalty(
                logits, generated_ids, self.config.rep_penalty
            )
        
        # 2-> temperature
        if self.config.temperature != 1.0:
            logits = apply_temperature(logits, self.config.temperature)

        # 3-> top-K
        if self.config.top_k > 0:
            logits = apply_top_k(logits, self.config.top_k)

        # 3 -> top-P
        if self.config.top_p < 1.0:
            logits = apply_top_p(logits, self.config.top_p)

        if self.config.min_p > 0.0:
            logits = apply_min_p(logits, self.config.min_p)

        result = multinomial_sample(logits)


    def should_stop(
            self,
            token_id: int,
            n_generated: int,
            eos_token_id: int,
    ) -> bool:

        if n_generated >= self.config.max_new_tokens:
            return True

        if token_id == eos_token_id and n_generated >= self.config.min_new_tokens:
            return True

        if token_id in self.config.stop_token_ids:
            return True

        return False

#batch sample
class BatchSampler:

    def __init__(self,
                 configs: List[SamplerConfig]):

                 self.configs = configs
                 self.processors = [LogitsProcessor(cfg) for cfg in configs]

    def sample(
            self,
            logits: torch.Tensor,
            generated_ids_per_seq: Optional[List[List[int]]] = None,
    ) -> List[int]:

        B = logits.size(0)
        next_tokens: List[int] = []

        for i in range(B):
            seq_logits = logits[i]
            past_ids = generated_ids_per_seq[i] if generated_ids_per_seq else None
            token_id = self.processors[i](seq_logits, past_ids).item()
            next_tokens.append(int(token_id))

        return next_tokens
    

                
        
        
        

        

