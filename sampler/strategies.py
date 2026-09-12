from __future__ import annotations
import torch
from typing import List, Optional

#temperature scaling 
def apply_temperature(logits: torch.Tensor, temperature:float) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError(f" Temperature must be >0, got {temperature}")

    if abs(temperature - 1.0) < 1e-6:
        return logits

    return logits/temperature


#Repitition penalty
def apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: List[int],
        penlaty: float = 1.0,
) -> torch.Tensor:

    if penlaty == 1.0 or not generated_ids:
        return logits

    if penlaty < 1.0:
        raise ValueError(f"Repetition penalty must be >=1.0, got{penlaty}")

    logits = logits.clone()

    # depulicate generated IDs (we only need to penalise each token once)
    unique_ids = list(set(generated_ids))

    #extract the logits at the postions of previously generated tokens
    score = logits[..., unique_ids]

    # Apply the penalty:
    # Positive logits: divide (make less positive → less probable)
    # Negative logits: multiply (make more negative → less probable)
    score = torch.where(score > 0, score / penlaty, score*penlaty)

    logits[..., unique_ids] = score
    return logits

# Top-K filtering
def apply_top_k(logits: torch.Tensor, k:int) -> torch.Tensor:
     # keep only the highest logits tokens: set all others -inf

     if k<=0:
         raise ValueError(f"top_k must be >=1, got{k}")

     k = min(k, logits.size(-1))

     threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]

     return logits.masked_fill(logits < threshold, float("-inf"))


# Top - P (nucleus) filtering
def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:

    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0,1], got {p}")

    if p == 1.0:
        return logits

    # Convert logits to probabilities for the cumsum computation
    # (we filter in logit space but need prob space for cumsum threshold)
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)


    # Find tokens to remove: those whose cumulative probability EXCEEDS p
    # after being added.
    # We shift by 1 (remove_mask[i] uses cumsum[i-1]) so the token that
    # pushes the cumsum over p is still included.
    remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= p

    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    filtered_logits = torch.zeros_like(logits).scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )

    return filtered_logits

#Min-p filtering
def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:

    if min_p == 0.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values

    threshold = min_p*max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))

#sampling
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)

def multinomial_sample(logits: torch.Tensor, num_samples: int =1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)

    zero_mask = (probs.sum(dim=-1) == 0)
    if zero_mask.any():
        probs[zero_mask] = torch.ones_like(probs[zero_mask]) / probs.size(-1)
 
    sampled = torch.multinomial(probs, num_samples=num_samples)
 
    if num_samples == 1:
        return sampled.squeeze(-1)  # (B,)
    return sampled  # (B, num_samples)

    