import sys, os
import math
import pytest
import torch

from sampler.strategies import(
    apply_temperature, apply_repetition_penalty,
    apply_top_k, apply_top_p, apply_min_p,
    greedy_sample, multinomial_sample,
)

from sampler.sampler import SamplerConfig, LogitsProcessor, BatchSampler

#fixures

@pytest.fixture
def vocab_size():
    return 100
 
@pytest.fixture
def logits(vocab_size):
    torch.manual_seed(42)
    return torch.randn(1, vocab_size)


class TestSamplerStrategies:
 
    def test_temperature_identity(self, logits):
        """Temperature=1.0 should not change logits."""
        out = apply_temperature(logits, 1.0)
        assert torch.allclose(out, logits)
 
    def test_temperature_sharpens(self, logits):
        """T < 1 should increase the gap between logits."""
        out = apply_temperature(logits, 0.5)
        # The ratio of any two logits doubles when T=0.5
        assert torch.allclose(out, logits / 0.5)
 
    def test_temperature_flattens(self, logits):
        """T > 1 should decrease the gap between logits."""
        out = apply_temperature(logits, 2.0)
        assert torch.allclose(out, logits / 2.0)
 
    def test_temperature_invalid(self, logits):
        with pytest.raises(ValueError):
            apply_temperature(logits, 0.0)
        with pytest.raises(ValueError):
            apply_temperature(logits, -1.0)
 
    def test_rep_penalty_identity(self, logits):
        """Penalty=1.0 should not change logits."""
        out = apply_repetition_penalty(logits, [1, 2, 3], penalty=1.0)
        assert torch.allclose(out, logits)
 
    def test_rep_penalty_reduces_positive(self, logits):
        """Positive logits for penalised tokens should become smaller."""
        # Force a positive logit at position 5
        logits_pos = logits.clone()
        logits_pos[0, 5] = 5.0
        out = apply_repetition_penalty(logits_pos, [5], penalty=1.5)
        assert out[0, 5] < logits_pos[0, 5]
 
    def test_rep_penalty_reduces_negative(self, logits):
        """Negative logits for penalised tokens should become more negative."""
        logits_neg = logits.clone()
        logits_neg[0, 5] = -3.0
        out = apply_repetition_penalty(logits_neg, [5], penalty=1.5)
        assert out[0, 5] < logits_neg[0, 5]
 
    def test_rep_penalty_empty_ids(self, logits):
        out = apply_repetition_penalty(logits, [], penalty=1.5)
        assert torch.allclose(out, logits)
 
    def test_top_k_keeps_k_tokens(self, logits):
        """After top-k, exactly k tokens should have finite logits."""
        k = 10
        out = apply_top_k(logits, k)
        finite = (out != float("-inf")).sum().item()
        assert finite == k
 
    def test_top_k_1_is_greedy(self, logits):
        """top_k=1 should keep only the argmax token."""
        out = apply_top_k(logits, 1)
        best_idx = logits.argmax().item()
        finite_idxs = (out != float("-inf")).nonzero(as_tuple=True)[-1].tolist()
        assert finite_idxs == [best_idx]
 
    def test_top_k_invalid(self, logits):
        with pytest.raises(ValueError):
            apply_top_k(logits, 0)
 
    def test_top_p_identity(self, logits):
        """top_p=1.0 should not filter anything."""
        out = apply_top_p(logits, 1.0)
        assert torch.allclose(out, logits)
 
    def test_top_p_reduces_vocab(self, logits):
        """top_p < 1 should filter some tokens."""
        out = apply_top_p(logits, 0.5)
        n_filtered = (out == float("-inf")).sum().item()
        assert n_filtered > 0
 
    def test_top_p_cumsum_property(self, logits):
        """
        After top-p filtering with p=0.9:
        The sum of softmax probabilities of surviving tokens should be >= 0.9.
        """
        p = 0.9
        out = apply_top_p(logits, p)
        surviving = out[out != float("-inf")]
        prob_sum = torch.softmax(surviving, dim=0).sum().item()
        # The surviving tokens' share of the original distribution should be >= p
        # (it's approximately p, but floating point makes it >= p - epsilon)
        assert prob_sum >= p - 0.01
 
    def test_top_p_invalid(self, logits):
        with pytest.raises(ValueError):
            apply_top_p(logits, 0.0)
        with pytest.raises(ValueError):
            apply_top_p(logits, 1.1)
 
    def test_greedy_sample_returns_argmax(self, logits):
        expected = logits.argmax(dim=-1)
        result = greedy_sample(logits)
        assert torch.equal(result, expected)
 
    def test_multinomial_sample_shape(self, logits):
        result = multinomial_sample(logits)
        assert result.shape == (1,)
 
    def test_multinomial_sample_valid_token(self, logits, vocab_size):
        result = multinomial_sample(logits)
        assert 0 <= result.item() < vocab_size
 
    def test_multinomial_sample_distribution(self, vocab_size):
        """
        Sample many times from a peaked distribution.
        The argmax token should be sampled most often.
        """
        torch.manual_seed(0)
        logits = torch.zeros(1, vocab_size)
        logits[0, 7] = 10.0   # strong peak at token 7
 
        counts = [0] * vocab_size
        for _ in range(500):
            tok = multinomial_sample(logits).item()
            counts[int(tok)] += 1
 
        assert counts[7] > 400, f"Token 7 should dominate; got {counts[7]}/500"
 
 

# 5. LogitsProcessor tests

 
class TestLogitsProcessor:
 
    def test_greedy_fast_path(self, logits, vocab_size):
        proc = LogitsProcessor(SamplerConfig(greedy=True))
        result = proc(logits)
        assert result.item() == logits.argmax().item()
 
    def test_output_shape_2d(self, logits):
        proc = LogitsProcessor(SamplerConfig(temperature=0.8))
        result = proc(logits)
        assert result.shape == (1,)
 
    def test_output_shape_1d(self, vocab_size):
        proc = LogitsProcessor(SamplerConfig(temperature=0.8))
        single_logits = torch.randn(vocab_size)
        result = proc(single_logits)
        assert result.shape == ()  # scalar for 1D input
 
    def test_valid_token_id(self, logits, vocab_size):
        proc = LogitsProcessor(SamplerConfig(temperature=0.9, top_k=10))
        result = proc(logits)
        assert 0 <= result.item() < vocab_size
 
    def test_top_k_constrains_output(self, vocab_size):
        """With top_k=1 and temperature=1, output must equal argmax."""
        torch.manual_seed(0)
        logits = torch.randn(1, vocab_size)
        proc = LogitsProcessor(SamplerConfig(top_k=1))
        result = proc(logits)
        assert result.item() == logits.argmax().item()
 
    def test_pipeline_applies_all_steps(self, vocab_size):
        """Smoke test: all strategies enabled, no crash."""
        logits = torch.randn(1, vocab_size)
        proc = LogitsProcessor(SamplerConfig(
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            min_p=0.05,
            rep_penalty=1.2,
        ))
        result = proc(logits, generated_ids=[1, 2, 3, 4])
        assert 0 <= result.item() < vocab_size
 
    def test_should_stop_max_tokens(self):
        cfg = SamplerConfig(max_new_tokens=10)
        proc = LogitsProcessor(cfg)
        assert proc.should_stop(5, 10, eos_token_id=2)
        assert not proc.should_stop(5, 9, eos_token_id=2)
 
    def test_should_stop_eos(self):
        cfg = SamplerConfig(max_new_tokens=100, min_new_tokens=0)
        proc = LogitsProcessor(cfg)
        assert proc.should_stop(2, 5, eos_token_id=2)   # EOS token
        assert not proc.should_stop(5, 5, eos_token_id=2)  # not EOS
 
    def test_should_stop_min_new_tokens(self):
        cfg = SamplerConfig(max_new_tokens=100, min_new_tokens=10)
        proc = LogitsProcessor(cfg)
        # EOS at step 5 < min_new_tokens=10: should NOT stop
        assert not proc.should_stop(2, 5, eos_token_id=2)
        # EOS at step 10 >= min_new_tokens=10: should stop
        assert proc.should_stop(2, 10, eos_token_id=2)
 
    def test_should_stop_custom_stop_tokens(self):
        cfg = SamplerConfig(stop_token_ids=[99, 100])
        proc = LogitsProcessor(cfg)
        assert proc.should_stop(99, 1, eos_token_id=2)
        assert proc.should_stop(100, 1, eos_token_id=2)
        assert not proc.should_stop(98, 1, eos_token_id=2)
 
 
# 6. BatchSampler tests

class TestBatchSampler:
 
    def test_batch_output_length(self, vocab_size):
        B = 3
        configs = [SamplerConfig(greedy=True) for _ in range(B)]
        sampler = BatchSampler(configs)
        logits = torch.randn(B, vocab_size)
        result = sampler.sample(logits)
        assert len(result) == B
 
    def test_batch_greedy_matches_argmax(self, vocab_size):
        B = 4
        configs = [SamplerConfig(greedy=True) for _ in range(B)]
        sampler = BatchSampler(configs)
        logits = torch.randn(B, vocab_size)
        result = sampler.sample(logits)
        expected = logits.argmax(dim=-1).tolist()
        assert result == expected
 
    def test_batch_mixed_configs(self, vocab_size):
        """Different configs per sequence — no crash."""
        configs = [
            SamplerConfig(greedy=True),
            SamplerConfig(temperature=0.9, top_p=0.9),
            SamplerConfig(temperature=0.7, top_k=10),
        ]
        sampler = BatchSampler(configs)
        logits = torch.randn(3, vocab_size)
        result = sampler.sample(logits, [[1,2,3], [4,5], []])
        assert len(result) == 3
        assert all(0 <= r < vocab_size for r in result)
 
 

# 7. SamplerConfig tests

 
class TestSamplerConfig:
 
    def test_defaults_are_identity(self):
        cfg = SamplerConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 1.0
        assert cfg.rep_penalty == 1.0
 
    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            SamplerConfig(temperature=0.0)
        with pytest.raises(ValueError):
            SamplerConfig(temperature=-1.0)
 
    def test_invalid_top_p(self):
        with pytest.raises(ValueError):
            SamplerConfig(top_p=0.0)
        with pytest.raises(ValueError):
            SamplerConfig(top_p=1.1)
 
    def test_invalid_rep_penalty(self):
        with pytest.raises(ValueError):
            SamplerConfig(rep_penalty=0.9)
 
    def test_greedy_preset(self):
        cfg = SamplerConfig.greedy_config()
        assert cfg.greedy is True
 
    def test_creative_preset(self):
        cfg = SamplerConfig.creative()
        assert cfg.temperature > 0.8
        assert cfg.top_p < 1.0
 
    def test_balanced_preset(self):
        cfg = SamplerConfig.balanced()
        assert cfg.top_k > 0
        assert cfg.top_p < 1.0
 
 
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
 