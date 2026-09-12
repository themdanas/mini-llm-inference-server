from linecache import cache
import math
import pytest
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from model.config import ModelConfig
from model.transformer import Transformer, TransformerBlock
from model.attention import (
    MultiHeadAttention, KVCache, precompute_rope_freq,
    rotate_half, apply_rope
)
from model.layers import RMSNorm, Embedding, SwiGLUFFN, GELUFFN, build_ffn

@pytest.fixture
def cfg():
    #tiny config that runs fast on cpu
    return ModelConfig.tiny()

@pytest.fixture
def model(cfg):
    torch.manual_seed(42)
    return Transformer(cfg)

@pytest.fixture
def device():
    return torch.device("cpu")


# Config Test
class TestModelConfig:
    def test_d_head_derived(self):
        cfg = ModelConfig(d_model=512, n_heads=8, n_kv_heads=8, n_layers=2, d_ff=256, vocab_size=100)
        assert cfg.d_head == 64

    def test_invalid_d_model_not_divisible(self):
        with pytest.raises(ValueError):
            ModelConfig(d_model=100, n_heads=8, n_kv_heads=4, n_layers=2, d_ff=256, vocab_size=100)

    def test_kv_group_size_mha(self):
        cfg = ModelConfig.tiny()
        assert cfg.kv_group_size == 1

    def test_kv_group_size_gqa(self):
        cfg = ModelConfig(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=64, vocab_size=100)
        assert cfg.kv_group_size == 2

    def test_param_count_reasonable(self):
        cfg = ModelConfig.small()
        # Should be in the range of 10M–100M for a small model
        assert 10_000_000 < cfg.n_params_approx < 200_000_000
 
    def test_tiny_factory(self):
        cfg = ModelConfig.tiny()
        assert cfg.n_layers == 2
        assert cfg.d_model == 64
 
    def test_repr_contains_size(self):
        cfg = ModelConfig.small()
        r = repr(cfg)
        assert "params" in r
 
    def test_json_roundtrip(self, tmp_path):
        cfg = ModelConfig.tiny()
        path = str(tmp_path / "config.json")
        cfg.to_json(path)
        cfg2 = ModelConfig.from_json(path)
        assert cfg.d_model == cfg2.d_model
        assert cfg.n_layers == cfg2.n_layers


#layers test
class TestRMSNorm:
    def test_output_shape(self, cfg):
        norm = RMSNorm(cfg.d_model)
        x = torch.randn(2, 10, cfg.d_model)
        out = norm(x)
        assert out.shape == x.shape
 
    def test_unit_rms(self, cfg):
       
        norm = RMSNorm(cfg.d_model)
        # Force weight to 1 so we can check pure normalisation
        with torch.no_grad():
            norm.weight.fill_(1.0)
        x = torch.randn(1, 1, cfg.d_model) * 100   # large magnitude
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
 
    def test_learnable_scale(self, cfg):
       
        norm = RMSNorm(cfg.d_model)
        assert isinstance(norm.weight, nn.Parameter)
        assert norm.weight.shape == (cfg.d_model,)
 
    def test_fp16_compatible(self, cfg):
        norm = RMSNorm(cfg.d_model).half()
        x = torch.randn(2, 5, cfg.d_model).half()
        out = norm(x)
        assert out.dtype == torch.float16


class TestTokenEmbedding:
    def test_output_shape(self, cfg):
        emb = Embedding(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, cfg.d_model)
 
    def test_scaling(self, cfg):
        
        emb = Embedding(cfg)
        ids = torch.tensor([[1]])
        with torch.no_grad():
            emb.embedding.weight.fill_(1.0)
        out = emb(ids)
        # Each output value should equal sqrt(d_model) (since weight=1, scaled by sqrt(d))
        expected = math.sqrt(cfg.d_model)
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-4)
 
    def test_weight_property(self, cfg):
        emb = Embedding(cfg)
        assert emb.weight.shape == (cfg.vocab_size, cfg.d_model)


class TestFFN:
    def test_swiglu_shape(self, cfg):
        ffn = SwiGLUFFN(cfg)
        x = torch.randn(2, 5, cfg.d_model)
        out = ffn(x)
        assert out.shape == (2, 5, cfg.d_model)
 
    def test_gelu_shape(self, cfg):
        cfg2 = ModelConfig.tiny()
        cfg2.activation = "gelu"
        ffn = GELUFFN(cfg2)
        x = torch.randn(2, 5, cfg2.d_model)
        out = ffn(x)
        assert out.shape == (2, 5, cfg2.d_model)
 
    def test_build_ffn_factory(self, cfg):
        ffn = build_ffn(cfg)
        assert isinstance(ffn, SwiGLUFFN)
 
    def test_invalid_activation(self, cfg):
        cfg_bad = ModelConfig.tiny()
        object.__setattr__(cfg_bad, 'activation', 'relu')
        with pytest.raises(ValueError):
            build_ffn(cfg_bad)


# RoPE Tests
class TestRoPE:
    def test_freq_table_shape(self, cfg):
        cos, sin = precompute_rope_freq(cfg.d_head, cfg.max_seq_len)
        assert cos.shape == (cfg.max_seq_len, cfg.d_head)
        assert sin.shape == (cfg.max_seq_len, cfg.d_head)

    def test_cos_sin_identity(self, cfg):
        cos, sin = precompute_rope_freq(cfg.d_head, cfg.max_seq_len)
        identity = cos.pow(2) + sin.pow(2)
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-6)

    def test_rotate_half_shape(self, cfg):
        x = torch.randn(2, 5, cfg.d_head)
        rot = rotate_half(x)
        assert rot.shape == x.shape

    def test_rotate_half_involution(self, cfg):
       
        x = torch.randn(1,1,1,8)
        assert torch.allclose(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x, atol=1e-6)

    def test_rope_output_shape(self, cfg):
        cos, sin = precompute_rope_freq(cfg.d_head, cfg.max_seq_len)
        B, T = 2, 5
        q = torch.randn(B, cfg.n_heads, T, cfg.d_head)
        k = torch.randn(B, cfg.n_kv_heads, T, cfg.d_head)
        pos = torch.arange(T).unsqueeze(0).expand(B, -1)
        q_rot, k_rot = apply_rope(q, k, cos, sin, pos)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
 
    def test_rope_preserves_magnitude(self, cfg):
       
        cos, sin = precompute_rope_freq (cfg.d_head, cfg.max_seq_len)
        q = torch.randn(1, cfg.n_heads, 3, cfg.d_head)
        k = torch.randn(1, cfg.n_kv_heads, 3, cfg.d_head)
        pos = torch.arange(3).unsqueeze(0)
        q_rot, k_rot = apply_rope(q, k, cos, sin, pos)
        # L2 norm before and after should match
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


#Attention test
class TestAttention:
    def test_output_shape(self, cfg):
        attn = MultiHeadAttention(cfg)
        cos, sin = precompute_rope_freq(cfg.d_head, cfg.max_seq_len)
        x = torch.randn(2, 5, cfg.d_model)
        pos = torch.arange(5).unsqueeze(0).expand(2,-1)
        out = attn(x, cos, sin, pos)
        assert out.shape == (2, 5, cfg.d_model)

    def test_causal_masking(self, cfg):
        torch.manual_seed(0)
        attn = MultiHeadAttention(cfg)
        attn.eval()
        cos, sin = precompute_rope_freq(cfg.d_head, cfg.max_seq_len)

        x1 = torch.randn(1, 5, cfg.d_model)
        x2 = x1.clone()

        x2[0, 3, :] += 100.0

        pos = torch.arange(5).unsqueeze(0)
        with torch.no_grad():
            out1 = attn(x1, cos, sin, pos)
            out2 = attn(x2, cos, sin, pos)

        #positions 0,1,2 must be identical - they can't see postion 3
        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5), \
            "Causal mask violated: position 3 leaked into positions 0-2"
 
        # Position 4 CAN see position 3, so it should differ
        assert not torch.allclose(out1[0, 4], out2[0, 4], atol=1e-5), \
            "Position 4 should be affected by position 3"


    def test_single_token_decode(self, cfg):
        
        attn = MultiHeadAttention(cfg)
        cos, sin = precompute_rope_freq(cfg.d_head, cfg.max_seq_len)
        cache = KVCache(cfg.n_kv_heads, cfg.d_head, cfg.max_seq_len)
 
        # Prefill 5 tokens
        x_prompt = torch.randn(1, 5, cfg.d_model)
        pos_prompt = torch.arange(5).unsqueeze(0)
        attn(x_prompt, cos, sin, pos_prompt, kv_cache=cache)
 
        # Decode step: 1 new token
        x_new = torch.randn(1, 1, cfg.d_model)
        pos_new = torch.tensor([[5]])
        out = attn(x_new, cos, sin, pos_new, kv_cache=cache)
        assert out.shape == (1, 1, cfg.d_model)
 
    def test_gqa_output_shape(self):
        
        cfg = ModelConfig(
            d_model=64, n_heads=4, n_kv_heads=2,
            n_layers=2, d_ff=128, vocab_size=100
        )
        attn = MultiHeadAttention(cfg)
        cos, sin = precompute_rope_freq(cfg.d_head, 128)
        x = torch.randn(2, 6, cfg.d_model)
        pos = torch.arange(6).unsqueeze(0).expand(2, -1)
        out = attn(x, cos, sin, pos)
        assert out.shape == (2, 6, cfg.d_model)


# KV Cache Tests
class TestKVCache:
    def test_update_grows_sequence(self, cfg):
        cache = KVCache(cfg.n_kv_heads, cfg.d_head,cfg.max_seq_len)
        k = torch.randn(1, cfg.n_kv_heads, 3, cfg.d_head)
        v = torch.randn(1, cfg.n_kv_heads, 3, cfg.d_head)
        k_full, v_full = cache.update(k, v)
        assert k_full.shape == (1, cfg.n_kv_heads, 3, cfg.d_head)
        assert cache.seq_len == 3

    def test_update_accumulates(self, cfg):
        cache = KVCache(cfg.n_kv_heads, cfg.d_head, cfg.max_seq_len)
        for step in range(4):
            k = torch.randn(1, cfg.n_kv_heads, 1, cfg.d_head)
            v = torch.randn(1, cfg.n_kv_heads, 1, cfg.d_head)
            k_full, v_full = cache.update(k, v)
        assert cache.seq_len == 4
        assert k_full.shape == (1, cfg.n_kv_heads, 4, cfg.d_head)
 
    def test_overflow_raises(self, cfg):
        cache = KVCache(cfg.n_kv_heads, cfg.d_head, max_seq_len=10)
        k = torch.randn(1, cfg.n_kv_heads, 11, cfg.d_head)
        v = torch.randn(1, cfg.n_kv_heads, 11, cfg.d_head)
        with pytest.raises(RuntimeError, match="KV cache overflow"):
            cache.update(k, v)
 
    def test_reset_clears_cache(self, cfg):
        cache = KVCache(cfg.n_kv_heads, cfg.d_head, cfg.max_seq_len)
        k = torch.randn(1, cfg.n_kv_heads, 5, cfg.d_head)
        v = torch.randn(1, cfg.n_kv_heads, 5, cfg.d_head)
        cache.update(k, v)
        cache.reset()
        assert cache.seq_len == 0
        assert cache.k_cache.sum().item() == 0.0



# Transformer tests
class TestTransformer:
    def test_output_shape(self, model, cfg):
        ids = torch.randint(0, cfg.vocab_size, (2,10))
        logits = model(ids)
        assert logits.shape == (2, 10, cfg.vocab_size)

    def test_single_token(self, model, cfg):
        ids = torch.randint(0, cfg.vocab_size, (1, 1))
        logits = model(ids)
        assert logits.shape == (1, 1, cfg.vocab_size)
 
    def test_no_nan_in_output(self, model, cfg):
        ids = torch.randint(0, cfg.vocab_size, (2, 20))
        logits = model(ids)
        assert not torch.isnan(logits).any(), "NaN in model output"
        assert not torch.isinf(logits).any(), "Inf in model output"
 
    def test_weight_tying(self, model):
        
        assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr(), \
            "Weight tying broken: lm_head and embedding have different weight tensors"
 
    def test_param_count(self, model):
        n = model.n_params()
        expected = model.config.n_params_approx
        # Actual count should be within 5% of approximation
        assert abs(n - expected) / expected < 0.05, \
            f"Param count {n} deviates >5% from estimate {expected}"
 
    def test_deterministic_with_seed(self, cfg):
        
        ids = torch.randint(0, cfg.vocab_size, (1, 5))
        torch.manual_seed(7)
        m1 = Transformer(cfg)
        m1.eval()
        torch.manual_seed(7)
        m2 = Transformer(cfg)
        m2.eval()
        with torch.no_grad():
            out1 = m1(ids)
            out2 = m2(ids)
        assert torch.allclose(out1, out2)
 
    def test_generate_returns_longer_sequence(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 3))
        out = model.generate(prompt, max_new_tokens=5)
        assert out.shape[1] >= 3 + 1   # at least one new token
 
    def test_rope_buffers_not_parameters(self, model):
        
        param_names = {n for n, _ in model.named_parameters()}
        assert "rope_cos" not in param_names
        assert "rope_sin" not in param_names 


# Critical: KV cache equilance 
class TestKVCacheEquivalence:
    @torch.no_grad()
    def test_cached_vs_uncached_prefill_logits(self, cfg):
       
        torch.manual_seed(42)
        model = Transformer(cfg)
        model.eval()

        ids = torch.randint(0, cfg.vocab_size, (1, 10))

        # no cache (standard forward)
        logits_no_cache = model(ids, kv_cache=None)

        #with cache (prefill only)
        caches = model.build_kv_cache(batch_size=1)
        logits_with_cache = model(ids, kv_cache=caches)

        assert torch.allclose(logits_no_cache, logits_with_cache, atol=1e-5), \
            "Prefill with KV cache does not match full forward without cache"

    @torch.no_grad()
    def test_cached_vs_uncached_decode_logits(self, cfg):
        torch.manual_seed(99)
        model = Transformer(cfg)
        model.eval()

        prompt_len = 5
        n_decode   = 8
        vocab      = cfg.vocab_size

        # Generate a fixed sequence (prompt + decode target)
        all_ids = torch.randint(0, vocab, (1, prompt_len + n_decode))

        #cached path
        caches = model.build_kv_cache(batch_size=1)

        #prefill : process entire prompt at once
        _ = model(all_ids[:, :prompt_len], kv_cache=caches)

        uncached_logits = []
        for step in range(n_decode):
            t = prompt_len + step
            full_ids = all_ids[:, :t+1]
            logits = model(full_ids, kv_cache=None)
            uncached_logits.append(logits[:, t, :])

        cached_logits = []
        for step in range(n_decode):
            t = prompt_len + step
            token = all_ids[:, t:t+1]
            logits = model(token, kv_cache=caches)
            cached_logits.append(logits[:, 0, :])  # decode step returns only the

        for step, (cached, uncached) in enumerate(zip(cached_logits, uncached_logits)):
            assert torch.allclose(cached, uncached, atol=1e-4), (
                f"KV cache equivalence FAILED at decode step {step}.\n"
                f"Max diff: {(cached - uncached).abs().max().item():.6f}\n"
                f"This means the KV cache is incorrect. Fix before proceeding."
            )

        @torch.no_grad()
        def test_position_ids_are_correct_in_decode(self, cfg):
            torch.manual_seed(11)
            model = Transformer(cfg)
            model.eval()

            # same token at position 0 vs postion 5 should give different logits
            token_id = torch.tensor([[7]])

            caches_a = model.build_kv_cache()
            caches_b = model.build_kv_cache()

            #pad cache_b with 5 dummy tokens so the next token is at position 5
            dummy = torch.randint(0, cfg.vocab_size,(1,5))
            model(dummy, kv_cache=caches_b)

            logits_pos0 = model(token_id, kv_cache=caches_a)
            logits_pos5 = model(token_id, kv_cache=caches_b)

            assert not torch.allclose(logits_pos0, logits_pos5, atol=1e-3), \
                "Position IDs are not being applied correctly in decode mode. " 

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

