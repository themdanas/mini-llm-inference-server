import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))

import math
import pytest
import torch

from tokenizer.vocabulary import (
    Vocabulary, BYTE_TO_UNICODE, UNICODE_TO_BYTE, _build_byte_to_unicode
)

from tokenizer.tokenizer import (
    BPETokenizer, get_pairs, apply_merge, bpe_encode_word, GPT2_SPLIT_PATTERN
)

#Fixure 
@pytest.fixture
def tiny_vocab():
    return Vocabulary.tiny_synthetic(vocab_size=256)

@pytest.fixture
def tiny_tok():
    return BPETokenizer(tiny_vocab, add_bos=False, add_eos=False)

@pytest.fixture
def logits(vocab_size):
    torch.manual_seed(42)
    return torch.randn(1, vocab_size)

#Vocabulary test
class TestBPEAlgorithm:
 
    def test_get_pairs_basic(self):
        pairs = get_pairs(["h", "e", "l", "l", "o"])
        assert pairs == [("h","e"), ("e","l"), ("l","l"), ("l","o")]
 
    def test_get_pairs_single(self):
        assert get_pairs(["x"]) == []
 
    def test_get_pairs_two(self):
        assert get_pairs(["a", "b"]) == [("a", "b")]
 
    def test_apply_merge_basic(self):
        result = apply_merge(["l", "o", "w", "e", "r"], ("l", "o"))
        assert result == ["lo", "w", "e", "r"]
 
    def test_apply_merge_multiple(self):
        """All non-overlapping occurrences of the pair should be merged."""
        result = apply_merge(["a", "b", "a", "b"], ("a", "b"))
        assert result == ["ab", "ab"]
 
    def test_apply_merge_overlapping(self):
        """Overlapping pairs: first match wins, greedy left-to-right."""
        result = apply_merge(["a", "a", "a"], ("a", "a"))
        # First "a","a" merged → ["aa", "a"]  (not ["a", "aa"])
        assert result == ["aa", "a"]
 
    def test_apply_merge_no_match(self):
        symbols = ["h", "e", "l", "l", "o"]
        result = apply_merge(symbols, ("x", "y"))
        assert result == symbols
 
    def test_bpe_encode_word_no_merges(self):
        """With no merge rules, each character stays separate."""
        result = bpe_encode_word(["h", "e", "l", "l", "o"], {})
        assert result == ["h", "e", "l", "l", "o"]
 
    def test_bpe_encode_word_with_merges(self):
        """Merges applied in priority order."""
        merge_ranks = {("h", "e"): 0, ("he", "l"): 1, ("hel", "l"): 2}
        result = bpe_encode_word(["h", "e", "l", "l", "o"], merge_ranks)
        assert result == ["hell", "o"]
 
    def test_bpe_encode_word_single_char(self):
        result = bpe_encode_word(["x"], {("x", "y"): 0})
        assert result == ["x"]
 
    def test_bpe_encode_word_full_merge(self):
        """All chars merged into one token."""
        merge_ranks = {("a", "b"): 0, ("ab", "c"): 1}
        result = bpe_encode_word(["a", "b", "c"], merge_ranks)
        assert result == ["abc"]

#Test BPE Algorithm
class TestBPEAlgorithm:
 
    def test_get_pairs_basic(self):
        pairs = get_pairs(["h", "e", "l", "l", "o"])
        assert pairs == [("h","e"), ("e","l"), ("l","l"), ("l","o")]
 
    def test_get_pairs_single(self):
        assert get_pairs(["x"]) == []
 
    def test_get_pairs_two(self):
        assert get_pairs(["a", "b"]) == [("a", "b")]
 
    def test_apply_merge_basic(self):
        result = apply_merge(["l", "o", "w", "e", "r"], ("l", "o"))
        assert result == ["lo", "w", "e", "r"]
 
    def test_apply_merge_multiple(self):
        """All non-overlapping occurrences of the pair should be merged."""
        result = apply_merge(["a", "b", "a", "b"], ("a", "b"))
        assert result == ["ab", "ab"]
 
    def test_apply_merge_overlapping(self):
        """Overlapping pairs: first match wins, greedy left-to-right."""
        result = apply_merge(["a", "a", "a"], ("a", "a"))
        # First "a","a" merged → ["aa", "a"]  (not ["a", "aa"])
        assert result == ["aa", "a"]
 
    def test_apply_merge_no_match(self):
        symbols = ["h", "e", "l", "l", "o"]
        result = apply_merge(symbols, ("x", "y"))
        assert result == symbols
 
    def test_bpe_encode_word_no_merges(self):
        """With no merge rules, each character stays separate."""
        result = bpe_encode_word(["h", "e", "l", "l", "o"], {})
        assert result == ["h", "e", "l", "l", "o"]
 
    def test_bpe_encode_word_with_merges(self):
        """Merges applied in priority order."""
        merge_ranks = {("h", "e"): 0, ("he", "l"): 1, ("hel", "l"): 2}
        result = bpe_encode_word(["h", "e", "l", "l", "o"], merge_ranks)
        assert result == ["hell", "o"]
 
    def test_bpe_encode_word_single_char(self):
        result = bpe_encode_word(["x"], {("x", "y"): 0})
        assert result == ["x"]
 
    def test_bpe_encode_word_full_merge(self):
        """All chars merged into one token."""
        merge_ranks = {("a", "b"): 0, ("ab", "c"): 1}
        result = bpe_encode_word(["a", "b", "c"], merge_ranks)
        assert result == ["abc"]

#Tokenizer test
class TestBPETokenizer:
 
    def test_encode_returns_list_of_ints(self, tiny_tok):
        ids = tiny_tok.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
 
    def test_encode_non_empty(self, tiny_tok):
        ids = tiny_tok.encode("test")
        assert len(ids) > 0
 
    def test_encode_empty_string(self, tiny_tok):
        ids = tiny_tok.encode("")
        assert ids == []
 
    def test_decode_returns_string(self, tiny_tok):
        ids = tiny_tok.encode("hello")
        text = tiny_tok.decode(ids)
        assert isinstance(text, str)
 
    def test_roundtrip_ascii(self, tiny_tok):
        """decode(encode(text)) == text for plain ASCII."""
        for text in ["hello", "Hello, world!", "test 123", "a b c"]:
            ids = tiny_tok.encode(text)
            recovered = tiny_tok.decode(ids)
            assert recovered == text, f"Round-trip failed for {text!r}: got {recovered!r}"
 
    def test_add_bos(self, tiny_vocab):
        tok = BPETokenizer(tiny_vocab, add_bos=True, add_eos=False)
        ids = tok.encode("hi")
        assert ids[0] == tiny_vocab.BOS_ID
 
    def test_add_eos(self, tiny_vocab):
        tok = BPETokenizer(tiny_vocab, add_bos=False, add_eos=True)
        ids = tok.encode("hi")
        assert ids[-1] == tiny_vocab.EOS_ID
 
    def test_bos_and_eos(self, tiny_vocab):
        tok = BPETokenizer(tiny_vocab, add_bos=True, add_eos=True)
        ids = tok.encode("hi")
        assert ids[0] == tiny_vocab.BOS_ID
        assert ids[-1] == tiny_vocab.EOS_ID
 
    def test_max_length_truncation(self, tiny_tok):
        ids = tiny_tok.encode("hello world this is a long sentence", max_length=5)
        assert len(ids) <= 5
 
    def test_skip_special_tokens_in_decode(self, tiny_vocab):
        tok = BPETokenizer(tiny_vocab, add_bos=True, add_eos=True)
        ids = tok.encode("hi")
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "<s>" not in decoded
        assert "</s>" not in decoded
 
    def test_encode_batch_same_length(self, tiny_tok):
        texts = ["hi", "hello world", "a"]
        padded, lengths = tiny_tok.encode_batch(texts, pad=True)
        target = max(lengths)
        for row in padded:
            assert len(row) == target
 
    def test_encode_batch_lengths_correct(self, tiny_tok):
        texts = ["hi", "hello world", "a"]
        _, lengths = tiny_tok.encode_batch(texts, pad=False)
        assert len(lengths) == 3
        assert lengths[0] < lengths[1]  # "hi" is shorter than "hello world"
 
    def test_decode_batch(self, tiny_tok):
        texts = ["hello", "world"]
        padded, _ = tiny_tok.encode_batch(texts, pad=False)
        recovered = tiny_tok.decode_batch(padded)
        assert recovered == texts
 
    def test_word_cache_consistency(self, tiny_tok):
        """Same word encoded twice should give same result (cache correctness)."""
        ids1 = tiny_tok.encode("hello")
        tiny_tok._word_cache.clear()
        ids2 = tiny_tok.encode("hello")
        assert ids1 == ids2
 
    def test_count_tokens(self, tiny_tok):
        text = "hello"
        assert tiny_tok.count_tokens(text) == len(tiny_tok.encode(text))
 
    def test_pretokenizer_regex(self):
        """GPT2 regex should handle contractions."""
        matches = GPT2_SPLIT_PATTERN.findall("don't stop")
        assert "don" in matches
        assert "'t" in matches

