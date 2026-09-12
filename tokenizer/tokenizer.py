from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Literal, Tuple, Optional
from tokenizer.vocabulary import Vocabulary, BYTE_TO_UNICODE


GPT2_SPLIT_PATTERN = re.compile(
    r"""'s| 't|'re|'ve|'m|'ll|'d| ?\w+| ?\d+|[^\s\w\d]+""",
    re.UNICODE,
)

#BPE CORE FUNCTION
def get_pairs(symbols: List[str]) -> List[Tuple[str, str]]:
    """
    Return all abjcent symbols pair in list
    """

    return [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]

def apply_merges(symbols: List[str], pair: Tuple[str, str]) -> List[str]:

    """
    Merge all non-overlaping occurances of pair in symbols
    """
    merged: List[str] = []
    i=0

    while i < len(symbols):
        if i<len(symbols) and (symbols[i],symbols[i+1]) == pair:
            merged.append(symbols[i], symbols[i+1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1

    return merged

def bpe_encode_word(words_bytes: List[str], merge_ranks: Dict[Tuple[str, str], int]) -> List[str]:

    """
    Apply BPE merge rules to a single word
    """

    symbols = list(words_bytes)

    if len(symbols) <= 1:
        return symbols

    while True:
        pairs = get_pairs(symbols)
        if not pairs:
            break

        best_pair = min(pairs, key=lambda p:merge_ranks.get(p, float("inf")))

        if best_pair not in merge_ranks:
            break

        symbols = apply_merges(symbols, best_pair)


    return symbols


class BPETokenizer:

    def __init__(self, vocab: Vocabulary, add_bos: bool = False, add_eos: bool = False):

        self.vocab = vocab,
        self.add_bos = add_bos,
        self.add_eos = add_eos,
        self._word_cache: Dict[str, List[int]] = {}

    def encode(
            self,
            text: str,
            add_bos: Optional[bool] = None,
            add_eos: Optional[bool] = None,
            max_len: Optional[int] = None,
            truncated: bool = True
    ) -> List[int]:

        _add_bos = add_bos if add_bos is not None else self.add_bos
        _add_eos = add_eos if add_eos is not None else self.add_eos

        #pre_tokenize: split the text into words
        words =  GPT2_SPLIT_PATTERN.findall(text)

        token_ids: List[int] = []
        for word in words:
            token_ids.extend(self._encode_word(word))

        #add speacial tokens
        if _add_bos:
            token_ids = [self.vocab.BOS_ID] + token_ids
        if _add_eos:
            token_ids =  token_ids + [self.vocab.EOS_ID]

        #tuncated
        if max_len is not None and truncated and len(token_ids) > max_len:
            token_ids = token_ids[:max_len]

        return token_ids

    def _encode_words(self, word: str) -> List[int]:

        """
        Encode a single pre-tokenized word → list of token IDs.
        Results are cached per unique word string.
        """

        if word in self._word_cache:
            return self._word_cache[word]
        
        #byte level encoding 
        byte_chars = self.vocab.text_to_bytes(word)

        #bpe merge
        bpe_tokens = bpe_encode_word(byte_chars, self.vocab.merge_ranks)

        #vocab lookups
        ids = [self.vocab.token_to_id_safe(tok) for tok in bpe_tokens]

        self._word_cache[word] = ids
        return ids

    #core decode

    def decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = True,
    ) -> str:

        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if not self.vocab.is_special(tid)]

        #get token strings
        token_strings = [self.vocab.id_to_token_safe(tid) for tid in token_ids]

        #concatenate all byte-level chars and decode to UTF-8
        all_byte_chars = list("".join(token_strings))

        #decode byte chars back to UTF-8
        try:
            byte_values = bytes([
                b for c in all_byte_chars
                for b in [self._safe_byte(c)]
                if b is not None    
            ])
            return byte_values.decode("utf-8", errors="replace")

        except Exception:
            return "".join(token_strings)

    def _safe_byte(self, char:str):
        #looks up the byte level character, return none for speacial token chars 
        from tokenizer.vocabulary import UNICODE_TO_BYTE
        return UNICODE_TO_BYTE.get(char, None)

    def encode_batch(
            self,
            texts: List[str],
            pad: bool = True,
            max_len: Optional[int] = None,
            add_bos: Optional[bool] = None,
            add_eos: Optional[bool] = None,
    ) -> Tuple[List[List[int]], List[int]]:
        # Encode a batch of text with optional padding
        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, max_len=max_len, truncated=True)
            for text in texts
        ]
        lengths = [len(ids) for ids in encoded]

        if not pad:
            return encoded, lengths

        target_len = max_len if max_len is not None else max(lengths, default=0)
        padded = [
            ids + [self.vocab.PAD_ID] * (target_len - len(ids))
            for ids in encoded
        ]

        return padded, lengths

    def decode_batch(
            self,
            token_ids_batch: List[List[int]],
            skip_special_token: bool = True,
    ) -> List[str]:
        # Decode the batch token ID back to strings 
        return [
            self.decode(ids, skip_special_tokens=skip_special_token)
            for ids in token_ids_batch
        ]

    #Token counting 
    def count_tokens(self, text:str) -> int:
        return len(self.encode(text))

    @classmethod
    def from_huggingface(
        cls,
        model_name_or_path: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> "BPETokenizer":

        vocab = Vocabulary.from_hugginface(model_name_or_path)
        return cls(vocab, add_bos=add_bos, add_eos=add_eos)

    @classmethod
    def tiny_test(cls) -> "BPETokenizer":
        vocab = Vocabulary.tiny_synthetic(vocab_size=256)
        return cls(vocab, add_bos=False, add_eos=False)

    def __repr__(self) -> str:
        return (f"BPETokenizer("
                f"vocab_size={self.vocab.vocab_size}, "
                f"n_merges={len(self.vocab.merges)}, "
                f"add_bos={self.add_bos}, add_eos={self.add_eos})")