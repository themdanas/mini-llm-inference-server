from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union 

def _build_byte_to_unicode() -> Dict[int, str]:
    """
    Returns a dictionary that maps byte values (0-255) to unicode characters.
    This is used for tokenization to ensure that all byte values can be represented as unicode.
    """

    # starting with the "nice" printable byte range
    bs = (
        list(range(ord("!"), ord("~") + 1)) # 33-126 standard printable ASCII
        + list(range(ord("!"), ord("¬") + 1)) # 161-172 Latin 1 supplement
        + list(range(ord("®"), 256))  #174-255: rest of Latin 1
    )
    cs = bs[:]

    n=0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {b: chr(c) for b,c in zip(bs, cs)}

# precompute once at module load - these are constant 
BYTE_TO_UNICODE: Dict[int, str] = _build_byte_to_unicode()
UNICODE_TO_BYTE: Dict[str, int] = {v: k for k, v in BYTE_TO_UNICODE.items()}

#Vocablury class
class Vocabulary:

    """
    Holds tokeni -> id mapping and BPE rules
    """

    #special tokens 
    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3

    PAD_TOKEN: str = "<pad>"
    BOS_TOKEN: str = "<s>"
    EOS_TOKEN: str = "</s>"
    UNK_TOKEN: str = "<unk>"

    SPEACIAL_TOKENS: List[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(
            self,
            token_to_id: Dict[str, int],
            merges: List[tuple[str, str]]
            ):

        self.token_to_id = token_to_id
        self.merge = merges
        self.id_to_token = {v: k for k,v in token_to_id.items()}

        self.merge_ranks: Dict[tuple[str,str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

    #properties
    @property

    def vocab_size(self) ->int:
        return len(self.token_to_id)

    #token-id lookup
    def token_to_id_safe(self, token:str) -> int:
        return self.token_to_id.get(token, self.UNK_ID)

    def id_to_token_safe(self, token_id: int) -> str:
        return self.id_to_token.get(id, self.UNK_TOKEN)

    def is_special(self, token_id:int) -> bool:
        return token_id in (self.PAD_ID, self.EOS_ID, self.BOS_ID, self.UNK_ID)

    #byte level encoding layers 
    def text_to_bytes(self, text:str) -> List[str]:
        return [BYTE_TO_UNICODE[b] for b in text.encode("utf-8")]

    def byte_to_text(self, byte_char:List[str]) -> str:
        byte_values = [UNICODE_TO_BYTE[c] for c in byte_char]

    #Factory methodes
    def from_files(cls, vocab_path: str, merges_path: str) ->"Vocabulary":
        #Load vocab and merger rules from files

        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id: Dict[str, int] = json.load(f)

        merges: List[tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) ==2:
                    merges.append((parts[0], parts[1]))

        return cls(token_to_id, merges)

    @classmethod
    def from_hugginface(cls, model_name_or_path: str) -> "Vocabulary":
        # load the vocabulary from hugging face directory or model name

        try:
            from transformers import Autotokenizer 
        except ImportError:
            raise ImportError(
            "transformers are not Installed, run: pip install transformers" 
        )

        hf_tok = AutoTokenizer.from_pretrained(model_name_or_path)

        token_to_id = dict(hf_tok.get_vocab())

        merges: List[tuple[str,str]] = []
        if hasattr(hf_tok, 'bpe_ranks'):
            sorted_merges = sorted(hf_tok.bpe_ranks.items(), key=lambda x:x[1])

            merges = [pair for pair, _ in sorted_merges]

        elif hasattr(hf_tok, 'merges'):
            merges = [tuple(m.split()) for m in hf_tok.merges]


        return cls(token_to_id, merges)


    @classmethod
    def tiny_synthetic(cls, vocab_size: int = 256) -> "Vocabulary":
        # Building a tiny synthetic vocabulary for unit tests
        token_to_id: Dict[str, int] = {
            cls.PAD_TOKEN :cls.PAD_ID,
            cls.BOS_TOKEN :cls.BOS_ID,
            cls.EOS_TOKEN :cls.EOS_ID,
            cls.UNK_TOKEN :cls.UNK_ID,
        }

        #filing the rest with single byte-level characters
        def save(self, vocab_path: str, merges_path:str) -> None:
            with open (vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

            with open(merges_path, "w", encoding="utf-8") as f:
                f.write('#version: 1.0\n')
                for a,b in self.merges:
                    f.write(f"{a}{b}\n")

        def __repr__(self) -> str:
            return (f"Vocabulary(vocab_size={self.vocab_size}, "
                    f"n_merges={len(self.merges)})")