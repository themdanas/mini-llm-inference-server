"""
memory/kv_cache.py
──────────────────
PagedKVCache: the VRAM tensor slab and scatter/gather operations.
"""

from __future__ import annotations
import torch
from typing import List, Optional, Tuple

from memory.block_allocator import BlockAllocator
from memory.page_table import BlockTable


class SingleSeqPagedView:
    """
    Per-layer view into PagedKVCache for a single sequence (Prefill or B=1 Decode).
    Duck-types KVCache (.seq_len and .update(k_new, v_new)) so MultiHeadAttention
    writes directly into the physical PagedKVCache slab.
    """

    def __init__(
        self,
        paged_cache: "PagedKVCache",
        block_table: BlockTable,
        layer_idx: int,
        start_pos: int,
    ) -> None:
        self.paged_cache = paged_cache
        self.block_table = block_table
        self.layer_idx = layer_idx
        self._seq_len = start_pos

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # k_new, v_new: (1, n_kv_heads, T_new, d_head)
        T_new = k_new.shape[2]
        k_seq = k_new.squeeze(0)  # (n_kv_heads, T_new, d_head)
        v_seq = v_new.squeeze(0)

        for offset in range(T_new):
            pos = self._seq_len + offset
            block_id, slot = self.block_table.logical_to_physical(pos)
            self.paged_cache.write(
                block_id,
                slot,
                self.layer_idx,
                k_seq[:, offset, :],
                v_seq[:, offset, :],
            )

        total_tokens = self._seq_len + T_new
        self._seq_len = total_tokens

        k_out, v_out = self.paged_cache.gather(
            block_ids=self.block_table.get_block_ids(),
            num_tokens=total_tokens,
            layer_idx=self.layer_idx,
        )
        return k_out.to(dtype=k_new.dtype, device=k_new.device), v_out.to(dtype=v_new.dtype, device=v_new.device)


class BatchedDecodePagedView:
    """
    Per-layer view into PagedKVCache for a batch of B decode sequences (T=1).
    Writes each sequence's new KV token into its assigned physical block slot
    and gathers a padded (B, n_kv_heads, max_S, d_head) tensor for batched attention.
    """

    def __init__(
        self,
        paged_cache: "PagedKVCache",
        block_tables: List[BlockTable],
        write_positions: List[int],
        layer_idx: int,
    ) -> None:
        self.paged_cache = paged_cache
        self.block_tables = block_tables
        self.write_positions = write_positions
        self.layer_idx = layer_idx
        self._seq_len = max(write_positions) if write_positions else 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # k_new, v_new: (B, n_kv_heads, 1, d_head)
        B = k_new.shape[0]
        seq_lens = [pos + 1 for pos in self.write_positions]
        max_s = max(seq_lens)

        k_batch = torch.zeros(
            (B, self.paged_cache.n_kv_heads, max_s, self.paged_cache.d_head),
            dtype=k_new.dtype,
            device=k_new.device,
        )
        v_batch = torch.zeros_like(k_batch)

        for i in range(B):
            bt = self.block_tables[i]
            pos = self.write_positions[i]
            s_len = seq_lens[i]

            block_id, slot = bt.logical_to_physical(pos)
            self.paged_cache.write(
                block_id,
                slot,
                self.layer_idx,
                k_new[i, :, 0, :],
                v_new[i, :, 0, :],
            )

            block_ids = bt.get_block_ids()
            k_seq = (
                self.paged_cache.k_slab[self.layer_idx, block_ids]
                .reshape(-1, self.paged_cache.n_kv_heads, self.paged_cache.d_head)[:s_len]
                .permute(1, 0, 2)
            )
            v_seq = (
                self.paged_cache.v_slab[self.layer_idx, block_ids]
                .reshape(-1, self.paged_cache.n_kv_heads, self.paged_cache.d_head)[:s_len]
                .permute(1, 0, 2)
            )
            k_batch[i, :, :s_len, :] = k_seq.to(dtype=k_new.dtype, device=k_new.device)
            v_batch[i, :, :s_len, :] = v_seq.to(dtype=v_new.dtype, device=v_new.device)

        return k_batch, v_batch


class PagedKVCache:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        d_head: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # Allocate the VRAM slab
        slab_shape = (n_layers, num_blocks, block_size, n_kv_heads, d_head)
        self.k_slab = torch.zeros(slab_shape, dtype=dtype, device=self.device)
        self.v_slab = torch.zeros(slab_shape, dtype=dtype, device=self.device)

        # Create the block allocator
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            cow_callback=self._copy_block,
        )

        bytes_per_element = torch.finfo(dtype).bits // 8
        vram_bytes = (
            2
            * slab_shape[0]
            * slab_shape[1]
            * slab_shape[2]
            * slab_shape[3]
            * slab_shape[4]
            * bytes_per_element
        )
        print(
            f"[KVCache] Allocated {vram_bytes / 1e6:.1f} MB "
            f"({num_blocks} blocks x {block_size} tokens, "
            f"{n_layers} layers, {dtype})"
        )

    def create_block_table(self, seq_id: int) -> BlockTable:
        return BlockTable(
            seq_id=seq_id,
            block_size=self.block_size,
            allocator=self.allocator,
        )

    def make_sequence_views(
        self,
        block_table: BlockTable,
        start_pos: int,
    ) -> List[SingleSeqPagedView]:
        return [
            SingleSeqPagedView(self, block_table, layer_idx=i, start_pos=start_pos)
            for i in range(self.n_layers)
        ]

    def make_batched_decode_views(
        self,
        block_tables: List[BlockTable],
        write_positions: List[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[BatchedDecodePagedView], torch.Tensor]:
        """
        Build per-layer batched decode KV views and an attention mask of shape (B, 1, 1, max_S)
        that masks out padding positions beyond each sequence's valid length.
        """
        target_device = device if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype

        views = [
            BatchedDecodePagedView(self, block_tables, write_positions, layer_idx=i)
            for i in range(self.n_layers)
        ]
        seq_lens = [pos + 1 for pos in write_positions]
        max_s = max(seq_lens) if seq_lens else 0
        B = len(seq_lens)

        # Build additive attention mask: 0.0 for valid positions, -inf for padded positions
        mask = torch.zeros((B, 1, 1, max_s), dtype=target_dtype, device=target_device)
        for i, s_len in enumerate(seq_lens):
            if s_len < max_s:
                mask[i, 0, 0, s_len:] = float("-inf")

        return views, mask

    def write(
        self,
        block_id: int,
        slot: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self.k_slab[layer_idx, block_id, slot] = k.to(self.dtype)
        self.v_slab[layer_idx, block_id, slot] = v.to(self.dtype)

    def write_prompt(
        self,
        block_table: BlockTable,
        layer_idx: int,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
    ) -> None:
        prompt_len = k_all.shape[2]
        k_all = k_all.squeeze(0)
        v_all = v_all.squeeze(0)

        for pos in range(prompt_len):
            block_id, slot = block_table.logical_to_physical(pos)
            self.write(
                block_id,
                slot,
                layer_idx,
                k_all[:, pos, :],
                v_all[:, pos, :],
            )

    def gather(
        self,
        block_ids: List[int],
        num_tokens: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not block_ids or num_tokens == 0:
            shape = (1, self.n_kv_heads, 0, self.d_head)
            empty = torch.zeros(shape, dtype=self.dtype, device=self.device)
            return empty, empty

        k_full = self.k_slab[layer_idx, block_ids].reshape(-1, self.n_kv_heads, self.d_head)[:num_tokens]
        v_full = self.v_slab[layer_idx, block_ids].reshape(-1, self.n_kv_heads, self.d_head)[:num_tokens]

        k_out = k_full.permute(1, 0, 2).unsqueeze(0)
        v_out = v_full.permute(1, 0, 2).unsqueeze(0)

        return k_out, v_out

    def gather_for_sequence(
        self,
        block_table: BlockTable,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gather(
            block_ids=block_table.get_block_ids(),
            num_tokens=block_table.num_tokens,
            layer_idx=layer_idx,
        )

    def _copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        self.k_slab[:, dst_block_id, :, :, :] = self.k_slab[:, src_block_id, :, :, :].clone()
        self.v_slab[:, dst_block_id, :, :, :] = self.v_slab[:, src_block_id, :, :, :].clone()

    @property
    def free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    @property
    def used_blocks(self) -> int:
        return self.allocator.num_used_blocks

    @property
    def utilization(self) -> float:
        return self.allocator.utilization

    def vram_bytes(self) -> int:
        return (self.k_slab.nelement() + self.v_slab.nelement()) * (
            torch.finfo(self.dtype).bits // 8
        )

    def __repr__(self) -> str:
        return (
            f"PagedKVCache("
            f"blocks={self.num_blocks}, "
            f"block_size={self.block_size}, "
            f"layers={self.n_layers}, "
            f"free={self.free_blocks}, "
            f"{self.vram_bytes()/1e6:.1f}MB)"
        )