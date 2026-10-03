"""
memory/kv_cache.py
──────────────────
PagedKVCache: the VRAM tensor slab and scatter/gather operations.
 
This is where the physics meets the math.
The BlockAllocator and BlockTable deal in metadata (block IDs, slot numbers).
The PagedKVCache deals in the actual GPU tensors.
"""


from __future__ import annotations
import torch
from typing import Dict, List, Optional, Tuple

from memory.block import PhysicalBlock
from memory.block_allocator import BlockAllocator
from memory.page_table import BlockTable

class PagedKVCache:

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 n_layers:   int,
                 n_kv_heads: int,
                 d_head:     int,
                 dtype:     torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 ):

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_layers   = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head     = d_head
        self.dtype      = dtype
        self.device     = device or torch.device("cpu")

    # Allocate the VRAM slab
        slab_shape = (n_layers, num_blocks, block_size, n_kv_heads, d_head)
        self.k_slab = torch.zeros(slab_shape, dtype=dtype, device=self.device)
        self.v_slab = torch.zeros(slab_shape, dtype=dtype, device=self.device)


        # creating the block allocator
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            cow_callback=self._copy_block,
        )

        bytes_per_element = torch.finfo(dtype).bits // 8
        vram_bytes = 2 * slab_shape[0] * slab_shape[1] * slab_shape[2] * \
                     slab_shape[3] * slab_shape[4] * bytes_per_element
        print(f"[KVCache] Allocated {vram_bytes / 1e6:.1f} MB "
              f"({num_blocks} blocks x {block_size} tokens, "
              f"{n_layers} layers, {dtype})")


    #Block Table management 
    def create_block_table(self, seq_id: int) -> BlockTable:
        """
        create a new BlockTable for a sequence
        Called by the scheduler when a request is admitted
        """

        return BlockTable(
            seq_id=seq_id,
            block_size=self.block_size,
            allocator=self.allocator
        )

    # write operations
    def write(
            self,
            block_id : int,
            slot: int,
            layer_idx: int,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> None:

        self.k_slab[layer_idx, block_id, slot] = k.to(self.dtype)
        self.v_slab[layer_idx, block_id, slot] = v.to(self.dtype)


    def write_prompt(
            self,
            block_table : BlockTable,
            layer_idx: int,
            k_all: torch.Tensor,
            v_all: torch.Tensor,
    ) -> None:
        """
        writing k v tensors for all prompt tokens at once (prefill)

        k_all:        (1, n_kv_heads, prompt_len, d_head) all prompt keys
        v_all:        (1, n_kv_heads, prompt_len, d_head) all prompt values

        """
        prompt_len = k_all.shape[2]
        k_all = k_all.squeeze(0) #(nv_kv_heads, prompt_len, d_head)
        v_all = v_all.squeeze(0)

        for pos in range(prompt_len):
            block_id, slot = block_table.logical_to_physical(pos)

            self.write(block_id, slot, layer_idx,
                       k_all[:, pos, :],
                       v_all[:, pos, :])

    # gather operations
    def gather(
            self,
            block_ids: List[int],
            num_tokens: int,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        reconstruct full k,v tensors from scatterd blocks 
        
        Returns :
            k: (1, n_kv_heads, num_tokens, d_head)
            v: (1, n_kv_heads, num_tokens, d_head)
 
        """
        if not block_ids:
            shape = (1, self.n_kv_heads, 0, self.d_head)
            empty = torch.zeros(shape, dtype=self.dtype, device=self.device)

            return empty, empty

        k_blocks = [
            self.k_slab[layer_idx, bid]

            for bid in block_ids
        
        ]

        v_blocks = [
            self.v_slab[layer_idx, bid]

            for bid in block_ids
        ]

        #concatenate along the dimensions
        k_full = torch.cat(k_blocks, dim=0)
        v_full = torch.cat(v_blocks, dim=0)

        k_full = k_full[:num_tokens]
        v_full = v_full[:num_tokens]

        k_out = k_full.permute(1,0,2).unsqueeze(0)
        v_out = v_full.permute(1,0,2).unsqueeze(0)

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
        # cpoy all the data from src_block to dst_block across all layers
         
        self.k_slab[:, dst_block_id, :, :, :] = \
            self.k_slab[:,src_block_id, :, :, :].clone()
        self.v_slab[:, dst_block_id, :, :, :] = \
            self.v_slab[:, src_block_id, :,:,:].clone()

    # Memory Statistics
    def free_blocks(self) -> int:
        return self.allocator.num_free_blocks
 
    @property
    def used_blocks(self) -> int:
        return self.allocator.num_used_blocks
 
    @property
    def utilization(self) -> float:
        return self.allocator.utilization
 
    def vram_bytes(self) -> int:
        """Total VRAM consumed by the KV cache slab."""
        return (self.k_slab.nelement() + self.v_slab.nelement()) * \
               (torch.finfo(self.dtype).bits // 8)
 
    def __repr__(self) -> str:
        return (
            f"PagedKVCache("
            f"blocks={self.num_blocks}, "
            f"block_size={self.block_size}, "
            f"layers={self.n_layers}, "
            f"free={self.free_blocks}, "
            f"{self.vram_bytes()/1e6:.1f}MB)"
        )
    

