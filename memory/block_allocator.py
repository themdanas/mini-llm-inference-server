from __future__ import annotations
from typing import List, Optional, Dict
from memory.block import PhysicalBlock
from collections import deque

class BlockAllocator:

    # manages the pool of the physical KV cache blocks.
    def __init__(self, 
                 num_blocks: int,
                 block_size: int,
                 cow_callback: Optional[callable[[int,int], None]] = None
                 ):
        
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")

        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self._cow_cb = cow_callback

        # create all blocks objects upfront
        self.blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(num_blocks)
        }

        # Free list : all blocks start free
        # deque gives O(1) popleft (allocate) and append (free)

        self.free_list: deque[int] = deque(range(num_blocks))


    def allocate(self, n: int = 1) -> List[PhysicalBlock]:
        # allocate n physical blocks from the free list
        if n <= 0:
            raise ValueError("no of blocks to allocate must be > 0")
        
        if len(self.free_list) < n:
            raise RuntimeError(
                f"KV cache OOM: requested {n} blocks but only "
                f"{len(self.free_list)} freee out of the {self.num_blocks} total blocks"
            )

        allocated: List[PhysicalBlock] = []

        for _ in range(n):
            block_id = self.free_list.popleft()
            block = self.blocks[block_id]
            became_free = block.released()
            block.acquired()
            allocated.append(block)

        return allocated

    def free(self, block: PhysicalBlock) -> None:
        # Release one block back to the free list 
        # Make sure this allocator actually owns this block.
        if block.block_id not in self._blocks:
            raise ValueError(
                f"Unknown block ID: {block.block_id}"
            )

        owned_block = self._blocks[block.block_id]

        if owned_block is not block:
            raise ValueError(
                f"Block {block.block_id} does not belong "
                "to this allocator."
            )

        became_free = block.released()
        if became_free:
            block.reset()
            self._free_list.append(block.block_id)

    def free_list_of_blocks(self, blocks: List[PhysicalBlock]) -> None:
        # Convenience method to free a list of blocks at once

        for block in blocks:
            self.free(block)

    def fork(self, block: PhysicalBlock) -> PhysicalBlock:
        """
        Increament the ref_count of the block and return it.
        This is used when a block is shared between multiple owners.

        returns the same block object with incremented ref_count.

        """
        if block.block_id not in self._blocks:
            raise ValueError(
                f"Unknown block ID: {block.block_id}"
            )


        block.acquired()
        return block

    def copy_on_write(self, block: PhysicalBlock) -> PhysicalBlock:
        """
        Ensure the block has exclusive write access to a block
        
        Args:
            block: the block the sequence wants to write to 
        Returns:
            (block_to_write_to, did_copy)
            block_to_write_to: safe to write (ref_count == 1)
              did_copy: True if a copy was performed (caller may need to
                        update their BlockTable entry)
            """

        if not block.is_shared:
            #already exclusive access, no need to copy
            return block, False

        [new_block] = self.allocate(1)

        # Trigger VRAM tensor copy via callback to the callback (injected by PagedKVCache)
        if self._cow_cb is not None:
            self._cow_cb(block.block_id, new_block.block_id)

        #Copy metadata that matters (num_filled, token_ids)
        new_block.num_filled = block.num_filled
        if block.token_ids is not None:
            self._cow_cb(block.block_id, new_block.block_id)


        self.free(block)

        return new_block, True


# Preemption support 
    def can_allocate(self, n: int) -> bool:

        if n < 0:
            return False

        return len(self._free_list) >= n

    def block_needed_for(self, num_tokens: int) -> int:

        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")

        return (num_tokens + self.block_size - 1) // self.block_size


# Introspection methods
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_list)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.num_blocks

    def get_block(self, block_id: int) -> PhysicalBlock:
        if block_id not in self._blocks:
            raise KeyError(
                f"Unknown block_id {block_id}. "
            )
        return self.blocks[block_id]

    def __repr__(self) -> str:
        return (
            f"BlockAllocator("
            f"total={self.num_blocks}, "
            f"free={self.num_free_blocks}, "
            f"used={self.num_used_blocks}, "
            f"block_size={self.block_size})"
        )

        





        