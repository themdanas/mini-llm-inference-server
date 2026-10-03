"""
memory/page_table.py

In OS a virtual memory page table maps
 virtual addresses to physical addresses
Here it maps logical sequences postions 
to physical block TDs in KV cache VRAM Slab

"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
from memory.block import PhysicalBlock, BlockList
from memory.block_allocator import BlockAllocator

class BlockTable:

    # Per squence logical-to-physical mapping

    def __init__(
            self,
            seq_id: int,
            block_size: int,
            allocator: BlockAllocator,
    ):
        self.seq_id = seq_id
        self.block_size = block_size
        self.allocator = allocator
        self._block_list = BlockList()
        self._num_tokens = 0

    def allocate_for_prompt(self, prompt_len: int) -> None:
        # allocate all blocks needed for the prompt of prompt_len tokens

        if self._num_tokens > 0:
            raise RuntimeError(
                f"Seq {self.seq_id}: allocate_for_prompt() called "
                f"but {self._num_tokens} tokens already allocated."
            )

        n_blocks = self.allocator.blocks_needed_for(prompt_len)
        new_blocks = self.allocator.allocate(n_blocks)
        for block in new_blocks:
            self._block_list.append(block)

        self._num_tokens = prompt_len

        remaining = prompt_len
        for block in self._block_list:
            block.num_filled = min(self.block_size, remaining)
            remaining -= block.num_filled

    def allocate_next_token(self) -> Optional[PhysicalBlock]:

        # Ensure the space for one more token (one decode step)

        last = self._block_list.last_block

        # case 1 : last block has room
        if last is not None and last.num_filled < self.block_size:
            last.num_filled += 1
            self._num_tokens += 1
            return last

        # case 2 : need a new block
        if not self.allocator.can_allocate(1):
            return None # OOM -- caller must handle

        [new_block] = self.allocator.allocate(1)
        new_block.num_filled = 1
        self._block_list.append(new_block)
        self._num_tokens += 1
        return new_block

    # logical to physical translation

    def logical_to_physical(self, token_pos: int) -> Tuple[int, int]:

        # convert the logical token position to (block_id, slot_within_block)

        if token_pos >= self.num_tokens:
            raise IndexError(
                f"Seq {self.seq_id}: token_pos={token_pos} out of range "
                f"(num_tokens={self._num_tokens})"
            )

        logical_block_idx = token_pos // self.block_size
        slot_within_block = token_pos % self.block_size

        block = self._block_list[logical_block_idx]
        return block.block_id, slot_within_block

    def get_block_ids(self) -> List[int]:

        #return all physical block IDs in logical order

        return self._block_list.block_ids

    def get_slot_for_new_tokens(self) -> Tuple[int, int]:

        # return (block_id, slot) where the next token should be written

        return self.logical_to_physical(self._num_tokens - 1)

    def get_slot_for_new_token(self) -> Tuple[int, int]:
        """Alias for get_slot_for_new_tokens for backward compatibility."""
        return self.get_slot_for_new_tokens()

    def ensure_cow_for_last_block(self) -> None:

        last = self._block_list.last_block
        if last is None or not last.is_shared:
            return

        new_block, did_copy = self.allocator.copy_on_write(last)
        if did_copy:
            self._block_list._blocks[-1] = new_block

    def fork(self) -> "BlockTable":

        # create a child block that shares all the current block

        child = BlockTable(
            seq_id=self.seq_id * 1000,
            block_size=self.block_size,
            allocator=self.allocator,
        )

        for block in self._block_list:
            shared = self.allocator.fork(block)
            child._block_list.append(shared)
        child._num_tokens = self._num_tokens
        return child

    def free_all(self) -> None:

        # release all the blocks to the block allocator
        
        self.allocator.free_list_of_blocks(list(self._block_list))
        self._block_list = BlockList()
        self._num_tokens = 0

    # Introspection

    @property
    def num_tokens(self) -> int:
        # total number of tokens stored (prompt + generated)

        return self._num_tokens

    @property
    def num_blocks(self) -> int:
        return len(self._block_list)

    @property
    def num_free_slots_in_last_block(self) -> int:
        """How many more tokens fit in the current last block."""
        last = self._block_list.last_block
        if last is None:
            return 0
        return self.block_size - last.num_filled

    def __repr__(self) -> str:
        return (
            f"BlockTable(seq={self.seq_id}, "
            f"tokens={self._num_tokens}, "
            f"blocks={self._block_list.block_ids})"
        )  
         