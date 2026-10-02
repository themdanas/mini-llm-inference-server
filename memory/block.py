from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional 

@dataclass
class PhysicalBlock:

    block_id : int
    ref_count : int = 0
    num_filled: int = 0
    token_ids: Optional[list[int]] = None

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    def acquired(self) -> None:
        self.ref_count += 1

    def released(self) -> None:
        if self.ref_count <= 0:
            raise RuntimeError(
                f"Block {self.block_id} released() called but ref_count is already 0"
            )
        self.ref_count -= 1
        return self.ref_count == 0
    
    def reset(self) -> None:
        self.ref_count = 0
        self.num_filled = 0
        self.token_ids = None

    def __repr__(self) -> str:
        status = "free" if self.is_free else "shared" if self.is_shared else f"ref={self.ref_count}"
        return f"PhysicalBlock(id={self.block_id}, status={status}, filled={self.num_filled})"


class BlockList:

    # Ordered collection of PhysicalBlocks to a sequence

    def __init__(self):
        self.blocks : List[PhysicalBlock] = []

    def append(self, block: PhysicalBlock) -> None:
        self.blocks.append(block)

    def __getitem__(self, index: int) -> PhysicalBlock:
        return self.blocks[index]

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    @property
    def block_ids(self) -> Optional[list[int]]:
        return [block.block_id for block in self.blocks]

    @property
    def last_block(self) -> Optional[PhysicalBlock]:
        return self.blocks[-1] if self.blocks else None

    def __repr__(self) -> str:
        return f"BlockList(blocks={self.blocks})"
    