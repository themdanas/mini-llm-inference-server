"""
tests/test_phase3.py
────────────────────
Tests for Phase 3: paged KV cache and memory management.

Run: pytest tests/test_phase3.py -v

Categories:
  1. PhysicalBlock        — ref counting, acquire/release, reset
  2. BlockList            — ordering and block_ids
  3. BlockAllocator       — free list, allocate, free, OOM, CoW
  4. BlockTable           — logical-to-physical mapping, prefill, decode
  5. PagedKVCache         — write, gather, equivalence with naive cache
  6. CRITICAL equivalence — paged gather == naive contiguous cache
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch

from memory.block import PhysicalBlock, BlockList
from memory.block_allocator import BlockAllocator
from memory.page_table import BlockTable
from memory.kv_cache import PagedKVCache


# ── Fixtures ──────────────────────────────────────────────────────────────────

BLOCK_SIZE  = 4
NUM_BLOCKS  = 16
N_LAYERS    = 2
N_KV_HEADS  = 2
D_HEAD      = 8

@pytest.fixture
def allocator():
    return BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)

@pytest.fixture
def kv_cache():
    return PagedKVCache(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
        n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, d_head=D_HEAD,
        dtype=torch.float32, device=torch.device("cpu"),
    )

@pytest.fixture
def block_table(allocator):
    return BlockTable(seq_id=0, block_size=BLOCK_SIZE, allocator=allocator)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PhysicalBlock
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysicalBlock:

    def test_initial_state(self):
        b = PhysicalBlock(block_id=5)
        assert b.block_id   == 5
        assert b.ref_count  == 0
        assert b.num_filled == 0
        assert b.is_free

    def test_acquire_increments_ref(self):
        b = PhysicalBlock(block_id=0)
        b.acquire()
        assert b.ref_count == 1
        assert not b.is_free

    def test_acquire_twice_shared(self):
        b = PhysicalBlock(block_id=0)
        b.acquire(); b.acquire()
        assert b.ref_count == 2
        assert b.is_shared

    def test_release_to_zero(self):
        b = PhysicalBlock(block_id=0)
        b.acquire()
        became_free = b.release()
        assert became_free
        assert b.is_free

    def test_release_shared(self):
        b = PhysicalBlock(block_id=0)
        b.acquire(); b.acquire()
        became_free = b.release()
        assert not became_free
        assert b.ref_count == 1

    def test_double_release_raises(self):
        b = PhysicalBlock(block_id=0)
        b.acquire()
        b.release()
        with pytest.raises(RuntimeError):
            b.release()

    def test_reset_clears_metadata(self):
        b = PhysicalBlock(block_id=0)
        b.acquire(); b.num_filled = 3; b.token_ids = [1,2,3]
        b.release(); b.reset()
        assert b.ref_count  == 0
        assert b.num_filled == 0
        assert b.token_ids  is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. BlockList
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockList:

    def test_empty(self):
        bl = BlockList()
        assert len(bl) == 0
        assert bl.last_block is None
        assert bl.block_ids == []

    def test_append_and_access(self):
        bl = BlockList()
        b0 = PhysicalBlock(0); b1 = PhysicalBlock(1)
        bl.append(b0); bl.append(b1)
        assert len(bl) == 2
        assert bl[0].block_id == 0
        assert bl[1].block_id == 1
        assert bl.last_block.block_id == 1

    def test_block_ids_order(self):
        bl = BlockList()
        for i in [5, 2, 9]:
            bl.append(PhysicalBlock(i))
        assert bl.block_ids == [5, 2, 9]


# ══════════════════════════════════════════════════════════════════════════════
# 3. BlockAllocator
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockAllocator:

    def test_initial_all_free(self, allocator):
        assert allocator.num_free_blocks == NUM_BLOCKS
        assert allocator.num_used_blocks == 0

    def test_allocate_one(self, allocator):
        blocks = allocator.allocate(1)
        assert len(blocks) == 1
        assert blocks[0].ref_count == 1
        assert allocator.num_free_blocks == NUM_BLOCKS - 1

    def test_allocate_many(self, allocator):
        blocks = allocator.allocate(5)
        assert len(blocks) == 5
        assert allocator.num_free_blocks == NUM_BLOCKS - 5

    def test_free_returns_to_pool(self, allocator):
        [block] = allocator.allocate(1)
        allocator.free(block)
        assert allocator.num_free_blocks == NUM_BLOCKS

    def test_oom_raises(self, allocator):
        allocator.allocate(NUM_BLOCKS)
        with pytest.raises(MemoryError):
            allocator.allocate(1)

    def test_can_allocate_true(self, allocator):
        assert allocator.can_allocate(NUM_BLOCKS)

    def test_can_allocate_false(self, allocator):
        allocator.allocate(NUM_BLOCKS)
        assert not allocator.can_allocate(1)

    def test_blocks_needed_exact(self, allocator):
        assert allocator.blocks_needed_for(BLOCK_SIZE)     == 1
        assert allocator.blocks_needed_for(BLOCK_SIZE + 1) == 2
        assert allocator.blocks_needed_for(1)              == 1

    def test_blocks_needed_zero(self, allocator):
        assert allocator.blocks_needed_for(0) == 0

    def test_fork_increments_ref(self, allocator):
        [block] = allocator.allocate(1)
        assert block.ref_count == 1
        allocator.fork(block)
        assert block.ref_count == 2

    def test_cow_exclusive_no_copy(self, allocator):
        [block] = allocator.allocate(1)
        result, did_copy = allocator.copy_on_write(block)
        assert not did_copy
        assert result.block_id == block.block_id

    def test_cow_shared_allocates_new(self, allocator):
        [block] = allocator.allocate(1)
        allocator.fork(block)              # ref_count -> 2
        new_block, did_copy = allocator.copy_on_write(block)
        assert did_copy
        assert new_block.block_id != block.block_id
        assert new_block.ref_count == 1
        assert block.ref_count == 1        # old block still has one owner

    def test_utilization(self, allocator):
        allocator.allocate(NUM_BLOCKS // 2)
        assert abs(allocator.utilization - 0.5) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 4. BlockTable
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockTable:

    def test_allocate_for_prompt_exact_block(self, block_table, allocator):
        block_table.allocate_for_prompt(BLOCK_SIZE)
        assert block_table.num_blocks == 1
        assert block_table.num_tokens == BLOCK_SIZE
        assert allocator.num_free_blocks == NUM_BLOCKS - 1

    def test_allocate_for_prompt_partial_block(self, block_table, allocator):
        block_table.allocate_for_prompt(BLOCK_SIZE - 1)
        assert block_table.num_blocks == 1
        assert block_table.num_tokens == BLOCK_SIZE - 1

    def test_allocate_for_prompt_multi_block(self, block_table, allocator):
        prompt_len = BLOCK_SIZE * 3 + 2
        block_table.allocate_for_prompt(prompt_len)
        assert block_table.num_blocks == 4
        assert block_table.num_tokens == prompt_len
        assert allocator.num_free_blocks == NUM_BLOCKS - 4

    def test_logical_to_physical_first_block(self, block_table):
        block_table.allocate_for_prompt(BLOCK_SIZE)
        block_id_0, slot_0 = block_table.logical_to_physical(0)
        block_id_3, slot_3 = block_table.logical_to_physical(BLOCK_SIZE - 1)
        assert slot_0 == 0
        assert slot_3 == BLOCK_SIZE - 1
        assert block_id_0 == block_id_3   # same block

    def test_logical_to_physical_second_block(self, block_table):
        block_table.allocate_for_prompt(BLOCK_SIZE * 2)
        bid0, s0 = block_table.logical_to_physical(0)
        bid1, s1 = block_table.logical_to_physical(BLOCK_SIZE)
        assert s0 == 0 and s1 == 0
        assert bid0 != bid1   # different blocks

    def test_logical_to_physical_out_of_range(self, block_table):
        block_table.allocate_for_prompt(BLOCK_SIZE)
        with pytest.raises(IndexError):
            block_table.logical_to_physical(BLOCK_SIZE)

    def test_allocate_next_token_same_block(self, block_table, allocator):
        block_table.allocate_for_prompt(BLOCK_SIZE - 1)
        n_before = allocator.num_used_blocks
        block = block_table.allocate_next_token()
        assert block is not None
        assert block_table.num_tokens == BLOCK_SIZE
        assert allocator.num_used_blocks == n_before   # no new block needed

    def test_allocate_next_token_new_block(self, block_table, allocator):
        block_table.allocate_for_prompt(BLOCK_SIZE)
        n_before = allocator.num_used_blocks
        block = block_table.allocate_next_token()
        assert block is not None
        assert block_table.num_blocks == 2
        assert allocator.num_used_blocks == n_before + 1

    def test_allocate_next_token_oom(self, allocator):
        allocator.allocate(NUM_BLOCKS - 1)   # leave only 1 block free
        bt = BlockTable(seq_id=1, block_size=BLOCK_SIZE, allocator=allocator)
        bt.allocate_for_prompt(BLOCK_SIZE)   # uses that last block
        result = bt.allocate_next_token()    # should fail
        assert result is None

    def test_free_all_releases_blocks(self, block_table, allocator):
        block_table.allocate_for_prompt(BLOCK_SIZE * 3)
        block_table.free_all()
        assert allocator.num_free_blocks == NUM_BLOCKS
        assert block_table.num_tokens == 0

    def test_fork_shares_blocks(self, block_table, allocator):
        block_table.allocate_for_prompt(BLOCK_SIZE * 2)
        child = block_table.fork()
        # All blocks should now have ref_count == 2
        for bid in block_table._block_list.block_ids:
            assert allocator.get_block(bid).ref_count == 2
        assert child.num_tokens == block_table.num_tokens

    def test_get_block_ids_order(self, block_table):
        block_table.allocate_for_prompt(BLOCK_SIZE * 3)
        ids = block_table.get_block_ids()
        assert len(ids) == 3
        # IDs should be unique
        assert len(set(ids)) == 3


# ══════════════════════════════════════════════════════════════════════════════
# 5. PagedKVCache
# ══════════════════════════════════════════════════════════════════════════════

class TestPagedKVCache:

    def test_slab_shape(self, kv_cache):
        expected = (N_LAYERS, NUM_BLOCKS, BLOCK_SIZE, N_KV_HEADS, D_HEAD)
        assert kv_cache.k_slab.shape == expected
        assert kv_cache.v_slab.shape == expected

    def test_initial_all_zeros(self, kv_cache):
        assert kv_cache.k_slab.sum().item() == 0
        assert kv_cache.v_slab.sum().item() == 0

    def test_write_and_read_single_token(self, kv_cache):
        torch.manual_seed(0)
        k = torch.randn(N_KV_HEADS, D_HEAD)
        v = torch.randn(N_KV_HEADS, D_HEAD)
        layer = 0; block_id = 3; slot = 2

        kv_cache.write(block_id, slot, layer, k, v)

        # Read back directly from slab
        assert torch.allclose(kv_cache.k_slab[layer, block_id, slot], k)
        assert torch.allclose(kv_cache.v_slab[layer, block_id, slot], v)

    def test_write_does_not_affect_other_slots(self, kv_cache):
        k = torch.ones(N_KV_HEADS, D_HEAD)
        v = torch.ones(N_KV_HEADS, D_HEAD)
        kv_cache.write(block_id=2, slot=1, layer_idx=0, k=k, v=v)
        # All other slots should still be zero
        assert kv_cache.k_slab[0, 2, 0].sum().item() == 0
        assert kv_cache.k_slab[0, 2, 2].sum().item() == 0
        assert kv_cache.k_slab[0, 3, 1].sum().item() == 0

    def test_gather_single_block(self, kv_cache):
        torch.manual_seed(1)
        # Write 3 tokens to block 0
        for slot in range(3):
            k = torch.randn(N_KV_HEADS, D_HEAD)
            v = torch.randn(N_KV_HEADS, D_HEAD)
            kv_cache.write(0, slot, 0, k, v)

        K, V = kv_cache.gather(block_ids=[0], num_tokens=3, layer_idx=0)
        assert K.shape == (1, N_KV_HEADS, 3, D_HEAD)
        assert V.shape == (1, N_KV_HEADS, 3, D_HEAD)

    def test_gather_multi_block(self, kv_cache):
        torch.manual_seed(2)
        # 2 full blocks = 2 * BLOCK_SIZE tokens
        num_tokens = BLOCK_SIZE * 2
        for slot in range(BLOCK_SIZE):
            kv_cache.write(0, slot, 0, torch.randn(N_KV_HEADS, D_HEAD), torch.randn(N_KV_HEADS, D_HEAD))
            kv_cache.write(1, slot, 0, torch.randn(N_KV_HEADS, D_HEAD), torch.randn(N_KV_HEADS, D_HEAD))

        K, V = kv_cache.gather(block_ids=[0, 1], num_tokens=num_tokens, layer_idx=0)
        assert K.shape == (1, N_KV_HEADS, num_tokens, D_HEAD)

    def test_gather_partial_last_block(self, kv_cache):
        # Write BLOCK_SIZE + 2 tokens across 2 blocks
        for slot in range(BLOCK_SIZE):
            kv_cache.write(0, slot, 0, torch.ones(N_KV_HEADS, D_HEAD), torch.ones(N_KV_HEADS, D_HEAD))
        for slot in range(2):
            kv_cache.write(1, slot, 0, torch.ones(N_KV_HEADS, D_HEAD) * 2, torch.ones(N_KV_HEADS, D_HEAD) * 2)

        K, V = kv_cache.gather(block_ids=[0, 1], num_tokens=BLOCK_SIZE + 2, layer_idx=0)
        assert K.shape == (1, N_KV_HEADS, BLOCK_SIZE + 2, D_HEAD)

    def test_gather_empty(self, kv_cache):
        K, V = kv_cache.gather(block_ids=[], num_tokens=0, layer_idx=0)
        assert K.shape == (1, N_KV_HEADS, 0, D_HEAD)

    def test_copy_block(self, kv_cache):
        torch.manual_seed(3)
        # Fill block 0 with random data
        for slot in range(BLOCK_SIZE):
            k = torch.randn(N_KV_HEADS, D_HEAD)
            v = torch.randn(N_KV_HEADS, D_HEAD)
            for layer in range(N_LAYERS):
                kv_cache.write(0, slot, layer, k, v)

        # Copy block 0 to block 5
        kv_cache._copy_block(src_block_id=0, dst_block_id=5)

        # All layers of block 5 should now match block 0
        for layer in range(N_LAYERS):
            assert torch.allclose(
                kv_cache.k_slab[layer, 0],
                kv_cache.k_slab[layer, 5]
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. CRITICAL — Paged gather == naive contiguous cache
# ══════════════════════════════════════════════════════════════════════════════

class TestPagedEquivalence:
    """
    THE critical correctness test for Phase 3.

    Verifies that kv_cache.gather() returns IDENTICAL tensors to what
    a naive contiguous KV cache would contain for the same sequence.

    If this fails, the block scatter/gather is broken and every
    module built on top (scheduler, serving) will produce wrong outputs.
    """

    def test_paged_gather_matches_naive_sequential_write(self):
        """
        Write tokens one by one using BlockTable + PagedKVCache.write().
        Then gather them. Verify they match the original token-by-token data.
        """
        torch.manual_seed(42)
        n_tokens = BLOCK_SIZE * 3 + 2   # crosses multiple blocks

        cache = PagedKVCache(
            num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
            n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS, d_head=D_HEAD,
            dtype=torch.float32,
        )

        bt = cache.create_block_table(seq_id=0)
        bt.allocate_for_prompt(n_tokens)

        # Reference: store ground-truth K,V in a plain list
        ref_k = []
        ref_v = []

        for pos in range(n_tokens):
            k = torch.randn(N_KV_HEADS, D_HEAD)
            v = torch.randn(N_KV_HEADS, D_HEAD)
            ref_k.append(k.clone())
            ref_v.append(v.clone())

            block_id, slot = bt.logical_to_physical(pos)
            for layer in range(N_LAYERS):
                cache.write(block_id, slot, layer, k, v)

        # Gather all tokens back
        for layer in range(N_LAYERS):
            K, V = cache.gather_for_sequence(bt, layer_idx=layer)
            # K shape: (1, N_KV_HEADS, n_tokens, D_HEAD)

            for pos in range(n_tokens):
                k_gathered = K[0, :, pos, :]   # (N_KV_HEADS, D_HEAD)
                v_gathered = V[0, :, pos, :]

                assert torch.allclose(k_gathered, ref_k[pos], atol=1e-5), \
                    f"Layer {layer}, token {pos}: K mismatch. " \
                    f"Max diff: {(k_gathered - ref_k[pos]).abs().max():.6f}"
                assert torch.allclose(v_gathered, ref_v[pos], atol=1e-5), \
                    f"Layer {layer}, token {pos}: V mismatch."

    def test_block_boundary_correctness(self):
        """
        Specifically test tokens at block boundaries -- the most common bug.

        Token at BLOCK_SIZE - 1 (last slot of block 0) and
        token at BLOCK_SIZE     (first slot of block 1) must be
        gathered correctly even though they live in different physical blocks.
        """
        torch.manual_seed(7)
        n_tokens = BLOCK_SIZE * 2

        cache = PagedKVCache(
            num_blocks=8, block_size=BLOCK_SIZE,
            n_layers=1, n_kv_heads=N_KV_HEADS, d_head=D_HEAD,
            dtype=torch.float32,
        )
        bt = cache.create_block_table(seq_id=0)
        bt.allocate_for_prompt(n_tokens)

        ref_k = []
        for pos in range(n_tokens):
            k = torch.randn(N_KV_HEADS, D_HEAD) * (pos + 1)   # distinct per position
            v = torch.zeros(N_KV_HEADS, D_HEAD)
            ref_k.append(k.clone())
            block_id, slot = bt.logical_to_physical(pos)
            cache.write(block_id, slot, 0, k, v)

        K, _ = cache.gather_for_sequence(bt, layer_idx=0)

        # Check boundary tokens explicitly
        boundary_pos = BLOCK_SIZE - 1
        assert torch.allclose(K[0, :, boundary_pos, :], ref_k[boundary_pos], atol=1e-5), \
            "Last token of first block gathered incorrectly"
        assert torch.allclose(K[0, :, boundary_pos + 1, :], ref_k[boundary_pos + 1], atol=1e-5), \
            "First token of second block gathered incorrectly"

    def test_allocate_next_and_gather(self):
        """
        Simulate a real decode loop:
        1. Allocate blocks for prompt
        2. Write prompt KV with write_prompt()
        3. For each decode step: allocate_next_token(), write(), gather()
        4. Verify gather output matches reference at every step
        """
        torch.manual_seed(99)
        prompt_len = BLOCK_SIZE + 2
        n_decode   = BLOCK_SIZE + 1   # enough to cross a block boundary

        cache = PagedKVCache(
            num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
            n_layers=1, n_kv_heads=N_KV_HEADS, d_head=D_HEAD,
            dtype=torch.float32,
        )
        bt = cache.create_block_table(seq_id=0)
        bt.allocate_for_prompt(prompt_len)

        # Generate all K,V values upfront (ground truth)
        all_k = [torch.randn(N_KV_HEADS, D_HEAD) for _ in range(prompt_len + n_decode)]
        all_v = [torch.randn(N_KV_HEADS, D_HEAD) for _ in range(prompt_len + n_decode)]

        # Write prompt using write_prompt (bulk write)
        k_prompt = torch.stack(all_k[:prompt_len], dim=1).unsqueeze(0)  # (1, H, T, D)
        v_prompt = torch.stack(all_v[:prompt_len], dim=1).unsqueeze(0)
        cache.write_prompt(bt, layer_idx=0, k_all=k_prompt, v_all=v_prompt)

        # Decode loop
        for step in range(n_decode):
            pos = prompt_len + step
            blk = bt.allocate_next_token()
            assert blk is not None, f"OOM at decode step {step}"

            block_id, slot = bt.get_slot_for_new_token()
            cache.write(block_id, slot, 0, all_k[pos], all_v[pos])

            # Gather and verify ALL tokens so far
            K, V = cache.gather_for_sequence(bt, layer_idx=0)
            assert K.shape == (1, N_KV_HEADS, pos + 1, D_HEAD)

            for p in range(pos + 1):
                assert torch.allclose(K[0, :, p, :], all_k[p], atol=1e-5), \
                    f"Decode step {step}: token {p} K mismatch"
                assert torch.allclose(V[0, :, p, :], all_v[p], atol=1e-5), \
                    f"Decode step {step}: token {p} V mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])