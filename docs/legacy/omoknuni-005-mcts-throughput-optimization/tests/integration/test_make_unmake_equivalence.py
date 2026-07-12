"""
Test that make_move/unmake_move is equivalent to clone/apply_move pattern.

This validates the core assumption of T024f-6: that we can replace
state pooling (copyFrom) with make/unmake and get identical results.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import after path setup
import alphazero_py


class TestMakeUnmakeEquivalence:
    """Test make/unmake pattern equivalence with clone/apply."""

    def test_gomoku_make_unmake_single_move(self):
        """Test single move make/unmake restores state exactly."""
        # Create initial state
        state = alphazero_py.GomokuState()

        # Capture initial state
        initial_hash = state.zobrist_hash()
        initial_legal = state.get_legal_moves()

        # Make a move
        move = 112  # Center position (7, 7) = 7*15 + 7
        undo_token = state.make_move(move)

        # Verify state changed
        assert state.zobrist_hash() != initial_hash
        assert move not in state.get_legal_moves()

        # Unmake the move
        state.unmake_move(move, undo_token)

        # Verify state restored exactly
        assert state.zobrist_hash() == initial_hash, f"Hash mismatch: {state.zobrist_hash()} != {initial_hash}"

        # Verify legal moves restored (set equality - order doesn't matter for MCTS)
        restored_legal = set(state.get_legal_moves())
        initial_legal_set = set(initial_legal)
        assert restored_legal == initial_legal_set, f"Legal moves mismatch: {len(restored_legal)} vs {len(initial_legal_set)}"

    def test_gomoku_make_unmake_sequence(self):
        """Test sequence of make/unmake restores state."""
        state = alphazero_py.GomokuState()
        initial_hash = state.zobrist_hash()

        # Make sequence of moves
        moves = [112, 113, 127, 128, 97]  # Play some moves
        undo_tokens = []

        for move in moves:
            undo_token = state.make_move(move)
            undo_tokens.append(undo_token)

        # Verify state changed
        assert state.zobrist_hash() != initial_hash

        # Unmake in reverse order
        for move, undo_token in zip(reversed(moves), reversed(undo_tokens)):
            state.unmake_move(move, undo_token)

        # Verify complete restoration
        assert state.zobrist_hash() == initial_hash

    def test_make_unmake_vs_clone_apply(self):
        """Test that make/unmake gives same result as clone/apply."""
        root_state = alphazero_py.GomokuState()

        # Path 1: Use clone + apply_move_inplace (current approach)
        state1 = root_state.clone()
        moves = [112, 113, 127, 128]
        for move in moves:
            state1.apply_move_inplace(move)
        hash1 = state1.zobrist_hash()
        legal1 = sorted(state1.get_legal_moves())

        # Path 2: Use make_move + unmake_move (new approach)
        state2 = root_state.clone()  # Will be thread-local in actual implementation
        undo_tokens = []
        for move in moves:
            undo_token = state2.make_move(move)
            undo_tokens.append(undo_token)

        hash2 = state2.zobrist_hash()
        legal2 = sorted(state2.get_legal_moves())

        # Both paths should give identical state
        assert hash1 == hash2, "Zobrist hash mismatch between clone/apply and make/unmake"
        assert legal1 == legal2, "Legal moves mismatch"

        # Verify unmake restores to root
        for move, undo_token in zip(reversed(moves), reversed(undo_tokens)):
            state2.unmake_move(move, undo_token)

        assert state2.zobrist_hash() == root_state.zobrist_hash()

    def test_make_unmake_performance(self):
        """Measure performance of make/unmake vs copyFrom."""
        import time

        root_state = alphazero_py.GomokuState()
        moves = [112, 113, 127, 128, 97, 98, 82, 83]
        iterations = 10000

        # Measure copyFrom + apply_move_inplace (current T018 approach)
        start = time.perf_counter()
        for _ in range(iterations):
            state = root_state.clone()
            for move in moves:
                state.apply_move_inplace(move)
            # State discarded
        clone_time = time.perf_counter() - start

        # Measure make_move + unmake_move (T024f approach)
        state = root_state.clone()  # Thread-local, allocated once
        start = time.perf_counter()
        for _ in range(iterations):
            undo_tokens = []
            for move in moves:
                undo_token = state.make_move(move)
                undo_tokens.append(undo_token)
            # Unwind
            for move, undo_token in zip(reversed(moves), reversed(undo_tokens)):
                state.unmake_move(move, undo_token)
        make_unmake_time = time.perf_counter() - start

        speedup = clone_time / make_unmake_time

        print(f"\nPerformance comparison ({iterations} iterations, {len(moves)} moves):")
        print(f"  copyFrom + apply:  {clone_time*1000:.2f}ms ({clone_time/iterations*1e6:.2f}μs per iteration)")
        print(f"  make + unmake:     {make_unmake_time*1000:.2f}ms ({make_unmake_time/iterations*1e6:.2f}μs per iteration)")
        print(f"  Speedup:           {speedup:.2f}×")

        # T024f target: make/unmake should be significantly faster
        # Expected: 418μs (copyFrom) vs ~120ns (8 make/unmake pairs @ 15ns each) = 3,483× speedup
        # But in practice, clone() includes more overhead, so expect 10-50× speedup
        assert speedup > 2.0, f"make/unmake should be faster, got {speedup:.2f}× speedup"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
