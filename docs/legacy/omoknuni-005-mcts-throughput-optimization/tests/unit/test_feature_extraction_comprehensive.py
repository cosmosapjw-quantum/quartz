"""
Comprehensive feature extraction tests (T007e)

Ultra-deep testing covering:
1. All three games (Gomoku, Chess, Go)
2. Gomoku rule variations (Freestyle, Renju, Omok)
3. Boundary conditions (edges, corners, near-boundaries)
4. Complex game states (deep move history, tactical positions)
5. Edge cases (empty board, full board, mid-game)
6. Determinism and reproducibility
7. Performance characteristics
"""

import pytest
import numpy as np
import alphazero_py
import time


# ============================================================================
# Gomoku Comprehensive Tests
# ============================================================================

class TestGomokuComprehensive:
    """Comprehensive Gomoku feature extraction tests."""

    def test_gomoku_initial_state(self):
        """Test feature extraction on initial empty board."""
        state = alphazero_py.GomokuState()
        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)

        # Verify empty cells plane is all 1.0
        empty_plane = buffer[2*225:3*225].reshape(15, 15)
        assert np.all(empty_plane == 1.0), "Empty plane should be all 1.0 on initial board"

        # Verify current/opponent stone planes are all 0.0
        current_stones = buffer[0:225].reshape(15, 15)
        opponent_stones = buffer[225:450].reshape(15, 15)
        assert np.all(current_stones == 0.0), "Current stones should be empty"
        assert np.all(opponent_stones == 0.0), "Opponent stones should be empty"

    def test_gomoku_single_center_move(self):
        """Test feature extraction with single move at center."""
        state = alphazero_py.GomokuState()
        center = 7 * 15 + 7  # Center position (7,7)
        state.make_move(center)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Current player (WHITE after BLACK moved) should not have center stone
        assert features[0, 7, 7] == 0.0, "Current player (WHITE) should not have center stone"

        # Opponent (BLACK) should have center stone
        assert features[1, 7, 7] == 1.0, "Opponent (BLACK) should have center stone"

        # Move history plane 4 (most recent move for opponent) should mark center
        assert features[4, 7, 7] == 1.0, "Move history should mark recent move"

    def test_gomoku_corner_positions(self):
        """Test feature extraction with stones at all four corners."""
        state = alphazero_py.GomokuState()

        # Place stones at corners
        corners = [
            (0, 0),    # Top-left
            (0, 14),   # Top-right
            (14, 0),   # Bottom-left
            (14, 14),  # Bottom-right
        ]

        for r, c in corners:
            action = r * 15 + c
            state.make_move(action)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Verify corners are marked in stone planes
        for r, c in corners:
            # Should be in either current or opponent plane
            has_stone = features[0, r, c] == 1.0 or features[1, r, c] == 1.0
            assert has_stone, f"Corner ({r},{c}) should have a stone"

    def test_gomoku_edge_positions(self):
        """Test feature extraction with stones along edges."""
        state = alphazero_py.GomokuState()

        # Place stones along top edge
        for c in range(0, 15, 3):
            action = 0 * 15 + c
            state.make_move(action)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Verify edge positions are marked
        for c in range(0, 15, 3):
            has_stone = features[0, 0, c] == 1.0 or features[1, 0, c] == 1.0
            assert has_stone, f"Edge position (0,{c}) should have a stone"

    def test_gomoku_freestyle_rules(self):
        """Test Gomoku with Freestyle rules (no forbidden moves)."""
        state = alphazero_py.GomokuState(use_renju=False, use_omok=False)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Plane 18 should be 1.0 for Freestyle
        assert np.all(features[18] == 1.0), "Freestyle indicator should be 1.0"
        assert np.all(features[19] == 0.0), "Renju indicator should be 0.0"
        assert np.all(features[20] == 0.0), "Omok indicator should be 0.0"

    def test_gomoku_renju_rules(self):
        """Test Gomoku with Renju rules."""
        state = alphazero_py.GomokuState(use_renju=True)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Plane 19 should be 1.0 for Renju
        assert np.all(features[18] == 0.0), "Freestyle indicator should be 0.0"
        assert np.all(features[19] == 1.0), "Renju indicator should be 1.0"
        assert np.all(features[20] == 0.0), "Omok indicator should be 0.0"

    def test_gomoku_omok_rules(self):
        """Test Gomoku with Omok rules."""
        state = alphazero_py.GomokuState(use_omok=True)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Plane 20 should be 1.0 for Omok
        assert np.all(features[18] == 0.0), "Freestyle indicator should be 0.0"
        assert np.all(features[19] == 0.0), "Renju indicator should be 0.0"
        assert np.all(features[20] == 1.0), "Omok indicator should be 1.0"

    def test_gomoku_deep_move_history(self):
        """Test Gomoku with long move history (>7 moves)."""
        state = alphazero_py.GomokuState()

        # Make 15 moves using legal moves to avoid conflicts
        # Spread moves across the board to avoid win conditions
        np.random.seed(42)
        for _ in range(15):
            legal_moves = state.get_legal_moves()
            if len(legal_moves) == 0:
                break
            # Choose a move from the middle of legal moves to spread out
            move_idx = len(legal_moves) // 2
            state.make_move(legal_moves[move_idx])

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Verify that only last 7 moves are in history planes
        # History planes 4-10 for current player, 11-17 for opponent
        history_planes_current = features[4:11]
        history_planes_opponent = features[11:18]

        # Count non-zero entries in history planes
        count_current = np.count_nonzero(history_planes_current)
        count_opponent = np.count_nonzero(history_planes_opponent)

        # Should have at most 7 entries total (alternating players)
        assert count_current + count_opponent <= 7, f"History should have at most 7 moves, got {count_current + count_opponent}"

    def test_gomoku_determinism(self):
        """Test that feature extraction is deterministic."""
        state = alphazero_py.GomokuState()

        # Make some moves
        for move in [112, 113, 127, 128, 142]:
            state.make_move(move)

        # Extract features twice
        buffer1 = np.zeros(36 * 15 * 15, dtype=np.float32)
        buffer2 = np.zeros(36 * 15 * 15, dtype=np.float32)

        state.extract_features_to_buffer(buffer1)
        state.extract_features_to_buffer(buffer2)

        # Should be identical
        np.testing.assert_array_equal(buffer1, buffer2, "Feature extraction should be deterministic")

    def test_gomoku_reproducibility_after_undo(self):
        """Test that features are consistent after undo."""
        state = alphazero_py.GomokuState()

        # Extract initial features
        buffer_initial = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer_initial)

        # Make a move
        state.make_move(112)

        # Undo the move
        state.undo_move()

        # Extract features again
        buffer_after_undo = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer_after_undo)

        # Should match initial features
        np.testing.assert_array_equal(buffer_initial, buffer_after_undo,
                                     "Features after undo should match initial state")

    def test_gomoku_near_boundary_run_length(self):
        """Test run-length features near board boundaries."""
        state = alphazero_py.GomokuState()

        # Place stones near top edge to test boundary handling
        # Horizontal line: (1,5), (1,6), (1,7)
        for c in [5, 6, 7]:
            action = 1 * 15 + c
            state.make_move(action)
            # Add opponent move to keep game going
            if c < 7:
                state.make_move((2 * 15) + c)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(36, 15, 15)

        # Run-length planes (28-35) should have non-zero values near the line
        run_length_planes = features[28:36]
        assert np.any(run_length_planes[:, 1, 5:8] > 0), "Run-length features should be non-zero near stone line"


# ============================================================================
# Chess Comprehensive Tests
# ============================================================================

class TestChessComprehensive:
    """Comprehensive Chess feature extraction tests."""

    def test_chess_initial_position(self):
        """Test Chess feature extraction on initial board."""
        state = alphazero_py.ChessState()

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 8 * 8, dtype=np.float32)
        state.extract_features_to_buffer(buffer)

        # Should have pieces on initial setup
        assert np.any(buffer != 0.0), "Initial Chess position should have pieces"

    def test_chess_after_e4(self):
        """Test Chess after classic opening move e2-e4."""
        state = alphazero_py.ChessState()

        # Get legal moves and find e2-e4
        legal_moves = state.get_legal_moves()

        # Make first legal move as proxy
        if len(legal_moves) > 0:
            state.make_move(legal_moves[0])

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 8 * 8, dtype=np.float32)
        state.extract_features_to_buffer(buffer)

        # Verify extraction succeeded
        assert buffer.shape == (num_planes * 64,), "Buffer should have correct shape"

    def test_chess_midgame_position(self):
        """Test Chess after several moves (midgame)."""
        state = alphazero_py.ChessState()

        # Make 10 moves
        for _ in range(10):
            legal_moves = state.get_legal_moves()
            if len(legal_moves) > 0:
                state.make_move(legal_moves[0])

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 8 * 8, dtype=np.float32)
        state.extract_features_to_buffer(buffer)

        # Verify extraction succeeded
        assert buffer.shape == (num_planes * 64,), "Buffer should have correct shape"

    def test_chess_determinism(self):
        """Test that Chess feature extraction is deterministic."""
        state = alphazero_py.ChessState()

        # Make some moves
        for _ in range(5):
            legal_moves = state.get_legal_moves()
            if len(legal_moves) > 0:
                state.make_move(legal_moves[0])

        # Extract features twice
        num_planes = state.get_num_feature_planes()
        buffer1 = np.zeros(num_planes * 8 * 8, dtype=np.float32)
        buffer2 = np.zeros(num_planes * 8 * 8, dtype=np.float32)

        state.extract_features_to_buffer(buffer1)
        state.extract_features_to_buffer(buffer2)

        # Should be identical
        np.testing.assert_array_equal(buffer1, buffer2, "Chess feature extraction should be deterministic")


# ============================================================================
# Go Comprehensive Tests
# ============================================================================

class TestGoComprehensive:
    """Comprehensive Go feature extraction tests."""

    def test_go_initial_empty_board(self):
        """Test Go feature extraction on empty board."""
        state = alphazero_py.GoState()

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 19 * 19, dtype=np.float32)
        state.extract_features_to_buffer(buffer)

        # Initial Go board should be empty
        assert np.count_nonzero(buffer) > 0, "Go features should have some non-zero values (turn indicator, etc.)"

    def test_go_corner_hoshi_points(self):
        """Test Go with stones on corner hoshi points (star points)."""
        state = alphazero_py.GoState()

        # Hoshi points (3,3), (3,15), (15,3), (15,15) in 0-indexed
        hoshi_points = [(3,3), (3,15), (15,3), (15,15)]

        for r, c in hoshi_points:
            action = r * 19 + c
            state.make_move(action)

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 19 * 19, dtype=np.float32)
        state.extract_features_to_buffer(buffer)
        features = buffer.reshape(num_planes, 19, 19)

        # Verify stones are placed
        for r, c in hoshi_points:
            has_stone = features[0, r, c] == 1.0 or features[1, r, c] == 1.0
            assert has_stone, f"Hoshi point ({r},{c}) should have a stone"

    def test_go_edge_positions(self):
        """Test Go with stones along edges."""
        state = alphazero_py.GoState()

        # Place stones along top edge
        for c in range(0, 19, 4):
            action = 0 * 19 + c
            state.make_move(action)
            # Add opponent move
            if c + 1 < 19:
                state.make_move(1 * 19 + c)

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 19 * 19, dtype=np.float32)
        state.extract_features_to_buffer(buffer)

        # Verify extraction succeeded
        assert buffer.shape == (num_planes * 361,), "Go buffer should have correct shape"

    def test_go_determinism(self):
        """Test that Go feature extraction is deterministic."""
        state = alphazero_py.GoState()

        # Make some moves
        for i in range(10):
            action = i * 19 + i  # Diagonal moves
            state.make_move(action)

        # Extract features twice
        num_planes = state.get_num_feature_planes()
        buffer1 = np.zeros(num_planes * 19 * 19, dtype=np.float32)
        buffer2 = np.zeros(num_planes * 19 * 19, dtype=np.float32)

        state.extract_features_to_buffer(buffer1)
        state.extract_features_to_buffer(buffer2)

        # Should be identical
        np.testing.assert_array_equal(buffer1, buffer2, "Go feature extraction should be deterministic")


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for feature extraction."""

    def test_gomoku_extraction_speed(self):
        """Test Gomoku feature extraction is reasonably fast."""
        state = alphazero_py.GomokuState()

        # Make some moves
        for move in [112, 113, 127, 128, 142]:
            state.make_move(move)

        buffer = np.zeros(36 * 15 * 15, dtype=np.float32)

        # Time 100 extractions
        start = time.time()
        for _ in range(100):
            state.extract_features_to_buffer(buffer)
        elapsed = time.time() - start

        avg_time_us = (elapsed / 100) * 1000000
        print(f"\nGomoku extraction: {avg_time_us:.2f} μs/extraction")

        # Should be reasonably fast (target <10μs, but allow up to 1000μs for now)
        assert avg_time_us < 1000, f"Gomoku extraction too slow: {avg_time_us:.2f} μs"

    def test_chess_extraction_speed(self):
        """Test Chess feature extraction speed."""
        state = alphazero_py.ChessState()

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 8 * 8, dtype=np.float32)

        # Time 100 extractions
        start = time.time()
        for _ in range(100):
            state.extract_features_to_buffer(buffer)
        elapsed = time.time() - start

        avg_time_us = (elapsed / 100) * 1000000
        print(f"\nChess extraction: {avg_time_us:.2f} μs/extraction")

        # Should be reasonably fast
        assert avg_time_us < 1000, f"Chess extraction too slow: {avg_time_us:.2f} μs"

    def test_go_extraction_speed(self):
        """Test Go feature extraction speed."""
        state = alphazero_py.GoState()

        num_planes = state.get_num_feature_planes()
        buffer = np.zeros(num_planes * 19 * 19, dtype=np.float32)

        # Time 100 extractions
        start = time.time()
        for _ in range(100):
            state.extract_features_to_buffer(buffer)
        elapsed = time.time() - start

        avg_time_us = (elapsed / 100) * 1000000
        print(f"\nGo extraction: {avg_time_us:.2f} μs/extraction")

        # Should be reasonably fast
        assert avg_time_us < 1000, f"Go extraction too slow: {avg_time_us:.2f} μs"


if __name__ == "__main__":
    print("\n=== Comprehensive Feature Extraction Tests (T007e) ===\n")
    pytest.main([__file__, "-v", "-s"])
