"""
Unit tests for direct feature extraction (T007e)

Tests verify:
1. extract_features_to_buffer() returns correct data for all game types
2. Output matches getEnhancedTensorRepresentation()
3. No crashes or memory corruption
4. All game types supported
"""

import pytest
import numpy as np
import alphazero_py


# ============================================================================
# Gomoku Feature Extraction Tests
# ============================================================================

def test_gomoku_feature_extraction_buffer_size():
    """Gomoku extract_features_to_buffer should use correct buffer size."""
    state = alphazero_py.GomokuState()

    # Gomoku: 36 planes, 15×15 board
    expected_size = 36 * 15 * 15
    buffer = np.zeros(expected_size, dtype=np.float32)

    # Should not crash
    state.extract_features_to_buffer(buffer)

    # Buffer should be modified (not all zeros for initial position)
    assert np.any(buffer != 0.0), "Buffer should contain non-zero values"


def test_gomoku_num_feature_planes():
    """Gomoku should report 36 feature planes."""
    state = alphazero_py.GomokuState()
    assert state.get_num_feature_planes() == 36


def test_gomoku_feature_extraction_vs_tensor_representation():
    """Gomoku buffer extraction should match tensor representation."""
    state = alphazero_py.GomokuState()

    # Make some moves to create non-trivial state
    state.make_move(112)  # Center move (7, 7)
    state.make_move(97)   # Adjacent move

    # Get tensor via old method
    tensor = state.get_enhanced_tensor_representation()

    # Get via buffer extraction
    buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
    state.extract_features_to_buffer(buffer)
    buffer_reshaped = buffer.reshape(36, 15, 15)

    # Convert tensor to numpy
    tensor_np = np.array(tensor, dtype=np.float32)

    # Should match (allowing small floating point differences)
    np.testing.assert_allclose(buffer_reshaped, tensor_np, rtol=1e-5, atol=1e-7)


# ============================================================================
# Chess Feature Extraction Tests
# ============================================================================




def test_chess_feature_extraction_buffer_size():
    """Chess extract_features_to_buffer should use correct buffer size."""
    state = alphazero_py.ChessState()

    # Chess: 21 planes (actual size), 8×8 board
    num_planes = state.get_num_feature_planes()
    expected_size = num_planes * 8 * 8
    buffer = np.zeros(expected_size, dtype=np.float32)

    # Should not crash
    state.extract_features_to_buffer(buffer)

    # Buffer should be modified
    assert np.any(buffer != 0.0), "Buffer should contain non-zero values"


def test_chess_num_feature_planes():
    """Chess should report 21 feature planes (actual implementation)."""
    state = alphazero_py.ChessState()
    assert state.get_num_feature_planes() == 21


def test_chess_feature_extraction_vs_tensor_representation():
    """Chess buffer extraction should match tensor representation."""
    state = alphazero_py.ChessState()

    # Make a move
    legal_moves = state.get_legal_moves()
    if len(legal_moves) > 0:
        state.make_move(legal_moves[0])

    # Get tensor via old method
    tensor = state.get_enhanced_tensor_representation()

    # Get via buffer extraction
    num_planes = state.get_num_feature_planes()
    buffer = np.zeros(num_planes * 8 * 8, dtype=np.float32)
    state.extract_features_to_buffer(buffer)
    buffer_reshaped = buffer.reshape(num_planes, 8, 8)

    # Convert tensor to numpy
    tensor_np = np.array(tensor, dtype=np.float32)

    # Should match
    np.testing.assert_allclose(buffer_reshaped, tensor_np, rtol=1e-5, atol=1e-7)


# ============================================================================
# Go Feature Extraction Tests
# ============================================================================




def test_go_feature_extraction_buffer_size():
    """Go extract_features_to_buffer should use correct buffer size."""
    state = alphazero_py.GoState()

    # Go: 21 planes (actual size), 19×19 board
    num_planes = state.get_num_feature_planes()
    expected_size = num_planes * 19 * 19
    buffer = np.zeros(expected_size, dtype=np.float32)

    # Should not crash
    state.extract_features_to_buffer(buffer)

    # Buffer should be modified
    assert np.any(buffer != 0.0), "Buffer should contain non-zero values"


def test_go_num_feature_planes():
    """Go should report 21 feature planes (actual implementation)."""
    state = alphazero_py.GoState()
    assert state.get_num_feature_planes() == 21


def test_go_feature_extraction_vs_tensor_representation():
    """Go buffer extraction should match tensor representation."""
    state = alphazero_py.GoState()

    # Make a move
    legal_moves = state.get_legal_moves()
    if len(legal_moves) > 0:
        state.make_move(legal_moves[0])

    # Get tensor via old method
    tensor = state.get_enhanced_tensor_representation()

    # Get via buffer extraction
    num_planes = state.get_num_feature_planes()
    buffer = np.zeros(num_planes * 19 * 19, dtype=np.float32)
    state.extract_features_to_buffer(buffer)
    buffer_reshaped = buffer.reshape(num_planes, 19, 19)

    # Convert tensor to numpy
    tensor_np = np.array(tensor, dtype=np.float32)

    # Should match
    np.testing.assert_allclose(buffer_reshaped, tensor_np, rtol=1e-5, atol=1e-7)


# ============================================================================
# Thread Safety Tests (Basic)
# ============================================================================

def test_feature_extraction_thread_safe():
    """Feature extraction should be thread-safe (read-only)."""
    import threading

    state = alphazero_py.GomokuState()
    state.make_move(112)  # Make it non-trivial

    errors = []

    def extract():
        try:
            buffer = np.zeros(36 * 15 * 15, dtype=np.float32)
            state.extract_features_to_buffer(buffer)
            # Verify buffer has expected properties
            assert buffer.shape == (36 * 15 * 15,)
            assert np.any(buffer != 0.0)
        except Exception as e:
            errors.append(e)

    # Run 10 concurrent extractions
    threads = [threading.Thread(target=extract) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety test failed with errors: {errors}"


if __name__ == "__main__":
    print("\n=== Feature Extraction Tests (T007e) ===\n")

    # Run tests manually
    test_gomoku_feature_extraction_buffer_size()
    print("✓ test_gomoku_feature_extraction_buffer_size")

    test_gomoku_num_feature_planes()
    print("✓ test_gomoku_num_feature_planes")

    test_gomoku_feature_extraction_vs_tensor_representation()
    print("✓ test_gomoku_feature_extraction_vs_tensor_representation")

    test_chess_feature_extraction_buffer_size()
    print("✓ test_chess_feature_extraction_buffer_size")

    test_chess_num_feature_planes()
    print("✓ test_chess_num_feature_planes")

    test_chess_feature_extraction_vs_tensor_representation()
    print("✓ test_chess_feature_extraction_vs_tensor_representation")

    test_go_feature_extraction_buffer_size()
    print("✓ test_go_feature_extraction_buffer_size")

    test_go_num_feature_planes()
    print("✓ test_go_num_feature_planes")

    test_go_feature_extraction_vs_tensor_representation()
    print("✓ test_go_feature_extraction_vs_tensor_representation")

    test_feature_extraction_thread_safe()
    print("✓ test_feature_extraction_thread_safe")

    print("\n=== All feature extraction tests passed! ===\n")
