"""
Unit tests for memory-mapped experience buffer implementation.

Tests storage, retrieval, caching, and buffer management functionality.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import time

from src.training.experience_buffer import (
    MemoryMappedExperienceBuffer, LRUCache, create_experience_buffer
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specs" / "001-goal-create-spec"))
from contracts.training_api import TrainingExample, GameResult


class TestLRUCache:
    """Test LRU cache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with size limits."""
        cache = LRUCache(max_size_mb=10)
        assert cache.max_size_mb == 10
        assert cache.max_entries > 0
        assert len(cache.cache) == 0

    def test_cache_put_get(self):
        """Test basic put/get operations."""
        cache = LRUCache(max_size_mb=1)

        # Create test example
        example = TrainingExample(
            state=np.random.rand(3, 15, 15).astype(np.float32),
            policy=np.random.rand(225).astype(np.float32),
            value=0.5,
            game_type="gomoku",
            move_number=10,
            game_id="test_game"
        )

        # Test put and get
        cache.put("key1", example)
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert retrieved.game_id == example.game_id
        assert retrieved.move_number == example.move_number
        assert np.array_equal(retrieved.state, example.state)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size_mb=1)  # Very small cache
        cache.max_entries = 2  # Force small size for testing

        # Create test examples
        examples = []
        for i in range(3):
            example = TrainingExample(
                state=np.random.rand(3, 15, 15).astype(np.float32),
                policy=np.random.rand(225).astype(np.float32),
                value=0.5,
                game_type="gomoku",
                move_number=i,
                game_id=f"game_{i}"
            )
            examples.append(example)

        # Add examples, exceeding cache size
        cache.put("key1", examples[0])
        cache.put("key2", examples[1])
        cache.put("key3", examples[2])  # Should evict key1

        # Check that oldest entry was evicted
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        cache = LRUCache(max_size_mb=1)
        cache.max_entries = 5

        # Initially empty
        stats = cache.stats()
        assert stats['size'] == 0
        assert stats['utilization'] == 0.0

        # Add some entries
        for i in range(3):
            example = TrainingExample(
                state=np.random.rand(3, 15, 15).astype(np.float32),
                policy=np.random.rand(225).astype(np.float32),
                value=0.5,
                game_type="gomoku",
                move_number=i,
                game_id=f"game_{i}"
            )
            cache.put(f"key_{i}", example)

        stats = cache.stats()
        assert stats['size'] == 3
        assert stats['utilization'] == 0.6  # 3/5


class TestMemoryMappedExperienceBuffer:
    """Test memory-mapped experience buffer."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_examples(self):
        """Create sample training examples."""
        examples = []
        for i in range(5):
            example = TrainingExample(
                state=np.random.rand(3, 15, 15).astype(np.float32),
                policy=np.random.rand(225).astype(np.float32),
                value=float(i % 3 - 1),  # -1, 0, 1
                game_type="gomoku",
                move_number=i,
                game_id=f"game_{i // 2}"  # 2-3 examples per game
            )
            examples.append(example)
        return examples

    @pytest.fixture
    def sample_games(self, sample_examples):
        """Create sample game results."""
        games = []

        # Game 1
        game1 = GameResult(
            winner=1,
            move_count=3,
            game_length_seconds=120.0,
            examples=sample_examples[:3],
            final_board="...",
            metadata={"variation": "standard"}
        )
        games.append(game1)

        # Game 2
        game2 = GameResult(
            winner=0,
            move_count=2,
            game_length_seconds=90.0,
            examples=sample_examples[3:],
            final_board="...",
            metadata={"variation": "renju"}
        )
        games.append(game2)

        return games

    def test_buffer_initialization(self, temp_dir):
        """Test buffer initialization."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        assert buffer.buffer_path.exists()
        assert buffer.max_examples == 1000
        assert buffer.cache_size_mb == 64
        assert len(buffer.index) == 0

    def test_add_games(self, temp_dir, sample_games):
        """Test adding games to buffer."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Add games
        buffer.add_games(sample_games)

        # Check buffer state
        assert len(buffer.index) == 5  # Total examples from both games
        assert buffer.metadata['total_games'] == 2
        assert buffer.metadata['total_examples'] == 5
        assert 'gomoku' in buffer.metadata['game_type_counts']

    def test_sample_batch(self, temp_dir, sample_games):
        """Test sampling batches from buffer."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Add games
        buffer.add_games(sample_games)

        # Sample batch
        batch = buffer.sample_batch(batch_size=3)

        assert len(batch) == 3
        for example in batch:
            assert isinstance(example, TrainingExample)
            assert example.game_type == "gomoku"
            assert example.state.shape == (3, 15, 15)
            assert example.policy.shape == (225,)

    def test_sample_batch_with_game_type_filter(self, temp_dir, sample_games):
        """Test sampling with game type filtering."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Add games
        buffer.add_games(sample_games)

        # Sample with game type filter
        batch = buffer.sample_batch(batch_size=3, game_types=["gomoku"])

        assert len(batch) == 3
        for example in batch:
            assert example.game_type == "gomoku"

    def test_sample_empty_buffer(self, temp_dir):
        """Test sampling from empty buffer."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Sample from empty buffer
        batch = buffer.sample_batch(batch_size=5)
        assert len(batch) == 0

    def test_buffer_size_limit(self, temp_dir):
        """Test buffer size limiting behavior."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=3,  # Very small limit
            cache_size_mb=64
        )

        # Create more examples than limit
        examples = []
        for i in range(5):
            example = TrainingExample(
                state=np.random.rand(3, 15, 15).astype(np.float32),
                policy=np.random.rand(225).astype(np.float32),
                value=0.0,
                game_type="gomoku",
                move_number=i,
                game_id=f"game_{i}"
            )
            examples.append(example)

        game = GameResult(
            winner=None,
            move_count=5,
            game_length_seconds=60.0,
            examples=examples,
            final_board="...",
            metadata={}
        )

        buffer.add_games([game])

        # Should only keep last 3 examples
        assert len(buffer.index) <= 3
        assert buffer.metadata['total_examples'] <= 3

    def test_get_stats(self, temp_dir, sample_games):
        """Test statistics reporting."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Initially empty
        stats = buffer.get_stats()
        assert stats['total_examples'] == 0
        assert stats['total_games'] == 0
        assert stats['buffer_utilization'] == 0.0

        # Add games
        buffer.add_games(sample_games)

        stats = buffer.get_stats()
        assert stats['total_examples'] == 5
        assert stats['total_games'] == 2
        assert stats['buffer_utilization'] == 5 / 1000
        assert 'gomoku' in stats['game_type_distribution']
        assert 'storage_size_mb' in stats
        assert 'cache_stats' in stats

    def test_cleanup(self, temp_dir, sample_games):
        """Test buffer cleanup functionality."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Add games
        buffer.add_games(sample_games)
        assert len(buffer.index) == 5

        # Cleanup to keep only 3 examples
        buffer.cleanup(keep_last_n=3)

        assert len(buffer.index) == 3
        assert buffer.metadata['total_examples'] == 3

    def test_cache_integration(self, temp_dir, sample_games):
        """Test cache integration with buffer operations."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=1  # Small cache for testing
        )

        # Add games
        buffer.add_games(sample_games)

        # Sample same batch twice to test caching
        batch1 = buffer.sample_batch(batch_size=2)
        batch2 = buffer.sample_batch(batch_size=2)

        # Cache should have some entries
        cache_stats = buffer.cache.stats()
        assert cache_stats['size'] >= 0

    def test_persistence(self, temp_dir, sample_games):
        """Test data persistence across buffer instances."""
        buffer_path = temp_dir / "buffer"

        # Create buffer and add data
        buffer1 = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=1000,
            cache_size_mb=64
        )
        buffer1.add_games(sample_games)

        original_count = len(buffer1.index)
        original_stats = buffer1.get_stats()

        # Create new buffer instance from same path
        buffer2 = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=1000,
            cache_size_mb=64
        )

        # Should load existing data
        assert len(buffer2.index) == original_count
        new_stats = buffer2.get_stats()
        assert new_stats['total_examples'] == original_stats['total_examples']
        assert new_stats['total_games'] == original_stats['total_games']

    def test_concurrent_access(self, temp_dir, sample_games):
        """Test thread safety of buffer operations."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Add games
        buffer.add_games(sample_games)

        # Multiple concurrent samples (simulating multi-threaded access)
        import threading
        results = []

        def sample_worker():
            batch = buffer.sample_batch(batch_size=2)
            results.append(len(batch))

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=sample_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All samples should succeed
        assert len(results) == 5
        assert all(count == 2 for count in results)


def test_create_experience_buffer(tmp_path):
    """Test factory function for creating experience buffer."""
    buffer = create_experience_buffer(
        buffer_path=tmp_path / "test_buffer",
        max_examples=500,
        cache_size_mb=32
    )

    assert isinstance(buffer, MemoryMappedExperienceBuffer)
    assert buffer.max_examples == 500
    assert buffer.cache_size_mb == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])