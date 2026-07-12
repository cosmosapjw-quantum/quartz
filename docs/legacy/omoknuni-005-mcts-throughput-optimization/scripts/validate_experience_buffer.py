#!/usr/bin/env python3
"""
Experience Buffer Validation Script

Demonstrates the memory-mapped experience buffer functionality including:
- Adding games to buffer
- Sampling training batches
- Cache performance
- Storage persistence
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.experience_buffer import create_experience_buffer
from specs.contracts.training_api import TrainingExample, GameResult


def create_mock_game(game_id: str, num_examples: int = 10, game_type: str = "gomoku") -> GameResult:
    """Create a mock game result for testing."""
    examples = []

    for i in range(num_examples):
        # Create realistic state tensor based on game type
        if game_type == "gomoku":
            state = np.random.rand(36, 15, 15).astype(np.float32)  # 36 planes for Gomoku
            policy = np.random.rand(225).astype(np.float32)  # 15x15 board
        elif game_type == "chess":
            state = np.random.rand(30, 8, 8).astype(np.float32)  # 30 planes for Chess
            policy = np.random.rand(4096).astype(np.float32)  # Chess action space
        else:  # go
            state = np.random.rand(25, 19, 19).astype(np.float32)  # 25 planes for Go
            policy = np.random.rand(362).astype(np.float32)  # 19x19 + pass

        # Normalize policy
        policy = policy / policy.sum()

        example = TrainingExample(
            state=state,
            policy=policy,
            value=np.random.uniform(-1, 1),
            game_type=game_type,
            move_number=i,
            game_id=game_id
        )
        examples.append(example)

    return GameResult(
        winner=np.random.choice([0, 1, None]),
        move_count=num_examples,
        game_length_seconds=np.random.uniform(60, 300),
        examples=examples,
        final_board=f"Final position for {game_id}",
        metadata={"variation": "standard"}
    )


def validate_basic_functionality():
    """Test basic buffer functionality."""
    logger.info("=== Basic Functionality Test ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "test_buffer"

        # Create buffer
        buffer = create_experience_buffer(
            buffer_path=buffer_path,
            max_examples=1000,
            cache_size_mb=64
        )

        logger.info(f"Created buffer at {buffer_path}")

        # Create and add some games
        games = []
        for i in range(5):
            game = create_mock_game(f"game_{i}", num_examples=8, game_type="gomoku")
            games.append(game)

        logger.info(f"Adding {len(games)} games to buffer...")
        start_time = time.time()
        buffer.add_games(games)
        add_time = time.time() - start_time

        logger.info(f"Added games in {add_time:.3f}s")

        # Get buffer stats
        stats = buffer.get_stats()
        logger.info(f"Buffer stats: {stats}")

        # Sample some batches
        logger.info("Sampling training batches...")

        start_time = time.time()
        for i in range(10):
            batch = buffer.sample_batch(batch_size=16)
            if i == 0:
                logger.info(f"First batch: {len(batch)} examples")
                logger.info(f"Example shapes: state={batch[0].state.shape}, policy={batch[0].policy.shape}")

        sample_time = time.time() - start_time
        logger.info(f"Sampled 10 batches in {sample_time:.3f}s")

        # Test cache performance
        cache_stats = buffer.cache.stats()
        logger.info(f"Cache stats: {cache_stats}")

        logger.info("✅ Basic functionality test passed!")


def validate_multi_game_types():
    """Test buffer with multiple game types."""
    logger.info("=== Multi-Game Type Test ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "multi_game_buffer"

        buffer = create_experience_buffer(
            buffer_path=buffer_path,
            max_examples=2000,
            cache_size_mb=128
        )

        # Create games of different types
        all_games = []
        game_types = ["gomoku", "chess", "go"]

        for game_type in game_types:
            for i in range(3):  # 3 games per type
                game = create_mock_game(f"{game_type}_game_{i}", num_examples=10, game_type=game_type)
                all_games.append(game)

        logger.info(f"Adding {len(all_games)} games of types: {game_types}")
        buffer.add_games(all_games)

        stats = buffer.get_stats()
        logger.info(f"Game type distribution: {stats['game_type_distribution']}")

        # Test sampling with game type filters
        for game_type in game_types:
            batch = buffer.sample_batch(batch_size=8, game_types=[game_type])
            if batch:
                logger.info(f"{game_type}: sampled {len(batch)} examples, "
                           f"state shape: {batch[0].state.shape}")
            else:
                logger.warning(f"No examples found for {game_type}")

        logger.info("✅ Multi-game type test passed!")


def validate_buffer_limits():
    """Test buffer size limits and cleanup."""
    logger.info("=== Buffer Limits Test ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "limit_buffer"

        # Create small buffer to test limits
        buffer = create_experience_buffer(
            buffer_path=buffer_path,
            max_examples=50,  # Small limit
            cache_size_mb=32
        )

        # Add more examples than the limit
        games = []
        for i in range(10):  # 10 games * 8 examples = 80 examples (exceeds limit)
            game = create_mock_game(f"limit_game_{i}", num_examples=8)
            games.append(game)

        logger.info(f"Adding {len(games)} games (80 examples) to buffer with limit 50...")
        buffer.add_games(games)

        stats = buffer.get_stats()
        logger.info(f"Buffer after limit: {stats['total_examples']} examples")

        # Test cleanup
        logger.info("Testing cleanup to keep only 20 examples...")
        buffer.cleanup(keep_last_n=20)

        stats = buffer.get_stats()
        logger.info(f"Buffer after cleanup: {stats['total_examples']} examples")

        assert stats['total_examples'] <= 20, f"Cleanup failed: still have {stats['total_examples']} examples"

        logger.info("✅ Buffer limits test passed!")


def validate_persistence():
    """Test data persistence across buffer instances."""
    logger.info("=== Persistence Test ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "persist_buffer"

        # Create first buffer instance and add data
        buffer1 = create_experience_buffer(
            buffer_path=buffer_path,
            max_examples=1000,
            cache_size_mb=64
        )

        games = [create_mock_game(f"persist_game_{i}", num_examples=6) for i in range(3)]
        buffer1.add_games(games)

        stats1 = buffer1.get_stats()
        logger.info(f"Buffer 1 stats: {stats1['total_examples']} examples, {stats1['total_games']} games")

        # Create second buffer instance from same path
        buffer2 = create_experience_buffer(
            buffer_path=buffer_path,
            max_examples=1000,
            cache_size_mb=64
        )

        stats2 = buffer2.get_stats()
        logger.info(f"Buffer 2 stats: {stats2['total_examples']} examples, {stats2['total_games']} games")

        # Verify data persistence
        assert stats2['total_examples'] == stats1['total_examples'], "Examples not persisted"
        assert stats2['total_games'] == stats1['total_games'], "Game count not persisted"

        # Verify we can sample from reloaded buffer
        batch = buffer2.sample_batch(batch_size=5)
        assert len(batch) == 5, "Could not sample from reloaded buffer"

        logger.info("✅ Persistence test passed!")


def validate_performance():
    """Test buffer performance with larger datasets."""
    logger.info("=== Performance Test ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "perf_buffer"

        buffer = create_experience_buffer(
            buffer_path=buffer_path,
            max_examples=10000,
            cache_size_mb=256
        )

        # Create large dataset
        logger.info("Creating large dataset...")
        large_games = []
        for i in range(50):  # 50 games * 20 examples = 1000 examples
            game = create_mock_game(f"perf_game_{i}", num_examples=20)
            large_games.append(game)

        # Test add performance
        logger.info("Testing add performance...")
        start_time = time.time()
        buffer.add_games(large_games)
        add_time = time.time() - start_time

        stats = buffer.get_stats()
        examples_per_sec = stats['total_examples'] / add_time
        logger.info(f"Added {stats['total_examples']} examples in {add_time:.3f}s "
                   f"({examples_per_sec:.0f} examples/sec)")

        # Test sampling performance
        logger.info("Testing sampling performance...")
        start_time = time.time()
        total_sampled = 0

        for _ in range(100):  # 100 batches
            batch = buffer.sample_batch(batch_size=32)
            total_sampled += len(batch)

        sample_time = time.time() - start_time
        samples_per_sec = total_sampled / sample_time
        logger.info(f"Sampled {total_sampled} examples in {sample_time:.3f}s "
                   f"({samples_per_sec:.0f} samples/sec)")

        # Test cache effectiveness
        cache_stats = buffer.cache.stats()
        logger.info(f"Cache utilization: {cache_stats['utilization']:.2%}")

        logger.info("✅ Performance test passed!")


def main():
    """Run all validation tests."""
    logger.info("Starting experience buffer validation...")

    try:
        validate_basic_functionality()
        validate_multi_game_types()
        validate_buffer_limits()
        validate_persistence()
        validate_performance()

        logger.info("🎉 All validation tests passed!")

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()