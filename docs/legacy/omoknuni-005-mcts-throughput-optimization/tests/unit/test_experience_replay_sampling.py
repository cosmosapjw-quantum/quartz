"""
Unit tests for advanced experience replay sampling functionality.

Tests balanced sampling, temporal uniformity, training iterators, and sampling statistics.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import time
import random

from src.training.experience_buffer import (
    MemoryMappedExperienceBuffer
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specs" / "001-goal-create-spec"))
from contracts.training_api import TrainingExample, GameResult


class TestAdvancedSampling:
    """Test advanced sampling methods in experience buffer."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def multi_game_buffer(self, temp_dir):
        """Create buffer with multiple game types for testing."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "multi_buffer",
            max_examples=5000,
            cache_size_mb=128
        )

        # Create games with different types and temporal ordering
        all_games = []
        game_types = ["gomoku", "chess", "go"]
        examples_per_game = 10

        for time_period in range(3):  # 3 temporal periods
            for game_type in game_types:
                for game_idx in range(3):  # 3 games per type per period
                    game_id = f"{game_type}_t{time_period}_g{game_idx}"
                    examples = []

                    for move_idx in range(examples_per_game):
                        # Create game-specific state shapes
                        if game_type == "gomoku":
                            state = np.random.rand(36, 15, 15).astype(np.float32)
                            policy = np.random.rand(225).astype(np.float32)
                        elif game_type == "chess":
                            state = np.random.rand(30, 8, 8).astype(np.float32)
                            policy = np.random.rand(4096).astype(np.float32)
                        else:  # go
                            state = np.random.rand(25, 19, 19).astype(np.float32)
                            policy = np.random.rand(362).astype(np.float32)

                        policy = policy / policy.sum()  # Normalize

                        example = TrainingExample(
                            state=state,
                            policy=policy,
                            value=np.random.uniform(-1, 1),
                            game_type=game_type,
                            move_number=move_idx,
                            game_id=game_id
                        )
                        examples.append(example)

                    game = GameResult(
                        winner=random.choice([0, 1, None]),
                        move_count=examples_per_game,
                        game_length_seconds=random.uniform(60, 300),
                        examples=examples,
                        final_board=f"Final position for {game_id}",
                        metadata={"time_period": time_period}
                    )
                    all_games.append(game)

            # Add games in batches to create temporal structure
            buffer.add_games(all_games[time_period * 9:(time_period + 1) * 9])

        return buffer

    def test_sample_balanced_batch_equal_distribution(self, multi_game_buffer):
        """Test balanced sampling with equal distribution."""
        buffer = multi_game_buffer

        # Sample a balanced batch
        batch = buffer.sample_balanced_batch(batch_size=90)  # 30 per game type

        # Count game types in batch
        game_type_counts = Counter(example.game_type for example in batch)

        # Should have roughly equal distribution
        assert len(game_type_counts) == 3
        for game_type, count in game_type_counts.items():
            assert 25 <= count <= 35, f"{game_type} has {count} examples, expected ~30"

        # Verify total batch size
        assert len(batch) == 90

    def test_sample_balanced_batch_custom_ratios(self, multi_game_buffer):
        """Test balanced sampling with custom game type ratios."""
        buffer = multi_game_buffer

        # Custom ratios: gomoku 50%, chess 30%, go 20%
        custom_ratios = {"gomoku": 0.5, "chess": 0.3, "go": 0.2}
        batch = buffer.sample_balanced_batch(
            batch_size=100,
            game_type_ratios=custom_ratios
        )

        # Count game types in batch
        game_type_counts = Counter(example.game_type for example in batch)

        # Verify approximately correct ratios
        total_samples = sum(game_type_counts.values())
        for game_type, expected_ratio in custom_ratios.items():
            actual_ratio = game_type_counts[game_type] / total_samples
            assert abs(actual_ratio - expected_ratio) < 0.1, \
                f"{game_type}: expected {expected_ratio:.1%}, got {actual_ratio:.1%}"

    def test_temporal_uniformity_sampling(self, multi_game_buffer):
        """Test temporal uniformity in sampling."""
        buffer = multi_game_buffer

        # Sample with temporal uniformity
        batch_uniform = buffer.sample_balanced_batch(
            batch_size=90,
            temporal_uniformity=True
        )

        # Sample without temporal uniformity
        batch_biased = buffer.sample_balanced_batch(
            batch_size=90,
            temporal_uniformity=False
        )

        # Both should have similar game type distributions
        uniform_counts = Counter(ex.game_type for ex in batch_uniform)
        biased_counts = Counter(ex.game_type for ex in batch_biased)

        assert len(uniform_counts) == len(biased_counts) == 3

        # Check that temporal uniformity spreads across time periods
        # (This is a statistical test, might occasionally fail due to randomness)
        uniform_time_periods = set()
        for example in batch_uniform:
            # Extract time period from game_id
            if "_t" in example.game_id:
                time_period = int(example.game_id.split("_t")[1].split("_")[0])
                uniform_time_periods.add(time_period)

        # Should sample from multiple time periods
        assert len(uniform_time_periods) >= 2, \
            f"Temporal uniformity should sample from multiple periods, got {uniform_time_periods}"

    def test_training_iterator(self, multi_game_buffer):
        """Test continuous training iterator."""
        buffer = multi_game_buffer

        # Create training iterator
        iterator = buffer.create_training_iterator(
            batch_size=32,
            game_type_ratios={"gomoku": 0.4, "chess": 0.4, "go": 0.2},
            shuffle_buffer_size=100
        )

        # Sample several batches
        batches = []
        for i, batch in enumerate(iterator):
            batches.append(batch)
            if i >= 5:  # Sample 6 batches
                break

        # Verify batch properties
        assert len(batches) == 6

        for batch in batches:
            assert len(batch) <= 32  # Should not exceed batch size
            assert len(batch) > 0   # Should not be empty

            # Check game type distribution in batch
            game_type_counts = Counter(ex.game_type for ex in batch)
            assert len(game_type_counts) <= 3  # At most 3 game types

    def test_get_sampling_stats(self, multi_game_buffer):
        """Test sampling statistics reporting."""
        buffer = multi_game_buffer

        stats = buffer.get_sampling_stats()

        # Verify basic statistics
        assert stats['total_examples'] > 0
        assert 'game_type_distribution' in stats
        assert 'game_type_percentages' in stats
        assert 'temporal_spread' in stats
        assert 'temporal_range' in stats
        assert 'sampling_capability' in stats

        # Check game type distribution
        game_types = stats['game_type_distribution']
        assert len(game_types) == 3
        assert 'gomoku' in game_types
        assert 'chess' in game_types
        assert 'go' in game_types

        # Each game type should have equal count (3 time periods * 3 games * 10 examples = 90)
        for game_type, count in game_types.items():
            assert count == 90, f"{game_type} should have 90 examples, got {count}"

        # Check percentages sum to 100
        percentages = stats['game_type_percentages']
        total_percentage = sum(percentages.values())
        assert abs(total_percentage - 100.0) < 0.1

        # Check temporal spread
        temporal_spread = stats['temporal_spread']
        assert len(temporal_spread) > 0

        # Check sampling capability
        capability = stats['sampling_capability']
        assert capability['max_balanced_batch_size'] == 270  # 90 * 3 game types
        assert capability['min_examples_per_type'] == 90
        assert capability['max_examples_per_type'] == 90

    def test_empty_buffer_sampling(self, temp_dir):
        """Test sampling behavior with empty buffer."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "empty_buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Test balanced sampling on empty buffer
        batch = buffer.sample_balanced_batch(batch_size=10)
        assert len(batch) == 0

        # Test training iterator on empty buffer
        iterator = buffer.create_training_iterator(batch_size=10)
        batches = list(iterator)
        assert len(batches) == 0

        # Test sampling stats on empty buffer
        stats = buffer.get_sampling_stats()
        assert stats['total_examples'] == 0
        assert stats['game_type_distribution'] == {}
        assert stats['sampling_capability']['max_balanced_batch_size'] == 0

    def test_single_game_type_sampling(self, temp_dir):
        """Test balanced sampling when only one game type is available."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "single_type_buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Add only gomoku games
        games = []
        for i in range(3):
            examples = []
            for j in range(10):
                example = TrainingExample(
                    state=np.random.rand(36, 15, 15).astype(np.float32),
                    policy=np.random.rand(225).astype(np.float32),
                    value=np.random.uniform(-1, 1),
                    game_type="gomoku",
                    move_number=j,
                    game_id=f"gomoku_game_{i}"
                )
                examples.append(example)

            game = GameResult(
                winner=random.choice([0, 1, None]),
                move_count=10,
                game_length_seconds=120.0,
                examples=examples,
                final_board="...",
                metadata={}
            )
            games.append(game)

        buffer.add_games(games)

        # Test balanced sampling
        batch = buffer.sample_balanced_batch(batch_size=15)
        assert len(batch) == 15

        # All should be gomoku
        game_types = set(ex.game_type for ex in batch)
        assert game_types == {"gomoku"}

        # Test with ratios for non-existent game types
        batch = buffer.sample_balanced_batch(
            batch_size=10,
            game_type_ratios={"gomoku": 0.5, "chess": 0.5}
        )
        assert len(batch) == 10
        assert all(ex.game_type == "gomoku" for ex in batch)

    def test_temporal_sampling_edge_cases(self, temp_dir):
        """Test temporal sampling with edge cases."""
        buffer = MemoryMappedExperienceBuffer(
            buffer_path=temp_dir / "temporal_edge_buffer",
            max_examples=1000,
            cache_size_mb=64
        )

        # Create buffer with very few examples
        examples = []
        for i in range(3):
            example = TrainingExample(
                state=np.random.rand(36, 15, 15).astype(np.float32),
                policy=np.random.rand(225).astype(np.float32),
                value=0.0,
                game_type="gomoku",
                move_number=i,
                game_id=f"test_game"
            )
            examples.append(example)

        game = GameResult(
            winner=None,
            move_count=3,
            game_length_seconds=60.0,
            examples=examples,
            final_board="...",
            metadata={}
        )

        buffer.add_games([game])

        # Test sampling more than available
        batch = buffer.sample_balanced_batch(batch_size=10)
        assert len(batch) == 3  # Should return all available examples

        # Test sampling with temporal uniformity on small dataset
        batch = buffer.sample_balanced_batch(
            batch_size=2,
            temporal_uniformity=True
        )
        assert len(batch) == 2

    def test_concurrent_sampling(self, multi_game_buffer):
        """Test thread safety of sampling operations."""
        buffer = multi_game_buffer

        import threading
        results = []
        errors = []

        def sample_worker():
            try:
                for _ in range(10):
                    batch = buffer.sample_balanced_batch(batch_size=20)
                    results.append(len(batch))
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple sampling threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=sample_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors during concurrent sampling: {errors}"
        assert len(results) == 50  # 5 threads * 10 samples each
        assert all(result > 0 for result in results), "All samples should be non-empty"

    def test_batch_distribution_consistency(self, multi_game_buffer):
        """Test that batch distributions are consistent over multiple samples."""
        buffer = multi_game_buffer

        # Sample multiple batches and check consistency
        batch_distributions = []
        for _ in range(10):
            batch = buffer.sample_balanced_batch(batch_size=60)
            distribution = Counter(ex.game_type for ex in batch)
            batch_distributions.append(distribution)

        # Check that distributions are reasonably consistent
        game_types = ["gomoku", "chess", "go"]
        for game_type in game_types:
            counts = [dist[game_type] for dist in batch_distributions]
            mean_count = sum(counts) / len(counts)
            std_dev = (sum((x - mean_count) ** 2 for x in counts) / len(counts)) ** 0.5

            # Standard deviation should be reasonable (not too high variance)
            assert std_dev < mean_count * 0.3, \
                f"{game_type} distribution too variable: mean={mean_count:.1f}, std={std_dev:.1f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])