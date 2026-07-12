#!/usr/bin/env python3
"""
Experience Replay Sampling Validation Script

Demonstrates advanced sampling capabilities including:
- Balanced game type distribution
- Temporal uniformity sampling
- Training iterator functionality
- Sampling statistics and analysis
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
import logging
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training.experience_buffer import MemoryMappedExperienceBuffer
from specs.contracts.training_api import TrainingExample, GameResult


def create_diverse_dataset(buffer: MemoryMappedExperienceBuffer,
                         num_time_periods: int = 5,
                         games_per_period: int = 6) -> None:
    """Create a diverse dataset with multiple game types and temporal structure."""
    logger.info(f"Creating dataset with {num_time_periods} time periods, {games_per_period} games per period")

    game_types = ["gomoku", "chess", "go"]
    all_games = []

    for time_period in range(num_time_periods):
        period_games = []

        for game_type in game_types:
            for game_idx in range(games_per_period // 3):  # Equal distribution
                game_id = f"{game_type}_t{time_period}_g{game_idx}"
                examples = []

                # Vary number of examples per game
                num_examples = random.randint(8, 15)

                for move_idx in range(num_examples):
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
                    move_count=num_examples,
                    game_length_seconds=random.uniform(60, 300),
                    examples=examples,
                    final_board=f"Final position for {game_id}",
                    metadata={"time_period": time_period, "strength": random.uniform(1000, 2000)}
                )
                period_games.append(game)

        # Add games from this time period
        buffer.add_games(period_games)
        logger.info(f"Added {len(period_games)} games from time period {time_period}")

    stats = buffer.get_stats()
    logger.info(f"Dataset created: {stats['total_examples']} total examples, {stats['total_games']} games")


def validate_balanced_sampling():
    """Test balanced sampling functionality."""
    logger.info("=== Balanced Sampling Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "balanced_buffer"

        buffer = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=10000,
            cache_size_mb=256
        )

        # Create diverse dataset
        create_diverse_dataset(buffer, num_time_periods=4, games_per_period=9)

        # Test equal distribution sampling
        logger.info("Testing equal distribution sampling...")
        batch_size = 120
        batch = buffer.sample_balanced_batch(batch_size=batch_size)

        game_type_counts = Counter(ex.game_type for ex in batch)
        logger.info(f"Equal distribution batch ({batch_size} examples): {dict(game_type_counts)}")

        # Verify roughly equal distribution
        expected_per_type = batch_size // 3
        for game_type, count in game_type_counts.items():
            ratio = count / batch_size
            logger.info(f"{game_type}: {count}/{batch_size} = {ratio:.1%}")
            assert abs(count - expected_per_type) <= 5, f"Unbalanced distribution for {game_type}"

        # Test custom ratio sampling
        logger.info("Testing custom ratio sampling...")
        custom_ratios = {"gomoku": 0.5, "chess": 0.3, "go": 0.2}
        batch = buffer.sample_balanced_batch(
            batch_size=100,
            game_type_ratios=custom_ratios
        )

        game_type_counts = Counter(ex.game_type for ex in batch)
        logger.info(f"Custom ratio batch: {dict(game_type_counts)}")

        for game_type, expected_ratio in custom_ratios.items():
            actual_ratio = game_type_counts[game_type] / len(batch)
            logger.info(f"{game_type}: expected {expected_ratio:.1%}, got {actual_ratio:.1%}")
            assert abs(actual_ratio - expected_ratio) < 0.15, f"Ratio mismatch for {game_type}"

        logger.info("✅ Balanced sampling validation passed!")


def validate_temporal_uniformity():
    """Test temporal uniformity in sampling."""
    logger.info("=== Temporal Uniformity Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "temporal_buffer"

        buffer = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=10000,
            cache_size_mb=256
        )

        # Create dataset with clear temporal structure
        create_diverse_dataset(buffer, num_time_periods=5, games_per_period=6)

        # Sample with temporal uniformity
        logger.info("Testing temporal uniformity...")
        uniform_batch = buffer.sample_balanced_batch(
            batch_size=150,
            temporal_uniformity=True
        )

        # Sample without temporal uniformity
        biased_batch = buffer.sample_balanced_batch(
            batch_size=150,
            temporal_uniformity=False
        )

        # Analyze temporal distribution
        def analyze_temporal_distribution(batch, name):
            time_periods = []
            for example in batch:
                if "_t" in example.game_id:
                    time_period = int(example.game_id.split("_t")[1].split("_")[0])
                    time_periods.append(time_period)

            period_counts = Counter(time_periods)
            logger.info(f"{name} temporal distribution: {dict(period_counts)}")

            return period_counts

        uniform_periods = analyze_temporal_distribution(uniform_batch, "Uniform")
        biased_periods = analyze_temporal_distribution(biased_batch, "Biased")

        # Temporal uniformity should spread more evenly
        uniform_std = np.std(list(uniform_periods.values()))
        biased_std = np.std(list(biased_periods.values()))

        logger.info(f"Temporal distribution standard deviation - Uniform: {uniform_std:.2f}, Biased: {biased_std:.2f}")

        # Uniform should have representation from multiple periods
        assert len(uniform_periods) >= 3, f"Uniform sampling should span multiple periods, got {len(uniform_periods)}"

        logger.info("✅ Temporal uniformity validation passed!")


def validate_training_iterator():
    """Test training iterator functionality."""
    logger.info("=== Training Iterator Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "iterator_buffer"

        buffer = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=10000,
            cache_size_mb=256
        )

        # Create dataset
        create_diverse_dataset(buffer, num_time_periods=3, games_per_period=12)

        # Test training iterator
        logger.info("Testing training iterator...")
        iterator = buffer.create_training_iterator(
            batch_size=32,
            game_type_ratios={"gomoku": 0.4, "chess": 0.4, "go": 0.2},
            shuffle_buffer_size=200
        )

        # Sample several batches
        batch_distributions = []
        start_time = time.time()

        for i, batch in enumerate(iterator):
            if i >= 10:  # Sample 10 batches
                break

            assert len(batch) <= 32, f"Batch {i} too large: {len(batch)}"
            assert len(batch) > 0, f"Batch {i} is empty"

            # Analyze distribution
            game_type_counts = Counter(ex.game_type for ex in batch)
            batch_distributions.append(game_type_counts)

            if i == 0:
                logger.info(f"First batch distribution: {dict(game_type_counts)}")

        iteration_time = time.time() - start_time
        logger.info(f"Generated 10 batches in {iteration_time:.3f}s ({len(batch_distributions) / iteration_time:.1f} batches/sec)")

        # Analyze overall distribution consistency
        total_counts = defaultdict(int)
        for dist in batch_distributions:
            for game_type, count in dist.items():
                total_counts[game_type] += count

        total_examples = sum(total_counts.values())
        ratios = {gt: count / total_examples for gt, count in total_counts.items()}
        logger.info(f"Overall iterator distribution: {ratios}")

        # Should roughly match target ratios
        expected_ratios = {"gomoku": 0.4, "chess": 0.4, "go": 0.2}
        for game_type, expected in expected_ratios.items():
            actual = ratios.get(game_type, 0)
            assert abs(actual - expected) < 0.2, f"Iterator ratio mismatch for {game_type}: {actual:.2f} vs {expected:.2f}"

        logger.info("✅ Training iterator validation passed!")


def validate_sampling_statistics():
    """Test sampling statistics and analysis."""
    logger.info("=== Sampling Statistics Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "stats_buffer"

        buffer = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=10000,
            cache_size_mb=256
        )

        # Create dataset
        create_diverse_dataset(buffer, num_time_periods=4, games_per_period=9)

        # Get sampling statistics
        stats = buffer.get_sampling_stats()

        logger.info("Sampling statistics:")
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Game type distribution: {stats['game_type_distribution']}")
        logger.info(f"Game type percentages: {stats['game_type_percentages']}")
        logger.info(f"Temporal range: {stats['temporal_range']}")
        logger.info(f"Sampling capability: {stats['sampling_capability']}")

        # Verify statistics
        assert stats['total_examples'] > 0
        assert len(stats['game_type_distribution']) == 3

        # Check percentages sum to 100
        total_percentage = sum(stats['game_type_percentages'].values())
        assert abs(total_percentage - 100.0) < 0.1

        # Check temporal spread
        temporal_spread = stats['temporal_spread']
        assert len(temporal_spread) > 0

        # Check sampling capability
        capability = stats['sampling_capability']
        assert capability['max_balanced_batch_size'] > 0
        assert capability['min_examples_per_type'] > 0

        logger.info("✅ Sampling statistics validation passed!")


def validate_performance():
    """Test sampling performance under load."""
    logger.info("=== Performance Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "perf_buffer"

        buffer = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=20000,
            cache_size_mb=512
        )

        # Create large dataset
        logger.info("Creating large dataset for performance testing...")
        create_diverse_dataset(buffer, num_time_periods=6, games_per_period=15)

        # Test balanced sampling performance
        logger.info("Testing balanced sampling performance...")
        start_time = time.time()

        total_sampled = 0
        for _ in range(100):  # 100 batch samples
            batch = buffer.sample_balanced_batch(batch_size=64)
            total_sampled += len(batch)

        sampling_time = time.time() - start_time
        samples_per_sec = total_sampled / sampling_time

        logger.info(f"Balanced sampling: {total_sampled} examples in {sampling_time:.3f}s "
                   f"({samples_per_sec:.0f} samples/sec)")

        # Test iterator performance
        logger.info("Testing iterator performance...")
        iterator = buffer.create_training_iterator(
            batch_size=64,
            shuffle_buffer_size=1000
        )

        start_time = time.time()
        total_iterated = 0

        for i, batch in enumerate(iterator):
            total_iterated += len(batch)
            if i >= 50:  # 50 batches
                break

        iterator_time = time.time() - start_time
        iterator_rate = total_iterated / iterator_time

        logger.info(f"Iterator: {total_iterated} examples in {iterator_time:.3f}s "
                   f"({iterator_rate:.0f} samples/sec)")

        # Performance should be reasonable (balanced sampling has more overhead than simple sampling)
        assert samples_per_sec > 300, f"Balanced sampling too slow: {samples_per_sec:.0f} samples/sec"
        assert iterator_rate > 1000, f"Iterator too slow: {iterator_rate:.0f} samples/sec"

        logger.info("✅ Performance validation passed!")


def create_sampling_visualization(output_dir: Path):
    """Create visualizations of sampling behavior."""
    logger.info("=== Creating Sampling Visualizations ===")

    output_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = Path(temp_dir) / "viz_buffer"

        buffer = MemoryMappedExperienceBuffer(
            buffer_path=buffer_path,
            max_examples=10000,
            cache_size_mb=256
        )

        # Create dataset
        create_diverse_dataset(buffer, num_time_periods=5, games_per_period=12)

        # Sample multiple batches for analysis
        logger.info("Generating samples for visualization...")

        # Test different sampling methods
        equal_batches = []
        custom_batches = []
        uniform_batches = []

        for _ in range(20):
            # Equal distribution
            batch = buffer.sample_balanced_batch(batch_size=60)
            equal_batches.append(Counter(ex.game_type for ex in batch))

            # Custom ratios
            batch = buffer.sample_balanced_batch(
                batch_size=60,
                game_type_ratios={"gomoku": 0.5, "chess": 0.3, "go": 0.2}
            )
            custom_batches.append(Counter(ex.game_type for ex in batch))

            # Temporal uniformity
            batch = buffer.sample_balanced_batch(
                batch_size=60,
                temporal_uniformity=True
            )
            uniform_batches.append(Counter(ex.game_type for ex in batch))

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Equal distribution consistency
        game_types = ["gomoku", "chess", "go"]
        equal_data = {gt: [batch[gt] for batch in equal_batches] for gt in game_types}

        axes[0, 0].boxplot([equal_data[gt] for gt in game_types], labels=game_types)
        axes[0, 0].set_title("Equal Distribution Sampling Consistency")
        axes[0, 0].set_ylabel("Examples per Batch")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Custom ratio consistency
        custom_data = {gt: [batch[gt] for batch in custom_batches] for gt in game_types}

        axes[0, 1].boxplot([custom_data[gt] for gt in game_types], labels=game_types)
        axes[0, 1].set_title("Custom Ratio Sampling (50:30:20)")
        axes[0, 1].set_ylabel("Examples per Batch")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Game type distribution over time
        stats = buffer.get_sampling_stats()
        game_dist = stats['game_type_distribution']

        axes[1, 0].bar(game_dist.keys(), game_dist.values())
        axes[1, 0].set_title("Overall Game Type Distribution")
        axes[1, 0].set_ylabel("Total Examples")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Temporal distribution
        temporal_spread = stats['temporal_spread']

        axes[1, 1].bar(range(len(temporal_spread)), list(temporal_spread.values()))
        axes[1, 1].set_title("Temporal Distribution")
        axes[1, 1].set_xlabel("Time Bucket")
        axes[1, 1].set_ylabel("Examples")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "sampling_analysis.png", dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_dir / 'sampling_analysis.png'}")

        plt.close()


def main():
    """Run all sampling validation tests."""
    logger.info("Starting experience replay sampling validation...")

    try:
        validate_balanced_sampling()
        validate_temporal_uniformity()
        validate_training_iterator()
        validate_sampling_statistics()
        validate_performance()

        # Create visualizations if matplotlib is available
        try:
            output_dir = Path("results/sampling_validation")
            create_sampling_visualization(output_dir)
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")

        logger.info("🎉 All sampling validation tests passed!")

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()