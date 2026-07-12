"""
Memory-mapped experience buffer implementation with Parquet storage and LRU caching.

Provides efficient storage and sampling of training examples from self-play games.
Optimized for large-scale training data with memory-mapped access patterns.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import mmap
import os
import pickle
import random
import time
from collections import OrderedDict, defaultdict
import logging
from threading import Lock
import math

import sys
from pathlib import Path
# Add specs directory to path to import contracts
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specs" / "001-goal-create-spec"))

from contracts.training_api import (
    ExperienceBuffer, TrainingExample, GameResult
)

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache for training examples."""

    def __init__(self, max_size_mb: int):
        self.max_size_mb = max_size_mb
        self.max_entries = max_size_mb * 1024 * 1024 // 4000  # ~4KB per example estimate
        self.cache = OrderedDict()
        self.lock = Lock()

    def get(self, key: str) -> Optional[TrainingExample]:
        """Get example from cache, moving to end if found."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key: str, value: TrainingExample) -> None:
        """Add example to cache, evicting LRU if needed."""
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.cache.pop(key)
            elif len(self.cache) >= self.max_entries:
                # Evict least recently used
                self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_entries,
                'utilization': len(self.cache) / self.max_entries if self.max_entries > 0 else 0.0
            }


class MemoryMappedExperienceBuffer(ExperienceBuffer):
    """Memory-mapped experience buffer with Parquet storage and LRU caching."""

    def __init__(self,
                 buffer_path: Path,
                 max_examples: int = 1_000_000,
                 cache_size_mb: int = 512):
        """Initialize memory-mapped experience buffer.

        Args:
            buffer_path: Directory for memory-mapped storage
            max_examples: Maximum training examples to store
            cache_size_mb: RAM cache size in megabytes
        """
        self.buffer_path = Path(buffer_path)
        self.max_examples = max_examples
        self.cache_size_mb = cache_size_mb

        # Create buffer directory
        self.buffer_path.mkdir(parents=True, exist_ok=True)

        # Initialize data structures
        self._parquet_file = self.buffer_path / "examples.parquet"
        self._metadata_file = self.buffer_path / "metadata.pkl"
        self._index_file = self.buffer_path / "index.pkl"

        # LRU cache for frequently accessed examples
        self.cache = LRUCache(cache_size_mb)

        # Thread safety
        self.lock = Lock()

        # Load existing data or initialize
        self._load_or_initialize()

        logger.info(f"ExperienceBuffer initialized: {len(self.index)} examples, "
                   f"cache size: {cache_size_mb}MB")

    def _load_or_initialize(self) -> None:
        """Load existing buffer data or initialize empty buffer."""
        try:
            # Load metadata
            if self._metadata_file.exists():
                with open(self._metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.metadata = {
                    'total_examples': 0,
                    'total_games': 0,
                    'game_type_counts': {},
                    'created_at': time.time(),
                    'last_modified': time.time()
                }

            # Load index
            if self._index_file.exists():
                with open(self._index_file, 'rb') as f:
                    self.index = pickle.load(f)
            else:
                self.index = []  # List of (game_id, example_idx, game_type, file_offset)

        except Exception as e:
            logger.warning(f"Failed to load existing buffer data: {e}. Initializing empty buffer.")
            self.metadata = {
                'total_examples': 0,
                'total_games': 0,
                'game_type_counts': {},
                'created_at': time.time(),
                'last_modified': time.time()
            }
            self.index = []

    def __len__(self) -> int:
        """Return number of examples in buffer."""
        with self.lock:
            return len(self.index)

    def add(self, example: TrainingExample) -> None:
        """Add single training example to buffer.

        Args:
            example: Training example to add
        """
        # Create a single-game result to use existing add_games method
        game_result = GameResult(
            winner=1 if example.value > 0 else (0 if example.value == 0 else -1),
            move_count=example.move_number + 1,
            game_length_seconds=1.0,
            examples=[example],
            final_board="",
            metadata={"game_type": example.game_type}
        )
        self.add_games([game_result])

    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample random batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            List of training examples
        """
        return self.sample_batch(batch_size)

    def _save_metadata(self) -> None:
        """Save metadata and index to disk."""
        self.metadata['last_modified'] = time.time()

        with open(self._metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        with open(self._index_file, 'wb') as f:
            pickle.dump(self.index, f)

    def add_games(self, games: List[GameResult]) -> None:
        """Add games to experience buffer.

        Args:
            games: List of completed self-play games
        """
        if not games:
            return

        with self.lock:
            # Convert games to training examples
            new_examples = []
            for game in games:
                game_type = game.examples[0].game_type if game.examples else "unknown"

                for example in game.examples:
                    # Create row data for Parquet
                    example_data = {
                        'game_id': example.game_id,
                        'game_type': example.game_type,
                        'move_number': example.move_number,
                        'value': example.value,
                        'state_data': example.state.tobytes(),  # Serialize numpy array
                        'state_shape': list(example.state.shape),
                        'policy_data': example.policy.tobytes(),  # Serialize numpy array
                        'policy_shape': list(example.policy.shape),
                        'timestamp': time.time()
                    }
                    new_examples.append(example_data)

                # Update game type counts
                self.metadata['game_type_counts'][game_type] = (
                    self.metadata['game_type_counts'].get(game_type, 0) + 1
                )

            if not new_examples:
                return

            # Convert to DataFrame for Parquet
            df = pd.DataFrame(new_examples)

            # Append to Parquet file
            table = pa.Table.from_pandas(df)

            if self._parquet_file.exists():
                # Read existing table and concatenate
                existing_table = pq.read_table(self._parquet_file)
                combined_table = pa.concat_tables([existing_table, table])
            else:
                combined_table = table

            # Handle buffer size limit
            if len(combined_table) > self.max_examples:
                # Keep only the most recent examples
                start_idx = len(combined_table) - self.max_examples
                combined_table = combined_table.slice(start_idx)

                # Clear cache as indices have changed
                self.cache.clear()
                logger.info(f"Buffer size limit reached. Keeping last {self.max_examples} examples.")

            # Write back to Parquet file
            pq.write_table(combined_table, self._parquet_file)

            # Update index
            start_offset = len(self.index)
            for i, example_data in enumerate(new_examples):
                if start_offset + i < self.max_examples:  # Only add if within limit
                    self.index.append((
                        example_data['game_id'],
                        example_data['move_number'],
                        example_data['game_type'],
                        start_offset + i  # File offset in Parquet table
                    ))

            # Trim index if needed
            if len(self.index) > self.max_examples:
                self.index = self.index[-self.max_examples:]

            # Update metadata
            self.metadata['total_examples'] = len(self.index)
            self.metadata['total_games'] += len(games)

            # Save metadata
            self._save_metadata()

            logger.info(f"Added {len(new_examples)} examples from {len(games)} games. "
                       f"Total: {len(self.index)} examples")

    def sample_batch(self,
                    batch_size: int,
                    game_types: Optional[List[str]] = None) -> List[TrainingExample]:
        """Sample training batch from buffer.

        Args:
            batch_size: Number of examples to sample
            game_types: Restrict to specific game types (None = all)

        Returns:
            List of training examples
        """
        with self.lock:
            if not self.index:
                return []

            # Filter indices by game type if specified
            if game_types:
                filtered_indices = [
                    (i, entry) for i, entry in enumerate(self.index)
                    if entry[2] in game_types  # entry[2] is game_type
                ]
            else:
                filtered_indices = list(enumerate(self.index))

            if not filtered_indices:
                return []

            # Sample random indices
            sample_size = min(batch_size, len(filtered_indices))
            sampled_indices = random.sample(filtered_indices, sample_size)

            # Load examples
            examples = []
            cache_hits = 0

            for idx, entry in sampled_indices:
                game_id, move_number, game_type, file_offset = entry
                cache_key = f"{game_id}_{move_number}"

                # Try cache first
                cached_example = self.cache.get(cache_key)
                if cached_example:
                    examples.append(cached_example)
                    cache_hits += 1
                    continue

                # Load from Parquet file
                try:
                    # Read single row from Parquet
                    table = pq.read_table(self._parquet_file)
                    if file_offset >= len(table):
                        logger.warning(f"File offset {file_offset} out of bounds for table length {len(table)}")
                        continue

                    row = table.slice(file_offset, 1).to_pandas().iloc[0]

                    # Reconstruct numpy arrays
                    state = np.frombuffer(row['state_data'], dtype=np.float32).reshape(row['state_shape'])
                    policy = np.frombuffer(row['policy_data'], dtype=np.float32).reshape(row['policy_shape'])

                    # Create TrainingExample
                    example = TrainingExample(
                        state=state,
                        policy=policy,
                        value=row['value'],
                        game_type=row['game_type'],
                        move_number=row['move_number'],
                        game_id=row['game_id']
                    )

                    # Cache the example
                    self.cache.put(cache_key, example)
                    examples.append(example)

                except Exception as e:
                    logger.warning(f"Failed to load example at offset {file_offset}: {e}")
                    continue

            logger.debug(f"Sampled {len(examples)} examples (cache hits: {cache_hits}/{sample_size})")
            return examples

    def sample_balanced_batch(self,
                            batch_size: int,
                            game_type_ratios: Optional[Dict[str, float]] = None,
                            temporal_uniformity: bool = True) -> List[TrainingExample]:
        """Sample training batch with balanced game type distribution.

        Args:
            batch_size: Number of examples to sample
            game_type_ratios: Desired ratios for each game type (None = equal distribution)
            temporal_uniformity: Whether to sample uniformly across time periods

        Returns:
            List of training examples with balanced distribution
        """
        with self.lock:
            if not self.index:
                return []

            # Get available game types
            available_games = set(entry[2] for entry in self.index)
            if not available_games:
                return []

            # Determine target ratios
            if game_type_ratios is None:
                # Equal distribution across all available game types
                target_ratios = {game_type: 1.0 / len(available_games)
                               for game_type in available_games}
            else:
                # Normalize provided ratios
                total_ratio = sum(game_type_ratios.values())
                target_ratios = {game_type: ratio / total_ratio
                               for game_type, ratio in game_type_ratios.items()
                               if game_type in available_games}

            # Group indices by game type
            game_type_indices = defaultdict(list)
            for i, entry in enumerate(self.index):
                game_type = entry[2]
                if game_type in target_ratios:
                    game_type_indices[game_type].append((i, entry))

            # Calculate samples per game type
            samples_per_type = {}
            total_allocated = 0

            for game_type, ratio in target_ratios.items():
                if game_type in game_type_indices:
                    target_count = max(1, int(batch_size * ratio))
                    available_count = len(game_type_indices[game_type])
                    actual_count = min(target_count, available_count)
                    samples_per_type[game_type] = actual_count
                    total_allocated += actual_count

            # If we allocated fewer than requested, distribute remainder
            if total_allocated < batch_size:
                remaining = batch_size - total_allocated
                game_types_with_capacity = [
                    gt for gt in game_type_indices.keys()
                    if len(game_type_indices[gt]) > samples_per_type.get(gt, 0)
                ]

                for _ in range(remaining):
                    if not game_types_with_capacity:
                        break

                    # Add one to the game type with the largest capacity
                    best_game_type = max(game_types_with_capacity,
                                       key=lambda gt: len(game_type_indices[gt]) - samples_per_type.get(gt, 0))
                    samples_per_type[best_game_type] += 1

                    # Remove from capacity list if at max
                    if len(game_type_indices[best_game_type]) <= samples_per_type[best_game_type]:
                        game_types_with_capacity.remove(best_game_type)

            # Sample from each game type
            sampled_indices = []
            for game_type, sample_count in samples_per_type.items():
                if sample_count == 0:
                    continue

                available_indices = game_type_indices[game_type]

                if temporal_uniformity:
                    # Sample uniformly across time periods
                    sampled = self._sample_temporally_uniform(available_indices, sample_count)
                else:
                    # Simple random sampling
                    sampled = random.sample(available_indices, sample_count)

                sampled_indices.extend(sampled)

            # Load examples (reuse existing loading logic)
            examples = []
            cache_hits = 0

            for idx, entry in sampled_indices:
                game_id, move_number, game_type, file_offset = entry
                cache_key = f"{game_id}_{move_number}"

                # Try cache first
                cached_example = self.cache.get(cache_key)
                if cached_example:
                    examples.append(cached_example)
                    cache_hits += 1
                    continue

                # Load from Parquet file
                try:
                    table = pq.read_table(self._parquet_file)
                    if file_offset >= len(table):
                        logger.warning(f"File offset {file_offset} out of bounds for table length {len(table)}")
                        continue

                    row = table.slice(file_offset, 1).to_pandas().iloc[0]

                    # Reconstruct numpy arrays
                    state = np.frombuffer(row['state_data'], dtype=np.float32).reshape(row['state_shape'])
                    policy = np.frombuffer(row['policy_data'], dtype=np.float32).reshape(row['policy_shape'])

                    # Create TrainingExample
                    example = TrainingExample(
                        state=state,
                        policy=policy,
                        value=row['value'],
                        game_type=row['game_type'],
                        move_number=row['move_number'],
                        game_id=row['game_id']
                    )

                    # Cache the example
                    self.cache.put(cache_key, example)
                    examples.append(example)

                except Exception as e:
                    logger.warning(f"Failed to load example at offset {file_offset}: {e}")
                    continue

            logger.debug(f"Sampled balanced batch: {len(examples)} examples "
                        f"(cache hits: {cache_hits}/{len(sampled_indices)})")

            # Log distribution for debugging
            if logger.isEnabledFor(logging.DEBUG):
                actual_distribution = defaultdict(int)
                for example in examples:
                    actual_distribution[example.game_type] += 1
                logger.debug(f"Batch distribution: {dict(actual_distribution)}")

            return examples

    def _sample_temporally_uniform(self,
                                 available_indices: List[Tuple[int, Tuple]],
                                 sample_count: int) -> List[Tuple[int, Tuple]]:
        """Sample uniformly across temporal periods to avoid recency bias.

        Args:
            available_indices: List of (index, entry) tuples
            sample_count: Number of samples to draw

        Returns:
            List of sampled (index, entry) tuples
        """
        if sample_count >= len(available_indices):
            return available_indices

        if sample_count == 0:
            return []

        # Sort by file offset (which correlates with time)
        sorted_indices = sorted(available_indices, key=lambda x: x[1][3])  # x[1][3] is file_offset

        # Divide into temporal buckets
        num_buckets = min(sample_count * 2, len(sorted_indices))  # At least as many buckets as samples
        bucket_size = len(sorted_indices) / num_buckets

        sampled = []
        samples_per_bucket = max(1, sample_count // num_buckets)
        remaining_samples = sample_count

        for bucket_idx in range(num_buckets):
            if remaining_samples <= 0:
                break

            # Define bucket boundaries
            start_idx = int(bucket_idx * bucket_size)
            end_idx = int((bucket_idx + 1) * bucket_size)
            if bucket_idx == num_buckets - 1:  # Last bucket gets remainder
                end_idx = len(sorted_indices)

            bucket_items = sorted_indices[start_idx:end_idx]
            if not bucket_items:
                continue

            # Sample from this bucket
            bucket_samples = min(samples_per_bucket, len(bucket_items), remaining_samples)
            if bucket_samples > 0:
                bucket_sampled = random.sample(bucket_items, bucket_samples)
                sampled.extend(bucket_sampled)
                remaining_samples -= bucket_samples

        # If we still need more samples, randomly fill from remaining
        if remaining_samples > 0:
            already_sampled = set(sampled)
            remaining_candidates = [idx_entry for idx_entry in sorted_indices
                                  if idx_entry not in already_sampled]
            if remaining_candidates:
                additional_samples = min(remaining_samples, len(remaining_candidates))
                additional = random.sample(remaining_candidates, additional_samples)
                sampled.extend(additional)

        return sampled

    def create_training_iterator(self,
                               batch_size: int,
                               game_type_ratios: Optional[Dict[str, float]] = None,
                               shuffle_buffer_size: int = 10000,
                               temporal_uniformity: bool = True) -> Iterator[List[TrainingExample]]:
        """Create an iterator for continuous training batch generation.

        Args:
            batch_size: Size of each training batch
            game_type_ratios: Desired ratios for each game type
            shuffle_buffer_size: Size of shuffle buffer for randomization
            temporal_uniformity: Whether to maintain temporal uniformity

        Yields:
            Training batches with balanced distribution
        """
        shuffle_buffer = []

        while True:
            # Fill shuffle buffer if needed
            while len(shuffle_buffer) < shuffle_buffer_size:
                # Sample a large batch to fill buffer
                buffer_batch_size = min(batch_size * 4, shuffle_buffer_size - len(shuffle_buffer))
                if buffer_batch_size <= 0:
                    break

                new_examples = self.sample_balanced_batch(
                    batch_size=buffer_batch_size,
                    game_type_ratios=game_type_ratios,
                    temporal_uniformity=temporal_uniformity
                )

                if not new_examples:
                    break  # No more examples available

                shuffle_buffer.extend(new_examples)
                random.shuffle(shuffle_buffer)  # Maintain randomness

            # Yield batches from shuffle buffer
            if len(shuffle_buffer) >= batch_size:
                batch = shuffle_buffer[:batch_size]
                shuffle_buffer = shuffle_buffer[batch_size:]
                yield batch
            else:
                # Not enough examples left, yield what we have if any
                if shuffle_buffer:
                    yield shuffle_buffer
                    shuffle_buffer.clear()
                break

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about sampling capability.

        Returns:
            dict: Statistics including game type distribution, temporal spread, etc.
        """
        with self.lock:
            if not self.index:
                return {
                    'total_examples': 0,
                    'game_type_distribution': {},
                    'game_type_percentages': {},
                    'temporal_spread': {},
                    'temporal_range': {'min_offset': 0, 'max_offset': 0, 'range': 0},
                    'sampling_capability': {
                        'max_balanced_batch_size': 0,
                        'min_examples_per_type': 0,
                        'max_examples_per_type': 0
                    }
                }

            # Game type distribution
            game_type_counts = defaultdict(int)
            for entry in self.index:
                game_type_counts[entry[2]] += 1

            # Temporal analysis (based on file offsets as proxy for time)
            file_offsets = [entry[3] for entry in self.index]
            min_offset, max_offset = min(file_offsets), max(file_offsets)

            # Divide into temporal buckets for analysis
            num_buckets = 10
            bucket_size = (max_offset - min_offset) / num_buckets if max_offset > min_offset else 1
            temporal_distribution = defaultdict(int)

            for offset in file_offsets:
                bucket = int((offset - min_offset) / bucket_size)
                bucket = min(bucket, num_buckets - 1)  # Ensure last bucket gets all remainder
                temporal_distribution[f"bucket_{bucket}"] += 1

            return {
                'total_examples': len(self.index),
                'game_type_distribution': dict(game_type_counts),
                'game_type_percentages': {
                    game_type: (count / len(self.index)) * 100
                    for game_type, count in game_type_counts.items()
                },
                'temporal_spread': dict(temporal_distribution),
                'temporal_range': {
                    'min_offset': min_offset,
                    'max_offset': max_offset,
                    'range': max_offset - min_offset
                },
                'sampling_capability': {
                    'max_balanced_batch_size': min(game_type_counts.values()) * len(game_type_counts) if game_type_counts else 0,
                    'min_examples_per_type': min(game_type_counts.values()) if game_type_counts else 0,
                    'max_examples_per_type': max(game_type_counts.values()) if game_type_counts else 0
                }
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            dict: Stats including size, distribution, memory usage
        """
        with self.lock:
            # Calculate storage size
            storage_size_mb = 0
            if self._parquet_file.exists():
                storage_size_mb = self._parquet_file.stat().st_size / (1024 * 1024)

            cache_stats = self.cache.stats()

            return {
                'total_examples': len(self.index),
                'total_games': self.metadata.get('total_games', 0),
                'game_type_distribution': dict(self.metadata.get('game_type_counts', {})),
                'storage_size_mb': round(storage_size_mb, 2),
                'cache_stats': cache_stats,
                'buffer_utilization': len(self.index) / self.max_examples,
                'created_at': self.metadata.get('created_at'),
                'last_modified': self.metadata.get('last_modified')
            }

    def cleanup(self, keep_last_n: int = 100_000) -> None:
        """Remove old examples to manage storage.

        Args:
            keep_last_n: Number of most recent examples to retain
        """
        with self.lock:
            if len(self.index) <= keep_last_n:
                logger.info(f"Buffer has {len(self.index)} examples, no cleanup needed (keeping {keep_last_n})")
                return

            logger.info(f"Cleaning up buffer: keeping last {keep_last_n} of {len(self.index)} examples")

            # Read current Parquet table
            if not self._parquet_file.exists():
                return

            table = pq.read_table(self._parquet_file)

            # Keep only the last keep_last_n examples
            if len(table) > keep_last_n:
                start_idx = len(table) - keep_last_n
                trimmed_table = table.slice(start_idx)

                # Write back to file
                pq.write_table(trimmed_table, self._parquet_file)

                # Update index
                self.index = self.index[-keep_last_n:]

                # Update metadata
                self.metadata['total_examples'] = len(self.index)
                self._save_metadata()

                # Clear cache as indices have changed
                self.cache.clear()

                logger.info(f"Cleanup complete: {len(self.index)} examples remaining")


def create_experience_buffer(buffer_path: Path,
                           max_examples: int = 1_000_000,
                           cache_size_mb: int = 512) -> ExperienceBuffer:
    """Factory function to create experience buffer.

    Args:
        buffer_path: Directory for buffer storage
        max_examples: Maximum examples to store
        cache_size_mb: RAM cache size in megabytes

    Returns:
        ExperienceBuffer instance
    """
    return MemoryMappedExperienceBuffer(
        buffer_path=buffer_path,
        max_examples=max_examples,
        cache_size_mb=cache_size_mb
    )