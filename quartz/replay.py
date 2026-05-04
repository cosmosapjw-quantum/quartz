"""Replay buffer and sparse policy utilities.

This module intentionally stays independent from the training loop so replay
storage, serialization, and batching can be tested and optimized in isolation.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
from dataclasses import dataclass

import numpy as np


class _RingBuffer:
    """Bounded FIFO with O(1) random-index access.

    Drop-in replacement for `collections.deque(maxlen=capacity)` for the
    operations replay actually uses (append, len, getitem, iter, maxlen
    attribute). `deque` is doubly-linked block-of-blocks: middle indexing
    is O(min(i, n-i)), which dominated `ReplayBuffer.sample` (358us per
    256-sample batch in profile). A pre-allocated cyclic list with a
    head pointer makes `__getitem__` a single modulo + array fetch.
    """

    __slots__ = ("_cap", "_buf", "_size", "_head")

    def __init__(self, capacity):
        self._cap = int(capacity)
        self._buf = [None] * self._cap
        self._size = 0
        self._head = 0

    @property
    def maxlen(self):
        return self._cap

    def append(self, item):
        if self._size < self._cap:
            self._buf[(self._head + self._size) % self._cap] = item
            self._size += 1
        else:
            self._buf[self._head] = item
            self._head = (self._head + 1) % self._cap

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._size))]
        if idx < 0:
            idx += self._size
        if not 0 <= idx < self._size:
            raise IndexError(idx)
        return self._buf[(self._head + idx) % self._cap]

    def __iter__(self):
        head, cap, size = self._head, self._cap, self._size
        buf = self._buf
        for i in range(size):
            yield buf[(head + i) % cap]

_TORCH_MODULE = None
_DATA_LOADER_CLS = None


def _torch_module():
    global _TORCH_MODULE
    if _TORCH_MODULE is None:
        import torch

        _TORCH_MODULE = torch
    return _TORCH_MODULE


def _data_loader_cls():
    global _DATA_LOADER_CLS
    if _DATA_LOADER_CLS is None:
        from torch.utils.data import DataLoader

        _DATA_LOADER_CLS = DataLoader
    return _DATA_LOADER_CLS


def iter_sparse_policy_entries(entries):
    for entry in entries or ():
        if isinstance(entry, str) and ":" in entry:
            idx_raw, val_raw = entry.split(":", 1)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            idx_raw, val_raw = entry[0], entry[1]
        else:
            continue
        try:
            idx = int(idx_raw)
            val = float(val_raw)
        except (TypeError, ValueError):
            continue
        yield idx, val


def dense_policy_from_sparse(entries, n_actions):
    policy = np.zeros(n_actions, dtype=np.float32)
    for idx, val in iter_sparse_policy_entries(entries):
        if 0 <= idx < n_actions:
            policy[idx] = val
    return policy


@dataclass
class SparsePolicyTarget:
    idx: np.ndarray
    val: np.ndarray
    n_actions: int

    def copy(self):
        return SparsePolicyTarget(
            idx=np.asarray(self.idx, dtype=np.int32).copy(),
            val=np.asarray(self.val, dtype=np.float32).copy(),
            n_actions=int(self.n_actions),
        )

    def __getitem__(self, key):
        return self.dense()[key]

    def __array__(self, dtype=None):
        dense = self.dense()
        if dtype is None:
            return dense
        return dense.astype(dtype, copy=False)

    def dense(self):
        policy = np.zeros(int(self.n_actions), dtype=np.float32)
        if self.idx.size:
            policy[self.idx] = self.val
        return policy

    def entropy(self):
        p = self.val[self.val > 1e-8]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))


@dataclass
class ReplayExample:
    state: np.ndarray
    policy: SparsePolicyTarget
    value: float
    metadata: dict | None = None

    def __getitem__(self, idx):
        if idx == 0:
            return self.state
        if idx == 1:
            return self.policy.dense()
        if idx == 2:
            return self.value
        raise IndexError(idx)


def sparse_policy_from_dense(policy):
    dense = np.asarray(policy, dtype=np.float32).reshape(-1)
    # `dense[idx]` already returns a fresh contiguous float32 array (fancy
    # indexing copies); the prior code chained `.astype(np.float32,
    # copy=True)` which forced a redundant second copy. flatnonzero already
    # yields int64 contiguous; one astype is enough.
    idx = np.flatnonzero(np.abs(dense) > 1e-12).astype(np.int32, copy=False)
    val = dense[idx]
    return SparsePolicyTarget(
        idx=idx,
        val=val,
        n_actions=int(dense.size),
    )


def sparse_policy_from_entries(entries, n_actions):
    idxs = []
    vals = []
    for idx, val in iter_sparse_policy_entries(entries):
        if 0 <= idx < int(n_actions) and abs(float(val)) > 1e-12:
            idxs.append(int(idx))
            vals.append(float(val))
    return SparsePolicyTarget(
        idx=np.asarray(idxs, dtype=np.int32),
        val=np.asarray(vals, dtype=np.float32),
        n_actions=int(n_actions),
    )


def normalize_sparse_policy(policy, n_actions=None):
    if isinstance(policy, SparsePolicyTarget):
        return SparsePolicyTarget(
            idx=np.asarray(policy.idx, dtype=np.int32),
            val=np.asarray(policy.val, dtype=np.float32),
            n_actions=int(policy.n_actions if n_actions is None else n_actions),
        )
    if isinstance(policy, np.ndarray):
        target = sparse_policy_from_dense(policy)
        if n_actions is not None and int(n_actions) != target.n_actions:
            raise ValueError("dense policy size mismatch")
        return target
    if n_actions is None:
        if isinstance(policy, (list, tuple)) and policy and not isinstance(policy[0], (list, tuple, str)):
            return sparse_policy_from_dense(policy)
        raise ValueError("n_actions is required for sparse policy entries")
    return sparse_policy_from_entries(policy, n_actions)


class ReplayBuffer:
    def __init__(self, capacity, recent_fraction=0.0, recent_window=0):
        self.buf = _RingBuffer(capacity)
        self._lock = threading.Lock()
        self.recent_fraction = float(max(0.0, min(1.0, recent_fraction)))
        self.recent_window = int(max(0, recent_window))

    def _make_example(self, state, policy, value, metadata=None):
        sparse_policy = normalize_sparse_policy(policy)
        return ReplayExample(
            state=np.asarray(state, dtype=np.float32),
            policy=sparse_policy,
            value=float(value),
            metadata=dict(metadata or {}),
        )

    def _make_sparse_example(self, state, policy_entries, value, n_actions, metadata=None):
        sparse_policy = normalize_sparse_policy(policy_entries, n_actions=n_actions)
        return ReplayExample(
            state=np.asarray(state, dtype=np.float32),
            policy=sparse_policy,
            value=float(value),
            metadata=dict(metadata or {}),
        )

    def add(self, state, policy, value, metadata=None):
        with self._lock:
            self.buf.append(self._make_example(state, policy, value, metadata=metadata))

    def add_sparse(self, state, policy_entries, value, n_actions, metadata=None):
        with self._lock:
            self.buf.append(
                self._make_sparse_example(state, policy_entries, value, n_actions, metadata=metadata)
            )

    @staticmethod
    def _stamp_actor_generation(metadata, actor_generation, actor_id=None):
        # Always return a dict so downstream ReplayExample stores a concrete
        # metadata object rather than None. When actor_generation is not
        # supplied we leave the field out rather than stamping 0, so readers
        # can tell "tagged with generation 0" from "not tagged at all".
        base = dict(metadata or {})
        if actor_generation is not None:
            base["actor_generation"] = int(actor_generation)
        if actor_id is not None:
            base["actor_id"] = str(actor_id)
        return base

    def add_game(self, states, policies, outcome, start_player=1, traces=None, actor_generation=None, actor_id=None):
        """Add a full game using side-to-move values at each ply.

        `actor_generation` and `actor_id`, when provided, are stamped into
        every per-sample metadata dict so replay samples remain traceable to
        the immutable actor snapshot that produced them.
        """
        with self._lock:
            for i, (state, policy) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                value = outcome * player_to_move
                metadata = traces[i] if traces is not None and i < len(traces) else None
                metadata = self._stamp_actor_generation(metadata, actor_generation, actor_id=actor_id)
                self.buf.append(self._make_example(state, policy, value, metadata=metadata))

    def add_sparse_game(self, states, policies, outcome, n_actions, start_player=1, traces=None, actor_generation=None, actor_id=None):
        with self._lock:
            for i, (state, policy) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                value = outcome * player_to_move
                metadata = traces[i] if traces is not None and i < len(traces) else None
                metadata = self._stamp_actor_generation(metadata, actor_generation, actor_id=actor_id)
                self.buf.append(
                    self._make_sparse_example(state, policy, value, n_actions, metadata=metadata)
                )

    def _sample_indices_from_size(self, total_size, n):
        sample_n = min(n, total_size)
        if sample_n <= 0:
            return []
        if self.recent_fraction <= 0.0 or self.recent_window <= 0:
            return random.sample(range(total_size), sample_n)

        recent_start = max(0, total_size - min(self.recent_window, total_size))
        recent_indices = range(recent_start, total_size)
        older_indices = range(0, recent_start)

        recent_target = min(sample_n, math.ceil(sample_n * self.recent_fraction))
        n_recent = min(recent_target, total_size - recent_start)
        n_older = min(sample_n - n_recent, recent_start)

        chosen = []
        if n_recent > 0:
            chosen.extend(random.sample(recent_indices, n_recent))
        if n_older > 0:
            chosen.extend(random.sample(older_indices, n_older))

        if len(chosen) < sample_n:
            chosen_set = set(chosen)
            remaining_pool = [i for i in range(total_size) if i not in chosen_set]
            chosen.extend(random.sample(remaining_pool, sample_n - len(chosen)))
        return chosen

    def _sample_indices_locked(self, n):
        return self._sample_indices_from_size(len(self.buf), n)

    def _sample_examples_locked(self, n):
        indices = self._sample_indices_locked(n)
        return [self.buf[i] for i in indices]

    def _sample_examples_from_snapshot(self, snapshot, n):
        indices = self._sample_indices_from_size(len(snapshot), n)
        return [snapshot[i] for i in indices]

    def sample(self, n):
        with self._lock:
            batch = self._sample_examples_locked(n)
        return collate_replay_samples(batch)

    def build_dataloader(self, batch_size, n_steps, pin_memory=False):
        with self._lock:
            snapshot = tuple(self.buf)
        n_steps = max(0, int(n_steps))
        total_needed = n_steps * batch_size
        if total_needed <= 0 or not snapshot:
            return None
        # [OPT] Vectorized index sampling — single numpy call instead of n_steps Python loops
        snap_len = len(snapshot)
        if self.recent_fraction <= 0.0 or self.recent_window <= 0:
            all_indices = np.random.randint(0, snap_len, size=total_needed)
            examples = [snapshot[i] for i in all_indices]
        else:
            examples = []
            for _ in range(n_steps):
                examples.extend(self._sample_examples_from_snapshot(snapshot, batch_size))
        if not examples:
            return None
        dataset = ReplayDataset(examples)
        return _data_loader_cls()(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_replay_samples,
            drop_last=False,
        )

    def __len__(self):
        return len(self.buf)

    def examples_at_indices(self, indices):
        with self._lock:
            return [self.buf[int(idx)] for idx in indices]

    def save(self, path):
        with self._lock:
            data = list(self.buf)
        policy_ptr = np.zeros(len(data) + 1, dtype=np.int64)
        nnz_total = 0
        for i, example in enumerate(data):
            nnz_total += int(example.policy.idx.size)
            policy_ptr[i + 1] = nnz_total
        policy_idx = np.empty(nnz_total, dtype=np.int32)
        policy_val = np.empty(nnz_total, dtype=np.float32)
        for i, example in enumerate(data):
            start = int(policy_ptr[i])
            end = int(policy_ptr[i + 1])
            policy_idx[start:end] = example.policy.idx
            policy_val[start:end] = example.policy.val
        np.savez_compressed(
            path,
            replay_format=np.array([2], dtype=np.int32),
            states=np.array([d.state for d in data], dtype=np.float32),
            policy_ptr=policy_ptr,
            policy_idx=policy_idx,
            policy_val=policy_val,
            n_actions=np.array([d.policy.n_actions for d in data], dtype=np.int32),
            values=np.array([d.value for d in data], dtype=np.float32),
            metadata_json=np.asarray(
                [
                    json.dumps(d.metadata or {}, separators=(",", ":"), sort_keys=True)
                    for d in data
                ],
                dtype=np.str_,
            ),
        )

    def load(self, path):
        if not os.path.exists(path):
            return 0
        loaded = np.load(path, allow_pickle=False)
        states = loaded["states"]
        values = loaded["values"]
        n = len(states)
        if "policy_ptr" in loaded and "policy_idx" in loaded and "policy_val" in loaded:
            ptr = loaded["policy_ptr"]
            idx = loaded["policy_idx"]
            val = loaded["policy_val"]
            n_actions = loaded["n_actions"] if "n_actions" in loaded else np.zeros(n, dtype=np.int32)
            metadata_json = loaded["metadata_json"] if "metadata_json" in loaded else None
            for i in range(n):
                start = int(ptr[i])
                end = int(ptr[i + 1])
                self.buf.append(
                    ReplayExample(
                        state=np.asarray(states[i], dtype=np.float32),
                        policy=SparsePolicyTarget(
                            idx=np.asarray(idx[start:end], dtype=np.int32),
                            val=np.asarray(val[start:end], dtype=np.float32),
                            n_actions=int(n_actions[i]),
                        ),
                        value=float(values[i]),
                        metadata=(
                            json.loads(str(metadata_json[i]))
                            if metadata_json is not None and str(metadata_json[i]).strip()
                            else {}
                        ),
                    )
                )
        else:
            policies = loaded["policies"]
            for i in range(n):
                self.buf.append(self._make_example(states[i], policies[i], float(values[i])))
        return n


class ReplayDataset:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_replay_samples(batch):
    torch = _torch_module()
    if not batch:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        )
    if all(isinstance(item, ReplayExample) for item in batch):
        examples = batch
    else:
        examples = []
        for item in batch:
            if isinstance(item, ReplayExample):
                examples.append(item)
            else:
                state, policy, value = item
                examples.append(
                    ReplayExample(
                        state=np.asarray(state, dtype=np.float32),
                        policy=normalize_sparse_policy(policy),
                        value=float(value),
                    )
                )
    n = len(examples)

    # States: one numpy stack instead of len(examples) per-row np.asarray
    # writes into a pre-allocated np.empty.
    states_np = np.stack([ex.state for ex in examples])
    if states_np.dtype != np.float32:
        states_np = states_np.astype(np.float32, copy=False)

    # Values: a single fromiter pass (avoids the per-row Python `float()` +
    # numpy scalar boxing of the prior loop).
    values_np = np.fromiter((ex.value for ex in examples), dtype=np.float32, count=n)

    # Policies: replace the per-row sparse→dense scatter Python loop with
    # one concatenated fancy-indexed write. For batch=256 this turns 256
    # `policies_np[row, idx] = val` calls into 3 numpy ops + 1 scatter.
    n_actions = max(int(ex.policy.n_actions) for ex in examples)
    policies_np = np.zeros((n, n_actions), dtype=np.float32)
    nnz_per_row = [int(ex.policy.idx.size) for ex in examples]
    total_nnz = sum(nnz_per_row)
    if total_nnz > 0:
        rows_concat = np.repeat(np.arange(n, dtype=np.int64), nnz_per_row)
        cols_concat = np.concatenate([ex.policy.idx for ex in examples])
        vals_concat = np.concatenate([ex.policy.val for ex in examples])
        policies_np[rows_concat, cols_concat] = vals_concat

    states = torch.from_numpy(states_np)
    policies = torch.from_numpy(policies_np)
    values = torch.from_numpy(values_np)
    return states, policies, values


def _percentile(sorted_values, p):
    # Linear interpolation percentile on a pre-sorted list; p in [0, 100].
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _summarize_sample(values):
    if not values:
        return None
    sv = sorted(float(x) for x in values)
    total = sum(sv)
    n = len(sv)
    return {
        "n": n,
        "min": float(sv[0]),
        "max": float(sv[-1]),
        "mean": float(total / n),
        "p50": _percentile(sv, 50.0),
        "p95": _percentile(sv, 95.0),
    }


def _finalize_halt_trace(per_mode):
    """Turn the per-penalty-mode accumulator into a disk-serializable block.

    Emits, per penalty mode actually observed, the halt-step distribution
    (root visits at halt), the p_flip distribution at halt, and the stop-reason
    counts. Ablation readers can use this to verify budget-fairness across
    modes — e.g., if Legacy stops at mean 400 visits and PFlipMixture at
    mean 180, the "same budget" claim is broken for that run.
    """
    out = {}
    for mode_key, bucket in per_mode.items():
        if bucket["moves"] <= 0:
            continue
        out[str(mode_key)] = {
            "moves": int(bucket["moves"]),
            "root_visits": _summarize_sample(bucket["root_visits"]),
            "p_flip_at_halt": _summarize_sample(bucket["p_flip_at_halt"]),
            "stop_reasons": dict(bucket["stop_reasons"]),
        }
    return out


class ReplayMetrics:
    """Track replay buffer health metrics."""

    @staticmethod
    def _examples_at_indices(replay, indices):
        if not indices:
            return []
        getter = getattr(replay, "examples_at_indices", None)
        if callable(getter):
            try:
                return list(getter(indices))
            except Exception:
                return []
        buf = getattr(replay, "buf", None)
        if buf is not None:
            try:
                return [buf[int(idx)] for idx in indices]
            except Exception:
                return []
        try:
            return [replay[int(idx)] for idx in indices]
        except Exception:
            return []

    @staticmethod
    def freshness(n_new, replay_size):
        return n_new / max(replay_size, 1)

    @staticmethod
    def freshness_summary(replay, current_generation, sample_n=200):
        """P03: per-replay-buffer age statistics relative to the current
        learner generation, with an exponential-decay freshness score.

        The legacy `freshness(n_new, replay_size)` (above) returned
        `n_new / replay_size` — a turnover rate, not a freshness. Two
        pipelines with the same turnover but very different `mean_age`
        (e.g. 5k-replay × 50/iter vs. 50k-replay × 500/iter) get the
        same legacy number, masking off-policy drift in the larger one.

        This summary instead returns a true monotone-in-mean-age score:

            half_life_gen   = max(1, capacity / 100)   # ~100 pos/game heuristic
            mean_age        = current_generation - sample_mean(actor_generation)
            freshness_score = exp(-mean_age / half_life_gen)

        Range: (0, 1]; 1.0 = freshly produced this generation; 0.5 = mean
        sample is one half-life old; below 0.1 = average sample is so
        off-policy that gradient direction is dominated by stale
        on-policy correlations.

        Returns dict with `schema_version: 1`, `oldest_gen`, `newest_gen`,
        `mean_age`, `freshness_score`, `sample_count`, `half_life_gen`.
        Empty replay → freshness_score=0.0 and all gen fields None.
        Rows without `actor_generation` metadata are silently skipped;
        if NO sampled row has the tag, returns the same empty shape.
        """
        if len(replay) <= 0:
            return {
                "schema_version": 1,
                "oldest_gen": None,
                "newest_gen": None,
                "mean_age": None,
                "freshness_score": 0.0,
                "sample_count": 0,
                "half_life_gen": None,
            }
        n = min(int(sample_n), len(replay))
        if n <= 0:
            return {
                "schema_version": 1,
                "oldest_gen": None,
                "newest_gen": None,
                "mean_age": None,
                "freshness_score": 0.0,
                "sample_count": 0,
                "half_life_gen": None,
            }
        indices = random.sample(range(len(replay)), n)
        examples = ReplayMetrics._examples_at_indices(replay, indices)
        gens = []
        for ex in examples:
            if isinstance(ex, ReplayExample) and ex.metadata:
                gen = ex.metadata.get("actor_generation")
                if gen is not None:
                    gens.append(int(gen))
        if not gens:
            return {
                "schema_version": 1,
                "oldest_gen": None,
                "newest_gen": None,
                "mean_age": None,
                "freshness_score": 0.0,
                "sample_count": 0,
                "half_life_gen": None,
            }
        oldest, newest = min(gens), max(gens)
        mean_gen = sum(gens) / len(gens)
        mean_age = float(current_generation) - mean_gen
        # Capacity-defined half-life. Assumes ~100 positions per game; if
        # the caller wants a tighter half-life they can divide capacity
        # by their actual positions-per-game ratio.
        capacity = getattr(replay.buf, "maxlen", len(replay)) or len(replay)
        half_life_gen = max(1.0, float(capacity) / 100.0)
        # max(0, mean_age): negative ages occur when a buffer was loaded
        # from disk with stamps from a future generation; clamp to keep
        # freshness in (0, 1].
        freshness = math.exp(-max(0.0, mean_age) / half_life_gen)
        return {
            "schema_version": 1,
            "oldest_gen": int(oldest),
            "newest_gen": int(newest),
            "mean_age": float(mean_age),
            "freshness_score": float(freshness),
            "sample_count": len(gens),
            "half_life_gen": float(half_life_gen),
        }

    @staticmethod
    def actor_generation_histogram(replay, sample_n=200):
        """Count samples per `actor_generation` tag in a random subsample.

        Paired with the `actor_generation` stamped at `add_game()` time, this
        lets a training-log consumer draw the distribution of actor identities
        currently influencing SGD — e.g. "iteration 7 trained on samples from
        generations {5: 12, 6: 60, 7: 128}". Samples without the tag are
        counted under the key `"untagged"`.
        """
        if len(replay) <= 0:
            return {}
        sample_n = min(int(sample_n), len(replay))
        if sample_n <= 0:
            return {}
        indices = random.sample(range(len(replay)), sample_n)
        examples = ReplayMetrics._examples_at_indices(replay, indices)
        hist: dict[str, int] = {}
        for sample in examples:
            gen_key = "untagged"
            if isinstance(sample, ReplayExample):
                gen = sample.metadata.get("actor_generation") if sample.metadata else None
                if gen is not None:
                    gen_key = str(int(gen))
            hist[gen_key] = hist.get(gen_key, 0) + 1
        return hist

    @staticmethod
    def policy_entropy(replay, sample_n=100):
        if len(replay) < sample_n:
            return 0.0
        indices = random.sample(range(len(replay)), sample_n)
        examples = ReplayMetrics._examples_at_indices(replay, indices)
        if len(examples) < sample_n:
            return 0.0
        total_ent = 0.0
        for sample in examples:
            if isinstance(sample, ReplayExample):
                total_ent += sample.policy.entropy()
            else:
                _, policy, _ = sample
                dense = np.array(policy, dtype=np.float32)
                dense = dense[dense > 1e-8]
                if len(dense) > 0:
                    total_ent += -np.sum(dense * np.log(dense))
        return total_ent / sample_n

    @staticmethod
    def value_std(replay, sample_n=100):
        if len(replay) < sample_n:
            return 0.0
        indices = random.sample(range(len(replay)), sample_n)
        examples = ReplayMetrics._examples_at_indices(replay, indices)
        if len(examples) < sample_n:
            return 0.0
        values = [
            sample.value if isinstance(sample, ReplayExample) else sample[2]
            for sample in examples
        ]
        return float(np.std(values))

    @staticmethod
    def search_summary(replay, sample_n=100):
        if len(replay) <= 0:
            return {}
        sample_n = min(int(sample_n), len(replay))
        indices = random.sample(range(len(replay)), sample_n)
        examples = ReplayMetrics._examples_at_indices(replay, indices)
        if not examples:
            return {}
        sample_n = len(examples)
        profile_counts = {}
        benchmark_safe = 0
        realized_iterations = []
        dup_rates = []
        p_flips = []
        halt_reason_hist = {}
        refresh_counts = []
        penalty_sums = []
        selection_root_selects = []
        selection_refresh_selected = []
        selection_penalty_abs_sums = []
        selection_effective_prior_l1_sums = []
        selection_mean_candidate_counts = []
        selection_max_candidate_counts = []
        penalty_mode_counts = {}
        prior_refresh_rates = []
        prior_q_divergences = []
        root_only_shaping_true = 0
        root_only_shaping_seen = 0
        telemetry_partial = 0
        refresh_metric_present = 0
        penalty_metric_present = 0
        selection_trace_present = 0
        halt_metric_present = 0
        # P6 (audit_codex_20260425.md W8): per-sample voc-channel
        # accumulators. Mean / count are surfaced in the summary so
        # ablation readers can see the QUARTZ-channel decomposition
        # alongside p_flip and halt_reason histograms.
        voc_total_samples = []
        voc_focus_samples = []
        voc_expand_samples = []
        voc_merge_samples = []
        # Q3 (audit_codex_20260428.md W'1): per-sample argmax-channel
        # histogram. Lets attribution readers verify that "VOC halt" is
        # not silently driven by a single channel — if this hist is
        # degenerate (one label dominates), the three-channel framing is
        # framing rather than decision-bearing structure.
        voc_argmax_channel_hist: dict[str, int] = {}
        controller_schema_versions: dict[str, int] = {}
        actuator_coverage_seen = 0
        prior_refresh_rate_configured = 0
        prior_refresh_rate_consumed_by_mode = 0
        prior_refresh_rate_inert_for_mode = 0
        prior_refresh_source_counts: dict[str, int] = {}
        # Per-penalty-mode halt-trace accumulators — needed so ablation readers
        # can verify same-budget fairness across penalty modes (see W1/F1 in
        # the audit review: default halt is p_flip-mediated, so mode-specific
        # realized_iterations distributions can reveal budget leakage).
        halt_trace_per_mode: dict[str, dict] = {}
        # P01: aggregators for the new `controller_summary.extended` block
        # (Rust schema_version 6+). The legacy `mean_prior_refresh_rate`
        # above was actually averaging the CONFIGURED rate (cfg field), not
        # a measured fired/eligible ratio. The extended block fixes this:
        # `refresh_active_count` and `refresh_eligible_count` are integer
        # counters incremented per root select. `measured_prior_refresh_rate`
        # below = active / eligible across the whole subsample.
        ext_refresh_active_total = 0
        ext_refresh_eligible_total = 0
        ext_penalty_mode_counts: dict[str, int] = {}
        ext_halt_reason_counts: dict[str, int] = {}
        ext_present_count = 0

        def _trace_bucket(mode_key: str) -> dict:
            bucket = halt_trace_per_mode.get(mode_key)
            if bucket is None:
                bucket = {
                    "moves": 0,
                    "root_visits": [],
                    "p_flip_at_halt": [],
                    "stop_reasons": {},
                }
                halt_trace_per_mode[mode_key] = bucket
            return bucket
        for sample in examples:
            metadata = sample.metadata if isinstance(sample, ReplayExample) else {}
            metadata = metadata or {}
            manifest = metadata.get("search_manifest") or {}
            realized = metadata.get("realized_budget") or {}
            ctrl = metadata.get("controller_summary") or {}
            profile = manifest.get("profile")
            if profile:
                profile_counts[profile] = int(profile_counts.get(profile, 0)) + 1
            if manifest.get("benchmark_safe"):
                benchmark_safe += 1
            if realized.get("realized_iterations") is not None:
                realized_iterations.append(float(realized["realized_iterations"]))
            if ctrl.get("dup_rate") is not None:
                dup_rates.append(float(ctrl["dup_rate"]))
            if ctrl.get("p_flip") is not None:
                p_flips.append(float(ctrl["p_flip"]))
            if ctrl.get("telemetry_partial"):
                telemetry_partial += 1
            if ctrl.get("refresh_count") is not None:
                refresh_counts.append(float(ctrl["refresh_count"]))
                refresh_metric_present += 1
            if ctrl.get("penalty_sum") is not None:
                penalty_sums.append(float(ctrl["penalty_sum"]))
                penalty_metric_present += 1
            selection_trace = ctrl.get("selection_trace") or {}
            if isinstance(selection_trace, dict) and selection_trace.get("root_selects") is not None:
                selection_trace_present += 1
                root_selects = float(selection_trace.get("root_selects") or 0.0)
                selection_root_selects.append(root_selects)
                selection_refresh_selected.append(float(selection_trace.get("refresh_selected_count") or 0.0))
                selection_penalty_abs_sums.append(float(selection_trace.get("selected_penalty_abs_sum") or 0.0))
                selection_effective_prior_l1_sums.append(float(selection_trace.get("selected_effective_prior_l1_sum") or 0.0))
                if selection_trace.get("selected_mean_candidate_count") is not None:
                    selection_mean_candidate_counts.append(float(selection_trace["selected_mean_candidate_count"]))
                if selection_trace.get("selected_max_candidate_count") is not None:
                    selection_max_candidate_counts.append(float(selection_trace["selected_max_candidate_count"]))
            penalty_mode = ctrl.get("penalty_mode")
            if penalty_mode:
                penalty_mode_counts[str(penalty_mode)] = int(penalty_mode_counts.get(str(penalty_mode), 0)) + 1
            if ctrl.get("prior_refresh_rate") is not None:
                prior_refresh_rates.append(float(ctrl["prior_refresh_rate"]))
            if ctrl.get("prior_q_divergence") is not None:
                prior_q_divergences.append(float(ctrl["prior_q_divergence"]))
            # P6: voc-channel accumulators + schema_version census.
            if ctrl.get("voc_total") is not None:
                voc_total_samples.append(float(ctrl["voc_total"]))
            if ctrl.get("voc_focus") is not None:
                voc_focus_samples.append(float(ctrl["voc_focus"]))
            if ctrl.get("voc_expand") is not None:
                voc_expand_samples.append(float(ctrl["voc_expand"]))
            if ctrl.get("voc_merge") is not None:
                voc_merge_samples.append(float(ctrl["voc_merge"]))
            # Q3: argmax-channel aggregation. Two acceptable shapes from
            # the Rust side:
            #   - per-sample scalar:        ctrl["voc_argmax_channel"] = "focus"
            #   - per-sample histogram:     ctrl["voc_argmax_channel_hist"] = {"focus": 3, ...}
            # The mcts_server attaches the per-game histogram; per-position
            # samples may also carry the scalar from a recent halt-check.
            argmax_hist = ctrl.get("voc_argmax_channel_hist")
            if isinstance(argmax_hist, dict) and argmax_hist:
                for label, count in argmax_hist.items():
                    if not label:
                        continue
                    voc_argmax_channel_hist[str(label)] = (
                        int(voc_argmax_channel_hist.get(str(label), 0)) + int(count or 0)
                    )
            else:
                argmax_scalar = ctrl.get("voc_argmax_channel")
                if argmax_scalar:
                    voc_argmax_channel_hist[str(argmax_scalar)] = (
                        int(voc_argmax_channel_hist.get(str(argmax_scalar), 0)) + 1
                    )
            if ctrl.get("schema_version") is not None:
                key = str(ctrl["schema_version"])
                controller_schema_versions[key] = (
                    int(controller_schema_versions.get(key, 0)) + 1
                )
            actuator_coverage = ctrl.get("actuator_coverage") or {}
            if isinstance(actuator_coverage, dict) and actuator_coverage:
                actuator_coverage_seen += 1
                if actuator_coverage.get("prior_refresh_rate_configured"):
                    prior_refresh_rate_configured += 1
                if actuator_coverage.get("prior_refresh_rate_consumed_by_mode"):
                    prior_refresh_rate_consumed_by_mode += 1
                if actuator_coverage.get("prior_refresh_rate_inert_for_mode"):
                    prior_refresh_rate_inert_for_mode += 1
                source = actuator_coverage.get("prior_refresh_source")
                if source:
                    key = str(source)
                    prior_refresh_source_counts[key] = (
                        int(prior_refresh_source_counts.get(key, 0)) + 1
                    )
            if ctrl.get("root_only_shaping") is not None:
                root_only_shaping_seen += 1
                if bool(ctrl.get("root_only_shaping")):
                    root_only_shaping_true += 1
            reason_hist = ctrl.get("halt_reason_hist") or {}
            if isinstance(reason_hist, dict) and reason_hist:
                halt_metric_present += 1
                for reason, count in reason_hist.items():
                    if not reason:
                        continue
                    halt_reason_hist[str(reason)] = int(halt_reason_hist.get(str(reason), 0)) + int(count or 0)
            else:
                stop_reason = ctrl.get("stop_reason") or metadata.get("stop_reason")
                if stop_reason:
                    halt_metric_present += 1
                    halt_reason_hist[str(stop_reason)] = int(halt_reason_hist.get(str(stop_reason), 0)) + 1
            # P01: aggregate the new `extended` block. Each game-row's
            # extended block carries cumulative counters for that game's
            # search; we sum across games to get a study-level rate. Older
            # Rust binaries (schema_version<=5) emit no extended block and
            # are silently skipped via the falsy guard.
            extended = ctrl.get("extended") or {}
            if isinstance(extended, dict) and extended:
                ext_present_count += 1
                if extended.get("refresh_active_count") is not None:
                    ext_refresh_active_total += int(extended["refresh_active_count"] or 0)
                if extended.get("refresh_eligible_count") is not None:
                    ext_refresh_eligible_total += int(extended["refresh_eligible_count"] or 0)
                pm_counts = extended.get("controller_penalty_mode_counts") or {}
                if isinstance(pm_counts, dict):
                    for k, v in pm_counts.items():
                        if not k:
                            continue
                        ext_penalty_mode_counts[str(k)] = (
                            int(ext_penalty_mode_counts.get(str(k), 0)) + int(v or 0)
                        )
                hr_counts = extended.get("halt_reason_count") or {}
                if isinstance(hr_counts, dict):
                    for k, v in hr_counts.items():
                        if not k:
                            continue
                        ext_halt_reason_counts[str(k)] = (
                            int(ext_halt_reason_counts.get(str(k), 0)) + int(v or 0)
                        )
            # Per-penalty-mode halt-trace bookkeeping. This is auditable
            # evidence for the budget-fairness check across controller modes.
            mode_key = str(penalty_mode) if penalty_mode else "unknown"
            realized = metadata.get("realized_budget") or {}
            rv = realized.get("realized_iterations")
            pflip_at_halt = ctrl.get("p_flip")
            stop_reason_tag = ctrl.get("stop_reason") or metadata.get("stop_reason")
            if rv is not None or pflip_at_halt is not None or stop_reason_tag:
                bucket = _trace_bucket(mode_key)
                bucket["moves"] += 1
                if rv is not None:
                    bucket["root_visits"].append(float(rv))
                if pflip_at_halt is not None:
                    bucket["p_flip_at_halt"].append(float(pflip_at_halt))
                if stop_reason_tag:
                    key = str(stop_reason_tag)
                    bucket["stop_reasons"][key] = int(bucket["stop_reasons"].get(key, 0)) + 1
        return {
            "samples": sample_n,
            "search_profile_counts": profile_counts,
            "benchmark_safe_frac": float(benchmark_safe / max(sample_n, 1)),
            "mean_realized_iterations": (
                float(sum(realized_iterations) / len(realized_iterations))
                if realized_iterations
                else None
            ),
            "mean_dup_rate": float(sum(dup_rates) / len(dup_rates)) if dup_rates else None,
            "mean_p_flip": float(sum(p_flips) / len(p_flips)) if p_flips else None,
            "halt_reason_hist": halt_reason_hist,
            "mean_refresh_count": float(sum(refresh_counts) / len(refresh_counts)) if refresh_counts else None,
            "mean_penalty_sum": float(sum(penalty_sums) / len(penalty_sums)) if penalty_sums else None,
            "mean_selection_root_selects": (
                float(sum(selection_root_selects) / len(selection_root_selects))
                if selection_root_selects
                else None
            ),
            "selection_refresh_selected_frac": (
                float(sum(selection_refresh_selected) / sum(selection_root_selects))
                if selection_root_selects and sum(selection_root_selects) > 0
                else None
            ),
            "mean_selection_penalty_abs_sum": (
                float(sum(selection_penalty_abs_sums) / len(selection_penalty_abs_sums))
                if selection_penalty_abs_sums
                else None
            ),
            "mean_selection_effective_prior_l1_sum": (
                float(sum(selection_effective_prior_l1_sums) / len(selection_effective_prior_l1_sums))
                if selection_effective_prior_l1_sums
                else None
            ),
            "mean_selection_candidate_count": (
                float(sum(selection_mean_candidate_counts) / len(selection_mean_candidate_counts))
                if selection_mean_candidate_counts
                else None
            ),
            "max_selection_candidate_count": (
                float(max(selection_max_candidate_counts))
                if selection_max_candidate_counts
                else None
            ),
            "controller_penalty_mode_counts": penalty_mode_counts,
            "mean_prior_refresh_rate": (
                float(sum(prior_refresh_rates) / len(prior_refresh_rates))
                if prior_refresh_rates
                else None
            ),
            "mean_prior_q_divergence": (
                float(sum(prior_q_divergences) / len(prior_q_divergences))
                if prior_q_divergences
                else None
            ),
            "root_only_shaping_frac": (
                float(root_only_shaping_true / max(root_only_shaping_seen, 1))
                if root_only_shaping_seen
                else None
            ),
            "controller_telemetry_partial_frac": float(telemetry_partial / max(sample_n, 1)),
            "halt_metric_coverage_frac": float(halt_metric_present / max(sample_n, 1)),
            "refresh_metric_coverage_frac": float(refresh_metric_present / max(sample_n, 1)),
            "penalty_metric_coverage_frac": float(penalty_metric_present / max(sample_n, 1)),
            "selection_trace_coverage_frac": float(selection_trace_present / max(sample_n, 1)),
            "halt_trace": _finalize_halt_trace(halt_trace_per_mode),
            # P6: voc-channel decomposition (mean across samples) and a
            # schema_version census so a downstream consumer can fail
            # fast on wire-format drift.
            "mean_voc_total": (
                float(sum(voc_total_samples) / len(voc_total_samples))
                if voc_total_samples
                else None
            ),
            "mean_voc_focus": (
                float(sum(voc_focus_samples) / len(voc_focus_samples))
                if voc_focus_samples
                else None
            ),
            "mean_voc_expand": (
                float(sum(voc_expand_samples) / len(voc_expand_samples))
                if voc_expand_samples
                else None
            ),
            "mean_voc_merge": (
                float(sum(voc_merge_samples) / len(voc_merge_samples))
                if voc_merge_samples
                else None
            ),
            # Q3 (audit_codex_20260428.md W'1): per-row argmax-channel
            # histogram. Empty dict means no per-position halt-check
            # records were available (older Rust binaries on
            # schema_version 1, or pre-halt smoke runs).
            "voc_argmax_channel_hist": dict(voc_argmax_channel_hist),
            "controller_schema_versions": dict(controller_schema_versions),
            "actuator_coverage_frac": float(actuator_coverage_seen / max(sample_n, 1)),
            "prior_refresh_rate_configured_frac": float(prior_refresh_rate_configured / max(actuator_coverage_seen, 1)),
            "prior_refresh_rate_consumed_by_mode_frac": float(prior_refresh_rate_consumed_by_mode / max(actuator_coverage_seen, 1)),
            "prior_refresh_rate_inert_for_mode_frac": float(prior_refresh_rate_inert_for_mode / max(actuator_coverage_seen, 1)),
            "prior_refresh_source_counts": dict(prior_refresh_source_counts),
            # P01: extended block aggregation. The legacy
            # `mean_prior_refresh_rate` above averages the configured
            # rate (stale claim from W3); these new fields aggregate the
            # actual measured fired/eligible counts emitted by Rust
            # schema_version 6+. `extended_coverage_frac` exposes how
            # many subsample rows carried the new block (0.0 means every
            # sample came from an older Rust binary; 1.0 means full
            # coverage).
            "extended_coverage_frac": float(ext_present_count / max(sample_n, 1)),
            "extended_refresh_active_total": int(ext_refresh_active_total),
            "extended_refresh_eligible_total": int(ext_refresh_eligible_total),
            "extended_measured_prior_refresh_rate": (
                float(ext_refresh_active_total / ext_refresh_eligible_total)
                if ext_refresh_eligible_total > 0
                else None
            ),
            "extended_controller_penalty_mode_counts": dict(ext_penalty_mode_counts),
            "extended_halt_reason_count": dict(ext_halt_reason_counts),
        }


__all__ = [
    "ReplayBuffer",
    "ReplayDataset",
    "ReplayExample",
    "ReplayMetrics",
    "SparsePolicyTarget",
    "collate_replay_samples",
    "dense_policy_from_sparse",
    "iter_sparse_policy_entries",
    "normalize_sparse_policy",
    "sparse_policy_from_dense",
    "sparse_policy_from_entries",
]
