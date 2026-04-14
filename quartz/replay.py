"""Replay buffer and sparse policy utilities.

This module intentionally stays independent from the training loop so replay
storage, serialization, and batching can be tested and optimized in isolation.
"""

from __future__ import annotations

import math
import os
import random
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


def _torch_module():
    import torch

    return torch


def _data_loader_cls():
    from torch.utils.data import DataLoader

    return DataLoader


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
    idx = np.flatnonzero(np.abs(dense) > 1e-12).astype(np.int32, copy=False)
    val = dense[idx].astype(np.float32, copy=True)
    return SparsePolicyTarget(
        idx=np.ascontiguousarray(idx, dtype=np.int32),
        val=np.ascontiguousarray(val, dtype=np.float32),
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
        self.buf = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.recent_fraction = float(max(0.0, min(1.0, recent_fraction)))
        self.recent_window = int(max(0, recent_window))

    def _make_example(self, state, policy, value):
        sparse_policy = normalize_sparse_policy(policy)
        return ReplayExample(
            state=np.asarray(state, dtype=np.float32),
            policy=sparse_policy,
            value=float(value),
        )

    def _make_sparse_example(self, state, policy_entries, value, n_actions):
        sparse_policy = normalize_sparse_policy(policy_entries, n_actions=n_actions)
        return ReplayExample(
            state=np.asarray(state, dtype=np.float32),
            policy=sparse_policy,
            value=float(value),
        )

    def add(self, state, policy, value):
        with self._lock:
            self.buf.append(self._make_example(state, policy, value))

    def add_sparse(self, state, policy_entries, value, n_actions):
        with self._lock:
            self.buf.append(self._make_sparse_example(state, policy_entries, value, n_actions))

    def add_game(self, states, policies, outcome, start_player=1):
        """Add a full game using side-to-move values at each ply."""
        with self._lock:
            for i, (state, policy) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                value = outcome * player_to_move
                self.buf.append(self._make_example(state, policy, value))

    def add_sparse_game(self, states, policies, outcome, n_actions, start_player=1):
        with self._lock:
            for i, (state, policy) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                value = outcome * player_to_move
                self.buf.append(self._make_sparse_example(state, policy, value, n_actions))

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
        examples = []
        for _ in range(max(0, int(n_steps))):
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

    def save(self, path):
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
    states_np = np.stack([np.asarray(ex.state, dtype=np.float32) for ex in examples])
    n_actions = max(int(ex.policy.n_actions) for ex in examples)
    policies_np = np.zeros((len(examples), n_actions), dtype=np.float32)
    for row, ex in enumerate(examples):
        if ex.policy.idx.size:
            policies_np[row, ex.policy.idx] = ex.policy.val
    states = torch.from_numpy(states_np)
    policies = torch.from_numpy(policies_np)
    values = torch.tensor([ex.value for ex in examples], dtype=torch.float32)
    return states, policies, values


class ReplayMetrics:
    """Track replay buffer health metrics."""

    @staticmethod
    def freshness(n_new, replay_size):
        return n_new / max(replay_size, 1)

    @staticmethod
    def policy_entropy(replay, sample_n=100):
        if len(replay) < sample_n:
            return 0.0
        indices = random.sample(range(len(replay)), sample_n)
        total_ent = 0.0
        for i in indices:
            sample = replay.buf[i]
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
        values = [
            replay.buf[i].value if isinstance(replay.buf[i], ReplayExample) else replay.buf[i][2]
            for i in indices
        ]
        return float(np.std(values))


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
