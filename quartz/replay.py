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
from collections import deque
from dataclasses import dataclass

import numpy as np

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

    def add_game(self, states, policies, outcome, start_player=1, traces=None):
        """Add a full game using side-to-move values at each ply."""
        with self._lock:
            for i, (state, policy) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                value = outcome * player_to_move
                metadata = traces[i] if traces is not None and i < len(traces) else None
                self.buf.append(self._make_example(state, policy, value, metadata=metadata))

    def add_sparse_game(self, states, policies, outcome, n_actions, start_player=1, traces=None):
        with self._lock:
            for i, (state, policy) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                value = outcome * player_to_move
                metadata = traces[i] if traces is not None and i < len(traces) else None
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
    first_state = np.asarray(examples[0].state, dtype=np.float32)
    states_np = np.empty((len(examples),) + first_state.shape, dtype=np.float32)
    values_np = np.empty(len(examples), dtype=np.float32)
    n_actions = max(int(ex.policy.n_actions) for ex in examples)
    policies_np = np.zeros((len(examples), n_actions), dtype=np.float32)
    for row, ex in enumerate(examples):
        states_np[row] = np.asarray(ex.state, dtype=np.float32)
        values_np[row] = float(ex.value)
        if ex.policy.idx.size:
            policies_np[row, ex.policy.idx] = ex.policy.val
    states = torch.from_numpy(states_np)
    policies = torch.from_numpy(policies_np)
    values = torch.from_numpy(values_np)
    return states, policies, values


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
        penalty_mode_counts = {}
        prior_refresh_rates = []
        prior_q_divergences = []
        root_only_shaping_true = 0
        root_only_shaping_seen = 0
        telemetry_partial = 0
        refresh_metric_present = 0
        penalty_metric_present = 0
        halt_metric_present = 0
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
            penalty_mode = ctrl.get("penalty_mode")
            if penalty_mode:
                penalty_mode_counts[str(penalty_mode)] = int(penalty_mode_counts.get(str(penalty_mode), 0)) + 1
            if ctrl.get("prior_refresh_rate") is not None:
                prior_refresh_rates.append(float(ctrl["prior_refresh_rate"]))
            if ctrl.get("prior_q_divergence") is not None:
                prior_q_divergences.append(float(ctrl["prior_q_divergence"]))
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
