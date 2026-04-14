"""Runtime helpers for NN evaluation batching and caching."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

import numpy as np

log = logging.getLogger(__name__)


class NNEvalCache:
    """LRU cache for NN evaluation results keyed by feature hash."""

    def __init__(self, max_entries=65536):
        self._cache = OrderedDict()
        self._max = max_entries
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, feat_hash):
        with self._lock:
            entry = self._cache.get(feat_hash)
            if entry is None:
                self._misses += 1
                return None
            try:
                self._cache.move_to_end(feat_hash)
            except KeyError:
                self._misses += 1
                return None
            self._hits += 1
            return entry

    def put(self, feat_hash, policy_np, value):
        with self._lock:
            if feat_hash in self._cache:
                self._cache.move_to_end(feat_hash)
            self._cache[feat_hash] = (policy_np, float(value))
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self):
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    @staticmethod
    def default_size(actions):
        entry_bytes = actions * 4 + 8
        return min(131072, max(4096, 256 * 1024 * 1024 // entry_bytes))


_NN_EVAL_CACHE = None


def get_nn_eval_cache(cfg):
    global _NN_EVAL_CACHE
    if os.environ.get("QUARTZ_DISABLE_NN_CACHE"):
        return None
    if _NN_EVAL_CACHE is None:
        actions = cfg.get("actions", 49)
        _NN_EVAL_CACHE = NNEvalCache(NNEvalCache.default_size(actions))
    return _NN_EVAL_CACHE


def clear_nn_eval_cache(logger=None):
    global _NN_EVAL_CACHE
    if _NN_EVAL_CACHE is not None:
        active_log = logger or log
        if _NN_EVAL_CACHE._hits + _NN_EVAL_CACHE._misses > 0:
            active_log.info(
                "NN cache: hit_rate=%.1f%% entries=%d",
                _NN_EVAL_CACHE.hit_rate * 100,
                len(_NN_EVAL_CACHE._cache),
            )
        _NN_EVAL_CACHE.clear()


def parse_eval_request(req):
    if len(req) >= 6:
        na, feats, model_tag, fp_lo, fp_hi, encoder_rev = req[:6]
    elif len(req) == 3:
        na, feats, model_tag = req
        fp_lo = fp_hi = encoder_rev = None
    else:
        na, feats = req
        model_tag = 0
        fp_lo = fp_hi = encoder_rev = None
    return (
        max(1, int(na)),
        feats,
        int(model_tag),
        None if fp_lo is None else int(fp_lo),
        None if fp_hi is None else int(fp_hi),
        None if encoder_rev is None else int(encoder_rev),
    )


def legacy_eval_cache_key(model_tag, num_actions, feat_array, ch_cfg, bs_cfg):
    return hash((
        int(model_tag),
        int(num_actions),
        int(ch_cfg),
        int(bs_cfg),
        feat_array.tobytes(),
    ))


def eval_request_cache_key(model_tag, num_actions, feat_array, ch_cfg, bs_cfg, fp_lo, fp_hi, encoder_rev):
    if fp_lo is not None and fp_hi is not None:
        return (int(model_tag), int(fp_hi), int(fp_lo), int(encoder_rev or 0))
    return legacy_eval_cache_key(model_tag, num_actions, feat_array, ch_cfg, bs_cfg)


def run_batched_eval_groups(eval_groups, model, device, cfg, run_model_batch, cache_key_fn=None):
    if not eval_groups:
        return []
    ch_cfg, bs_cfg = cfg["ch"], cfg["board"]
    expected = ch_cfg * bs_cfg * bs_cfg
    nn_cache = get_nn_eval_cache(cfg)
    flat_requests = []
    batch_features = []
    cached_results = {}
    for group in eval_groups:
        for request in group["requests"]:
            na, feats, model_tag, fp_lo, fp_hi, encoder_rev = parse_eval_request(request)
            idx = len(flat_requests)
            if len(feats) == expected:
                features = np.asarray(feats, dtype=np.float32).reshape(ch_cfg, bs_cfg, bs_cfg)
                key_builder = cache_key_fn or eval_request_cache_key
                cache_key = key_builder(
                    model_tag,
                    na,
                    features,
                    ch_cfg,
                    bs_cfg,
                    fp_lo,
                    fp_hi,
                    encoder_rev,
                )
            else:
                features = np.zeros((ch_cfg, bs_cfg, bs_cfg), dtype=np.float32)
                cache_key = None
            flat_requests.append((na, int(model_tag), cache_key))
            if nn_cache is not None and cache_key is not None:
                hit = nn_cache.get(cache_key)
                if hit is not None:
                    cached_results[idx] = hit
                    batch_features.append(None)
                    continue
            batch_features.append(features)

    model_map = {int(k): v for k, v in model.items()} if isinstance(model, dict) else None

    all_policies = [None] * len(flat_requests)
    all_values = [0.0] * len(flat_requests)
    for idx, (policy, value) in cached_results.items():
        na = flat_requests[idx][0]
        all_policies[idx] = policy[:na] if len(policy) >= na else policy
        all_values[idx] = value

    gpu_indices = [i for i in range(len(flat_requests)) if i not in cached_results]
    gpu_features = [batch_features[i] for i in gpu_indices if batch_features[i] is not None]

    if gpu_features:
        if model_map is not None:
            features_np = np.stack(gpu_features, axis=0)
            by_tag = {}
            for local_i, global_i in enumerate(gpu_indices):
                na, model_tag, _cache_key = flat_requests[global_i]
                by_tag.setdefault(int(model_tag), []).append((local_i, global_i, int(na)))
            for model_tag, entries in by_tag.items():
                model_obj = model_map.get(int(model_tag))
                local_idxs = [li for li, _, _ in entries]
                if model_obj is not None:
                    probs_batch, vals_np = run_model_batch(model_obj, device, features_np[local_idxs])
                    for bi, (_, global_i, na) in enumerate(entries):
                        all_policies[global_i] = probs_batch[bi][:na]
                        all_values[global_i] = float(vals_np[bi])
                else:
                    for _, global_i, na in entries:
                        all_policies[global_i] = np.full(na, 1.0 / na, dtype=np.float32)
                        all_values[global_i] = 0.0
        elif model is not None:
            probs_batch, vals_np = run_model_batch(model, device, np.stack(gpu_features, axis=0))
            for bi, global_i in enumerate(gpu_indices):
                na = flat_requests[global_i][0]
                all_policies[global_i] = probs_batch[bi][:na]
                all_values[global_i] = float(vals_np[bi])
        else:
            for global_i in gpu_indices:
                na = flat_requests[global_i][0]
                all_policies[global_i] = np.full(na, 1.0 / na, dtype=np.float32)
                all_values[global_i] = 0.0

    for i in range(len(all_policies)):
        if all_policies[i] is None:
            na = flat_requests[i][0]
            all_policies[i] = np.full(na, 1.0 / na, dtype=np.float32)

    if nn_cache is not None:
        for global_i in gpu_indices:
            _na, _model_tag_i, cache_key = flat_requests[global_i]
            if cache_key is not None:
                nn_cache.put(cache_key, all_policies[global_i], all_values[global_i])

    responses = []
    offset = 0
    for group in eval_groups:
        count = len(group["requests"])
        responses.append({
            "gi": int(group.get("gi", 0)),
            "kind": group["kind"],
            "policies": all_policies[offset:offset + count],
            "values": all_values[offset:offset + count],
        })
        offset += count
    return responses


def make_eval_request_group(kind, requests, gi=0, prefer_shm=False):
    normalized = [parse_eval_request(req) for req in requests]
    group = {
        "gi": int(gi),
        "kind": kind,
        "requests": normalized,
    }
    if prefer_shm:
        group["prefer_shm"] = True
    return group


__all__ = [
    "NNEvalCache",
    "clear_nn_eval_cache",
    "eval_request_cache_key",
    "get_nn_eval_cache",
    "legacy_eval_cache_key",
    "make_eval_request_group",
    "parse_eval_request",
    "run_batched_eval_groups",
]
