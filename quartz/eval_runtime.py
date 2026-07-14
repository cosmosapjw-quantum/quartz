"""Runtime helpers for NN evaluation batching and caching."""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import OrderedDict

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fused multi-model forward (Phase 7 follow-up, 2026-04-27)
#
# **Opt-in via `QUARTZ_FUSED_EVAL=1`.** Default OFF.
#
# Goal: when a `PersistentRustNNEvalCampaign` runs M ≥ 2 engines (e.g.
# cross-tournament with paired-seed eval, M = 12 in smoke_multiseed),
# the legacy code path issues one `run_model_batch` call PER unique
# model_tag in each batch step. With M = 12 active tags per move and
# ~50 moves × 144 games this multiplied per-move dispatch overhead by
# ~12×, dominating per-game cost (71 s/game vs 21 s/game with 4 engines).
#
# This helper replaces the per-tag for-loop with a single
# `torch.func.vmap(functional_call(...))` over a batched module state.
# All M models — required to share architecture — are stacked once via
# `stack_module_state` and cached keyed by `id(model_map)`. A
# `weakref.finalize` anchored on the FIRST model (not the dict — dicts
# don't support weakrefs and the bug caused process-lifetime cache
# leaks of ~180 MiB per 12-engine campaign) purges the cache entry
# when the model is GC'd.
#
# Why opt-in: initial deployment on RX 6950 XT / ROCm 7.2 caused GPU
# paralysis. Three suspected contributing factors documented in the
# `run_batched_eval_groups` inline comment. Re-enable per-host once
# VRAM behavior is profiled. Production runs unaffected by this
# helper unless the env var is set.
# ─────────────────────────────────────────────────────────────────────────────


_FUSED_CACHE: dict = {}
# Reentrant: a lock-held eviction/clear can drop the last reference to an
# entry's anchor model, firing the `weakref.finalize(_purge_fused_cache)`
# callback synchronously on the same thread. That callback re-acquires this
# lock, so a plain non-reentrant Lock self-deadlocks (see the eviction loop
# in `_get_or_build_fused_state` and `clear_fused_cache`). RLock permits the
# same-thread re-entry; the finalizer's critical section is a no-op-safe pop.
_FUSED_CACHE_LOCK = threading.RLock()


def _purge_fused_cache(cache_key: int) -> None:
    with _FUSED_CACHE_LOCK:
        _FUSED_CACHE.pop(cache_key, None)


def _build_fused_state(model_map: dict):
    """Stack the M models' parameters/buffers along a leading dim.

    Returns (stacked_params, stacked_buffers, base_module, tag_to_idx)
    or None if the models do not share an architecture (different
    state_dict keys/shapes).
    """
    try:
        import torch  # noqa: F401
        from torch.func import stack_module_state
    except Exception:  # pragma: no cover - torch.func should be present on >=2.0
        return None

    sorted_tags = sorted(int(k) for k in model_map.keys())
    if not sorted_tags:
        return None
    models = [model_map[k] for k in sorted_tags]

    # Equivalence check on state_dict keys; if mismatched, bail and let
    # the caller fall back to the per-tag loop.
    keys0 = set(models[0].state_dict().keys())
    for m in models[1:]:
        if set(m.state_dict().keys()) != keys0:
            return None

    # Set every model to eval() so BN running stats are used and dropout
    # is disabled. functional_call honors module mode for buffers.
    for m in models:
        try:
            m.eval()
        except Exception:
            pass

    try:
        stacked_params, stacked_buffers = stack_module_state(models)
    except Exception:
        # Heterogeneous shape (e.g. one model has different filter count)
        # — bail to per-tag fallback.
        return None

    base = models[0]
    tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
    return stacked_params, stacked_buffers, base, tag_to_idx


def _get_or_build_fused_state(model_map: dict):
    cache_key = id(model_map)
    with _FUSED_CACHE_LOCK:
        cached = _FUSED_CACHE.get(cache_key)
        if cached is not None and cached.get("size") == len(model_map):
            return cached
        # Bug fix (2026-04-27): if a stale entry exists for a different
        # size or any prior id-collision, evict it before building a
        # fresh one. Belt-and-suspenders against the dict-weakref
        # silent-fail described below.
        _FUSED_CACHE.pop(cache_key, None)
        # Cap the cache to ~2 entries to bound VRAM use across multiple
        # campaigns within a single Python process. Older entries are
        # evicted FIFO; their stacked tensors (180+ MiB per 12-engine
        # entry) become eligible for GC immediately.
        while len(_FUSED_CACHE) >= 2:
            _FUSED_CACHE.pop(next(iter(_FUSED_CACHE)))
    built = _build_fused_state(model_map)
    if built is None:
        return None
    stacked_params, stacked_buffers, base, tag_to_idx = built
    cached = {
        "size": len(model_map),
        "stacked_params": stacked_params,
        "stacked_buffers": stacked_buffers,
        "base": base,
        "tag_to_idx": tag_to_idx,
    }
    with _FUSED_CACHE_LOCK:
        _FUSED_CACHE[cache_key] = cached
    # Bug fix (2026-04-27): `weakref.finalize(model_map, ...)` on a
    # plain `dict` raises TypeError because dicts have no
    # `__weakref__` slot. Previous code silently swallowed it,
    # leaving the cache entry resident for the lifetime of the
    # process — ~180 MiB per 12-engine campaign in stacked GPU
    # tensors. Anchor the weakref to one of the contained models
    # (an `nn.Module`, which DOES support weakrefs); the cache
    # entry is purged when the last reference to that model drops.
    try:
        sentinel = next(iter(model_map.values()))
        weakref.finalize(sentinel, _purge_fused_cache, cache_key)
    except (TypeError, StopIteration):
        pass

    return cached


def clear_fused_cache():
    """Public entry point to drop all stacked-state cache entries.

    Callers (e.g. PersistentRustNNEvalCampaign.close) can invoke this
    when the upstream model_map is no longer used so the GPU memory
    held by stacked params/buffers is reclaimed promptly without
    waiting for the weakref-anchor model to GC.
    """
    with _FUSED_CACHE_LOCK:
        _FUSED_CACHE.clear()


def _run_fused_multi_model_batch(model_map, device, features_np, by_tag):
    """Single fused forward for the multi-model eval batch.

    Returns dict {global_i: (policy_np, value_float)} or None if the
    fused path is not available for this model_map (forces caller
    fallback to the per-tag loop).
    """
    if features_np is None or features_np.size == 0 or not by_tag:
        return {}

    cached = _get_or_build_fused_state(model_map)
    if cached is None:
        return None

    try:
        import torch
        from torch.func import functional_call, vmap
    except Exception:
        return None

    stacked_params = cached["stacked_params"]
    stacked_buffers = cached["stacked_buffers"]
    base = cached["base"]
    tag_to_idx = cached["tag_to_idx"]
    M = len(tag_to_idx)

    # Ignore tags that are not in the cached map (caller will fall back
    # for those — keeps semantics aligned with the per-tag loop where
    # missing model_obj produces uniform priors).
    usable_by_tag = {t: e for t, e in by_tag.items() if int(t) in tag_to_idx}
    if not usable_by_tag:
        return {}

    max_K = max(len(entries) for entries in usable_by_tag.values())
    if max_K == 0:
        return {}

    features_t = torch.from_numpy(features_np).to(device)
    feat_shape = features_t.shape[1:]

    # Padded (M, max_K, *feat_shape) input — empty slots are zeros,
    # whose outputs we discard in the scatter step below.
    x_padded = torch.zeros((M, max_K) + tuple(feat_shape), device=device, dtype=features_t.dtype)
    request_index = [[] for _ in range(M)]
    for tag, entries in usable_by_tag.items():
        m_idx = tag_to_idx[int(tag)]
        for k, (local_i, global_i, na) in enumerate(entries):
            x_padded[m_idx, k] = features_t[local_i]
            request_index[m_idx].append((k, global_i, na))

    def call_one(params, buffers, x):
        return functional_call(base, (params, buffers), (x,))

    try:
        with torch.inference_mode():
            logits, values = vmap(call_one)(stacked_params, stacked_buffers, x_padded)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            values_np = values.cpu().numpy()
    except Exception as exc:
        # Any vmap-time failure → invalidate cache and fall back.
        log.warning(
            "fused multi-model forward failed (%s); falling back to per-tag loop",
            exc,
        )
        with _FUSED_CACHE_LOCK:
            _FUSED_CACHE.pop(id(model_map), None)
        return None

    out = {}
    for m_idx in range(M):
        for k, global_i, na in request_index[m_idx]:
            out[global_i] = (
                np.asarray(probs[m_idx, k, :na], dtype=np.float32),
                float(values_np[m_idx, k]),
            )
    return out


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

            # Phase 7 follow-up (2026-04-27): fused multi-model forward
            # via `torch.func.vmap(functional_call(...))` is implemented
            # in `_run_fused_multi_model_batch` but is **opt-in** behind
            # the `QUARTZ_FUSED_EVAL=1` env var until validated against
            # ROCm 7.x VRAM behavior. Initial deployment caused GPU
            # paralysis on RX 6950 XT — three suspected causes:
            #   1. `weakref.finalize` on the `model_map` dict silently
            #      no-ops (dicts have no __weakref__), so the
            #      stacked-state cache leaks ~180 MiB per 12-engine
            #      campaign across runs.
            #   2. `stack_module_state` clones every param/buffer to
            #      a new stacked tensor — original models stay live on
            #      GPU via engine refs, doubling effective VRAM.
            #   3. vmap activations multiply per-block working set by
            #      M; ROCm caching-allocator fragmentation amplifies
            #      this and can stall the device under sustained load.
            # The per-tag for-loop below is the validated production
            # path. The helper + unit tests stay in tree so the fused
            # path can be re-enabled after profile-driven fixes.
            use_fused = (
                len(model_map) >= 2
                and str(os.environ.get("QUARTZ_FUSED_EVAL", "0")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            fused_out = None
            if use_fused:
                fused_out = _run_fused_multi_model_batch(
                    model_map, device, features_np, by_tag
                )
            if fused_out is not None:
                for global_i, (policy_np, value_f) in fused_out.items():
                    all_policies[global_i] = policy_np
                    all_values[global_i] = value_f
                # Fill uniform priors for tags missing from model_map.
                model_keys = {int(k) for k in model_map.keys()}
                for tag, entries in by_tag.items():
                    if int(tag) in model_keys:
                        continue
                    for _, global_i, na in entries:
                        all_policies[global_i] = np.full(na, 1.0 / na, dtype=np.float32)
                        all_values[global_i] = 0.0
            else:
                # Per-tag legacy path (default). One forward pass per
                # unique model_tag in this batch step.
                for model_tag, entries in by_tag.items():
                    model_obj = model_map.get(int(model_tag))
                    local_idxs = [li for li, _, _ in entries]
                    if model_obj is not None:
                        probs_batch, vals_np = run_model_batch(
                            model_obj, device, features_np[local_idxs]
                        )
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
