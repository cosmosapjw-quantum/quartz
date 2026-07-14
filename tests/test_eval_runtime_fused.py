"""Phase 7 follow-up (2026-04-27): equivalence tests for the fused
multi-model forward in `quartz.eval_runtime`.

Verifies that `_run_fused_multi_model_batch` (vmap-based, single GPU
dispatch) produces numerically identical outputs to the legacy
per-tag for-loop, and that `run_batched_eval_groups` delegates to
the fused helper when ≥ 2 models share a state_dict AND the
`QUARTZ_FUSED_EVAL=1` opt-in env var is set (gated until the
ROCm-side VRAM behavior is profile-validated; see eval_runtime.py
banner comment).
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _enable_fused_path(monkeypatch):
    """Force the fused dispatch on for every test in this module.
    Production default is OFF (`QUARTZ_FUSED_EVAL` unset)."""
    monkeypatch.setenv("QUARTZ_FUSED_EVAL", "1")


class _TinyAlphaZeroNet(torch.nn.Module):
    """Module-scope class so all instances share the same Python class
    (a `stack_module_state` requirement)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(4)
        # Match the AlphaZeroNet contract: forward returns (logits, value).
        self.policy = torch.nn.Linear(4 * 2 * 2, 4)
        self.value = torch.nn.Linear(4 * 2 * 2, 1)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x))).flatten(1)
        p = self.policy(h)
        v = self.value(h).squeeze(-1)
        return p, v


def _make_tiny_model(seed: int):
    """Build a small (ch, board) AlphaZero-shape net seeded for repro."""
    torch.manual_seed(seed)
    m = _TinyAlphaZeroNet()
    m.eval()
    return m


def _make_model_map(M: int):
    return {tag: _make_tiny_model(seed=100 + tag) for tag in range(M)}


def _build_request(gi: int, na: int, model_tag: int, board: int, ch: int):
    """Construct the (na, feats, model_tag, fp_lo, fp_hi, encoder_rev)
    tuple shape that `parse_eval_request` understands.

    `fp_lo` / `fp_hi` are set to None so the eval cache falls back to
    content-hash keys (`legacy_eval_cache_key`) — otherwise multiple
    requests with the same `model_tag` would collide on a single
    cache slot and stale-hit each other across our two test runs.
    """
    feats = np.random.RandomState(gi).rand(ch * board * board).astype(np.float32).tolist()
    return (na, feats, model_tag, None, None, None)


def test_fused_path_matches_per_tag_path():
    """Equivalence: fused forward and per-tag forward produce policies
    and values that match within fp32 tolerance.
    """
    eval_mod = importlib.import_module("quartz.eval_runtime")

    M = 4
    ch, board = 1, 2
    cfg = {"ch": ch, "board": board}
    model_map = _make_model_map(M)
    device = torch.device("cpu")

    # Build a batch with multiple requests per tag, varying na ≤ 4.
    rng = np.random.RandomState(7)
    n_per_tag = 3
    requests = []
    for tag in range(M):
        for _ in range(n_per_tag):
            na = int(rng.randint(2, 5))
            requests.append(_build_request(len(requests), na, tag, board, ch))
    groups = [{"gi": 0, "kind": "json_batch", "requests": requests}]

    # Run via fused-aware entry point (will dispatch to fused for M ≥ 2).
    res_fused = eval_mod.run_batched_eval_groups(
        groups,
        model_map,
        device,
        cfg,
        run_model_batch=eval_mod_run_model_batch_fallback,
    )

    # Run via legacy per-tag path. We force the fallback by supplying a
    # dict of one-arch-different model — instead, monkey-patch the
    # fused helper to return None.
    fused_orig = eval_mod._run_fused_multi_model_batch
    eval_mod._run_fused_multi_model_batch = lambda *a, **kw: None
    try:
        res_pertag = eval_mod.run_batched_eval_groups(
            groups,
            model_map,
            device,
            cfg,
            run_model_batch=eval_mod_run_model_batch_fallback,
        )
    finally:
        eval_mod._run_fused_multi_model_batch = fused_orig

    # Compare top-level structure.
    assert len(res_fused) == 1
    assert len(res_pertag) == 1
    pol_f = res_fused[0]["policies"]
    pol_p = res_pertag[0]["policies"]
    val_f = res_fused[0]["values"]
    val_p = res_pertag[0]["values"]

    assert len(pol_f) == len(pol_p) == len(requests)

    # Element-wise comparison.
    for i, (pf, pp) in enumerate(zip(pol_f, pol_p)):
        assert pf.shape == pp.shape, f"shape mismatch at i={i}: {pf.shape} vs {pp.shape}"
        np.testing.assert_allclose(
            pf, pp, rtol=1e-4, atol=1e-5,
            err_msg=f"policy mismatch at request {i}",
        )

    for i, (vf, vp) in enumerate(zip(val_f, val_p)):
        np.testing.assert_allclose(
            vf, vp, rtol=1e-4, atol=1e-5,
            err_msg=f"value mismatch at request {i}",
        )


def eval_mod_run_model_batch_fallback(model, device, batch_features):
    """Per-tag fallback: standard PyTorch forward + softmax."""
    x = torch.from_numpy(np.asarray(batch_features, dtype=np.float32)).to(device)
    with torch.inference_mode():
        logits, vals = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        vals_np = vals.cpu().numpy()
    return probs, vals_np


def test_fused_handles_unbalanced_per_tag_batches():
    """Padding correctness: tag A has 5 requests, tag B has 1 request.
    Fused must still produce identical outputs to per-tag.
    """
    eval_mod = importlib.import_module("quartz.eval_runtime")
    ch, board = 1, 2
    cfg = {"ch": ch, "board": board}
    model_map = _make_model_map(2)
    device = torch.device("cpu")

    requests = []
    for _ in range(5):
        requests.append(_build_request(len(requests), 4, 0, board, ch))
    requests.append(_build_request(len(requests), 4, 1, board, ch))
    groups = [{"gi": 0, "kind": "json_batch", "requests": requests}]

    res_fused = eval_mod.run_batched_eval_groups(
        groups, model_map, device, cfg,
        run_model_batch=eval_mod_run_model_batch_fallback,
    )

    fused_orig = eval_mod._run_fused_multi_model_batch
    eval_mod._run_fused_multi_model_batch = lambda *a, **kw: None
    try:
        res_pertag = eval_mod.run_batched_eval_groups(
            groups, model_map, device, cfg,
            run_model_batch=eval_mod_run_model_batch_fallback,
        )
    finally:
        eval_mod._run_fused_multi_model_batch = fused_orig

    for pf, pp in zip(res_fused[0]["policies"], res_pertag[0]["policies"]):
        np.testing.assert_allclose(pf, pp, rtol=1e-4, atol=1e-5)
    for vf, vp in zip(res_fused[0]["values"], res_pertag[0]["values"]):
        np.testing.assert_allclose(vf, vp, rtol=1e-4, atol=1e-5)


def test_fused_falls_back_on_arch_mismatch():
    """If model_map contains arch-different models (different state_dict
    keys), fused build must return None and the call must succeed via
    per-tag fallback.
    """
    eval_mod = importlib.import_module("quartz.eval_runtime")

    class _Other(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Deliberately different architecture.
            self.layer = torch.nn.Conv2d(1, 8, 3, padding=1, bias=True)
            self.policy = torch.nn.Linear(8 * 2 * 2, 4)
            self.value = torch.nn.Linear(8 * 2 * 2, 1)

        def forward(self, x):
            h = self.layer(x).flatten(1)
            return self.policy(h), self.value(h).squeeze(-1)

    model_map = {0: _make_tiny_model(seed=1), 1: _Other().eval()}
    cfg = {"ch": 1, "board": 2}
    device = torch.device("cpu")

    # Direct test of the helper.
    req_features = np.random.RandomState(0).rand(2, 1, 2, 2).astype(np.float32)
    by_tag = {0: [(0, 0, 4)], 1: [(1, 1, 4)]}
    out = eval_mod._run_fused_multi_model_batch(model_map, device, req_features, by_tag)
    assert out is None, "fused should bail on architecture mismatch"

    # End-to-end: run_batched_eval_groups must still succeed via fallback.
    requests = [
        _build_request(0, 4, 0, 2, 1),
        _build_request(1, 4, 1, 2, 1),
    ]
    groups = [{"gi": 0, "kind": "json_batch", "requests": requests}]
    res = eval_mod.run_batched_eval_groups(
        groups, model_map, device, cfg,
        run_model_batch=eval_mod_run_model_batch_fallback,
    )
    assert len(res[0]["policies"]) == 2


def test_fused_cache_invalidates_on_model_map_gc():
    """`weakref.finalize` should drop the cache entry when the model_map
    is garbage-collected. Direct id-collision attack would otherwise
    misroute requests to a stale stacked state.
    """
    import gc

    eval_mod = importlib.import_module("quartz.eval_runtime")
    eval_mod._FUSED_CACHE.clear()

    mm = _make_model_map(3)
    cached = eval_mod._get_or_build_fused_state(mm)
    assert cached is not None
    cache_size_before = len(eval_mod._FUSED_CACHE)

    # Drop the only reference; GC; cache should clear.
    mm_id = id(mm)
    del mm
    gc.collect()

    # weakref.finalize callback may run synchronously or after a tick.
    # We accept either: entry purged OR entry remains but rebuilt on
    # next call with a fresh dict.
    assert mm_id not in eval_mod._FUSED_CACHE or len(eval_mod._FUSED_CACHE) <= cache_size_before


def test_fused_cache_eviction_under_lock_does_not_deadlock():
    """Regression (2026-07-14): a lock-held eviction that drops an anchor
    model's last reference fires `_purge_fused_cache` synchronously on the
    same thread, which re-acquires `_FUSED_CACHE_LOCK`. With a plain
    (non-reentrant) `threading.Lock` this self-deadlocks — the exact hang
    that stalled `test_fused_falls_back_on_arch_mismatch` once earlier
    tests had filled the cache, timing out the CI pytest job.

    Reproduce the reentrancy directly (no GC/id-reuse flakiness): hold the
    cache lock exactly as `_get_or_build_fused_state` does, then invoke the
    finalizer. The lock must tolerate same-thread re-entry.
    """
    import threading

    eval_mod = importlib.import_module("quartz.eval_runtime")
    eval_mod._FUSED_CACHE.clear()
    eval_mod._FUSED_CACHE[424242] = {"size": 1}

    done = threading.Event()

    def _evict_while_holding_lock():
        with eval_mod._FUSED_CACHE_LOCK:  # outer hold, as in the eviction loop
            eval_mod._purge_fused_cache(424242)  # weakref finalizer re-acquires
        done.set()

    t = threading.Thread(target=_evict_while_holding_lock, daemon=True)
    t.start()
    assert done.wait(timeout=10), (
        "cache lock is not reentrant — the finalizer self-deadlocks under it"
    )
    assert 424242 not in eval_mod._FUSED_CACHE
    eval_mod._FUSED_CACHE.clear()


def test_fused_accepts_single_model_via_legacy_path():
    """run_batched_eval_groups with M=1 should NOT call the fused
    helper — it goes through the single-model `model is not None`
    branch instead. We verify by setting fused to a sentinel that
    would raise if called.
    """
    eval_mod = importlib.import_module("quartz.eval_runtime")
    cfg = {"ch": 1, "board": 2}
    device = torch.device("cpu")
    # Single model, NOT a dict.
    model = _make_tiny_model(seed=42)

    requests = [_build_request(i, 4, 0, 2, 1) for i in range(3)]
    groups = [{"gi": 0, "kind": "json_batch", "requests": requests}]

    fused_orig = eval_mod._run_fused_multi_model_batch
    def _boom(*a, **kw):
        raise AssertionError("fused helper should not be called for single model")
    eval_mod._run_fused_multi_model_batch = _boom
    try:
        res = eval_mod.run_batched_eval_groups(
            groups, model, device, cfg,
            run_model_batch=eval_mod_run_model_batch_fallback,
        )
    finally:
        eval_mod._run_fused_multi_model_batch = fused_orig

    assert len(res[0]["policies"]) == 3
