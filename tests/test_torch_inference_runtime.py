import gc
import weakref
from typing import Any

import numpy as np
import pytest

from quartz import torch_inference_runtime as tir


def test_predictor_path_coerces_contiguous_writable_float32_batch():
    seen = {}

    class Predictor:
        def predict(self, batch):
            seen["dtype"] = batch.dtype
            seen["contiguous"] = batch.flags.c_contiguous
            seen["writeable"] = batch.flags.writeable
            return np.ones((batch.shape[0], 3), dtype=np.float64), np.array([[0.25], [-0.5]])

    batch = np.arange(2 * 1 * 3 * 3, dtype=np.float64).reshape(2, 1, 3, 3)[:, :, :, ::-1]
    batch.setflags(write=False)

    policies, values = tir.run_model_batch(Predictor(), "cpu", batch)

    assert seen == {"dtype": np.dtype("float32"), "contiguous": True, "writeable": True}
    assert policies.dtype == np.float32
    assert values.dtype == np.float32
    assert values.tolist() == [0.25, -0.5]


def test_torch_cpu_run_model_batch_matches_manual_forward(monkeypatch):
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)
            self.value = torch.nn.Linear(4, 1)

        def forward(self, x):
            flat = x.flatten(1)
            return self.fc(flat), self.value(flat).squeeze(-1)

    monkeypatch.setenv("QUARTZ_DISABLE_COMPILE", "1")
    torch.manual_seed(7)
    model = TinyModel().eval()
    batch = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2)

    policies, values = tir.run_model_batch(model, torch.device("cpu"), batch, torch_module=torch)

    with torch.inference_mode():
        logits, manual_values = model(torch.from_numpy(batch))
        manual_policies = torch.softmax(logits, dim=-1).numpy()

    np.testing.assert_allclose(policies, manual_policies, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(values, manual_values.numpy(), rtol=1e-6, atol=1e-7)


class _StubTorchModule:
    """Fake torch module exposing only what get_compiled_model needs,
    so these tests exercise the CACHING logic without paying for a
    real torch.compile (slow, and Triton/inductor may be unavailable
    on a GPU-less/ROCm CI box). compile() deliberately returns a
    tuple that holds `model` itself — mirroring the real
    torch.compile's OptimizedModule, which stores the original module
    as `self._orig_mod = model` (a strong reference back to the
    input). That back-reference is exactly what makes a
    WeakKeyDictionary-based cache fail to evict (see the module
    docstring in torch_inference_runtime.py) and is the reason the fix
    uses attribute-based caching instead."""

    def __init__(self):
        self.compile_calls: list[Any] = []

    def compile(self, model, mode=None, dynamic=None):
        self.compile_calls.append(model)
        return ("compiled", model, mode, dynamic)


class _FakeModel:
    """A plain object standing in for an nn.Module — supports both
    weak references and arbitrary attribute assignment (no
    __slots__), like the real thing, without importing torch."""


class _NoAttributeModel:
    """A model-like object that CANNOT have an arbitrary attribute
    set on it, to exercise get_compiled_model's fallback path."""

    __slots__ = ()


def test_get_compiled_model_caches_across_calls_for_same_model_instance():
    """A3-a: repeated calls for the SAME model object must compile
    only once and return the cached artifact thereafter."""
    stub = _StubTorchModule()
    model = _FakeModel()

    first = tir.get_compiled_model(model, torch_module=stub)
    second = tir.get_compiled_model(model, torch_module=stub)

    assert len(stub.compile_calls) == 1
    assert first is second
    assert getattr(model, tir._COMPILED_MODEL_ATTR) is first


def test_get_compiled_model_compiles_separately_for_distinct_model_instances():
    """Two distinct model objects must not share a cache entry."""
    stub = _StubTorchModule()
    model_a = _FakeModel()
    model_b = _FakeModel()

    compiled_a = tir.get_compiled_model(model_a, torch_module=stub)
    compiled_b = tir.get_compiled_model(model_b, torch_module=stub)

    assert len(stub.compile_calls) == 2
    assert compiled_a != compiled_b


def test_get_compiled_model_cache_evicts_when_model_is_garbage_collected():
    """A3-a regression, and the reason a WeakKeyDictionary does NOT
    work here (see the module docstring): the compiled wrapper holds
    a strong reference back to the original model (torch's own
    `_orig_mod`, mirrored by _StubTorchModule.compile above). A
    WeakKeyDictionary keyed by `model` would still be pinned forever,
    because the dict's OWN value slot (holding `compiled`, which
    holds `model`) is always reachable from the module-level dict —
    key-side weakness never even comes into play. This test was
    written FIRST against a WeakKeyDictionary implementation and
    failed exactly this way, which is what drove the fix to
    attribute-based caching instead: model and compiled form an
    explicit reference CYCLE (attribute one way, _orig_mod the
    other) with no anchor beyond whatever external code holds
    `model` — Python's cycle-collecting GC reclaims that once the
    caller drops its own reference."""
    stub = _StubTorchModule()
    model = _FakeModel()

    compiled = tir.get_compiled_model(model, torch_module=stub)
    assert compiled is not None
    model_ref = weakref.ref(model)

    stub.compile_calls.clear()  # drop the stub's own strong ref to model
    del compiled
    del model
    del stub
    gc.collect()

    assert model_ref() is None, "model must be collectible once external references are dropped"


def test_get_compiled_model_falls_back_gracefully_when_attribute_assignment_fails():
    """A model-like object that can't have an attribute set on it must
    still get a compiled result (just uncached) rather than raising —
    and every call recompiles, since nothing persists the result."""
    stub = _StubTorchModule()
    model = _NoAttributeModel()

    first = tir.get_compiled_model(model, torch_module=stub)
    second = tir.get_compiled_model(model, torch_module=stub)

    assert len(stub.compile_calls) == 2, "uncacheable model must recompile on every call"
    assert first[1] is model
    assert second[1] is model


def test_get_compiled_model_skips_compilation_when_disabled(monkeypatch):
    monkeypatch.setenv("QUARTZ_DISABLE_COMPILE", "1")
    stub = _StubTorchModule()
    model = _FakeModel()

    result = tir.get_compiled_model(model, torch_module=stub)

    assert result is model
    assert stub.compile_calls == []


def test_get_compiled_model_skips_compilation_on_cpu_device_by_default():
    stub = _StubTorchModule()
    model = _FakeModel()

    result = tir.get_compiled_model(model, device="cpu", torch_module=stub)

    assert result is model
    assert stub.compile_calls == []
