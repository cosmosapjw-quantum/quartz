"""Versioned JSON contracts for idea-foundry replay and Rust hand-off.

The serializers are deliberately explicit.  ``dataclasses.asdict`` would make
wire compatibility depend on field order and future implementation details;
these functions keep the v1 names pinned and validate numerical inputs before
they can become live-controller evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from .contracts import (
    FOUNDRY_CONTRACT_SCHEMA_VERSION,
    CostVector,
    EdgeObservation,
    FoundryRootExtras,
    FreshnessIdentity,
    MetaAction,
    MetaActionKind,
    MetaProposal,
    ProposalEstimate,
    RootObservation,
    RuntimeObservation,
)


class ContractValidationError(ValueError):
    """Raised when a foundry payload violates the fail-closed wire contract."""


_AXIS_RE = re.compile(r"^A(?:0[1-9]|1[0-9]|2[0-6])(?:\.[a-z0-9_]+)?$")
_FRESHNESS_STRING_FIELDS = (
    "checkpoint_id",
    "evaluator_id",
    "edge_set_hash",
    "tt_identity_policy",
)
_FRESHNESS_INTEGER_LIMITS = {
    "root_hash": (64, 0),
    "candidate_epoch": (64, 0),
    "cache_schema_version": (16, 1),
    "root_visits": (32, 0),
    "iteration": (64, 0),
}
_ROOT_INTEGER_WIDTHS = {
    "root_hash": 64,
    "root_visits": 32,
    "iteration": 64,
    "elapsed_ms": 64,
    "remaining_visits": 32,
    "n_children": 16,
    "n_visible": 16,
    "revision_count": 16,
}
_EDGE_INTEGER_WIDTHS = {
    "edge_pos": 16,
    "action_id": 32,
    "visits": 32,
    "virtual_visits": 32,
    "pending": 32,
}
_RUNTIME_INTEGER_WIDTHS = {
    "threads": 16,
    "batch_size": 16,
    "inflight": 16,
    "max_pending": 16,
    "tt_wait_ns": 64,
}
_ROOT_FLOAT_FIELDS = (
    "entropy",
    "effective_branching",
    "top2_margin",
    "margin_slope",
    "entropy_slope",
    "prior_visit_js",
    "candidate_omission_bound",
)
_ROOT_OPTIONAL_FLOAT_FIELDS = ("h1_stability", "p_flip")
_FOUNDRY_EXTRAS_FLOAT_FIELDS = (
    "entropy",
    "effective_branching",
    "top2_margin",
    "margin_slope",
    "entropy_slope",
    "prior_visit_js",
    "omission_bound",
)
_EDGE_FLOAT_FIELDS = (
    "prior_anchor",
    "prior_current",
    "q_mean",
    "q_sum",
    "m2",
    "last_value",
    "mc_radius",
    "epistemic_radius",
    "drift_radius",
    "bias_radius",
    "lower",
    "upper",
)
_RUNTIME_FLOAT_FIELDS = (
    "queue_wait_ms",
    "eval_latency_ms",
    "nps",
    "edge_duplicate_rate",
    "semantic_path_overlap",
)

# Fields absent from a rule are required to retain their canonical wire
# default: null for primary/secondary/value and zero for amount.  This mirrors
# MetaActionWire::into_action(), including its round-trip canonicality check.
_ACTION_FIELD_RULES = {
    MetaActionKind.STOP: {"primary": "optional_u16"},
    MetaActionKind.SAMPLE: {"primary": "required_u16", "amount": "u32"},
    MetaActionKind.CHALLENGE: {
        "primary": "required_u16",
        "secondary": "required_u16",
        "amount": "u32",
    },
    MetaActionKind.WIDEN: {"amount": "u16"},
    MetaActionKind.DEEPEN: {"primary": "required_u16", "amount": "u32"},
    MetaActionKind.PROVE: {"primary": "required_u16", "amount": "u32"},
    MetaActionKind.RESAMPLE_MODE: {
        "primary": "required_u16",
        "amount": "u16",
    },
    MetaActionKind.MERGE_OR_SHARE: {"primary": "required_u64"},
    MetaActionKind.SET_BATCH: {"amount": "u16"},
    MetaActionKind.SET_INFLIGHT: {"amount": "u16"},
    MetaActionKind.SET_THREADS: {"amount": "u16"},
    MetaActionKind.REANALYSE: {"primary": "required_u64"},
    MetaActionKind.ARCHIVE_STATE: {"value": "required_finite"},
    MetaActionKind.NOOP: {},
}
_ACTION_CANONICAL_DEFAULTS = {
    "primary": None,
    "secondary": None,
    "amount": 0,
    "value": None,
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{label} must be an object")
    return value


def _required_member(row: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in row:
        raise ContractValidationError(f"{label}.{key} is required")
    return row[key]


def _wire_uint(value: Any, label: str, bits: int, *, minimum: int = 0) -> int:
    """Validate an exact Python int against a Rust unsigned wire width."""

    maximum = (1 << bits) - 1
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractValidationError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractValidationError(f"{label} must be a non-empty string")
    return value


def _wire_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ContractValidationError(f"{label} must be a string")
    return value


def _wire_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ContractValidationError(f"{label} must be a boolean")
    return value


def _wire_string_list(value: Any, label: str) -> list[str]:
    if type(value) is not list:
        raise ContractValidationError(f"{label} must be an array of strings")
    if any(type(item) is not str for item in value):
        raise ContractValidationError(f"{label} must contain only strings")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ContractValidationError(f"{label} must be an int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError(f"{label} must be finite")
    return number


def _nonnegative(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0:
        raise ContractValidationError(f"{label} must be non-negative")
    return number


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-safe value with string mapping keys."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value, "telemetry value")
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ContractValidationError("telemetry object keys must be strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ContractValidationError(
        f"unsupported telemetry value type: {type(value).__name__}"
    )


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def cost_vector_to_payload(cost: CostVector) -> dict[str, float]:
    payload = {
        "nn_evals": _nonnegative(cost.nn_evals, "cost.nn_evals"),
        "cpu_ms": _nonnegative(cost.cpu_ms, "cost.cpu_ms"),
        "gpu_ms": _nonnegative(cost.gpu_ms, "cost.gpu_ms"),
        "energy_proxy": _nonnegative(cost.energy_proxy, "cost.energy_proxy"),
    }
    return payload


def cost_vector_from_payload(payload: Mapping[str, Any]) -> CostVector:
    row = _mapping(payload, "cost")
    return CostVector(
        nn_evals=_nonnegative(
            _required_member(row, "nn_evals", "cost"), "cost.nn_evals"
        ),
        cpu_ms=_nonnegative(
            _required_member(row, "cpu_ms", "cost"), "cost.cpu_ms"
        ),
        gpu_ms=_nonnegative(
            _required_member(row, "gpu_ms", "cost"), "cost.gpu_ms"
        ),
        energy_proxy=_nonnegative(
            _required_member(row, "energy_proxy", "cost"),
            "cost.energy_proxy",
        ),
    )


def validate_freshness_identity(value: FreshnessIdentity) -> None:
    """Require the same identity invariants as the Rust wire contract."""

    if not isinstance(value, FreshnessIdentity):
        raise ContractValidationError("freshness must be a FreshnessIdentity")
    for field_name in _FRESHNESS_STRING_FIELDS:
        _nonempty_string(getattr(value, field_name), f"freshness.{field_name}")
    for field_name, (bits, minimum) in _FRESHNESS_INTEGER_LIMITS.items():
        _wire_uint(
            getattr(value, field_name),
            f"freshness.{field_name}",
            bits,
            minimum=minimum,
        )


def freshness_to_payload(value: FreshnessIdentity) -> dict[str, Any]:
    validate_freshness_identity(value)
    return {
        "root_hash": value.root_hash,
        "checkpoint_id": value.checkpoint_id,
        "evaluator_id": value.evaluator_id,
        "edge_set_hash": value.edge_set_hash,
        "candidate_epoch": value.candidate_epoch,
        "tt_identity_policy": value.tt_identity_policy,
        "cache_schema_version": value.cache_schema_version,
        "root_visits": value.root_visits,
        "iteration": value.iteration,
    }


def freshness_from_payload(payload: Mapping[str, Any]) -> FreshnessIdentity:
    row = _mapping(payload, "freshness")
    value = FreshnessIdentity(
        root_hash=row["root_hash"],
        checkpoint_id=row["checkpoint_id"],
        evaluator_id=row["evaluator_id"],
        edge_set_hash=row["edge_set_hash"],
        candidate_epoch=row["candidate_epoch"],
        tt_identity_policy=row["tt_identity_policy"],
        cache_schema_version=row["cache_schema_version"],
        root_visits=row["root_visits"],
        iteration=row["iteration"],
    )
    validate_freshness_identity(value)
    return value


def _edge_to_payload(edge: EdgeObservation) -> dict[str, Any]:
    _validate_edge_observation(edge)
    return _edge_to_payload_unchecked(edge)


def _edge_from_payload(payload: Mapping[str, Any]) -> EdgeObservation:
    row = _mapping(payload, "edge")
    edge = EdgeObservation(
        edge_pos=_wire_uint(row["edge_pos"], "edge.edge_pos", 16),
        action_id=_wire_uint(row["action_id"], "edge.action_id", 32),
        visible=_wire_bool(row["visible"], "edge.visible"),
        prior_anchor=_finite(row["prior_anchor"], "edge.prior_anchor"),
        prior_current=_finite(row["prior_current"], "edge.prior_current"),
        visits=_wire_uint(row["visits"], "edge.visits", 32),
        virtual_visits=_wire_uint(
            row["virtual_visits"], "edge.virtual_visits", 32
        ),
        pending=_wire_uint(row["pending"], "edge.pending", 32),
        q_mean=_finite(row["q_mean"], "edge.q_mean"),
        q_sum=_finite(row["q_sum"], "edge.q_sum"),
        m2=_finite(row["m2"], "edge.m2"),
        last_value=_finite(row.get("last_value", 0.0), "edge.last_value"),
        mc_radius=_finite(row.get("mc_radius", 0.0), "edge.mc_radius"),
        epistemic_radius=_finite(
            row.get("epistemic_radius", 0.0), "edge.epistemic_radius"
        ),
        drift_radius=_finite(row.get("drift_radius", 0.0), "edge.drift_radius"),
        bias_radius=_finite(row.get("bias_radius", 0.0), "edge.bias_radius"),
        lower=_finite(row.get("lower", -1.0), "edge.lower"),
        upper=_finite(row.get("upper", 1.0), "edge.upper"),
        tactical_flags=tuple(
            _wire_string_list(row.get("tactical_flags", []), "edge.tactical_flags")
        ),
    )
    _validate_edge_observation(edge)
    return edge


def _validate_edge_observation(edge: EdgeObservation) -> None:
    if not isinstance(edge, EdgeObservation):
        raise ContractValidationError("edge must be an EdgeObservation")
    for field_name, bits in _EDGE_INTEGER_WIDTHS.items():
        _wire_uint(getattr(edge, field_name), f"edge.{field_name}", bits)
    _wire_bool(edge.visible, "edge.visible")
    if type(edge.tactical_flags) is not tuple or any(
        type(item) is not str for item in edge.tactical_flags
    ):
        raise ContractValidationError(
            "edge.tactical_flags must be a tuple containing only strings"
        )
    for field_name in _EDGE_FLOAT_FIELDS:
        _finite(getattr(edge, field_name), f"edge.{field_name}")
    if edge.prior_anchor < 0.0 or edge.prior_current < 0.0:
        raise ContractValidationError("edge priors must be non-negative")
    if edge.lower > edge.upper:
        raise ContractValidationError("edge lower bound exceeds upper bound")


def _edge_to_payload_unchecked(edge: EdgeObservation) -> dict[str, Any]:
    """Return an edge body after the caller has validated the edge."""

    return {
        "edge_pos": edge.edge_pos,
        "action_id": edge.action_id,
        "visible": edge.visible,
        "prior_anchor": _finite(edge.prior_anchor, "edge.prior_anchor"),
        "prior_current": _finite(edge.prior_current, "edge.prior_current"),
        "visits": edge.visits,
        "virtual_visits": edge.virtual_visits,
        "pending": edge.pending,
        "q_mean": _finite(edge.q_mean, "edge.q_mean"),
        "q_sum": _finite(edge.q_sum, "edge.q_sum"),
        "m2": _finite(edge.m2, "edge.m2"),
        "last_value": _finite(edge.last_value, "edge.last_value"),
        "mc_radius": _finite(edge.mc_radius, "edge.mc_radius"),
        "epistemic_radius": _finite(
            edge.epistemic_radius, "edge.epistemic_radius"
        ),
        "drift_radius": _finite(edge.drift_radius, "edge.drift_radius"),
        "bias_radius": _finite(edge.bias_radius, "edge.bias_radius"),
        "lower": _finite(edge.lower, "edge.lower"),
        "upper": _finite(edge.upper, "edge.upper"),
        "tactical_flags": list(edge.tactical_flags),
    }


def _runtime_to_payload(runtime: RuntimeObservation) -> dict[str, Any]:
    _validate_runtime_observation(runtime)
    return {
        "threads": runtime.threads,
        "batch_size": runtime.batch_size,
        "inflight": runtime.inflight,
        "queue_wait_ms": _finite(runtime.queue_wait_ms, "runtime.queue_wait_ms"),
        "eval_latency_ms": _finite(
            runtime.eval_latency_ms, "runtime.eval_latency_ms"
        ),
        "nps": _finite(runtime.nps, "runtime.nps"),
        "edge_duplicate_rate": _finite(
            runtime.edge_duplicate_rate, "runtime.edge_duplicate_rate"
        ),
        "semantic_path_overlap": _finite(
            runtime.semantic_path_overlap, "runtime.semantic_path_overlap"
        ),
        "max_pending": runtime.max_pending,
        "tt_wait_ns": runtime.tt_wait_ns,
    }


def _runtime_from_payload(payload: Mapping[str, Any]) -> RuntimeObservation:
    row = _mapping(payload, "runtime")
    runtime = RuntimeObservation(
        threads=_wire_uint(
            _required_member(row, "threads", "runtime"), "runtime.threads", 16
        ),
        batch_size=_wire_uint(
            _required_member(row, "batch_size", "runtime"),
            "runtime.batch_size",
            16,
        ),
        inflight=_wire_uint(
            _required_member(row, "inflight", "runtime"),
            "runtime.inflight",
            16,
        ),
        queue_wait_ms=_finite(
            _required_member(row, "queue_wait_ms", "runtime"),
            "runtime.queue_wait_ms",
        ),
        eval_latency_ms=_finite(
            _required_member(row, "eval_latency_ms", "runtime"),
            "runtime.eval_latency_ms",
        ),
        nps=_finite(
            _required_member(row, "nps", "runtime"), "runtime.nps"
        ),
        edge_duplicate_rate=_finite(
            _required_member(row, "edge_duplicate_rate", "runtime"),
            "runtime.edge_duplicate_rate",
        ),
        semantic_path_overlap=_finite(
            _required_member(row, "semantic_path_overlap", "runtime"),
            "runtime.semantic_path_overlap",
        ),
        max_pending=_wire_uint(
            _required_member(row, "max_pending", "runtime"),
            "runtime.max_pending",
            16,
        ),
        tt_wait_ns=_wire_uint(
            _required_member(row, "tt_wait_ns", "runtime"),
            "runtime.tt_wait_ns",
            64,
        ),
    )
    _validate_runtime_observation(runtime)
    return runtime


def _validate_runtime_observation(runtime: RuntimeObservation) -> None:
    if not isinstance(runtime, RuntimeObservation):
        raise ContractValidationError("runtime must be a RuntimeObservation")
    for field_name, bits in _RUNTIME_INTEGER_WIDTHS.items():
        _wire_uint(getattr(runtime, field_name), f"runtime.{field_name}", bits)
    for field_name in _RUNTIME_FLOAT_FIELDS:
        _finite(getattr(runtime, field_name), f"runtime.{field_name}")


def validate_foundry_root_extras(extras: FoundryRootExtras) -> None:
    """Validate the direct JSON mirror of Rust ``FoundryRootExtras``."""

    if not isinstance(extras, FoundryRootExtras):
        raise ContractValidationError("foundry extras must be FoundryRootExtras")
    _wire_uint(extras.schema_version, "foundry_extras.schema_version", 16)
    if extras.schema_version != FOUNDRY_CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError("unsupported foundry extras schema_version")
    validate_freshness_identity(extras.freshness)
    for field_name in _FOUNDRY_EXTRAS_FLOAT_FIELDS:
        _finite(getattr(extras, field_name), f"foundry_extras.{field_name}")
    for field_name in _ROOT_OPTIONAL_FLOAT_FIELDS:
        field_value = getattr(extras, field_name)
        if field_value is not None:
            _finite(field_value, f"foundry_extras.{field_name}")
    _wire_uint(extras.revision_count, "foundry_extras.revision_count", 16)
    _validate_runtime_observation(extras.runtime)


def foundry_root_extras_to_payload(extras: FoundryRootExtras) -> dict[str, Any]:
    validate_foundry_root_extras(extras)
    return {
        "schema_version": extras.schema_version,
        "freshness": freshness_to_payload(extras.freshness),
        "entropy": _finite(extras.entropy, "foundry_extras.entropy"),
        "effective_branching": _finite(
            extras.effective_branching, "foundry_extras.effective_branching"
        ),
        "top2_margin": _finite(
            extras.top2_margin, "foundry_extras.top2_margin"
        ),
        "margin_slope": _finite(
            extras.margin_slope, "foundry_extras.margin_slope"
        ),
        "entropy_slope": _finite(
            extras.entropy_slope, "foundry_extras.entropy_slope"
        ),
        "h1_stability": (
            None
            if extras.h1_stability is None
            else _finite(extras.h1_stability, "foundry_extras.h1_stability")
        ),
        "p_flip": (
            None
            if extras.p_flip is None
            else _finite(extras.p_flip, "foundry_extras.p_flip")
        ),
        "prior_visit_js": _finite(
            extras.prior_visit_js, "foundry_extras.prior_visit_js"
        ),
        "omission_bound": _finite(
            extras.omission_bound, "foundry_extras.omission_bound"
        ),
        "revision_count": extras.revision_count,
        "runtime": _runtime_to_payload(extras.runtime),
    }


def foundry_root_extras_from_payload(
    payload: Mapping[str, Any],
) -> FoundryRootExtras:
    row = _mapping(payload, "foundry extras")
    schema_version = _wire_uint(
        _required_member(row, "schema_version", "foundry_extras"),
        "foundry_extras.schema_version",
        16,
    )
    extras = FoundryRootExtras(
        schema_version=schema_version,
        freshness=freshness_from_payload(
            _mapping(
                _required_member(row, "freshness", "foundry_extras"),
                "foundry_extras.freshness",
            )
        ),
        entropy=_finite(
            _required_member(row, "entropy", "foundry_extras"),
            "foundry_extras.entropy",
        ),
        effective_branching=_finite(
            _required_member(row, "effective_branching", "foundry_extras"),
            "foundry_extras.effective_branching",
        ),
        top2_margin=_finite(
            _required_member(row, "top2_margin", "foundry_extras"),
            "foundry_extras.top2_margin",
        ),
        margin_slope=_finite(
            _required_member(row, "margin_slope", "foundry_extras"),
            "foundry_extras.margin_slope",
        ),
        entropy_slope=_finite(
            _required_member(row, "entropy_slope", "foundry_extras"),
            "foundry_extras.entropy_slope",
        ),
        h1_stability=(
            None
            if row.get("h1_stability") is None
            else _finite(row["h1_stability"], "foundry_extras.h1_stability")
        ),
        p_flip=(
            None
            if row.get("p_flip") is None
            else _finite(row["p_flip"], "foundry_extras.p_flip")
        ),
        prior_visit_js=_finite(
            _required_member(row, "prior_visit_js", "foundry_extras"),
            "foundry_extras.prior_visit_js",
        ),
        omission_bound=_finite(
            _required_member(row, "omission_bound", "foundry_extras"),
            "foundry_extras.omission_bound",
        ),
        revision_count=_wire_uint(
            _required_member(row, "revision_count", "foundry_extras"),
            "foundry_extras.revision_count",
            16,
        ),
        runtime=_runtime_from_payload(
            _mapping(
                _required_member(row, "runtime", "foundry_extras"),
                "foundry_extras.runtime",
            )
        ),
    )
    validate_foundry_root_extras(extras)
    return extras


def validate_root_observation(observation: RootObservation) -> None:
    if not isinstance(observation, RootObservation):
        raise ContractValidationError("root must be a RootObservation")
    for field_name, bits in _ROOT_INTEGER_WIDTHS.items():
        _wire_uint(getattr(observation, field_name), f"root.{field_name}", bits)
    for field_name in _ROOT_FLOAT_FIELDS:
        _finite(getattr(observation, field_name), f"root.{field_name}")
    for field_name in _ROOT_OPTIONAL_FLOAT_FIELDS:
        field_value = getattr(observation, field_name)
        if field_value is not None:
            _finite(field_value, f"root.{field_name}")
    _nonempty_string(observation.checkpoint_id, "root.checkpoint_id")
    _wire_string(observation.position_id, "root.position_id")
    _wire_string(observation.game, "root.game")
    _validate_runtime_observation(observation.runtime)
    if observation.n_children < observation.n_visible:
        raise ContractValidationError("visible child count exceeds total child count")
    positions = [edge.edge_pos for edge in observation.edges]
    if len(set(positions)) != len(positions):
        raise ContractValidationError("edge_pos values must be unique")
    for edge in observation.edges:
        _validate_edge_observation(edge)
        if edge.edge_pos >= observation.n_children:
            raise ContractValidationError("edge.edge_pos must be less than root.n_children")
    fresh = observation.freshness_identity()
    validate_freshness_identity(fresh)
    if (
        fresh.root_hash != observation.root_hash
        or fresh.checkpoint_id != observation.checkpoint_id
        or fresh.root_visits != observation.root_visits
        or fresh.iteration != observation.iteration
    ):
        raise ContractValidationError("freshness identity disagrees with root observation")


def root_observation_to_payload(observation: RootObservation) -> dict[str, Any]:
    validate_root_observation(observation)
    body = {
        "root_hash": observation.root_hash,
        "checkpoint_id": observation.checkpoint_id,
        "position_id": observation.position_id,
        "game": observation.game,
        "root_visits": observation.root_visits,
        "iteration": observation.iteration,
        "elapsed_ms": observation.elapsed_ms,
        "remaining_visits": observation.remaining_visits,
        "n_children": observation.n_children,
        "n_visible": observation.n_visible,
        "entropy": _finite(observation.entropy, "root.entropy"),
        "effective_branching": _finite(
            observation.effective_branching, "root.effective_branching"
        ),
        "top2_margin": _finite(observation.top2_margin, "root.top2_margin"),
        "margin_slope": _finite(observation.margin_slope, "root.margin_slope"),
        "entropy_slope": _finite(observation.entropy_slope, "root.entropy_slope"),
        "h1_stability": (
            None
            if observation.h1_stability is None
            else _finite(observation.h1_stability, "root.h1_stability")
        ),
        "p_flip": (
            None
            if observation.p_flip is None
            else _finite(observation.p_flip, "root.p_flip")
        ),
        "prior_visit_js": _finite(observation.prior_visit_js, "root.prior_visit_js"),
        "candidate_omission_bound": _finite(
            observation.candidate_omission_bound,
            "root.candidate_omission_bound",
        ),
        "revision_count": observation.revision_count,
        "edges": [_edge_to_payload(edge) for edge in observation.edges],
        "runtime": _runtime_to_payload(observation.runtime),
        "extras": _json_value(observation.extras),
        "freshness": freshness_to_payload(observation.freshness_identity()),
    }
    return {
        "schema_version": FOUNDRY_CONTRACT_SCHEMA_VERSION,
        "kind": "root_observation",
        "observation": body,
    }


def root_observation_from_payload(payload: Mapping[str, Any]) -> RootObservation:
    top = _mapping(payload, "root payload")
    schema_version = _wire_uint(
        top.get("schema_version"), "root.schema_version", 16
    )
    if schema_version != FOUNDRY_CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError("unsupported root observation schema_version")
    if top.get("kind") != "root_observation":
        raise ContractValidationError("payload kind is not root_observation")
    row = _mapping(top.get("observation"), "observation")
    runtime = (
        _runtime_from_payload(_mapping(row["runtime"], "runtime"))
        if "runtime" in row
        else RuntimeObservation()
    )
    observation = RootObservation(
        root_hash=_wire_uint(row["root_hash"], "root.root_hash", 64),
        checkpoint_id=_nonempty_string(row["checkpoint_id"], "root.checkpoint_id"),
        position_id=_wire_string(row["position_id"], "root.position_id"),
        game=_wire_string(row["game"], "root.game"),
        root_visits=_wire_uint(row["root_visits"], "root.root_visits", 32),
        iteration=_wire_uint(row["iteration"], "root.iteration", 64),
        elapsed_ms=_wire_uint(row["elapsed_ms"], "root.elapsed_ms", 64),
        remaining_visits=_wire_uint(
            row["remaining_visits"], "root.remaining_visits", 32
        ),
        n_children=_wire_uint(row["n_children"], "root.n_children", 16),
        n_visible=_wire_uint(row["n_visible"], "root.n_visible", 16),
        entropy=_finite(row["entropy"], "root.entropy"),
        effective_branching=_finite(
            row["effective_branching"], "root.effective_branching"
        ),
        top2_margin=_finite(row["top2_margin"], "root.top2_margin"),
        margin_slope=_finite(row["margin_slope"], "root.margin_slope"),
        entropy_slope=_finite(row["entropy_slope"], "root.entropy_slope"),
        h1_stability=(
            None
            if row.get("h1_stability") is None
            else _finite(row["h1_stability"], "root.h1_stability")
        ),
        p_flip=(
            None
            if row.get("p_flip") is None
            else _finite(row["p_flip"], "root.p_flip")
        ),
        prior_visit_js=_finite(row["prior_visit_js"], "root.prior_visit_js"),
        candidate_omission_bound=_finite(
            row["candidate_omission_bound"], "root.candidate_omission_bound"
        ),
        revision_count=_wire_uint(
            row["revision_count"], "root.revision_count", 16
        ),
        edges=tuple(_edge_from_payload(item) for item in row["edges"]),
        runtime=runtime,
        extras=dict(_mapping(row.get("extras", {}), "extras")),
        freshness=freshness_from_payload(_mapping(row["freshness"], "freshness")),
    )
    validate_root_observation(observation)
    return observation


def _action_to_payload(action: MetaAction) -> dict[str, Any]:
    validate_meta_action(action)
    return {
        "kind": action.kind.value,
        "primary": action.primary,
        "secondary": action.secondary,
        "amount": action.amount,
        "value": (
            None
            if action.value is None
            else _finite(action.value, "action.value")
        ),
        "label": action.label,
    }


def _action_from_payload(payload: Mapping[str, Any]) -> MetaAction:
    row = _mapping(payload, "action")
    try:
        raw_kind = row["kind"]
        if not isinstance(raw_kind, str):
            raise ValueError
        kind = MetaActionKind(raw_kind)
    except (KeyError, ValueError) as exc:
        raise ContractValidationError("unknown action kind") from exc
    action = MetaAction(
        kind=kind,
        primary=row.get("primary"),
        secondary=row.get("secondary"),
        amount=row.get("amount", 0),
        value=row.get("value"),
        label=row.get("label"),
    )
    validate_meta_action(action)
    return action


def validate_meta_action(action: MetaAction) -> None:
    """Mirror Rust MetaActionWire widths and per-kind canonical fields."""

    if not isinstance(action, MetaAction) or not isinstance(action.kind, MetaActionKind):
        raise ContractValidationError("action must have a known MetaActionKind")
    rules = _ACTION_FIELD_RULES[action.kind]

    # Rust first deserializes these generic wire integers as u64, before
    # narrowing them according to the typed action variant.
    for field_name in ("primary", "secondary"):
        field_value = getattr(action, field_name)
        if field_value is not None:
            _wire_uint(field_value, f"action.{field_name}", 64)
    _wire_uint(action.amount, "action.amount", 64)

    for field_name, canonical_default in _ACTION_CANONICAL_DEFAULTS.items():
        field_value = getattr(action, field_name)
        rule = rules.get(field_name)
        if rule is None:
            if field_value != canonical_default:
                raise ContractValidationError(
                    f"action.{field_name} is forbidden for {action.kind.value}"
                )
            continue
        if rule == "required_finite":
            if field_value is None:
                raise ContractValidationError(
                    f"action.{field_name} is required for {action.kind.value}"
                )
            _finite(field_value, f"action.{field_name}")
            continue
        optional = rule.startswith("optional_")
        if field_value is None:
            if optional:
                continue
            raise ContractValidationError(
                f"action.{field_name} is required for {action.kind.value}"
            )
        bits = int(rule.rsplit("u", 1)[1])
        _wire_uint(field_value, f"action.{field_name}", bits)

    if action.label is not None:
        _nonempty_string(action.label, "action.label")


def validate_proposal(proposal: MetaProposal) -> None:
    _wire_uint(proposal.schema_version, "proposal.schema_version", 16)
    if proposal.schema_version != FOUNDRY_CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError("unsupported proposal schema_version")
    _wire_string(proposal.axis_id, "proposal.axis_id")
    if not _AXIS_RE.fullmatch(proposal.axis_id):
        raise ContractValidationError(f"invalid axis_id: {proposal.axis_id!r}")
    _nonempty_string(proposal.activation_guard, "proposal.activation_guard")
    _nonempty_string(proposal.explanation, "proposal.explanation")
    _nonempty_string(proposal.evidence_scope, "proposal.evidence_scope")
    validate_meta_action(proposal.action)
    estimate = proposal.estimate
    for label, value in (
        ("regret_reduction_mean", estimate.regret_reduction_mean),
        ("regret_reduction_lcb", estimate.regret_reduction_lcb),
        ("confidence", estimate.confidence),
    ):
        _finite(value, f"estimate.{label}")
    if not 0.0 <= estimate.confidence <= 1.0:
        raise ContractValidationError("estimate confidence must be in [0, 1]")
    cost_vector_to_payload(estimate.cost)
    _json_value(proposal.telemetry)


def proposal_to_payload(proposal: MetaProposal) -> dict[str, Any]:
    validate_proposal(proposal)
    return {
        "schema_version": FOUNDRY_CONTRACT_SCHEMA_VERSION,
        "kind": "meta_proposal",
        "proposal": {
            "axis_id": proposal.axis_id,
            "action": _action_to_payload(proposal.action),
            "estimate": {
                "regret_reduction_mean": _finite(
                    proposal.estimate.regret_reduction_mean,
                    "estimate.regret_reduction_mean",
                ),
                "regret_reduction_lcb": _finite(
                    proposal.estimate.regret_reduction_lcb,
                    "estimate.regret_reduction_lcb",
                ),
                "confidence": _finite(
                    proposal.estimate.confidence, "estimate.confidence"
                ),
                "cost": cost_vector_to_payload(proposal.estimate.cost),
            },
            "activation_guard": proposal.activation_guard,
            "explanation": proposal.explanation,
            "evidence_scope": proposal.evidence_scope,
            "telemetry": _json_value(proposal.telemetry),
        },
    }


def proposal_from_payload(payload: Mapping[str, Any]) -> MetaProposal:
    top = _mapping(payload, "proposal payload")
    schema_version = _wire_uint(
        top.get("schema_version"), "proposal.schema_version", 16
    )
    if schema_version != FOUNDRY_CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError("unsupported proposal schema_version")
    if top.get("kind") != "meta_proposal":
        raise ContractValidationError("payload kind is not meta_proposal")
    row = _mapping(top.get("proposal"), "proposal")
    estimate_row = _mapping(row.get("estimate"), "estimate")
    cost_row = _mapping(
        _required_member(estimate_row, "cost", "estimate"), "estimate.cost"
    )
    proposal = MetaProposal(
        axis_id=row["axis_id"],
        action=_action_from_payload(_mapping(row.get("action"), "action")),
        estimate=ProposalEstimate(
            regret_reduction_mean=_finite(
                _required_member(
                    estimate_row, "regret_reduction_mean", "estimate"
                ),
                "estimate.regret_reduction_mean",
            ),
            regret_reduction_lcb=_finite(
                _required_member(
                    estimate_row, "regret_reduction_lcb", "estimate"
                ),
                "estimate.regret_reduction_lcb",
            ),
            confidence=_finite(
                _required_member(estimate_row, "confidence", "estimate"),
                "estimate.confidence",
            ),
            cost=cost_vector_from_payload(cost_row),
        ),
        activation_guard=row.get("activation_guard"),
        explanation=row.get("explanation"),
        schema_version=schema_version,
        evidence_scope=row.get("evidence_scope"),
        telemetry=dict(_mapping(row.get("telemetry", {}), "telemetry")),
    )
    validate_proposal(proposal)
    return proposal
