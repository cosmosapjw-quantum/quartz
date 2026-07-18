"""Contract and numerical tests for the A15 matched CPU/CUDA diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quartz.experiment_manifest import file_sha256
from quartz.experiments import a15_matched_service_curve as a15

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "a15_matched_service_curve.v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _timing_row(
    backend: str, *, input_hash: str = "a" * 64, rate: float = 10.0
) -> dict:
    return {
        "schema_version": 1,
        "axis_id": "A15",
        "role": a15.ROLE,
        "evidence_status": a15.EVIDENCE_STATUS,
        "claim_scope": a15.CLAIM_SCOPE,
        "workload_identity_sha256": "b" * 64,
        "cell_id": "b8_i1",
        "backend": backend,
        "batch_size": 8,
        "inflight": 1,
        "repetition": 0,
        "first_backend": "cpu",
        "input_seed": 17,
        "input_sha256": input_hash,
        "n_waves": 2,
        "warmup_waves": 1,
        "items_per_s": rate,
        "amortized_ms_per_batch": 2.0,
    }


def test_profiles_are_versioned_and_full_matrix_is_separate():
    cfg = _config()
    diagnostic = a15.validate_config(cfg, "diagnostic")
    full = a15.validate_config(cfg, "full")
    assert len(diagnostic["batch_sizes"]) * len(diagnostic["inflight_grid"]) == 4
    assert len(full["batch_sizes"]) * len(full["inflight_grid"]) == 24
    assert full["repetitions"] >= 5
    assert diagnostic["name"] == "diagnostic"
    assert full["name"] == "full"


def test_profile_validation_rejects_duplicate_cells():
    cfg = _config()
    cfg["profiles"]["diagnostic"]["batch_sizes"] = [8, 8]
    with pytest.raises(ValueError, match="duplicates"):
        a15.validate_config(cfg, "diagnostic")


def test_seed_order_and_determinism_contract_are_explicit():
    cfg = _config()
    assert a15.validate_config(cfg, "diagnostic")["name"] == "diagnostic"
    cfg["seed_contract"]["backend_order"] = "cpu_then_cuda"
    with pytest.raises(ValueError, match="backend_order"):
        a15.validate_config(cfg, "diagnostic")
    cfg = _config()
    del cfg["runtime_contract"]["deterministic_algorithms"]
    with pytest.raises(ValueError, match="deterministic_algorithms"):
        a15.validate_config(cfg, "diagnostic")


def test_model_state_and_inputs_are_deterministic():
    torch.manual_seed(123)
    model_a = a15.build_eval_model(4, 8, 1, 5, 25).eval()
    torch.manual_seed(123)
    model_b = a15.build_eval_model(4, 8, 1, 5, 25).eval()
    assert a15.model_state_sha256(model_a) == a15.model_state_sha256(model_b)

    first, first_hash, first_seed = a15.make_matched_inputs(
        batch_size=3, inflight=2, in_ch=4, board=5, base_seed=44, repetition=0
    )
    second, second_hash, second_seed = a15.make_matched_inputs(
        batch_size=3, inflight=2, in_ch=4, board=5, base_seed=44, repetition=0
    )
    assert first_hash == second_hash
    assert first_seed == second_seed
    assert all(torch.equal(left, right) for left, right in zip(first, second))

    _, other_hash, other_seed = a15.make_matched_inputs(
        batch_size=3, inflight=2, in_ch=4, board=5, base_seed=44, repetition=1
    )
    assert other_hash != first_hash
    assert other_seed != first_seed


def test_workload_identity_binds_state_grid_threads_and_source():
    contract, digest = a15.build_workload_identity(
        model_spec={
            "label": "test",
            "in_ch": 4,
            "channels": 8,
            "blocks": 1,
            "board": 5,
            "actions": 25,
        },
        model_state_hash="a" * 64,
        profile={
            "name": "diagnostic",
            "batch_sizes": [1],
            "inflight_grid": [1],
            "repetitions": 2,
            "n_waves": 2,
            "warmup_waves": 1,
        },
        seed_contract={"model_seed": 1, "input_seed": 2},
        runtime_contract={"cpu_intraop_threads": 1, "cpu_interop_threads": 1},
        builder_source_sha256="b" * 64,
    )
    changed, changed_digest = a15.build_workload_identity(
        model_spec=contract["model_spec"],
        model_state_hash="a" * 64,
        profile=contract["profile"],
        seed_contract=contract["seed_contract"],
        runtime_contract={"cpu_intraop_threads": 2, "cpu_interop_threads": 1},
        builder_source_sha256="b" * 64,
    )
    assert len(digest) == 64
    assert digest != changed_digest
    assert contract["backend_submission_semantics"]["cpu"].startswith("serial")
    assert changed["runtime_contract"]["cpu_intraop_threads"] == 2


def test_measurement_excludes_setup_and_computes_units():
    model = a15.build_eval_model(4, 8, 1, 5, 25).eval()
    inputs, _, _ = a15.make_matched_inputs(
        batch_size=10, inflight=2, in_ch=4, board=5, base_seed=2, repetition=0
    )
    ticks = iter([4.0, 6.0])
    measured = a15.measure_prepared_inputs(
        model,
        torch.device("cpu"),
        inputs,
        n_waves=5,
        warmup_waves=1,
        time_fn=lambda: next(ticks),
    )
    assert measured["total_items"] == 100
    assert measured["total_batches"] == 10
    assert measured["items_per_s"] == pytest.approx(50.0)
    assert measured["amortized_ms_per_batch"] == pytest.approx(200.0)
    assert measured["wave_latency_ms"] == pytest.approx(400.0)


def test_exact_pairing_is_descriptive_and_fail_closed():
    cpu = _timing_row("cpu", rate=10.0)
    cuda = _timing_row("cuda", rate=25.0)
    paired = a15.pair_rows([cpu, cuda])
    assert paired[0]["cuda_to_cpu_throughput_ratio"] == pytest.approx(2.5)
    assert paired[0]["claim_scope"] == "ablation_readiness_only"

    mismatched = dict(cuda, input_sha256="c" * 64)
    with pytest.raises(ValueError, match="input_sha256"):
        a15.pair_rows([cpu, mismatched])
    with pytest.raises(ValueError, match="incomplete"):
        a15.pair_rows([cpu])


def test_summary_never_promotes_and_marks_diagnostic_profile_incomplete_as_full():
    profile = {
        "name": "diagnostic",
        "batch_sizes": [8],
        "inflight_grid": [1],
        "repetitions": 1,
        "n_waves": 2,
        "warmup_waves": 1,
    }
    rows = [_timing_row("cpu", rate=10.0), _timing_row("cuda", rate=20.0)]
    paired = a15.pair_rows(rows)
    summary = a15.diagnostic_summary(
        rows,
        paired,
        profile=profile,
        parity={"passed": True},
        workload_identity_sha256="b" * 64,
    )
    assert summary["promotion"] == {"auto": False, "eligible": False}
    assert summary["matched_pair_contract_complete"] is True
    assert summary["pipeline_ready_for_full_profile"] is True
    assert summary["full_matrix_executed"] is False
    assert summary["role"] == "ablation_readiness"


def test_diagnostic_plot_is_labeled_and_nonempty(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [_timing_row("cpu", rate=10.0), _timing_row("cuda", rate=20.0)]
    aggregates = a15.aggregate_rows(rows)
    output = tmp_path / "diagnostic.png"
    a15.render_diagnostic_plot(output, aggregates, workload_identity_sha256="b" * 64)
    assert output.is_file()
    assert output.stat().st_size > 1000


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA parity requires a visible CUDA device"
)
def test_cpu_cuda_semantic_parity_on_fixed_model_and_input():
    torch.manual_seed(99)
    cpu_model = a15.build_eval_model(4, 8, 1, 5, 25).eval()
    cuda_model = copy.deepcopy(cpu_model).to("cuda").eval()
    probe = torch.randn(4, 4, 5, 5, generator=torch.Generator().manual_seed(100))
    parity = a15.check_backend_parity(
        cpu_model, cuda_model, probe, atol=1e-4, rtol=1e-4
    )
    assert parity["passed"] is True
    assert parity["policy_top1_agreement"] == 1.0
    assert parity["policy_max_abs_error"] <= 1e-4
    assert parity["value_max_abs_error"] <= 1e-4


def test_config_and_existing_builder_are_hashable_inputs():
    builder = REPO_ROOT / "quartz" / "experiments" / "service_curve.py"
    assert len(file_sha256(CONFIG_PATH)) == 64
    assert len(file_sha256(builder)) == 64


def test_published_output_retry_is_hash_verified(tmp_path):
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "a15_matched_service_curve.py"
    spec = importlib.util.spec_from_file_location("a15_retry_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "published"
    output.mkdir()
    rows = output / "rows.jsonl"
    summary = output / "summary.json"
    plot = output / "diagnostic.png"
    service_rows = output / "service_curve_rows.v1.csv"
    paired_rows = output / "paired_backend_rows.v1.csv"
    plot_metadata = output / "plot_metadata.v1.json"
    rows.write_text("{}\n", encoding="utf-8")
    summary.write_text(
        json.dumps({"execution_status": "completed_no_promotion"}), encoding="utf-8"
    )
    plot.write_bytes(b"png")
    service_rows.write_text("backend,items_per_s\n", encoding="utf-8")
    paired_rows.write_text("cell_id,cpu,cuda\n", encoding="utf-8")
    plot_metadata.write_text("{}\n", encoding="utf-8")
    source_hash = file_sha256(script_path)
    artifacts = [
        {"path": path.name, "sha256": file_sha256(path)}
        for path in (
            rows,
            summary,
            plot,
            service_rows,
            paired_rows,
            plot_metadata,
        )
    ]
    manifest = {
        "status": "completed_no_promotion",
        "evidence_status": a15.EVIDENCE_STATUS,
        "resolved_config": {"profile": {"name": "diagnostic"}},
        "input_hashes": [{"name": "a15_config", "sha256": file_sha256(CONFIG_PATH)}],
        "source_hashes": [
            {
                "path": str(script_path.relative_to(REPO_ROOT)),
                "sha256": source_hash,
            }
        ],
        "artifacts": artifacts,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        module.validate_complete_output(
            output, config_path=CONFIG_PATH, profile_name="diagnostic"
        )["status"]
        == "completed_no_promotion"
    )
    rows.write_text('{"drift": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash drift"):
        module.validate_complete_output(
            output, config_path=CONFIG_PATH, profile_name="diagnostic"
        )
