from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from quartz.experiment_manifest import file_sha256
from quartz.idea_foundry.meta_analysis import (
    MetaAnalysisError,
    pool_effect_group,
    pool_effect_records,
    run_meta_analysis,
)


def _effect(
    *,
    run_id: str,
    independent_group_id: str,
    effect: float,
    standard_error: float,
    estimand_id: str = "paired_win_rate_delta",
) -> dict[str, object]:
    return {
        "axis_id": "A15",
        "estimand_id": estimand_id,
        "effect_scale": "risk_difference",
        "reference_id": "fixed_controller_v1",
        "unit": "fraction",
        "higher_is_better": True,
        "run_id": run_id,
        "independent_group_id": independent_group_id,
        "effect": effect,
        "standard_error": standard_error,
        "claim_scope": "paired_ablation_analysis_only",
        "evidence_status": "preregistered_ablation",
        "source_artifact_path": "source.json",
        "source_artifact_sha256": "0" * 64,
    }


def test_known_inverse_variance_and_random_effects_result() -> None:
    pooled = pool_effect_group(
        [
            _effect(
                run_id="r1", independent_group_id="g1", effect=1.0, standard_error=1.0
            ),
            _effect(
                run_id="r2", independent_group_id="g2", effect=2.0, standard_error=1.0
            ),
        ]
    )
    assert pooled["status"] == "POOLED_ANALYSIS_ONLY"
    assert pooled["fixed_effect"] == pytest.approx(1.5)
    assert pooled["fixed_standard_error"] == pytest.approx(math.sqrt(0.5))
    assert pooled["cochran_q"] == pytest.approx(0.5)
    assert pooled["tau_squared_dl"] == pytest.approx(0.0)
    assert pooled["random_effect"] == pytest.approx(1.5)


def test_meta_results_are_invariant_to_record_order() -> None:
    records = [
        _effect(run_id="r1", independent_group_id="g1", effect=0.2, standard_error=0.1),
        _effect(run_id="r2", independent_group_id="g2", effect=0.4, standard_error=0.2),
        _effect(
            run_id="r3",
            independent_group_id="g3",
            effect=-0.1,
            standard_error=0.3,
            estimand_id="latency_delta",
        ),
    ]
    expected = pool_effect_records(records)
    for seed in range(20):
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        assert pool_effect_records(shuffled) == expected


def test_incompatible_estimands_are_never_pooled_together() -> None:
    groups = pool_effect_records(
        [
            _effect(
                run_id="r1", independent_group_id="g1", effect=0.2, standard_error=0.1
            ),
            _effect(
                run_id="r2",
                independent_group_id="g2",
                effect=0.2,
                standard_error=0.1,
                estimand_id="latency_delta",
            ),
        ]
    )
    assert len(groups) == 2
    assert {row["status"] for row in groups} == {"INSUFFICIENT_INDEPENDENT_EFFECTS"}


def test_duplicate_independent_group_is_rejected() -> None:
    with pytest.raises(MetaAnalysisError, match="double-count"):
        pool_effect_group(
            [
                _effect(
                    run_id="r1",
                    independent_group_id="g1",
                    effect=0.2,
                    standard_error=0.1,
                ),
                _effect(
                    run_id="r2",
                    independent_group_id="g1",
                    effect=0.3,
                    standard_error=0.1,
                ),
            ]
        )


def test_contract_diagnostic_cannot_be_submitted_as_effect() -> None:
    record = _effect(
        run_id="r1", independent_group_id="g1", effect=1.0, standard_error=0.1
    )
    record["claim_scope"] = "synthetic_contract_analysis_only"
    with pytest.raises(MetaAnalysisError, match="not admissible"):
        pool_effect_group([record])


def test_meta_analysis_verifies_source_hash_and_emits_no_promotion(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text('{"paired_seeds": [1, 2]}\n', encoding="utf-8")
    records = [
        _effect(run_id="r1", independent_group_id="g1", effect=0.2, standard_error=0.1),
        _effect(run_id="r2", independent_group_id="g2", effect=0.4, standard_error=0.2),
    ]
    for record in records:
        record["source_artifact_sha256"] = file_sha256(source_path)
    input_path = tmp_path / "effects.json"
    input_path.write_text(json.dumps({"effect_records": records}), encoding="utf-8")
    payload = run_meta_analysis([input_path], tmp_path / "meta")
    assert payload["status"] == "COMPLETED_ANALYSIS_ONLY"
    assert payload["pooled_group_count"] == 1
    assert payload["promotion"]["eligible"] is False

    source_path.write_text('{"paired_seeds": [9]}\n', encoding="utf-8")
    with pytest.raises(MetaAnalysisError, match="hash mismatch"):
        run_meta_analysis([input_path], tmp_path / "meta-tampered")
