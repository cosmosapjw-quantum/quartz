import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "idea_lab.local.v2.json"
AXIS_PATH = REPO_ROOT / "configs" / "idea_foundry.axes.v1.json"

EXPECTED_AXIS_IDS = [f"A{idx:02d}" for idx in range(1, 27)]
EXPECTED_FIRST_GATE_ORDER = [
    "A03.trace",
    "A09.trace",
    "A01.trace",
    "A02.shadow",
    "A17.trace",
    "A21.analysis",
    "A22.analysis",
    "A06.synthetic",
    "A07.synthetic",
    "A12.synthetic",
    "A25.synthetic",
    "A26.analysis",
    "A05.counterfactual",
    "A04.shadow",
    "A08.conditional_audit",
    "A13.synthetic",
    "A15.system",
    "A14.shadow",
    "A16.cache_only",
    "A11.synthetic",
    "A18.representation",
    "A19.representation",
    "A23.deployment",
    "A20.training_control",
    "A24.training_control",
    "A10.conditional_audit",
]

REQUIRED_LANE_KEYS = {
    "axis_id",
    "role",
    "execution_status",
    "evidence_status",
    "claim_scope",
    "depends_on",
    "dependency_condition",
    "resource_profile",
    "inputs",
    "seed_contract",
    "steps",
    "expected_artifacts",
    "promotion_gate",
    "resume_policy",
    "prohibited_inferences",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_idea_lab_module():
    path = REPO_ROOT / "scripts" / "idea_lab.py"
    spec = importlib.util.spec_from_file_location("idea_lab_registry_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_orchestrator_accepts_v2_registry():
    idea_lab = load_idea_lab_module()
    registry = idea_lab.load_config(REGISTRY_PATH)

    assert registry["format_version"] == 2
    assert registry["_axis_ids"] == EXPECTED_AXIS_IDS
    assert idea_lab._topological_ids(registry["lanes"])


def test_v2_registry_covers_exact_axis_set_and_required_lane_contract():
    registry = load_json(REGISTRY_PATH)
    axes = load_json(AXIS_PATH)

    assert registry["format_version"] == 2
    assert registry["default_suite"] == "first-gate-all"
    assert [axis["id"] for axis in axes["axes"]] == EXPECTED_AXIS_IDS
    assert registry["axis_registry"] == "configs/idea_foundry.axes.v1.json"
    assert registry["axis_registry_contract"]["required_axis_ids"] == EXPECTED_AXIS_IDS
    assert registry["axis_registry_contract"]["sha256"] == hashlib.sha256(
        AXIS_PATH.read_bytes()
    ).hexdigest()

    lanes = registry["lanes"]
    lane_ids = [lane["id"] for lane in lanes]
    assert len(lane_ids) == len(set(lane_ids))
    assert {lane["axis_id"] for lane in lanes} == set(EXPECTED_AXIS_IDS)
    for lane in lanes:
        assert REQUIRED_LANE_KEYS <= lane.keys(), lane["id"]
        assert lane["axis_id"] in EXPECTED_AXIS_IDS
        assert lane["execution_status"] in {"available", "blocked"}
        assert lane["evidence_status"] in {
            "skeleton_only",
            "mechanism_valid",
            "shadow_only",
            "conditional_only",
            "analysis_only",
        }
        assert lane["promotion_gate"]["allow_auto_promotion"] is False
        assert lane["resume_policy"]["overwrite_is_resume"] is False
        assert lane["prohibited_inferences"]


def test_first_gate_suite_has_exact_preregistered_order_and_acyclic_dependencies():
    registry = load_json(REGISTRY_PATH)
    lanes = {lane["id"]: lane for lane in registry["lanes"]}
    ordered = registry["suites"]["first-gate-all"]

    assert ordered == EXPECTED_FIRST_GATE_ORDER
    assert [lanes[lane_id]["axis_id"] for lane_id in ordered]
    assert len({lanes[lane_id]["axis_id"] for lane_id in ordered}) == 26
    for index, lane_id in enumerate(ordered):
        assert lanes[lane_id]["execution_status"] == "available"
        expected = [] if index == 0 else [ordered[index - 1]]
        assert lanes[lane_id]["depends_on"] == expected
        assert (
            lanes[lane_id]["dependency_condition"]
            == "terminal_without_technical_failure"
        )

    visiting = set()
    visited = set()

    def visit(lane_id):
        if lane_id in visiting:
            raise AssertionError(f"dependency cycle through {lane_id}")
        if lane_id in visited:
            return
        visiting.add(lane_id)
        for dependency in lanes[lane_id]["depends_on"]:
            assert dependency in lanes
            visit(dependency)
        visiting.remove(lane_id)
        visited.add(lane_id)

    for lane_id in lanes:
        visit(lane_id)


def test_stage_contracts_pin_scientific_first_gate_invariants():
    contracts = load_json(REGISTRY_PATH)["stage_contracts"]

    assert contracts["trace_shadow"]["order"] == [
        "A03",
        "A09",
        "A01",
        "A02",
        "A17",
        "A21",
        "A22",
    ]
    assert contracts["synthetic_candidate_bank"]["fixture_bank"] == [
        "hidden_best",
        "near_tie",
        "multimodal",
    ]
    assert "total budget conserved" in contracts["synthetic_candidate_bank"][
        "invariants"
    ]
    assert contracts["counterfactual"]["actions"] == [
        "STOP",
        "SAMPLE_INCUMBENT",
        "SAMPLE_CHALLENGER",
        "WIDEN",
    ]
    assert contracts["live_allocation"]["promotion_order"] == [
        "A04.live",
        "A06.live",
        "A07.live",
        "A08.live",
    ]
    assert contracts["conditional_audit"]["no_eligible_outcome"] == (
        "DORMANT_NO_ELIGIBLE_SLICE"
    )
    assert contracts["conditional_audit"]["default_policy"] == "no_refresh"


def test_first_gate_steps_call_axis_gate_and_require_versioned_artifacts():
    registry = load_json(REGISTRY_PATH)
    lanes = {lane["id"]: lane for lane in registry["lanes"]}

    for lane_id in registry["suites"]["first-gate-all"]:
        lane = lanes[lane_id]
        assert len(lane["steps"]) == 1
        command = lane["steps"][0]["command"]
        assert command[:2] == ["{python}", "scripts/idea_foundry_axis_gate.py"]
        assert command[command.index("--axis") + 1] == lane["axis_id"]
        assert command[command.index("--role") + 1] == lane["role"]
        assert command[command.index("--seed") + 1] == "20260718"
        assert lane["seed_contract"]["seeds"] == [20260718]
        artifacts = {item["path"]: item for item in lane["expected_artifacts"]}
        assert set(artifacts) == {
            "{output}/run_manifest.json",
            "{output}/rows.jsonl",
            "{output}/summary.json",
        }
        assert all(item["required"] for item in artifacts.values())
        assert all(item["schema"]["required_keys"] for item in artifacts.values())
        assert lane["seed_contract"]["trace_salt_count"] == 1
        assert lane["seed_contract"][
            "restart_and_continuation_kept_separate"
        ]


def test_live_multirole_chain_and_hardware_lanes_are_fail_closed():
    registry = load_json(REGISTRY_PATH)
    lanes = {lane["id"]: lane for lane in registry["lanes"]}

    for axis_id in ("A06", "A07"):
        assert {lane["role"] for lane in registry["lanes"] if lane["axis_id"] == axis_id} == {
            "synthetic",
            "live",
        }
    assert registry["suites"]["live-promotion-blocked"] == [
        "A04.live",
        "A06.live",
        "A07.live",
        "A08.live",
    ]
    assert lanes["A06.live"]["depends_on"] == ["A04.live", "A06.synthetic"]
    assert lanes["A07.live"]["depends_on"] == ["A06.live", "A07.synthetic"]
    assert lanes["A08.live"]["depends_on"] == [
        "A07.live",
        "A08.conditional_audit",
    ]
    for lane_id in registry["suites"]["live-promotion-blocked"]:
        lane = lanes[lane_id]
        assert lane["execution_status"] == "blocked"
        assert lane["dependency_condition"] == "all_succeeded"
        assert lane["blocked_by"]

    for lane_id in ("A15.cuda", "A18.cuda", "A19.cuda"):
        lane = lanes[lane_id]
        assert lane["execution_status"] == "blocked"
        assert lane["claim_scope"] == "blocked_promotion_prerequisites_only"
        assert lane["resource_profile"]["hardware_gate"] == "cuda"
        assert lane["resource_profile"]["allowed"] == ["cuda"]
        assert not any("hardware" in blocker.lower() for blocker in lane["blocked_by"])


def test_a15_a19_ablation_readiness_chain_is_executable_but_non_promoting():
    registry = load_json(REGISTRY_PATH)
    lanes = {lane["id"]: lane for lane in registry["lanes"]}
    ordered = registry["suites"]["a15-a19-ablation-readiness"]

    assert ordered == [
        "A15.ablation_readiness",
        "A18.ablation_readiness",
        "A19.ablation_readiness",
    ]
    assert len(registry["lanes"]) == 36
    for index, lane_id in enumerate(ordered):
        lane = lanes[lane_id]
        assert lane["execution_status"] == "available"
        assert lane["role"] == "ablation_readiness"
        assert lane["evidence_status"] == "skeleton_only"
        assert lane["claim_scope"] == "ablation_readiness_only"
        assert lane["resource_profile"]["hardware_gate"] == "cuda"
        assert lane["promotion_gate"]["allow_auto_promotion"] is False
        assert lane["steps"]
        expected_dependency = [] if index == 0 else [ordered[index - 1]]
        assert lane["depends_on"] == expected_dependency

    assert lanes["A15.cuda"]["depends_on"] == ["A15.ablation_readiness"]
    assert lanes["A18.cuda"]["depends_on"][0] == "A18.ablation_readiness"
    assert lanes["A19.cuda"]["depends_on"][0] == "A19.ablation_readiness"
    assert any(
        "no measured graph-seed shortlist" in blocker
        for blocker in lanes["A19.cuda"]["blocked_by"]
    )


def test_analysis_only_and_conditional_lanes_cannot_drift_into_live_claims():
    registry = load_json(REGISTRY_PATH)
    lanes = {lane["id"]: lane for lane in registry["lanes"]}

    for lane_id in ("A21.analysis", "A22.analysis", "A26.analysis"):
        lane = lanes[lane_id]
        assert lane["role"] == "analysis"
        assert lane["evidence_status"] == "analysis_only"
        assert lane["claim_scope"] == "synthetic_contract_gate_only"
        assert "live" not in lane["groups"]
        assert any(
            "live meta-action" in inference or "online controller" in inference
            for inference in lane["prohibited_inferences"]
        )
    a10 = lanes["A10.conditional_audit"]
    assert a10["execution_status"] == "available"
    assert a10["role"] == "conditional_audit"
    assert a10["conditional_outcomes"] == {
        "no_eligible_slice": "DORMANT_NO_ELIGIBLE_SLICE",
        "runtime_terminal_status": "completed_no_promotion",
        "default_policy_after_audit": "no_refresh",
    }
    assert a10["promotion_gate"]["allow_auto_promotion"] is False


def test_import_receipt_and_payload_manifest_are_pinned():
    receipt = load_json(REPO_ROOT / "docs" / "idea_foundry" / "IMPORT_RECEIPT.json")
    archives = {item["path"]: item for item in receipt["archives"]}

    assert receipt["applied_commit"]["commit"] == (
        "7f332d60f7152548717a671cbda690697e06a040"
    )
    assert archives["quartz_idea_foundry_skeleton.zip"]["sha256"] == (
        "b2919ea8ee4e9451fb44fee6b35b3b4a25540008dc79e83127215f78444ca464"
    )
    assert archives["quartz_idea_foundry_skeleton.patch"]["sha256"] == (
        "3563cf51a373138a8ed6a72ba03acb70b844ab9f885e710055cd53b49d86d014"
    )
    for archive_name, metadata in archives.items():
        archive_path = REPO_ROOT / archive_name
        if archive_path.exists():
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == metadata[
                "sha256"
            ]

    payloads = (REPO_ROOT / "BUNDLE_FILE_LIST.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(payloads) == len(set(payloads)) == 21
    assert all(payload and not payload.startswith("./.git/") for payload in payloads)
    assert payloads[0:3] == [
        "APPLY_TO_REPO.md",
        "BUNDLE_FILE_LIST.txt",
        "configs/idea_foundry.axes.v1.json",
    ]
