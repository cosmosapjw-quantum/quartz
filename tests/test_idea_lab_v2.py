import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "idea_lab.py"
    spec = importlib.util.spec_from_file_location("idea_lab_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def artifact_writer(*, promotion_eligible=True, omit_rows=False, execution_status=None):
    execution_status = execution_status or (
        "succeeded" if promotion_eligible else "completed_no_promotion"
    )
    script = [
        "import json,sys",
        "from pathlib import Path",
        "out=Path(sys.argv[1])",
        "out.mkdir(parents=True,exist_ok=True)",
        "(out/'run_manifest.json').write_text(json.dumps({'axis_id':'A01','role':'synthetic','claim_scope':'synthetic_contract_gate_only'}))",
        f"(out/'summary.json').write_text(json.dumps({{'promotion_eligible':{promotion_eligible!r},'axis_id':'A01','role':'synthetic','claim_scope':'synthetic_contract_gate_only','evidence_status':'skeleton_only','execution_status':{execution_status!r}}}))",
    ]
    if not omit_rows:
        script.append(
            "(out/'rows.jsonl').write_text(json.dumps({'axis_id':'A01','role':'synthetic','value':1})+'\\n')"
        )
    return ";".join(script)


def write_v2_registry(
    tmp_path,
    *,
    command=None,
    timeout_s=5,
    promotion_eligible=True,
    execution_status=None,
):
    axes = [{"id": f"A{idx:02d}"} for idx in range(1, 27)]
    axis_path = tmp_path / "axes.json"
    axis_path.write_text(json.dumps({"axes": axes}), encoding="utf-8")
    source_path = tmp_path / "producer_source.py"
    source_path.write_text("SOURCE_VERSION = 1\n", encoding="utf-8")
    command = command or [
        sys.executable,
        "-c",
        artifact_writer(
            promotion_eligible=promotion_eligible,
            execution_status=execution_status,
        ),
        "{output}",
    ]

    lanes = []
    for idx in range(1, 27):
        axis_id = f"A{idx:02d}"
        available = idx == 1
        lane = {
            "id": f"{axis_id}.synthetic",
            "axis_id": axis_id,
            "title": axis_id,
            "groups": ["test"],
            "profiles": ["cpu"],
            "description": "test lane",
            "role": "synthetic",
            "execution_status": "available" if available else "blocked",
            "evidence_status": "skeleton_only",
            "claim_scope": "synthetic_contract_gate_only",
            "depends_on": [],
            "dependency_condition": "terminal_without_technical_failure",
            "resource_profile": {
                "allowed": ["cpu"],
                "accelerator": None,
                "hardware_gate": None,
            },
            "requirements": {},
            "inputs": [
                {"name": "fixed-contract", "kind": "semantic", "required": True}
            ],
            "seed_contract": {
                "mode": "fixed",
                "seeds": [101],
                "grouping": "position",
                "trace_salt_count": 1,
            },
            "steps": (
                [{"name": "gate", "command": command, "timeout_s": timeout_s}]
                if available
                else []
            ),
            "expected_artifacts": [
                {
                    "path": "{output}/run_manifest.json",
                    "kind": "json",
                    "required": True,
                    "schema": {
                        "required_keys": ["axis_id"],
                        "equals": {"axis_id": "A01"},
                    },
                },
                {
                    "path": "{output}/rows.jsonl",
                    "kind": "jsonl",
                    "required": True,
                    "schema": {
                        "min_rows": 1,
                        "required_keys": ["axis_id", "value"],
                        "equals": {"axis_id": "A01"},
                    },
                },
                {
                    "path": "{output}/summary.json",
                    "kind": "json",
                    "required": True,
                    "schema": {"required_keys": ["promotion_eligible"]},
                },
            ],
            "promotion_gate": {
                "mode": "manual_claim_ledger",
                "allow_auto_promotion": False,
                "negative_result_status": "completed_no_promotion",
            },
            "resume_policy": {
                "mode": "verified_artifacts_only",
                "require_registry_hash": True,
                "require_git_commit": True,
                "require_source_hashes": True,
                "require_input_hashes": True,
            },
            "prohibited_inferences": ["play-strength improvement"],
        }
        if not available:
            lane["blocked_by"] = ["not selected in this fixture"]
        lanes.append(lane)
    registry = {
        "format_version": 2,
        "experiment_id": "v2-test",
        "source_dependencies": [str(source_path)],
        "axis_registry": str(axis_path),
        "default_suite": "first-gate-all",
        "suites": {"first-gate-all": ["A01.synthetic"]},
        "campaign_policy": {"technical_failures": "fail_closed"},
        "lanes": lanes,
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def make_plan(
    lab,
    cfg_path,
    tmp_path,
    run_id="campaign",
    variables=None,
    python=sys.executable,
):
    cfg = lab.load_config(cfg_path)
    lanes = lab.expand_lanes(cfg, ["first-gate-all"], [])
    plans = [
        lab.plan_lane(
            lane,
            repo_root=Path(__file__).resolve().parents[1],
            profile="cpu",
            output_root=tmp_path / "out",
            run_id=run_id,
            python=python,
            device=None,
            variables=variables or {},
            campaign_source_paths=cfg.get("_source_dependencies", []),
        )
        for lane in lanes
    ]
    return cfg, plans


def run_campaign(
    lab,
    cfg_path,
    cfg,
    plans,
    tmp_path,
    *,
    run_id="campaign",
    resume=False,
    overwrite=False,
    python=sys.executable,
):
    return lab.run_plans(
        plans,
        repo_root=Path(__file__).resolve().parents[1],
        output_root=tmp_path / "out",
        run_id=run_id,
        cfg_path=cfg_path,
        cfg=cfg,
        profile="cpu",
        python=python,
        doctor={"ok": True, "checks": []},
        keep_going=False,
        overwrite=overwrite,
        resume=resume,
    )


def test_v2_requires_exact_axis_coverage_and_rejects_cycles(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][-1]["axis_id"] = "A01"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="axis coverage mismatch"):
        lab.load_config(cfg_path)

    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["depends_on"] = ["A02.synthetic"]
    payload["lanes"][1]["depends_on"] = ["A01.synthetic"]
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="dependency cycle"):
        lab.load_config(cfg_path)


def test_dependency_closure_is_ordered_before_selected_lane(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][1]["depends_on"] = ["A01.synthetic"]
    payload["suites"]["ordered"] = ["A02.synthetic"]
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = lab.load_config(cfg_path)
    assert [lane["id"] for lane in lab.expand_lanes(cfg, ["ordered"], [])] == [
        "A01.synthetic",
        "A02.synthetic",
    ]


def test_v2_rejects_disabled_resume_hashes_and_blocked_commands(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["resume_policy"]["require_source_hashes"] = False
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="must require source hash matching"):
        lab.load_config(cfg_path)

    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][1]["steps"] = payload["lanes"][0]["steps"]
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        lab.ConfigError, match="blocked lane .* must not register executable steps"
    ):
        lab.load_config(cfg_path)


def test_v2_rejects_duplicate_steps_and_unknown_artifact_owner(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["steps"].append(payload["lanes"][0]["steps"][0])
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="duplicate step names"):
        lab.load_config(cfg_path)

    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["expected_artifacts"][0]["step"] = "typo-step"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="unknown step owner"):
        lab.load_config(cfg_path)

    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["steps"][0]["expected_artifacts"] = [
        {"path": "{output}/bad.json", "kind": "jsoon"}
    ]
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="invalid artifact kind 'jsoon'"):
        lab.load_config(cfg_path)

    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["expected_artifacts"][0]["required"] = 0
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="artifact required must be boolean"):
        lab.load_config(cfg_path)


def test_v2_confines_artifacts_and_rejects_unsafe_run_ids(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["expected_artifacts"][0]["path"] = "../../escape.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert not plans[0].ready
    assert any("escapes campaign root" in blocker for blocker in plans[0].blockers)
    with pytest.raises(lab.ConfigError, match="run id"):
        lab.make_run_id("../escape")


def test_campaign_lock_rejects_a_second_controller(tmp_path):
    lab = load_module()
    run_root = tmp_path / "campaign"
    first = lab._acquire_campaign_lock(run_root)
    try:
        with pytest.raises(lab.ConfigError, match="campaign is locked"):
            lab._acquire_campaign_lock(run_root)
    finally:
        lab._release_campaign_lock(first)
    second = lab._acquire_campaign_lock(run_root)
    lab._release_campaign_lock(second)


def test_artifact_identity_is_bound_to_lane_and_conflicts_fail(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    _, plans = make_plan(lab, cfg_path, tmp_path)
    manifest_schema = plans[0].expected_artifacts[0]["schema"]
    assert manifest_schema["equals"]["axis_id"] == "A01"

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["expected_artifacts"][0]["schema"]["equals"] = {
        "axis_id": "A02"
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(lab.ConfigError, match="identity conflict"):
        make_plan(lab, cfg_path, tmp_path)


def test_v2_artifacts_are_validated_and_resume_skips_only_verified_steps(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 0
    state_path = tmp_path / "out" / "campaign" / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "succeeded"
    assert state["lanes"]["A01.synthetic"]["status"] == "succeeded"
    assert len(state["lanes"]["A01.synthetic"]["steps"][0]["artifacts"]) == 3

    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path, resume=True) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert (
        state["lanes"]["A01.synthetic"]["steps"][0]["resume_action"] == "verified_skip"
    )
    assert len(state["lanes"]["A01.synthetic"]["artifacts"]) == 3

    rows = tmp_path / "out" / "campaign" / "A01.synthetic" / "rows.jsonl"
    rows.write_text('{"axis_id":"tampered","value":1}\n', encoding="utf-8")
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path, resume=True) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "resume_action" not in state["lanes"]["A01.synthetic"]["steps"][0]
    assert json.loads(rows.read_text(encoding="utf-8"))["axis_id"] == "A01"


def test_resume_treats_relative_and_absolute_venv_launcher_as_same_interpreter(
    tmp_path,
):
    lab = load_module()
    command = ["{python}", "-c", artifact_writer(), "{output}"]
    cfg_path = write_v2_registry(tmp_path, command=command)
    cfg, plans = make_plan(lab, cfg_path, tmp_path, python=sys.executable)
    assert (
        run_campaign(
            lab,
            cfg_path,
            cfg,
            plans,
            tmp_path,
            python=sys.executable,
        )
        == 0
    )

    relative_python = os.path.relpath(sys.executable, Path.cwd())
    cfg, resume_plans = make_plan(
        lab,
        cfg_path,
        tmp_path,
        python=relative_python,
    )
    assert (
        run_campaign(
            lab,
            cfg_path,
            cfg,
            resume_plans,
            tmp_path,
            resume=True,
            python=relative_python,
        )
        == 0
    )
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["python"] == sys.executable
    assert (
        state["lanes"]["A01.synthetic"]["steps"][0]["resume_action"] == "verified_skip"
    )


def test_failed_attempt_rows_logs_and_stdout_survive_successful_resume(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    fail_once = (
        "from pathlib import Path;import sys;"
        "out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);"
        "marker=out/'fail_once.marker';first=not marker.exists();"
        "marker.write_text('attempted');"
        "print('first-attempt-failure' if first else 'second-attempt-success');"
        "sys.exit(9) if first else None;"
    )
    payload["lanes"][0]["steps"][0]["command"] = [
        sys.executable,
        "-c",
        fail_once + artifact_writer(),
        "{output}",
    ]
    payload["lanes"][0]["steps"][0]["stdout"] = "{output}/stdout.log"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 9
    run_root = tmp_path / "out" / "campaign"
    state_path = run_root / "campaign_state.json"
    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    first_attempt = first_state["lanes"]["A01.synthetic"]["attempts"][0]
    first_log = Path(first_attempt["log"])
    first_stdout = Path(first_attempt["stdout"])
    first_log_bytes = first_log.read_bytes()
    first_stdout_bytes = first_stdout.read_bytes()
    assert first_attempt["attempt_id"] == "A01.synthetic:gate:001"
    assert first_attempt["status"] == "failed"
    assert b"first-attempt-failure" in first_log_bytes

    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path, resume=True) == 0
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    lane = resumed["lanes"]["A01.synthetic"]
    assert len(lane["attempts"]) == 2
    assert lane["attempts"][0] == first_attempt
    assert first_log.read_bytes() == first_log_bytes
    assert first_stdout.read_bytes() == first_stdout_bytes
    assert lane["attempts"][1]["attempt_id"] == "A01.synthetic:gate:002"
    assert lane["attempts"][1]["status"] == "succeeded"
    assert lane["steps"][0]["attempt"] == 2
    assert Path(lane["attempts"][1]["log"]) != first_log
    assert Path(lane["attempts"][1]["stdout"]) != first_stdout
    assert (run_root / "A01.synthetic" / "stdout.log").read_text() == (
        "second-attempt-success\n"
    )


def test_attempt_evidence_path_collision_fails_closed_without_overwrite(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 0
    run_root = tmp_path / "out" / "campaign"
    state_path = run_root / "campaign_state.json"
    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    first_attempt = first_state["lanes"]["A01.synthetic"]["attempts"][0]
    first_log_bytes = Path(first_attempt["log"]).read_bytes()

    rows = run_root / "A01.synthetic" / "rows.jsonl"
    rows.write_text('{"axis_id":"tampered","value":1}\n', encoding="utf-8")
    collision = run_root / "A01.synthetic" / "01_gate.attempt-002.log"
    collision.write_bytes(b"preexisting-evidence\n")

    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path, resume=True) == 126
    failed = json.loads(state_path.read_text(encoding="utf-8"))
    lane = failed["lanes"]["A01.synthetic"]
    assert failed["status"] == "failed"
    assert lane["status"] == "failed"
    assert len(lane["attempts"]) == 2
    assert "FileExistsError" in lane["attempts"][1]["runner_error"]
    assert collision.read_bytes() == b"preexisting-evidence\n"
    assert Path(first_attempt["log"]).read_bytes() == first_log_bytes


def test_resume_refuses_declared_source_dependency_drift(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 0

    (tmp_path / "producer_source.py").write_text(
        "SOURCE_VERSION = 2\n", encoding="utf-8"
    )
    cfg, changed_plans = make_plan(lab, cfg_path, tmp_path)
    with pytest.raises(lab.ConfigError, match="source_hashes changed"):
        run_campaign(lab, cfg_path, cfg, changed_plans, tmp_path, resume=True)


def test_resume_refuses_changed_expanded_template_plan(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["lanes"][0]["steps"][0]["command"].append("{variant}")
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg, plans = make_plan(lab, cfg_path, tmp_path, variables={"variant": "one"})
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 0
    cfg, changed_plans = make_plan(
        lab,
        cfg_path,
        tmp_path,
        variables={"variant": "two"},
    )
    with pytest.raises(lab.ConfigError, match="expanded_plan_sha256 changed"):
        run_campaign(lab, cfg_path, cfg, changed_plans, tmp_path, resume=True)


def test_resume_reexecutes_downstream_after_upstream_step_runs(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    counter_code = (
        "from pathlib import Path;import sys;"
        "p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
        "p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"
    )
    artifact_command = payload["lanes"][0]["steps"][0]["command"]
    payload["lanes"][0]["steps"] = [
        {
            "name": "upstream",
            "command": [
                sys.executable,
                "-c",
                counter_code,
                "{output}/upstream_count.txt",
            ],
            "timeout_s": 5,
        },
        {"name": "downstream", "command": artifact_command, "timeout_s": 5},
    ]
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 0
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path, resume=True) == 0
    run_root = tmp_path / "out" / "campaign"
    state = json.loads((run_root / "campaign_state.json").read_text(encoding="utf-8"))
    steps = state["lanes"]["A01.synthetic"]["steps"]
    assert [row["name"] for row in steps] == ["upstream", "downstream"]
    assert all("resume_action" not in row for row in steps)
    assert (run_root / "A01.synthetic" / "upstream_count.txt").read_text() == "2"


def test_missing_step_local_artifact_stops_before_downstream_step(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    downstream = payload["lanes"][0]["steps"][0]
    payload["lanes"][0]["steps"] = [
        {
            "name": "upstream",
            "command": [sys.executable, "-c", "pass"],
            "timeout_s": 5,
            "expected_artifacts": [
                {
                    "path": "{output}/must_exist.json",
                    "kind": "json",
                    "required": True,
                    "schema": {"required_keys": ["ok"]},
                }
            ],
        },
        {**downstream, "name": "downstream"},
    ]
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 3
    lane_root = tmp_path / "out" / "campaign" / "A01.synthetic"
    assert not (lane_root / "run_manifest.json").exists()
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert [step["name"] for step in state["lanes"]["A01.synthetic"]["steps"]] == [
        "upstream"
    ]


def test_embedded_source_hash_mismatch_fails_technically(tmp_path):
    lab = load_module()
    command_code = (
        artifact_writer()
        + ";"
        + ";".join(
            [
                "p=out/'run_manifest.json'",
                "payload=json.loads(p.read_text())",
                "payload['source_hashes']=[{'path':'scripts/idea_lab.py','sha256':'0'*64}]",
                "p.write_text(json.dumps(payload))",
            ]
        )
    )
    cfg_path = write_v2_registry(
        tmp_path,
        command=[sys.executable, "-c", command_code, "{output}"],
    )
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 3
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["lanes"]["A01.synthetic"]["status"] == "failed"
    assert "embedded source hash drift" in state["lanes"]["A01.synthetic"]["reason"]


def test_negative_scientific_gate_is_not_a_technical_failure(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path, promotion_eligible=False)
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 0
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "succeeded"
    assert state["lanes"]["A01.synthetic"]["status"] == "completed_no_promotion"


def test_rc_zero_summary_failure_is_a_technical_failure(tmp_path):
    lab = load_module()
    cfg_path = write_v2_registry(tmp_path, execution_status="failed")
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 3
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    lane = state["lanes"]["A01.synthetic"]
    assert lane["status"] == "failed"
    assert "non-success execution_status='failed'" in lane["reason"]


def test_directory_artifact_rejects_descendant_symlink_escape(tmp_path):
    lab = load_module()
    campaign_root = tmp_path / "campaign"
    artifact_dir = campaign_root / "artifact"
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (artifact_dir / "escape").symlink_to(outside)
    with pytest.raises(lab.ArtifactError, match="forbidden symlink"):
        lab.validate_expected_artifact(
            {"path": str(artifact_dir), "kind": "directory", "required": True},
            campaign_root=campaign_root,
        )


@pytest.mark.parametrize(
    ("kind", "contents"),
    [
        ("json", '{"axis_id":"A01","value":NaN}'),
        ("json", '{"axis_id":"A01","value":1e9999}'),
        ("jsonl", '{"axis_id":"A01","value":Infinity}\n'),
    ],
)
def test_json_artifacts_reject_nonfinite_numbers(tmp_path, kind, contents):
    lab = load_module()
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    artifact = campaign_root / (
        "artifact.jsonl" if kind == "jsonl" else "artifact.json"
    )
    artifact.write_text(contents, encoding="utf-8")
    schema = {"required_keys": ["axis_id", "value"]}
    if kind == "jsonl":
        schema["min_rows"] = 1
    with pytest.raises(lab.ArtifactError, match="non-finite JSON"):
        lab.validate_expected_artifact(
            {"path": str(artifact), "kind": kind, "required": True, "schema": schema},
            campaign_root=campaign_root,
        )


def test_missing_artifact_fails_closed_and_v2_overwrite_is_forbidden(tmp_path):
    lab = load_module()
    command = [sys.executable, "-c", artifact_writer(omit_rows=True), "{output}"]
    cfg_path = write_v2_registry(tmp_path, command=command)
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 3
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["lanes"]["A01.synthetic"]["status"] == "failed"
    assert "required artifact is missing" in state["lanes"]["A01.synthetic"]["reason"]
    with pytest.raises(lab.ConfigError, match="overwrite"):
        run_campaign(
            lab, cfg_path, cfg, plans, tmp_path, run_id="other", overwrite=True
        )


def test_timeout_state_is_persisted(tmp_path):
    lab = load_module()
    command = [sys.executable, "-c", "import time;time.sleep(5)"]
    cfg_path = write_v2_registry(tmp_path, command=command, timeout_s=1)
    cfg, plans = make_plan(lab, cfg_path, tmp_path)
    assert run_campaign(lab, cfg_path, cfg, plans, tmp_path) == 124
    state = json.loads(
        (tmp_path / "out" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "failed"
    assert state["lanes"]["A01.synthetic"]["status"] == "timeout"


def test_doctor_reports_target_interpreter(tmp_path):
    lab = load_module()
    report = lab.collect_doctor(
        tmp_path, profile="cpu", python=sys.executable, strict=False
    )
    assert report["target_python"]["executable"] == sys.executable
    assert report["target_python"]["modules"]["numpy"] is True


def test_driver_probe_requires_successful_subprocess(monkeypatch):
    lab = load_module()
    monkeypatch.setattr(lab.shutil, "which", lambda _name: "/usr/bin/vendor-smi")
    monkeypatch.setattr(
        lab.subprocess,
        "run",
        lambda *_args, **_kwargs: lab.subprocess.CompletedProcess(
            ["vendor-smi"],
            returncode=1,
            stdout="",
            stderr="driver unavailable",
        ),
    )
    ok, detail = lab._probe_driver((("vendor-smi", "-L"),))
    assert ok is False
    assert detail["attempts"][0]["returncode"] == 1


def test_cuda_doctor_rejects_non_nvidia_device_name(monkeypatch, tmp_path):
    lab = load_module()
    target = {
        "executable": sys.executable,
        "version": "test",
        "modules": {"numpy": True},
        "torch": {
            "cuda_available": True,
            "device_count": 1,
            "device_names": ["Generic Accelerator"],
            "device_properties_ok": True,
            "cuda_build": "12.8",
            "hip_build": None,
        },
    }
    monkeypatch.setattr(lab, "_target_python_report", lambda *_args: (target, None))
    monkeypatch.setattr(
        lab, "_probe_driver", lambda *_args: (True, {"successful_probe": {}})
    )
    report = lab.collect_doctor(
        tmp_path, profile="cuda", python=sys.executable, strict=False
    )
    accelerator = next(
        item for item in report["checks"] if item["kind"] == "accelerator"
    )
    assert accelerator["ok"] is False
    assert json.loads(accelerator["detail"])["nvidia_name_contract"] is False
