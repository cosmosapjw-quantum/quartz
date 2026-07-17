import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "idea_lab.py"
    spec = importlib.util.spec_from_file_location("idea_lab", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_registry(tmp_path):
    registry = {
        "format_version": 1,
        "experiment_id": "test_lab",
        "default_suite": "smoke",
        "suites": {
            "smoke": ["available.one", "planned.two"],
            "single": ["available.one"],
        },
        "lanes": [
            {
                "id": "available.one",
                "title": "Available",
                "status": "available",
                "groups": ["smoke"],
                "profiles": ["cpu"],
                "requirements": {
                    "commands": ["python3"],
                    "paths": ["{needed_path}"],
                    "variables": ["needed_path"],
                },
                "steps": [
                    {
                        "name": "echo",
                        "command": ["{python}", "-c", "print('hello')"],
                        "timeout_s": 5,
                    }
                ],
                "artifact_globs": ["{output}/summary.json"],
            },
            {
                "id": "planned.two",
                "title": "Planned",
                "status": "planned",
                "groups": ["foundry"],
                "blocked_by": ["implement it"],
            },
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_load_config_and_expand_lanes(tmp_path):
    lab = load_module()
    cfg = lab.load_config(write_registry(tmp_path))
    lanes = lab.expand_lanes(cfg, ["smoke"], [])
    assert [lane["id"] for lane in lanes] == ["available.one", "planned.two"]


def test_plan_marks_missing_variable(tmp_path):
    lab = load_module()
    cfg = lab.load_config(write_registry(tmp_path))
    lane = lab.lane_index(cfg)["available.one"]
    plan = lab.plan_lane(
        lane,
        repo_root=tmp_path,
        profile="cpu",
        output_root=tmp_path / "out",
        run_id="run",
        python="python3",
        device=None,
        variables={},
    )
    assert plan.status == "BLOCKED"
    assert any("missing --set needed_path" in blocker for blocker in plan.blockers)


def test_plan_resolves_ready_command_and_output(tmp_path):
    lab = load_module()
    cfg = lab.load_config(write_registry(tmp_path))
    required = tmp_path / "required.txt"
    required.write_text("ok", encoding="utf-8")
    lane = lab.lane_index(cfg)["available.one"]
    plan = lab.plan_lane(
        lane,
        repo_root=tmp_path,
        profile="cpu",
        output_root=tmp_path / "out",
        run_id="abc",
        python="python3",
        device=None,
        variables={"needed_path": str(required)},
    )
    assert plan.status == "READY", plan.blockers
    assert plan.commands[0]["command"][:2] == ["python3", "-c"]
    assert plan.output_dir.endswith("available.one")
    assert plan.artifact_globs == [str(tmp_path / "out" / "abc" / "available.one" / "summary.json")]


def test_planned_lane_exposes_blockers(tmp_path):
    lab = load_module()
    cfg = lab.load_config(write_registry(tmp_path))
    lane = lab.lane_index(cfg)["planned.two"]
    plan = lab.plan_lane(
        lane,
        repo_root=tmp_path,
        profile="cpu",
        output_root=tmp_path / "out",
        run_id="run",
        python="python3",
        device=None,
        variables={},
    )
    assert plan.status == "BLOCKED"
    assert plan.blockers == ["planned: implement it"]


def test_doctor_report_shape(tmp_path):
    lab = load_module()
    report = lab.collect_doctor(tmp_path, profile="cpu", python="python3", strict=False)
    assert report["profile"] == "cpu"
    assert isinstance(report["checks"], list)
    assert any(row["name"] == "git" for row in report["checks"])
