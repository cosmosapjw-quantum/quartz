import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_controller_sweep_module():
    root = Path(__file__).resolve().parents[1]
    return load_module(
        "controller_sweep_script", root / "scripts" / "controller_sweep.py"
    )


def test_build_default_search_space_centers_on_base_cfg():
    sweep = load_controller_sweep_module()
    base_cfg = {
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
    }

    space = sweep.build_default_search_space(base_cfg)

    assert 0.3 in space["hbar_penalty_cap"]
    assert 0.3 in space["sigma_0"]
    assert 15 in space["min_visits"]
    assert 20 in space["check_interval"]
    assert 2.0 in space["c_puct"]


def test_sample_candidate_pool_includes_anchors_and_unique_rows():
    sweep = load_controller_sweep_module()
    base_cfg = {
        "penalty_mode": "GatedRefresh",
        "prior_refresh_rate": 0.0,
        "prior_refresh_temp": 1.0,
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
    }

    rows = sweep.sample_candidate_pool(base_cfg, n_random=6, seed=7)

    assert [row["id"] for row in rows[:4]] == [
        "A1_legacy_base",
        "A2_legacy_krefresh",
        "A3_theory_base",
        "A4_theory_krefresh",
    ]
    assert len({sweep.candidate_key(row["overrides"]) for row in rows}) == len(rows)


def test_select_stage2_candidates_keeps_anchor_rows():
    sweep = load_controller_sweep_module()
    candidates = [
        {
            "id": "A1_legacy_base",
            "source": "anchor",
            "label": "anchor-1",
            "overrides": {},
        },
        {
            "id": "A2_legacy_krefresh",
            "source": "anchor",
            "label": "anchor-2",
            "overrides": {},
        },
        {
            "id": "A3_theory_base",
            "source": "anchor",
            "label": "anchor-3",
            "overrides": {},
        },
        {
            "id": "A4_theory_krefresh",
            "source": "anchor",
            "label": "anchor-4",
            "overrides": {},
        },
        {"id": "R01_alpha", "source": "random", "label": "random-1", "overrides": {}},
        {"id": "R02_beta", "source": "random", "label": "random-2", "overrides": {}},
    ]
    summary = [
        {
            "candidate_id": "R01_alpha",
            "stage1_score": 0.7,
            "agreement_rate": 0.7,
            "reference_policy_mass": 0.5,
        },
        {
            "candidate_id": "R02_beta",
            "stage1_score": 0.6,
            "agreement_rate": 0.6,
            "reference_policy_mass": 0.4,
        },
    ]

    selected = sweep.select_stage2_candidates(candidates, summary, topk=5)

    assert [row["id"] for row in selected[:4]] == [
        "A1_legacy_base",
        "A2_legacy_krefresh",
        "A3_theory_base",
        "A4_theory_krefresh",
    ]
    assert selected[4]["id"] == "R01_alpha"


def test_select_stage2_candidates_honors_exact_anchor_cap():
    sweep = load_controller_sweep_module()
    candidates = [
        {
            "id": "A1_legacy_base",
            "source": "anchor",
            "label": "anchor-1",
            "overrides": {},
        },
        {
            "id": "A2_legacy_krefresh",
            "source": "anchor",
            "label": "anchor-2",
            "overrides": {},
        },
        {
            "id": "A3_theory_base",
            "source": "anchor",
            "label": "anchor-3",
            "overrides": {},
        },
        {
            "id": "A4_theory_krefresh",
            "source": "anchor",
            "label": "anchor-4",
            "overrides": {},
        },
        {"id": "R01_alpha", "source": "random", "label": "random-1", "overrides": {}},
    ]
    summary = [
        {
            "candidate_id": "R01_alpha",
            "stage1_score": 0.9,
            "agreement_rate": 0.9,
            "reference_policy_mass": 0.5,
        },
    ]

    selected = sweep.select_stage2_candidates(candidates, summary, topk=4)

    assert [row["id"] for row in selected] == [
        "A1_legacy_base",
        "A2_legacy_krefresh",
        "A3_theory_base",
        "A4_theory_krefresh",
    ]


def test_aggregate_stage2_matches_builds_overall_leaderboard():
    sweep = load_controller_sweep_module()
    candidates = [
        {
            "id": "A1_legacy_base",
            "source": "anchor",
            "label": "legacy",
            "overrides": {},
        },
        {
            "id": "A4_theory_krefresh",
            "source": "anchor",
            "label": "theory",
            "overrides": {},
        },
    ]
    matches = [
        {
            "checkpoint_path": "/tmp/ckpt_a.pt",
            "candidate_a": "A1_legacy_base",
            "candidate_b": "A4_theory_krefresh",
            "games": 8,
            "wins_a": 5,
            "wins_b": 2,
            "draws": 1,
        },
        {
            "checkpoint_path": "/tmp/ckpt_b.pt",
            "candidate_a": "A1_legacy_base",
            "candidate_b": "A4_theory_krefresh",
            "games": 8,
            "wins_a": 3,
            "wins_b": 4,
            "draws": 1,
        },
    ]

    payload = sweep.aggregate_stage2_matches(candidates, matches)

    assert payload["overall"][0]["candidate_id"] == "A1_legacy_base"
    assert payload["overall"][0]["points"] == 9.0
    assert payload["overall"][0]["games"] == 16
    assert set(payload["by_checkpoint"]) == {"/tmp/ckpt_a.pt", "/tmp/ckpt_b.pt"}


def test_stage2_search_telemetry_tracks_budget_and_halt_reason():
    sweep = load_controller_sweep_module()
    bucket = {
        "search_count": 0,
        "benchmark_safe_count": 0,
        "root_visits": [],
        "halt_reason_hist": {},
        "selection_root_selects": 0,
    }

    sweep._record_stage2_search_telemetry(
        bucket,
        {
            "search_manifest": {"benchmark_safe": True},
            "realized_budget": {"root_visits": 16, "stop_reason": "BudgetExhausted"},
            "controller_summary": {"selection_trace": {"root_selects": 4}},
        },
    )
    summary = sweep._finalize_stage2_search_telemetry(bucket)

    assert summary["search_count"] == 1
    assert summary["benchmark_safe_frac"] == 1.0
    assert summary["root_visits"]["mean"] == 16.0
    assert summary["halt_reason_hist"] == {"BudgetExhausted": 1}
    assert summary["selection_root_selects"] == 4


def test_generate_random_positions_returns_non_terminal_gomoku_positions():
    sweep = load_controller_sweep_module()
    cfg = {"board": 7, "win": 4}

    rows = sweep.generate_random_positions(
        "gomoku7", cfg, count=6, seed=13, min_moves=4, max_moves=8
    )

    assert len(rows) == 6
    for row in rows:
        board = row["board"]
        assert len(board) == 49
        assert row["player"] in (-1, 1)
        assert board.count(0) >= 2


def test_discover_checkpoint_paths_recurses_and_dedupes_best_first(tmp_path):
    sweep = load_controller_sweep_module()
    root = tmp_path / "models"
    (root / "a" / "seed_41").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "seed_41" / "best.pt").write_bytes(b"a")
    (root / "b" / "latest.pt").write_bytes(b"b")

    rows = sweep.discover_checkpoint_paths(root)

    assert rows[0].endswith("best.pt")
    assert rows[1].endswith("latest.pt")


def test_discover_checkpoint_paths_prefers_latest_for_bootstrap_only_runs(tmp_path):
    sweep = load_controller_sweep_module()
    root = tmp_path / "models"
    run_dir = root / "a" / "seed_41"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"bootstrap")
    (run_dir / "latest.pt").write_bytes(b"latest")
    (run_dir / "checkpoint_status.json").write_text(
        __import__("json").dumps(
            {
                "best_checkpoint_bootstrap_seeded": True,
                "preferred_posttrain_checkpoint": "latest.pt",
            }
        ),
        encoding="utf-8",
    )

    rows = sweep.discover_checkpoint_paths(root)

    assert rows == [str(run_dir / "latest.pt")]


def test_load_resume_state_reads_report_and_shortlist(tmp_path):
    sweep = load_controller_sweep_module()
    base = tmp_path / "resume"
    base.mkdir()
    payload = {
        "manifest": {
            "game": "gomoku7",
            "checkpoints": ["a.pt", "b.pt"],
            "candidates": [
                {"id": "A1", "label": "anchor", "source": "anchor", "overrides": {}}
            ],
        },
        "stage1": {
            "shortlist": [
                {"id": "A1", "label": "anchor", "source": "anchor", "overrides": {}}
            ],
        },
    }
    (base / "sweep_report.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )

    report_dir, manifest, candidates, checkpoints, shortlist = sweep.load_resume_state(
        str(base)
    )

    assert report_dir == base
    assert manifest["game"] == "gomoku7"
    assert candidates[0]["id"] == "A1"
    assert checkpoints == ["a.pt", "b.pt"]
    assert shortlist[0]["id"] == "A1"


def test_parse_selected_candidate_ids_validates_subset():
    sweep = load_controller_sweep_module()

    assert sweep.parse_selected_candidate_ids(None, ["A1", "A2"]) == ["A1", "A2"]
    assert sweep.parse_selected_candidate_ids("A2", ["A1", "A2"]) == ["A2"]

    try:
        sweep.parse_selected_candidate_ids("A3", ["A1", "A2"])
    except ValueError as exc:
        assert "unknown candidate ids" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown candidate id")


def test_resolve_explicit_checkpoint_paths_rejects_directories_and_missing(tmp_path):
    sweep = load_controller_sweep_module()
    good = tmp_path / "good.pt"
    good.write_bytes(b"x")
    bad_dir = tmp_path / "models"
    bad_dir.mkdir()

    try:
        sweep.resolve_explicit_checkpoint_paths(
            f"{good},{bad_dir},{tmp_path / 'missing.pt'}"
        )
    except ValueError as exc:
        text = str(exc)
        assert "--checkpoints expects checkpoint files" in text
        assert "checkpoint paths do not exist" in text
    else:
        raise AssertionError(
            "expected ValueError for invalid explicit checkpoint inputs"
        )


def test_resolve_checkpoint_paths_accepts_explicit_files(tmp_path):
    sweep = load_controller_sweep_module()
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    args = SimpleNamespace(
        checkpoints=f"{a},{b}",
        checkpoint_dir=None,
        max_checkpoints=3,
        bootstrap_if_empty=False,
    )

    rows = sweep.resolve_checkpoint_paths(args, tmp_path)

    assert rows == [str(a), str(b)]
