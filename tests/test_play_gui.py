from pathlib import Path

from quartz.play_gui import (
    apply_checkpoint_architecture,
    discover_models,
    grid_coord_label,
    parse_chess_fen_board,
    parse_side,
)


def test_discover_models_lists_best_and_latest(tmp_path):
    model_dir = tmp_path / "alphazero_gomoku7"
    model_dir.mkdir()
    (model_dir / "best.pt").write_bytes(b"x")
    (model_dir / "latest.pt").write_bytes(b"y")

    rows = discover_models(tmp_path)

    assert [row["kind"] for row in rows] == ["best", "latest"]
    assert all(row["game"] == "gomoku7" for row in rows)
    assert all(row["sha256"] for row in rows)
    assert rows[0]["bytes"] == 1


def test_parse_chess_fen_board_returns_a1_indexed_board():
    board = parse_chess_fen_board("8/8/8/8/8/8/8/R3K2R w KQ - 0 1")

    assert board[0] == "R"
    assert board[4] == "K"
    assert board[7] == "R"
    assert board[56] == ""


def test_parse_side_matches_game_conventions():
    assert parse_side("chess", "white") == 1
    assert parse_side("chess", "black") == -1
    assert parse_side("gomoku7", "black") == 1
    assert parse_side("gomoku7", "white") == -1


def test_grid_coord_label_handles_go_pass_and_board_coords():
    assert grid_coord_label("go9", 9, 81) == "pass"
    assert grid_coord_label("gomoku7", 7, 0) == "A7"


def test_apply_checkpoint_architecture_uses_saved_shapes():
    cfg = {"board": 7, "ch": 3, "actions": 49, "filters": 64, "blocks": 4, "vh": 64}
    state_dict = {
        "input_conv.0.weight": type("Shape", (), {"shape": (96, 3, 3, 3)})(),
        "p_fc.weight": type("Shape", (), {"shape": (49, 196)})(),
        "v_fc.0.weight": type("Shape", (), {"shape": (64, 32)})(),
        "tower.0.net.0.weight": 0,
        "tower.5.net.0.weight": 0,
    }

    tuned = apply_checkpoint_architecture(cfg, state_dict)

    assert tuned["filters"] == 96
    assert tuned["blocks"] == 6
