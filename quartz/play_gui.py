#!/usr/bin/env python3
"""Browser GUI for playing trained QUARTZ models."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from quartz.alphazero_train import (
    AlphaZeroNet,
    GAME_CONFIGS,
    NNSearchClient,
    build_rust_state_meta,
    build_training_game_adapter,
    initial_chess_fen,
    is_chess_game,
    is_go_game,
)
from quartz.backend import load_torch_state_dict
from quartz.encoders import get_encoder

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static" / "play"
GO_FILES = "ABCDEFGHJKLMNOPQRST"


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None:
            try:
                if bool(mps_backend.is_available()):
                    return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")
    return torch.device(name)


def side_name(game: str, side: int) -> str:
    if is_chess_game(game):
        return "white" if side == 1 else "black"
    return "black" if side == 1 else "white"


def parse_side(game: str, value: str) -> int:
    v = (value or "").strip().lower()
    if is_chess_game(game):
        return -1 if v in {"black", "-1"} else 1
    return -1 if v in {"white", "-1"} else 1


def outcome_label(game: str, outcome: float | None, void_result: bool = False) -> str:
    if void_result:
        return "No result"
    if outcome is None:
        return "In progress"
    if outcome > 0:
        return "White wins" if is_chess_game(game) else "Black wins"
    if outcome < 0:
        return "Black wins" if is_chess_game(game) else "White wins"
    return "Draw"


def grid_coord_label(game: str, board_size: int, action: int) -> str:
    if is_go_game(game):
        if action == board_size * board_size:
            return "pass"
        file_char = GO_FILES[action % board_size]
        rank = board_size - (action // board_size)
        return f"{file_char}{rank}"
    file_char = chr(ord("A") + (action % board_size))
    rank = board_size - (action // board_size)
    return f"{file_char}{rank}"


def parse_uci_square(text: str) -> int:
    file_idx = ord(text[0]) - ord("a")
    rank_idx = int(text[1]) - 1
    return rank_idx * 8 + file_idx


def parse_chess_legal_move(uci: str, action: int) -> dict[str, Any]:
    return {
        "uci": uci,
        "action": action,
        "from": parse_uci_square(uci[:2]),
        "to": parse_uci_square(uci[2:4]),
        "promotion": uci[4] if len(uci) >= 5 else "",
    }


def parse_chess_fen_board(fen: str) -> list[str]:
    board = [""] * 64
    board_part = fen.split()[0]
    rank = 7
    file_idx = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file_idx = 0
            continue
        if ch.isdigit():
            file_idx += int(ch)
        else:
            board[rank * 8 + file_idx] = ch
            file_idx += 1
    return board


def infer_game_from_model_dir(path: Path) -> str | None:
    name = path.name
    prefix = "alphazero_"
    if not name.startswith(prefix):
        return None
    game = name[len(prefix):]
    return game if game in GAME_CONFIGS else None


def discover_models(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists():
        return []
    entries = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        game = infer_game_from_model_dir(child)
        if game is None:
            continue
        for kind, filename in (("best", "best.pt"), ("latest", "latest.pt")):
            model_path = child / filename
            if model_path.exists():
                entries.append(
                    {
                        "id": f"{game}:{kind}",
                        "game": game,
                        "kind": kind,
                        "label": f"{game} {kind}",
                        "path": str(model_path.resolve()),
                    }
                )
    return entries


@dataclass
class LoadedModel:
    game: str
    path: str
    cfg: dict[str, Any]
    model: torch.nn.Module


def apply_checkpoint_architecture(cfg: dict[str, Any], state_dict: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(cfg)
    conv_w = state_dict.get("input_conv.0.weight")
    p_fc_w = state_dict.get("p_fc.weight")
    v_fc_w = state_dict.get("v_fc.0.weight")
    if conv_w is not None:
        tuned["filters"] = int(conv_w.shape[0])
        tuned["ch"] = int(conv_w.shape[1])
    if p_fc_w is not None:
        tuned["actions"] = int(p_fc_w.shape[0])
        flat_policy = int(p_fc_w.shape[1])
        board_sq = flat_policy // 4 if flat_policy % 4 == 0 else 0
        board = int(round(board_sq ** 0.5)) if board_sq > 0 else 0
        if board > 0 and board * board == board_sq:
            tuned["board"] = board
    tower_indices = [
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("tower.") and key.split(".")[1].isdigit()
    ]
    if tower_indices:
        tuned["blocks"] = max(tower_indices) + 1
    if v_fc_w is not None:
        tuned["vh"] = int(v_fc_w.shape[0])
    return tuned


class ModelStore:
    def __init__(self, device: torch.device):
        self.device = device
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], LoadedModel] = {}

    def base_cfg(self, game: str) -> dict[str, Any]:
        cfg = dict(GAME_CONFIGS[game])
        cfg["_name"] = game
        try:
            cfg["_encoder"] = get_encoder(game)
        except Exception:
            cfg["_encoder"] = None
        return cfg

    def load(self, game: str, model_path: str) -> LoadedModel:
        resolved = str(Path(model_path).resolve())
        key = (game, resolved)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        state_dict = load_torch_state_dict(resolved, torch, map_location=self.device)
        cfg = apply_checkpoint_architecture(self.base_cfg(game), state_dict)
        model = AlphaZeroNet(cfg).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        loaded = LoadedModel(game=game, path=resolved, cfg=cfg, model=model)
        with self._lock:
            self._cache[key] = loaded
        return loaded


class RustControlClient:
    def __init__(self, rust_binary: str):
        self.rust_binary = rust_binary
        self.proc = None

    def start(self):
        if self.proc is None:
            from quartz.alphazero_train import launch_rust_server

            self.proc = launch_rust_server(self.rust_binary)

    def stop(self):
        if self.proc is not None:
            try:
                from quartz.alphazero_train import proc_write_json_line

                proc_write_json_line(self.proc, {"cmd": "quit"})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self.proc is not None
        from quartz.alphazero_train import proc_write_json_line, proc_read_json_line

        proc_write_json_line(self.proc, payload)
        line = proc_read_json_line(self.proc)
        if not line:
            raise RuntimeError("Rust control server returned empty response")
        return json.loads(line)

    def chess_state(self, game: str, fen: str) -> dict[str, Any]:
        return self.request({"cmd": "chess_state", "game": game, "fen": fen})

    def chess_apply(self, game: str, fen: str, move_uci: str) -> dict[str, Any]:
        return self.request({"cmd": "chess_apply", "game": game, "fen": fen, "move_uci": move_uci})


class GameSession:
    def __init__(
        self,
        session_id: str,
        game: str,
        model_path: str,
        human_side: int,
        cfg: dict[str, Any],
        model: torch.nn.Module,
        device: torch.device,
        rust_binary: str,
    ):
        self.id = session_id
        self.game = game
        self.model_path = model_path
        self.human_side = human_side
        self.ai_side = -human_side
        self.cfg = cfg
        self.model = model
        self.device = device
        self.rust_binary = rust_binary
        self.move_history: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.search_client = NNSearchClient(self.model, self.cfg, self.device, self.rust_binary)
        self.rust_control = RustControlClient(self.rust_binary) if is_chess_game(game) else None
        self.state = None
        self.fen = ""
        self.chess_info: dict[str, Any] = {}
        self.restart()

    def close(self):
        self.search_client.stop()
        if self.rust_control is not None:
            self.rust_control.stop()

    def restart(self):
        self.move_history.clear()
        self.snapshots.clear()
        if is_chess_game(self.game):
            self.fen = initial_chess_fen(self.cfg)
            assert self.rust_control is not None
            self.chess_info = self.rust_control.chess_state(self.game, self.fen)
        else:
            self.state = build_training_game_adapter(self.cfg)

    def current_side(self) -> int:
        if is_chess_game(self.game):
            return 1 if self.chess_info.get("side_to_move") == "w" else -1
        assert self.state is not None
        return getattr(self.state, "_player", 1)

    def terminal(self) -> bool:
        if is_chess_game(self.game):
            return bool(self.chess_info.get("terminal", False))
        assert self.state is not None
        return bool(self.state.is_terminal())

    def result_label(self) -> str:
        if is_chess_game(self.game):
            return outcome_label(self.game, self.chess_info.get("outcome_white"))
        assert self.state is not None
        void_result = bool(getattr(self.state, "is_void_result", lambda: False)())
        return outcome_label(self.game, self.state.outcome_for_black(), void_result=void_result)

    def human_to_move(self) -> bool:
        return (not self.terminal()) and self.current_side() == self.human_side

    def ai_to_move(self) -> bool:
        return (not self.terminal()) and self.current_side() == self.ai_side

    def _push_snapshot(self):
        if is_chess_game(self.game):
            self.snapshots.append({"fen": self.fen, "move_log_len": len(self.move_history)})
        else:
            assert self.state is not None
            self.snapshots.append({"state": self.state.clone(), "move_log_len": len(self.move_history)})

    def undo(self, count: int = 1):
        restore = None
        for _ in range(min(max(1, count), len(self.snapshots))):
            restore = self.snapshots.pop()
        if restore is None:
            return
        if is_chess_game(self.game):
            self.fen = restore["fen"]
            assert self.rust_control is not None
            self.chess_info = self.rust_control.chess_state(self.game, self.fen)
        else:
            self.state = restore["state"]
        self.move_history = self.move_history[: restore["move_log_len"]]

    def resign(self, side: str):
        if self.terminal():
            return
        resigning = parse_side(self.game, side)
        if is_chess_game(self.game):
            winner = -1.0 if resigning == 1 else 1.0
            self.chess_info = {
                "fen": self.fen,
                "side_to_move": self.chess_info.get("side_to_move", "w"),
                "terminal": True,
                "outcome_white": winner,
                "legal_moves": [],
                "legal_actions": [],
            }
        else:
            assert self.state is not None
            self.state._terminal = True
            if self.game.startswith("go"):
                self.state._void_result = False
            self.state._outcome = -1.0 if resigning == 1 else 1.0

    def _append_move(self, actor: str, label: str):
        side = side_name(self.game, self.current_side() * -1)
        self.move_history.append(
            {
                "ply": len(self.move_history) + 1,
                "actor": actor,
                "side": side,
                "label": label,
            }
        )

    def _apply_human_grid_move(self, action: int):
        assert self.state is not None
        legal = set(self.state.legal_moves())
        if action not in legal:
            raise ValueError("illegal move")
        self._push_snapshot()
        self.state.apply_move(action)
        self._append_move("human", grid_coord_label(self.game, self.cfg["board"], action))

    def _apply_human_chess_move(self, move_uci: str):
        legal = {mv["uci"] for mv in self._chess_legal_moves()}
        if move_uci not in legal:
            raise ValueError("illegal move")
        self._push_snapshot()
        assert self.rust_control is not None
        result = self.rust_control.chess_apply(self.game, self.fen, move_uci)
        if result.get("status") != "ok":
            raise ValueError(result.get("error", "failed to apply move"))
        self.fen = result["fen"]
        self.chess_info = result
        self._append_move("human", result.get("applied_move", move_uci))

    def apply_human_move(self, payload: dict[str, Any]):
        if not self.human_to_move():
            raise ValueError("not human turn")
        if is_chess_game(self.game):
            move_uci = str(payload.get("move_uci", "")).strip().lower()
            if not move_uci:
                raise ValueError("missing move_uci")
            self._apply_human_chess_move(move_uci)
        else:
            self._apply_human_grid_move(int(payload.get("action")))

    def apply_ai_move(self):
        if not self.ai_to_move():
            raise ValueError("not AI turn")
        if is_chess_game(self.game):
            player = 1 if self.chess_info.get("side_to_move") == "w" else -1
            result = self.search_client.search_move(None, player, fen=self.fen)
            move_uci = result.get("best_move_uci", "")
            next_fen = result.get("result_fen", "")
            if not move_uci or not next_fen:
                raise ValueError("AI search returned no move")
            self._push_snapshot()
            self.fen = next_fen
            assert self.rust_control is not None
            self.chess_info = self.rust_control.chess_state(self.game, self.fen)
            self._append_move("ai", move_uci)
            return

        assert self.state is not None
        player = 1 if self.state.current_player() == 0 else -1
        board = list(self.state._board)
        result = self.search_client.search_move(
            board,
            player,
            state_meta=build_rust_state_meta(self.game, self.state, self.cfg),
        )
        if "best_move" not in result:
            raise ValueError("AI search returned no move")
        action = int(result["best_move"])
        legal = set(self.state.legal_moves())
        if action not in legal:
            raise ValueError("AI produced illegal move")
        self._push_snapshot()
        self.state.apply_move(action)
        self._append_move("ai", grid_coord_label(self.game, self.cfg["board"], action))

    def _chess_legal_moves(self) -> list[dict[str, Any]]:
        ucis = self.chess_info.get("legal_moves", [])
        actions = self.chess_info.get("legal_actions", [])
        return [
            parse_chess_legal_move(uci, actions[idx] if idx < len(actions) else -1)
            for idx, uci in enumerate(ucis)
        ]

    def payload(self) -> dict[str, Any]:
        base = {
            "sessionId": self.id,
            "game": self.game,
            "modelPath": self.model_path,
            "boardSize": self.cfg["board"],
            "humanSide": side_name(self.game, self.human_side),
            "aiSide": side_name(self.game, self.ai_side),
            "currentPlayer": side_name(self.game, self.current_side()),
            "humanToMove": self.human_to_move(),
            "aiToMove": self.ai_to_move(),
            "terminal": self.terminal(),
            "resultLabel": self.result_label(),
            "moveHistory": self.move_history,
            "searchIterations": self.cfg["iters"],
        }
        if is_chess_game(self.game):
            base.update(
                {
                    "render": "chess",
                    "fen": self.fen,
                    "board": parse_chess_fen_board(self.fen),
                    "legalMoves": self._chess_legal_moves(),
                }
            )
            return base

        assert self.state is not None
        legal_actions = self.state.legal_moves()
        base.update(
            {
                "render": "grid",
                "board": list(self.state._board),
                "legalActions": legal_actions,
                "passAction": self.cfg["board"] * self.cfg["board"] if is_go_game(self.game) else None,
            }
        )
        if is_go_game(self.game):
            base.update(
                {
                    "captures": {
                        "black": int(getattr(self.state, "_black_caps", 0)),
                        "white": int(getattr(self.state, "_white_caps", 0)),
                    },
                    "passes": int(getattr(self.state, "_passes", 0)),
                    "komi": float(getattr(self.state, "_komi", 7.5)),
                    "ruleset": getattr(self.state, "_ruleset", "chinese"),
                    "scoring": getattr(self.state, "_scoring", "area"),
                    "voidResult": bool(getattr(self.state, "is_void_result", lambda: False)()),
                }
            )
        return base


class PlayApp:
    def __init__(self, models_dir: Path, device: torch.device, rust_binary: str):
        self.models_dir = models_dir
        self.device = device
        self.rust_binary = rust_binary
        self.model_store = ModelStore(device)
        self._lock = threading.Lock()
        self.sessions: dict[str, GameSession] = {}

    def close(self):
        with self._lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            session.close()

    def available_models(self) -> list[dict[str, Any]]:
        return discover_models(self.models_dir)

    def default_model_for_game(self, game: str) -> str | None:
        candidates = [m for m in self.available_models() if m["game"] == game]
        if not candidates:
            return None
        candidates.sort(key=lambda row: (0 if row["kind"] == "best" else 1, row["path"]))
        return candidates[0]["path"]

    def create_session(self, payload: dict[str, Any]) -> GameSession:
        game = str(payload.get("game", "")).strip()
        if game not in GAME_CONFIGS:
            raise ValueError(f"unsupported game: {game}")
        model_path = str(payload.get("modelPath") or "").strip()
        if not model_path:
            default_model = self.default_model_for_game(game)
            if default_model is None:
                raise ValueError(f"no trained model found for {game}")
            model_path = default_model
        model_path = str(Path(model_path).resolve())
        if not Path(model_path).exists():
            raise ValueError(f"model not found: {model_path}")
        human_side = parse_side(game, str(payload.get("humanSide", "")))
        loaded = self.model_store.load(game, model_path)
        cfg = dict(loaded.cfg)
        search_iterations = int(payload.get("searchIterations") or cfg["iters"])
        cfg["iters"] = max(1, search_iterations)
        if cfg.get("chess960", False) and payload.get("chess960Index") is not None:
            cfg["chess960_index"] = int(payload["chess960Index"])
        session = GameSession(
            session_id=uuid.uuid4().hex[:12],
            game=game,
            model_path=model_path,
            human_side=human_side,
            cfg=cfg,
            model=loaded.model,
            device=self.device,
            rust_binary=self.rust_binary,
        )
        with self._lock:
            self.sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> GameSession:
        with self._lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        return session


def read_static_asset(name: str) -> bytes:
    path = STATIC_DIR / name
    return path.read_bytes()


def make_handler(app: PlayApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QuartzPlay/0.1"

        def log_message(self, fmt: str, *args):
            return

        def _send_json(self, status: int, payload: dict[str, Any]):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, status: int, data: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def _path_parts(self) -> list[str]:
            return [part for part in urllib.parse.urlparse(self.path).path.split("/") if part]

        def do_GET(self):
            parts = self._path_parts()
            try:
                if parts == []:
                    self._send_bytes(200, read_static_asset("index.html"), "text/html; charset=utf-8")
                    return
                if parts == ["app.js"]:
                    self._send_bytes(200, read_static_asset("app.js"), "text/javascript; charset=utf-8")
                    return
                if parts == ["styles.css"]:
                    self._send_bytes(200, read_static_asset("styles.css"), "text/css; charset=utf-8")
                    return
                if parts == ["api", "models"]:
                    self._send_json(200, {"models": app.available_models(), "games": sorted(GAME_CONFIGS)})
                    return
                if len(parts) == 3 and parts[:2] == ["api", "session"]:
                    session = app.get_session(parts[2])
                    self._send_json(200, session.payload())
                    return
                self._send_json(404, {"error": "not found"})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self):
            parts = self._path_parts()
            try:
                if parts == ["api", "session"]:
                    session = app.create_session(self._body_json())
                    self._send_json(200, session.payload())
                    return
                if len(parts) == 4 and parts[:2] == ["api", "session"]:
                    session = app.get_session(parts[2])
                    body = self._body_json()
                    action = parts[3]
                    if action == "move":
                        session.apply_human_move(body)
                    elif action == "ai":
                        session.apply_ai_move()
                    elif action == "undo":
                        session.undo(int(body.get("count", 1)))
                    elif action == "restart":
                        session.restart()
                    elif action == "resign":
                        session.resign(str(body.get("side") or side_name(session.game, session.human_side)))
                    else:
                        self._send_json(404, {"error": "not found"})
                        return
                    self._send_json(200, session.payload())
                    return
                self._send_json(404, {"error": "not found"})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="QUARTZ human-play browser GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    args = parser.parse_args()

    device = choose_device(args.device)
    app = PlayApp(Path(args.models_dir).resolve(), device, args.rust_binary)
    handler = make_handler(app)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"QUARTZ Play GUI listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.close()


if __name__ == "__main__":
    main()
