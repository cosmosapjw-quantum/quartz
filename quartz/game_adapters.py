"""Local game adapters used by training, arena evaluation, and play GUI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quartz.selfplay_runtime import encode_chess_fen as _encode_chess_fen_impl
from quartz.selfplay_runtime import initial_chess_fen as _initial_chess_fen_impl
from quartz.selfplay_runtime import is_chess_game as _is_chess_game_impl
from quartz.selfplay_runtime import is_go_game as _is_go_game_impl
from quartz.training_catalog import GOMOKU15_VARIANTS, STANDARD_CHESS_FEN


DEFAULT_CHESS_POLICY_ACTIONS = 4672
DEFAULT_STANDARD_CHESS_FEN = STANDARD_CHESS_FEN


def _default_is_chess_game(game_name):
    return bool(_is_chess_game_impl(game_name))


def _default_is_go_game(game_name):
    return bool(_is_go_game_impl(game_name))


def _default_initial_chess_fen(cfg):
    return _initial_chess_fen_impl(cfg, standard_chess_fen=DEFAULT_STANDARD_CHESS_FEN)


def _default_encode_chess_fen(fen):
    return _encode_chess_fen_impl(fen)


def _default_gomoku15_variants():
    return set(GOMOKU15_VARIANTS)


def _encode_board_fallback(cfg, board_flat, player):
    bs = int(cfg["board"])
    n2 = bs * bs
    ch = int(cfg.get("ch", 17))
    enc = np.zeros((ch, bs, bs), dtype=np.float32)
    # [OPT] Vectorized board encoding
    board_arr = np.asarray(board_flat, dtype=np.int8).ravel()[:n2]
    my_val = np.int8(player)
    enc[0].ravel()[:len(board_arr)] = (board_arr == my_val).astype(np.float32)
    opp_mask = (board_arr != 0) & (board_arr != my_val)
    enc[1].ravel()[:len(board_arr)] = opp_mask.astype(np.float32)
    if player == 1:
        enc[ch - 1] = 1.0
    return enc


@dataclass(frozen=True)
class GameAdapterRuntimeHooks:
    is_chess_game: object
    is_go_game: object
    initial_chess_fen: object
    encode_chess_fen: object
    gomoku15_variants: object
    chess_policy_actions: int = DEFAULT_CHESS_POLICY_ACTIONS
    gomoku_adapter_cls: object | None = None
    go_adapter_cls: object | None = None
    tictactoe_adapter_cls: object | None = None
    chess_adapter_cls: object | None = None


class GomokuGameAdapter:
    """Flat-board adapter for gomoku-style games and variants."""

    def __init__(self, board_size=7, win_len=4, encoder=None, variant="gomoku7"):
        self._bs = board_size
        self._wl = win_len
        self._variant = variant
        self._board = [0] * (board_size * board_size)
        self._player = 1
        self._terminal = False
        self._outcome = None
        self._encoder = encoder
        self._ch = encoder.n_channels if encoder else 3

    def clone(self):
        g = GomokuGameAdapter(self._bs, self._wl, self._encoder, self._variant)
        g._board = self._board[:]
        g._player = self._player
        g._terminal = self._terminal
        g._outcome = self._outcome
        return g

    def _line_count(self, action, dr, dc):
        r0, c0 = action // self._bs, action % self._bs
        cnt = 1
        for sign in (1, -1):
            nr, nc = r0 + sign * dr, c0 + sign * dc
            while 0 <= nr < self._bs and 0 <= nc < self._bs and self._board[nr * self._bs + nc] == self._player:
                cnt += 1
                nr += sign * dr
                nc += sign * dc
        return cnt

    def _line_ends(self, action, dr, dc):
        r0, c0 = action // self._bs, action % self._bs
        stone = self._player
        forward = backward = 0

        nr, nc = r0 + dr, c0 + dc
        while 0 <= nr < self._bs and 0 <= nc < self._bs and self._board[nr * self._bs + nc] == stone:
            forward += 1
            nr += dr
            nc += dc
        forward_blocked = not (0 <= nr < self._bs and 0 <= nc < self._bs) or self._board[nr * self._bs + nc] == -stone

        nr, nc = r0 - dr, c0 - dc
        while 0 <= nr < self._bs and 0 <= nc < self._bs and self._board[nr * self._bs + nc] == stone:
            backward += 1
            nr -= dr
            nc -= dc
        backward_blocked = not (0 <= nr < self._bs and 0 <= nc < self._bs) or self._board[nr * self._bs + nc] == -stone

        return 1 + forward + backward, forward_blocked, backward_blocked

    def _is_winning_move(self, action):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt, forward_blocked, backward_blocked = self._line_ends(action, dr, dc)
            if self._variant == "gomoku15_std":
                if cnt == self._wl:
                    return True
                continue
            if self._variant in {"gomoku15_omok", "gomoku15_renju"}:
                if self._player == 1:
                    if cnt == self._wl:
                        return True
                elif cnt >= self._wl:
                    return True
                continue
            if self._variant == "gomoku15_caro":
                if cnt == self._wl and not (forward_blocked and backward_blocked):
                    return True
                continue
            if cnt >= self._wl:
                return True
        return False

    def apply_move(self, action):
        self._board[action] = self._player
        if self._is_winning_move(action):
            self._terminal = True
            self._outcome = 1.0 if self._player == 1 else -1.0
            self._player = -self._player
            return
        move_limit = 200 if self._variant == "gomoku15_renju" else self._bs * self._bs
        played = sum(1 for value in self._board if value != 0)
        if played >= move_limit or not any(b == 0 for b in self._board):
            self._terminal = True
            self._outcome = 0.0
        self._player = -self._player

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def current_player(self):
        return 0 if self._player == 1 else 1

    def legal_moves(self):
        if self._terminal:
            return []
        return [i for i, value in enumerate(self._board) if value == 0]

    def _encode(self):
        if self._encoder is not None:
            return self._encoder.encode(np.array(self._board, dtype=np.int8), self._player)
        enc = np.zeros((self._ch, self._bs, self._bs), dtype=np.float32)
        for i in range(self._bs * self._bs):
            r, c = i // self._bs, i % self._bs
            if self._board[i] == self._player:
                enc[0, r, c] = 1.0
            elif self._board[i] != 0:
                enc[1, r, c] = 1.0
        if self._ch >= 3 and self._player == 1:
            enc[2] = 1.0
        return enc


class TicTacToeGameAdapter:
    def __init__(self, encoder=None):
        self._bs = 3
        self._board = [0] * 9
        self._player = 1
        self._terminal = False
        self._outcome = None
        self._encoder = encoder

    def clone(self):
        g = TicTacToeGameAdapter(self._encoder)
        g._board = self._board[:]
        g._player = self._player
        g._terminal = self._terminal
        g._outcome = self._outcome
        return g

    def apply_move(self, action):
        self._board[action] = self._player
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        for a, b, c in lines:
            if self._board[a] != 0 and self._board[a] == self._board[b] == self._board[c]:
                self._terminal = True
                self._outcome = 1.0 if self._player == 1 else -1.0
                self._player = -self._player
                return
        if not any(v == 0 for v in self._board):
            self._terminal = True
            self._outcome = 0.0
        self._player = -self._player

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def current_player(self):
        return 0 if self._player == 1 else 1

    def legal_moves(self):
        if self._terminal:
            return []
        return [i for i, value in enumerate(self._board) if value == 0]

    def _encode(self):
        if self._encoder is not None:
            return self._encoder.encode(np.array(self._board, dtype=np.int8), self._player)
        enc = np.zeros((3, 3, 3), dtype=np.float32)
        for i, value in enumerate(self._board):
            r, c = divmod(i, 3)
            if value == self._player:
                enc[0, r, c] = 1.0
            elif value != 0:
                enc[1, r, c] = 1.0
        if self._player == 1:
            enc[2] = 1.0
        return enc


class GoGameAdapter:
    """Local Go state used by training/eval for configurable Go rulesets."""

    def __init__(
        self,
        board_size=9,
        komi=7.5,
        encoder=None,
        scoring="area",
        allow_suicide=False,
        ruleset="chinese",
        _encode_board_fn=None,
    ):
        self._bs = board_size
        self._board = [0] * (board_size * board_size)
        self._player = 1
        self._passes = 0
        self._ko_point = None
        self._terminal = False
        self._outcome = None
        self._komi = komi
        self._encoder = encoder
        self._scoring = scoring
        self._allow_suicide = allow_suicide
        self._ruleset = ruleset
        self._black_caps = 0
        self._white_caps = 0
        self._cycle_terminal = False
        self._history_hashes = {self._position_hash()}
        self._void_result = False
        self._encode_board_fn = _encode_board_fn or _encode_board_fallback

    def clone(self):
        g = GoGameAdapter(
            self._bs,
            self._komi,
            self._encoder,
            scoring=self._scoring,
            allow_suicide=self._allow_suicide,
            ruleset=self._ruleset,
            _encode_board_fn=self._encode_board_fn,
        )
        g._board = self._board[:]
        g._player = self._player
        g._passes = self._passes
        g._ko_point = self._ko_point
        g._terminal = self._terminal
        g._outcome = self._outcome
        g._black_caps = self._black_caps
        g._white_caps = self._white_caps
        g._cycle_terminal = self._cycle_terminal
        g._history_hashes = set(self._history_hashes)
        g._void_result = self._void_result
        return g

    def _position_hash(self, board=None, player=None):
        state_board = tuple(self._board if board is None else board)
        state_player = self._player if player is None else player
        return (state_board, state_player)

    def _neighbors(self, pos):
        r, c = divmod(pos, self._bs)
        if r > 0:
            yield pos - self._bs
        if r + 1 < self._bs:
            yield pos + self._bs
        if c > 0:
            yield pos - 1
        if c + 1 < self._bs:
            yield pos + 1

    def _group_and_liberties(self, board, pos):
        color = board[pos]
        if color == 0:
            return [], set()
        group = []
        liberties = set()
        stack = [pos]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            for nb in self._neighbors(cur):
                if board[nb] == 0:
                    liberties.add(nb)
                elif board[nb] == color and nb not in visited:
                    stack.append(nb)
        return group, liberties

    def _is_legal(self, pos):
        if self._cycle_terminal:
            return False
        if pos < 0 or pos >= len(self._board):
            return False
        if self._board[pos] != 0:
            return False
        if self._ko_point == pos:
            return False
        board = self._board[:]
        board[pos] = self._player
        opp = -self._player
        captured = 0
        for nb in self._neighbors(pos):
            if board[nb] == opp:
                group, liberties = self._group_and_liberties(board, nb)
                if not liberties:
                    captured += len(group)
                    for stone in group:
                        board[stone] = 0
        _, liberties = self._group_and_liberties(board, pos)
        if not (liberties or captured > 0 or self._allow_suicide):
            return False
        if self._ruleset == "chinese":
            next_player = -self._player
            if self._position_hash(board=board, player=next_player) in self._history_hashes:
                return False
        return True

    def _score(self):
        if self._scoring == "area":
            black = 0.0
            white = self._komi
            visited = set()
            for pos, value in enumerate(self._board):
                if value == 1:
                    black += 1.0
                elif value == -1:
                    white += 1.0
                elif pos not in visited:
                    region = []
                    owners = set()
                    stack = [pos]
                    while stack:
                        cur = stack.pop()
                        if cur in visited or self._board[cur] != 0:
                            continue
                        visited.add(cur)
                        region.append(cur)
                        for nb in self._neighbors(cur):
                            if self._board[nb] == 0:
                                stack.append(nb)
                            else:
                                owners.add(self._board[nb])
                    if owners == {1}:
                        black += float(len(region))
                    elif owners == {-1}:
                        white += float(len(region))
            return black, white

        board = self._board[:]
        black = float(self._black_caps)
        white = float(self._white_caps) + self._komi

        def classify_empty_regions(state_board):
            region_ids = {}
            owners = []
            for pos, value in enumerate(state_board):
                if value != 0 or pos in region_ids:
                    continue
                rid = len(owners)
                stack = [pos]
                region_ids[pos] = rid
                border = set()
                while stack:
                    cur = stack.pop()
                    for nb in self._neighbors(cur):
                        if state_board[nb] == 0 and nb not in region_ids:
                            region_ids[nb] = rid
                            stack.append(nb)
                        elif state_board[nb] != 0:
                            border.add(state_board[nb])
                if border == {1}:
                    owners.append(1)
                elif border == {-1}:
                    owners.append(-1)
                else:
                    owners.append(0)
            return region_ids, owners

        if self._ruleset in {"japanese", "korean"}:
            while True:
                region_ids, owners = classify_empty_regions(board)
                visited = set()
                removed_any = False
                for pos, color in enumerate(board):
                    if color == 0 or pos in visited:
                        continue
                    stack = [pos]
                    group = []
                    adj_regions = set()
                    touches_opponent = False
                    touches_edge = False
                    while stack:
                        cur = stack.pop()
                        if cur in visited:
                            continue
                        visited.add(cur)
                        group.append(cur)
                        row, col = divmod(cur, self._bs)
                        if row == 0 or row + 1 == self._bs or col == 0 or col + 1 == self._bs:
                            touches_edge = True
                        for nb in self._neighbors(cur):
                            if board[nb] == color:
                                stack.append(nb)
                            elif board[nb] == 0 and nb in region_ids:
                                adj_regions.add(region_ids[nb])
                            elif board[nb] == -color:
                                touches_opponent = True
                    eye_count = 0
                    touches_neutral = False
                    for rid in adj_regions:
                        owner = owners[rid]
                        if owner == color:
                            eye_count += 1
                        elif owner == 0:
                            touches_neutral = True
                    if eye_count < 2 and not touches_neutral and touches_opponent and not touches_edge:
                        removed_any = True
                        for stone in group:
                            board[stone] = 0
                        if color == 1:
                            white += float(len(group))
                        else:
                            black += float(len(group))
                if not removed_any:
                    break

        visited = set()
        for pos, value in enumerate(board):
            if value != 0 or pos in visited:
                continue
            region = []
            owners = set()
            stack = [pos]
            while stack:
                cur = stack.pop()
                if cur in visited or board[cur] != 0:
                    continue
                visited.add(cur)
                region.append(cur)
                for nb in self._neighbors(cur):
                    if board[nb] == 0:
                        stack.append(nb)
                    else:
                        owners.add(board[nb])
            if owners == {1}:
                black += float(len(region))
            elif owners == {-1}:
                white += float(len(region))
        return black, white

    def apply_move(self, action):
        pass_action = self._bs * self._bs
        if action == pass_action:
            self._passes += 1
            self._ko_point = None
            self._player = -self._player
            if self._ruleset in {"japanese", "korean"} and self._passes < 2 and self._position_hash() in self._history_hashes:
                self._cycle_terminal = True
            self._history_hashes.add(self._position_hash())
        else:
            self._passes = 0
            self._board[action] = self._player
            opp = -self._player
            captured_points = []
            for nb in self._neighbors(action):
                if self._board[nb] == opp:
                    group, liberties = self._group_and_liberties(self._board, nb)
                    if not liberties:
                        captured_points.extend(group)
            for stone in captured_points:
                self._board[stone] = 0
            if captured_points:
                if self._player == 1:
                    self._black_caps += len(captured_points)
                else:
                    self._white_caps += len(captured_points)
            group, liberties = self._group_and_liberties(self._board, action)
            self._ko_point = None
            if len(captured_points) == 1 and len(liberties) == 1:
                self._ko_point = captured_points[0]
            if not liberties:
                if self._allow_suicide:
                    for stone in group:
                        self._board[stone] = 0
                    if self._player == 1:
                        self._white_caps += len(group)
                    else:
                        self._black_caps += len(group)
                else:
                    raise ValueError(f"illegal suicide move at {action}")
            self._player = -self._player
            if self._ruleset in {"japanese", "korean"} and self._position_hash() in self._history_hashes:
                self._cycle_terminal = True
            self._history_hashes.add(self._position_hash())
        if self._passes >= 2 or self._cycle_terminal:
            self._terminal = True
            if self._cycle_terminal:
                self._void_result = self._ruleset == "japanese"
                self._outcome = None if self._void_result else 0.0
            else:
                self._void_result = False
                black, white = self._score()
                if black > white:
                    self._outcome = 1.0
                elif white > black:
                    self._outcome = -1.0
                else:
                    self._outcome = 0.0

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def is_void_result(self):
        return self._terminal and self._void_result

    def current_player(self):
        return 0 if self._player == 1 else 1

    def legal_moves(self):
        if self._terminal:
            return []
        moves = [i for i, value in enumerate(self._board) if value == 0 and self._is_legal(i)]
        moves.append(self._bs * self._bs)
        return moves

    def _encode(self):
        if self._encoder is not None:
            return self._encoder.encode(np.array(self._board, dtype=np.int8), self._player)
        return self._encode_board_fn({"board": self._bs, "ch": 17}, np.array(self._board, dtype=np.int8), self._player)


class ChessEvaluationAdapter:
    """Engine-driven chess state for evaluator matches."""

    supports_random_baseline = False

    def __init__(
        self,
        actions=DEFAULT_CHESS_POLICY_ACTIONS,
        encoder=None,
        start_fen=DEFAULT_STANDARD_CHESS_FEN,
        _encode_chess_fen_fn=None,
    ):
        self._actions = actions
        self._encoder = encoder
        self._fen = start_fen
        self._chess_history_hashes = None
        self._terminal = False
        self._outcome = None
        self._encode_chess_fen_fn = _encode_chess_fen_fn or _default_encode_chess_fen

    def clone(self):
        g = ChessEvaluationAdapter(
            self._actions,
            self._encoder,
            self._fen,
            _encode_chess_fen_fn=self._encode_chess_fen_fn,
        )
        g._chess_history_hashes = (
            list(self._chess_history_hashes) if self._chess_history_hashes is not None else None
        )
        g._terminal = self._terminal
        g._outcome = self._outcome
        return g

    def _side_part(self):
        parts = self._fen.split()
        return parts[1] if len(parts) >= 2 else "w"

    def current_player(self):
        return 1 if self._side_part() == "w" else 0

    def legal_moves(self):
        if self._terminal:
            return []
        return list(range(self._actions))

    def apply_engine_meta(self, action, meta):
        if meta.get("terminal", False):
            self._terminal = True
            self._outcome = float(meta.get("outcome_for_black", 0.0))
            return True
        new_fen = meta.get("result_fen", "")
        if not new_fen or new_fen == self._fen:
            return False
        self._fen = new_fen
        history_hashes = meta.get("result_history_hashes")
        if history_hashes is not None:
            self._chess_history_hashes = [int(v) for v in history_hashes]
        return True

    def apply_move(self, action):
        raise RuntimeError("Chess evaluator requires engine-provided state transitions")

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def _encode(self):
        return self._encode_chess_fen_fn(self._fen)


def build_training_game_adapter(cfg, runtime_hooks=None):
    hooks = runtime_hooks or GameAdapterRuntimeHooks(
        is_chess_game=_default_is_chess_game,
        is_go_game=_default_is_go_game,
        initial_chess_fen=_default_initial_chess_fen,
        encode_chess_fen=_default_encode_chess_fen,
        gomoku15_variants=_default_gomoku15_variants(),
        chess_policy_actions=DEFAULT_CHESS_POLICY_ACTIONS,
    )
    gomoku_adapter_cls = hooks.gomoku_adapter_cls or GomokuGameAdapter
    go_adapter_cls = hooks.go_adapter_cls or GoGameAdapter
    tictactoe_adapter_cls = hooks.tictactoe_adapter_cls or TicTacToeGameAdapter
    chess_adapter_cls = hooks.chess_adapter_cls or ChessEvaluationAdapter

    game_name = cfg.get("_name")
    encoder = cfg.get("_encoder")
    if game_name == "tictactoe":
        return tictactoe_adapter_cls(encoder=encoder)
    if hooks.is_chess_game(game_name):
        return chess_adapter_cls(
            actions=cfg.get("actions", hooks.chess_policy_actions),
            encoder=encoder,
            start_fen=hooks.initial_chess_fen(cfg),
            _encode_chess_fen_fn=hooks.encode_chess_fen,
        )
    if hooks.is_go_game(game_name):
        return go_adapter_cls(
            board_size=cfg["board"],
            komi=cfg.get("go_komi", 7.5),
            encoder=encoder,
            scoring=cfg.get("go_scoring", "area"),
            allow_suicide=cfg.get("go_allow_suicide", False),
            ruleset=cfg.get("go_ruleset", "chinese"),
            _encode_board_fn=_encode_board_fallback,
        )
    if game_name in hooks.gomoku15_variants or game_name == "gomoku7":
        return gomoku_adapter_cls(
            board_size=cfg["board"],
            win_len=cfg["win"],
            encoder=encoder,
            variant=game_name,
        )
    raise ValueError(f"No local game adapter for {game_name}")
