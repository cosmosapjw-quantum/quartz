"""
Game-Agnostic Board Encoders for QUARTZ AlphaZero
==================================================

Each encoder transforms a game-specific state into a (C, H, W) tensor
suitable for the AlphaZero neural network. The encoder also provides
the inverse (tensor → flat board) for TreeMCTS board reconstruction.

Encoder contract:
  - encode(board_flat, player) → np.ndarray (C, H, W)
  - decode(enc, player) → np.ndarray (board_size², dtype=int8)
  - heuristic_prior(board_flat, player) → np.ndarray (n_actions,)
  - fast_leaf_value(board_flat, last_move, mover) → float
  - n_channels: int
  - board_size: int
  - n_actions: int
"""

import numpy as np
import math
from abc import ABC, abstractmethod


class GameEncoder(ABC):
    """Base class for all game encoders."""

    @property
    @abstractmethod
    def n_channels(self) -> int: ...

    @property
    @abstractmethod
    def board_size(self) -> int: ...

    @property
    @abstractmethod
    def n_actions(self) -> int: ...

    @property
    def win_length(self) -> int:
        """Win condition length (0 for non-connect games like Go/Chess)."""
        return 0

    @abstractmethod
    def encode(self, board_flat: np.ndarray, player: int) -> np.ndarray:
        """Encode board to (C, H, W) tensor."""
        ...

    @abstractmethod
    def decode(self, enc: np.ndarray, player: int) -> np.ndarray:
        """Reconstruct flat board from encoded tensor."""
        ...

    def heuristic_prior(self, board_flat: np.ndarray, player: int) -> np.ndarray:
        """Domain-specific heuristic prior. Default: uniform over legal."""
        legal = (board_flat == 0).astype(np.float32)
        s = legal.sum()
        return legal / s if s > 0 else np.ones(self.n_actions, dtype=np.float32) / self.n_actions

    def fast_leaf_value(self, board_flat: np.ndarray, last_move: int, mover: int) -> float:
        """O(1) leaf evaluation from last move. Default: 0."""
        return 0.0

    def legal_mask(self, board_flat: np.ndarray) -> np.ndarray:
        """Binary mask of legal actions."""
        bs2 = self.board_size ** 2
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for i in range(min(bs2, self.n_actions)):
            if board_flat[i] == 0:
                mask[i] = 1.0
        return mask


# ════════════════════════════════════════════
# § Gomoku Encoder (gomoku7, gomoku15, etc.)
# ════════════════════════════════════════════

class GomokuEncoder(GameEncoder):
    """3-plane encoder for Gomoku variants.

    Planes:
      0: current player's stones
      1: opponent's stones
      2: color indicator (1.0 if current player is black)
    """

    def __init__(self, board_size: int = 15, win_len: int = 5):
        self._board_size = board_size
        self._win_len = win_len

    @property
    def n_channels(self) -> int:
        return 3

    @property
    def board_size(self) -> int:
        return self._board_size

    @property
    def n_actions(self) -> int:
        return self._board_size ** 2

    @property
    def win_length(self) -> int:
        return self._win_len

    def encode(self, board_flat, player):
        bs = self._board_size
        enc = np.zeros((3, bs, bs), dtype=np.float32)
        for i in range(bs * bs):
            r, c = i // bs, i % bs
            if board_flat[i] == player:
                enc[0, r, c] = 1.0
            elif board_flat[i] != 0:
                enc[1, r, c] = 1.0
        if player == 1:
            enc[2] = 1.0
        return enc

    def decode(self, enc, player):
        bs = self._board_size
        board = np.zeros(bs * bs, dtype=np.int8)
        for r in range(bs):
            for c in range(bs):
                if enc[0, r, c] > 0.5:
                    board[r * bs + c] = player
                elif enc[1, r, c] > 0.5:
                    board[r * bs + c] = -player
        return board

    def heuristic_prior(self, board_flat, player):
        """Pattern-based prior: adjacency + threat detection.

        Called once at root. O(n² × 4 directions).
        """
        bs = self._board_size
        n2 = bs * bs
        wl = self._win_len
        scores = np.zeros(n2, dtype=np.float32)

        for pos in range(n2):
            if board_flat[pos] != 0:
                continue
            r, c = pos // bs, pos % bs

            # Adjacency bonus
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < bs and 0 <= nc < bs and board_flat[nr * bs + nc] != 0:
                        scores[pos] += 0.5

            # Threat patterns in 4 directions
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                for side in (player, -player):
                    cnt = 0
                    oe = 0
                    for sign in (1, -1):
                        nr, nc = r + sign * dr, c + sign * dc
                        while 0 <= nr < bs and 0 <= nc < bs:
                            if board_flat[nr * bs + nc] == side:
                                cnt += 1
                                nr += sign * dr
                                nc += sign * dc
                            elif board_flat[nr * bs + nc] == 0:
                                oe += 1
                                break
                            else:
                                break
                    mult = 1.0 if side == player else 0.8
                    if cnt >= wl - 1:
                        scores[pos] += 100 * mult
                    elif cnt == wl - 2 and oe >= 1:
                        scores[pos] += 15 * mult
                    elif cnt >= 1:
                        scores[pos] += cnt * mult

            # Center bias
            scores[pos] += max(0, (bs - abs(r - bs // 2) - abs(c - bs // 2))) * 0.2

        s = scores.sum()
        if s > 0:
            scores /= s
        return scores

    def fast_leaf_value(self, board_flat, last_move, mover):
        """O(4 directions) leaf evaluation from last move."""
        if last_move < 0:
            return 0.0
        bs = self._board_size
        wl = self._win_len
        r0, c0 = last_move // bs, last_move % bs
        threat = 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt = 1
            open_ends = 0
            for sign in (1, -1):
                nr, nc = r0 + sign * dr, c0 + sign * dc
                while 0 <= nr < bs and 0 <= nc < bs and board_flat[nr * bs + nc] == mover:
                    cnt += 1
                    nr += sign * dr
                    nc += sign * dc
                if 0 <= nr < bs and 0 <= nc < bs and board_flat[nr * bs + nc] == 0:
                    open_ends += 1
            if cnt >= wl:
                return 1.0
            if cnt == wl - 1 and open_ends >= 1:
                threat += 50
            elif cnt == wl - 2 and open_ends >= 2:
                threat += 10
            elif cnt >= 2:
                threat += cnt
        return float(np.tanh(threat / 50.0))


# ════════════════════════════════════════════
# § Go Encoder (9×9, 19×19)
# ════════════════════════════════════════════

class GoEncoder(GameEncoder):
    """17-plane encoder for Go.

    Planes 0-7:   current player's stones by liberties (1,2,3,4,5,6,7,8+)
    Planes 8-15:  opponent's stones by liberties (1,2,3,4,5,6,7,8+)
    Plane 16:     color indicator (1.0 if current player is black)

    This matches the AlphaGo Zero encoding (simplified — no history planes).
    """

    def __init__(self, board_size: int = 9):
        self._board_size = board_size

    @property
    def n_channels(self) -> int:
        return 17

    @property
    def board_size(self) -> int:
        return self._board_size

    @property
    def n_actions(self) -> int:
        return self._board_size ** 2 + 1  # +1 for pass

    def _count_liberties(self, board_flat, pos, color):
        """Count liberties of the group containing `pos`."""
        bs = self._board_size
        visited = set()
        liberties = set()
        stack = [pos]
        while stack:
            p = stack.pop()
            if p in visited:
                continue
            visited.add(p)
            r, c = p // bs, p % bs
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < bs and 0 <= nc < bs:
                    np_ = nr * bs + nc
                    if board_flat[np_] == 0:
                        liberties.add(np_)
                    elif board_flat[np_] == color and np_ not in visited:
                        stack.append(np_)
        return len(liberties)

    def encode(self, board_flat, player):
        bs = self._board_size
        n2 = bs * bs
        enc = np.zeros((17, bs, bs), dtype=np.float32)

        for pos in range(n2):
            if board_flat[pos] == 0:
                continue
            r, c = pos // bs, pos % bs
            color = board_flat[pos]
            libs = self._count_liberties(board_flat, pos, color)
            lib_idx = min(libs, 8) - 1  # 0-7
            if lib_idx < 0:
                lib_idx = 0

            if color == player:
                enc[lib_idx, r, c] = 1.0
            else:
                enc[8 + lib_idx, r, c] = 1.0

        if player == 1:
            enc[16] = 1.0
        return enc

    def decode(self, enc, player):
        bs = self._board_size
        board = np.zeros(bs * bs, dtype=np.int8)
        for r in range(bs):
            for c in range(bs):
                # Current player: any of planes 0-7
                if np.any(enc[0:8, r, c] > 0.5):
                    board[r * bs + c] = player
                # Opponent: any of planes 8-15
                elif np.any(enc[8:16, r, c] > 0.5):
                    board[r * bs + c] = -player
        return board


# ════════════════════════════════════════════
# § Chess Encoder (8×8)
# ════════════════════════════════════════════

class ChessEncoder(GameEncoder):
    """16-plane encoder for Chess.

    Planes 0-5:   white pieces (P,N,B,R,Q,K)
    Planes 6-11:  black pieces (p,n,b,r,q,k)
    Plane 12:     white kingside castling rights
    Plane 13:     white queenside castling rights
    Plane 14:     en passant square
    Plane 15:     side to move (1.0 if black)

    Matches Rust Chess::encode_planes() output (16 × 64 = 1024).
    """

    @property
    def n_channels(self) -> int:
        return 16

    @property
    def board_size(self) -> int:
        return 8

    @property
    def n_actions(self) -> int:
        return 4096

    def encode(self, board_flat, player):
        # Chess encoding is done by Rust encode_planes().
        # This Python encoder handles the simple case for TreeMCTS fallback.
        enc = np.zeros((16, 8, 8), dtype=np.float32)
        # board_flat for chess is the Rust-encoded 1024-element binary vector
        # reshaped to (16, 8, 8)
        if len(board_flat) == 1024:
            enc = board_flat.astype(np.float32).reshape(16, 8, 8)
        elif len(board_flat) == 64:
            # Simple piece encoding: positive = white, negative = black
            for i in range(64):
                r, c = i // 8, i % 8
                v = board_flat[i]
                if v > 0 and v <= 6:
                    enc[v - 1, r, c] = 1.0
                elif v < 0 and v >= -6:
                    enc[5 + abs(v), r, c] = 1.0
            if player == -1:
                enc[15] = 1.0
        return enc

    def decode(self, enc, player):
        board = np.zeros(64, dtype=np.int8)
        for r in range(8):
            for c in range(8):
                for p in range(6):
                    if enc[p, r, c] > 0.5:
                        board[r * 8 + c] = p + 1
                        break
                    elif enc[6 + p, r, c] > 0.5:
                        board[r * 8 + c] = -(p + 1)
                        break
        return board


# ════════════════════════════════════════════
# § Encoder Registry
# ════════════════════════════════════════════

ENCODERS = {
    "gomoku7": lambda: GomokuEncoder(board_size=7, win_len=4),
    "gomoku15": lambda: GomokuEncoder(board_size=15, win_len=5),
    "gomoku15_free": lambda: GomokuEncoder(board_size=15, win_len=5),
    "gomoku15_std": lambda: GomokuEncoder(board_size=15, win_len=5),
    "gomoku15_omok": lambda: GomokuEncoder(board_size=15, win_len=5),
    "gomoku15_renju": lambda: GomokuEncoder(board_size=15, win_len=5),
    "gomoku15_caro": lambda: GomokuEncoder(board_size=15, win_len=5),
    "go9": lambda: GoEncoder(board_size=9),
    "go9_cn": lambda: GoEncoder(board_size=9),
    "go9_jp": lambda: GoEncoder(board_size=9),
    "go9_kr": lambda: GoEncoder(board_size=9),
    "go13": lambda: GoEncoder(board_size=13),
    "go13_cn": lambda: GoEncoder(board_size=13),
    "go13_jp": lambda: GoEncoder(board_size=13),
    "go13_kr": lambda: GoEncoder(board_size=13),
    "go19": lambda: GoEncoder(board_size=19),
    "go19_cn": lambda: GoEncoder(board_size=19),
    "go19_jp": lambda: GoEncoder(board_size=19),
    "go19_kr": lambda: GoEncoder(board_size=19),
    "chess": lambda: ChessEncoder(),
    "chess960": lambda: ChessEncoder(),
    "tictactoe": lambda: GomokuEncoder(board_size=3, win_len=3),
}


def get_encoder(game_name: str) -> GameEncoder:
    """Get encoder by game name. Raises KeyError if not found."""
    if game_name not in ENCODERS:
        raise KeyError(f"Unknown game: {game_name}. Available: {list(ENCODERS.keys())}")
    return ENCODERS[game_name]()


# ════════════════════════════════════════════
# § Self-tests
# ════════════════════════════════════════════

def _run_tests():
    print("Testing encoders...")

    # Gomoku encode/decode roundtrip
    enc = GomokuEncoder(7, 4)
    board = np.zeros(49, dtype=np.int8)
    board[24] = 1; board[25] = -1
    t = enc.encode(board, 1)
    assert t.shape == (3, 7, 7), f"Wrong shape: {t.shape}"
    assert t[0, 3, 3] == 1.0, "Current player stone missing"
    assert t[1, 3, 4] == 1.0, "Opponent stone missing"
    assert t[2, 0, 0] == 1.0, "Color plane missing"
    b2 = enc.decode(t, 1)
    assert b2[24] == 1 and b2[25] == -1, f"Roundtrip failed: {b2[24]}, {b2[25]}"
    print("  [PASS] GomokuEncoder 7×7 roundtrip")

    # Gomoku 15×15
    enc15 = GomokuEncoder(15, 5)
    assert enc15.n_channels == 3 and enc15.n_actions == 225
    board15 = np.zeros(225, dtype=np.int8)
    board15[112] = 1
    t15 = enc15.encode(board15, 1)
    b15 = enc15.decode(t15, 1)
    assert b15[112] == 1
    print("  [PASS] GomokuEncoder 15×15 roundtrip")

    # Gomoku heuristic prior
    board[21] = 1; board[22] = 1; board[23] = 1  # 3-in-a-row
    prior = enc.heuristic_prior(board, 1)
    assert prior.shape == (49,)
    # Position 24 should be winning (already has a stone, skip to another test)
    board2 = np.zeros(49, dtype=np.int8)
    board2[21] = 1; board2[22] = 1; board2[23] = 1
    prior2 = enc.heuristic_prior(board2, 1)
    best = np.argmax(prior2)
    assert best == 24 or best == 20, f"Expected winning move, got {best}"
    print("  [PASS] GomokuEncoder heuristic prior")

    # Fast leaf value
    lv = enc.fast_leaf_value(board2, 23, 1)
    assert lv > 0.5, f"Should detect strong threat, got {lv}"
    print("  [PASS] GomokuEncoder fast_leaf_value")

    # Go encoder
    go_enc = GoEncoder(9)
    assert go_enc.n_channels == 17 and go_enc.n_actions == 82
    go_board = np.zeros(81, dtype=np.int8)
    go_board[40] = 1; go_board[41] = -1
    gt = go_enc.encode(go_board, 1)
    assert gt.shape == (17, 9, 9)
    gb = go_enc.decode(gt, 1)
    assert gb[40] == 1 and gb[41] == -1, f"Go roundtrip failed"
    print("  [PASS] GoEncoder 9×9 roundtrip")

    # Go liberty counting
    # Stone at center with 4 liberties
    go_b2 = np.zeros(81, dtype=np.int8)
    go_b2[40] = 1  # center of 9×9
    gt2 = go_enc.encode(go_b2, 1)
    # Should be on liberty plane 3 (4 liberties → index 3)
    assert gt2[3, 4, 4] == 1.0, f"Center stone should have 4 libs (plane 3)"
    print("  [PASS] GoEncoder liberty counting")

    # Chess encoder
    ch_enc = ChessEncoder()
    assert ch_enc.n_channels == 16 and ch_enc.n_actions == 4096
    ch_board = np.zeros(64, dtype=np.int8)
    ch_board[0] = 4  # white rook at a1
    ch_board[63] = -4  # black rook at h8
    ct = ch_enc.encode(ch_board, 1)
    assert ct.shape == (16, 8, 8)
    cb = ch_enc.decode(ct, 1)
    assert cb[0] == 4 and cb[63] == -4, f"Chess roundtrip failed"
    print("  [PASS] ChessEncoder 8×8 roundtrip")

    # Registry
    for name in ENCODERS:
        e = get_encoder(name)
        assert isinstance(e, GameEncoder)
    print("  [PASS] Encoder registry")

    print(f"\n[ALL PASS] 8 encoder tests passed.")


if __name__ == "__main__":
    _run_tests()
