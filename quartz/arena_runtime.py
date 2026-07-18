"""Arena helpers: head-to-head evaluation, Glicko-2, and TreeMCTS engine wrappers."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
from quartz import runtime_support


def arena_compare(model_a_path, model_b_path, cfg, device, n_games=50):
    """Play N games with SPRT early termination."""
    alphazero_net_cls = runtime_support.AlphaZeroNet
    load_torch_state_dict = runtime_support.load_torch_state_dict
    torch_module = runtime_support.torch
    tree_mcts_cls = TreeMCTS
    # Use tqdm_factory (which auto-disables on non-TTY stderr) instead
    # of raw tqdm; otherwise smoke_e2e's subprocess capture floods stdout
    # with one progress-bar line per update.
    tqdm_cls = runtime_support.tqdm_factory

    model_a = alphazero_net_cls(cfg).to(device)
    model_b = alphazero_net_cls(cfg).to(device)
    model_a.load_state_dict(
        load_torch_state_dict(model_a_path, torch_module, map_location=device)
    )
    model_b.load_state_dict(
        load_torch_state_dict(model_b_path, torch_module, map_location=device)
    )
    model_a.eval()
    model_b.eval()

    wins_a, wins_b, draws = 0, 0, 0
    mcts_a = tree_mcts_cls(cfg, model_a, device)
    mcts_b = tree_mcts_cls(cfg, model_b, device)

    p0, p1 = 0.5, 0.55
    alpha, beta = 0.05, 0.05
    lower_bound = math.log(beta / (1 - alpha))
    upper_bound = math.log((1 - beta) / alpha)
    sprt_decided = False
    sprt_result = None

    with tqdm_cls(total=n_games, desc="Arena", leave=False) as pbar:
        for game_idx in range(n_games):
            if game_idx % 2 == 0:
                first, second = mcts_a, mcts_b
                first_is_a = True
            else:
                first, second = mcts_b, mcts_a
                first_is_a = False

            board = np.zeros(cfg["board"] ** 2, dtype=np.int8)
            player = 1
            n2 = cfg["board"] ** 2
            winner = 0

            for _move_n in range(n2):
                legal_mask = np.array(
                    [
                        1.0 if board[i] == 0 else 0.0
                        for i in range(min(n2, cfg["actions"]))
                    ]
                )
                if cfg["actions"] > n2:
                    legal_mask = np.concatenate(
                        [legal_mask, np.zeros(cfg["actions"] - n2)]
                    )
                legal = [i for i in range(n2) if board[i] == 0]
                if not legal:
                    break

                encoded = np.zeros(
                    (cfg["ch"], cfg["board"], cfg["board"]), dtype=np.float32
                )
                for i in range(n2):
                    r, c = divmod(i, cfg["board"])
                    if board[i] == player:
                        encoded[0, r, c] = 1.0
                    elif board[i] != 0:
                        encoded[1, r, c] = 1.0
                if cfg["ch"] >= 3 and player == 1:
                    encoded[2] = 1.0

                mcts = first if player == 1 else second
                policy = mcts.search(encoded, player, legal_mask, cfg["iters"] // 4)
                chosen = max(legal, key=lambda i: policy[i] if i < len(policy) else 0)
                board[chosen] = player

                if cfg["win"] > 0:
                    r0, c0 = divmod(chosen, cfg["board"])
                    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        cnt = 1
                        for sign in [1, -1]:
                            nr, nc = r0 + sign * dr, c0 + sign * dc
                            while (
                                0 <= nr < cfg["board"]
                                and 0 <= nc < cfg["board"]
                                and board[nr * cfg["board"] + nc] == player
                            ):
                                cnt += 1
                                nr += sign * dr
                                nc += sign * dc
                        if cnt >= cfg["win"]:
                            winner = player
                            break
                    if winner:
                        break
                player = -player

            if winner == 1:
                if first_is_a:
                    wins_a += 1
                else:
                    wins_b += 1
            elif winner == -1:
                if first_is_a:
                    wins_b += 1
                else:
                    wins_a += 1
            else:
                draws += 1
            pbar.update(1)
            pbar.set_postfix_str(f"A:{wins_a} B:{wins_b} D:{draws}")

            decisive = wins_a + wins_b
            if decisive > 0 and not sprt_decided:
                w = wins_a
                n_dec = decisive
                llr = w * math.log(p1 / p0) + (n_dec - w) * math.log(
                    (1 - p1) / (1 - p0)
                )
                if llr >= upper_bound:
                    sprt_decided = True
                    sprt_result = "H1_accept"
                    pbar.set_postfix_str(f"SPRT: A wins (LLR={llr:.2f})")
                    break
                if llr <= lower_bound:
                    sprt_decided = True
                    sprt_result = "H0_accept"
                    pbar.set_postfix_str(f"SPRT: equal (LLR={llr:.2f})")
                    break

    total = wins_a + wins_b + draws
    wr = wins_a / max(total, 1)
    z = 1.96
    n = max(total, 1)
    p_hat = wr
    ci_lo = (
        p_hat
        + z * z / (2 * n)
        - z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    ) / (1 + z * z / n)
    ci_hi = (
        p_hat
        + z * z / (2 * n)
        + z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    ) / (1 + z * z / n)
    sprt_str = sprt_result or "inconclusive"
    return wins_a, wins_b, draws, wr, (ci_lo, ci_hi), sprt_str


class MCTSNode:
    """Array-based MCTS node used by legacy arena TreeMCTS."""

    __slots__ = [
        "move",
        "child_moves",
        "child_n",
        "child_w",
        "child_prior",
        "child_nodes",
        "n_children",
        "is_expanded",
        "total_n",
    ]

    def __init__(self, parent=None, move=None, prior=0.0):
        self.move = move if move is not None else -1
        self.child_moves = None
        self.child_n = None
        self.child_w = None
        self.child_prior = None
        self.child_nodes = None
        self.n_children = 0
        self.is_expanded = False
        self.total_n = 0

    def expand(self, legal_moves, priors):
        k = len(legal_moves)
        self.child_moves = np.array(legal_moves, dtype=np.int32)
        self.child_n = np.zeros(k, dtype=np.int32)
        self.child_w = np.zeros(k, dtype=np.float32)
        p = np.array(
            [priors[m] if m < len(priors) else 1.0 / max(k, 1) for m in legal_moves],
            dtype=np.float32,
        )
        ps = p.sum()
        if ps > 0:
            p /= ps
        self.child_prior = p
        self.child_nodes = {}
        self.n_children = k
        self.is_expanded = True
        self.total_n = 0

    def backup_child(self, ci, value):
        self.child_n[ci] += 1
        self.child_w[ci] += value
        self.total_n += 1

    @property
    def n(self):
        return self.total_n

    @property
    def w(self):
        return float(self.child_w.sum()) if self.child_w is not None else 0.0

    @property
    def children(self):
        if self.child_nodes is None:
            return []
        return list(self.child_nodes.values())


class TreeMCTS:
    """Legacy arena-only MCTS helper."""

    FPU_OFFSET = 0.25
    FPU_PRIOR_WEIGHT = 3.0

    def __init__(self, cfg, model=None, device="cpu"):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.n_actions = cfg["actions"]
        self.board_size = cfg["board"]
        self.penalty_mode = cfg.get("penalty_mode", "GatedRefresh")
        self.c_puct = cfg.get("c_puct", 2.0)
        self._win_len = cfg.get("win", 0)
        self._encoder = cfg.get("_encoder")
        self._has_heuristic = (
            hasattr(self._encoder, "heuristic_prior") if self._encoder else False
        )

    def _gomoku_heuristic_prior(self, board, player):
        if self._encoder is not None:
            return self._encoder.heuristic_prior(board, player)
        n2 = self.board_size**2
        legal = np.zeros(self.n_actions, dtype=np.float32)
        for i in range(min(n2, self.n_actions)):
            if board[i] == 0:
                legal[i] = 1.0
        s = legal.sum()
        return (
            legal / s
            if s > 0
            else np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        )

    def _fast_leaf_value(self, board, last_move, player_who_moved):
        if self._encoder is not None:
            return self._encoder.fast_leaf_value(board, last_move, player_who_moved)
        return 0.0

    def _encode(self, board, player):
        return runtime_support.encode_board(self.cfg, board, player)

    def _evaluate_leaf(self, board, player):
        if self.model is not None:
            torch_module = runtime_support.torch
            encode = self._encode(board, player)
            with torch_module.no_grad():
                x = (
                    torch_module.tensor(encode, dtype=torch_module.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits, val = self.model(x)
                probs = torch_module.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return probs, val.item()
        probs = np.ones(self.n_actions, dtype=np.float32)
        n2 = self.board_size**2
        for i in range(min(n2, self.n_actions)):
            if board[i] != 0:
                probs[i] = 0
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs, 0.0

    def search(self, board_enc, player, legal_mask, n_iters):
        board = runtime_support.decode_board(self.cfg, board_enc, player)
        bs = self.board_size
        legal_indices = [
            i for i in range(min(self.n_actions, bs * bs)) if board[i] == 0
        ]
        if not legal_indices:
            return np.zeros(self.n_actions, dtype=np.float32)

        if self.model is not None:
            priors, root_val = self._evaluate_leaf(board, player)
        elif self._has_heuristic:
            priors = self._gomoku_heuristic_prior(board, player)
            root_val = 0.0
        else:
            priors, root_val = self._evaluate_leaf(board, player)

        root = MCTSNode()
        root.expand(legal_indices, priors)
        state = _SearchState(
            root=root, board=board, player=player, root_val=root_val, cfg=self
        )

        for _ in range(n_iters):
            leaf = state.select_to_leaf()
            if leaf.is_terminal:
                state.backup(leaf, leaf.terminal_value)
            elif self.model is not None:
                priors, val = self._evaluate_leaf(leaf.board, leaf.player)
                state.expand_and_backup(leaf, priors, val)
            else:
                child_uniform = np.ones(self.n_actions, dtype=np.float32) / max(
                    self.n_actions, 1
                )
                val = self._fast_leaf_value(leaf.board, leaf.last_move, -leaf.player)
                state.expand_and_backup(leaf, child_uniform, val)

        return state.extract_policy(self.n_actions)


@dataclass
class _LeafInfo:
    node: MCTSNode
    ci: int
    path: list
    board: np.ndarray
    player: int
    last_move: int
    is_terminal: bool = False
    terminal_value: float = 0.0


class _SearchState:
    def __init__(self, root, board, player, root_val, cfg):
        self.root = root
        self.board = board
        self.player = player
        self.root_val = root_val
        self.cfg = cfg
        self._wl = cfg._win_len
        self._bs = cfg.board_size
        self._c_puct = cfg.c_puct
        self._penalty = cfg.penalty_mode
        self._fpu_off = cfg.FPU_OFFSET
        self._fpu_pw = cfg.FPU_PRIOR_WEIGHT

    def select_to_leaf(self):
        node = self.root
        cur_board = self.board.copy()
        cur_player = self.player
        path = []
        parent_value = self.root_val
        last_move = -1
        bs = self._bs
        wl = self._wl
        c_puct = self._c_puct
        ci = -1

        while node.is_expanded and node.n_children > 0:
            cp = c_puct
            if node is self.root and self._penalty == "GatedRefresh":
                cp = c_puct * 0.85
            elif node is self.root and self._penalty == "SelfAdaptive":
                cp = c_puct * 0.80

            n = node.child_n
            p = node.child_prior
            unvisited = n == 0
            if unvisited.any():
                fpu_scores = parent_value - self._fpu_off + p * self._fpu_pw
                scores = np.where(
                    unvisited,
                    fpu_scores + np.random.random(node.n_children) * 0.001,
                    -1e9,
                )
                ci = int(np.argmax(scores))
            else:
                q = node.child_w / np.maximum(n.astype(np.float32), 1)
                sqrt_total = math.sqrt(node.total_n + 1)
                u = cp * p * sqrt_total / (1 + n.astype(np.float32))
                ci = int(np.argmax(q + u))
                parent_value = float(q[ci])

            path.append((node, ci))
            move = int(node.child_moves[ci])
            cur_board[move] = cur_player
            last_move = move

            won = False
            if wl > 0:
                r0, c0 = move // bs, move % bs
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cnt = 1
                    for sign in (1, -1):
                        nr, nc = r0 + sign * dr, c0 + sign * dc
                        while (
                            0 <= nr < bs
                            and 0 <= nc < bs
                            and cur_board[nr * bs + nc] == cur_player
                        ):
                            cnt += 1
                            nr += sign * dr
                            nc += sign * dc
                    if cnt >= wl:
                        won = True
                        break
                if won:
                    return _LeafInfo(
                        node=node,
                        ci=ci,
                        path=path,
                        board=cur_board,
                        player=cur_player,
                        last_move=last_move,
                        is_terminal=True,
                        terminal_value=-1.0,
                    )

            cur_player = -cur_player
            if ci in node.child_nodes:
                node = node.child_nodes[ci]
            else:
                break

        return _LeafInfo(
            node=node,
            ci=ci,
            path=path,
            board=cur_board,
            player=cur_player,
            last_move=last_move,
        )

    def expand_and_backup(self, leaf, priors, leaf_val):
        bs = self._bs
        na = self.cfg.n_actions
        child_legal = [i for i in range(min(na, bs * bs)) if leaf.board[i] == 0]
        if child_legal:
            new_node = MCTSNode()
            new_node.expand(child_legal, priors)
            leaf.node.child_nodes[leaf.ci] = new_node
            value = -leaf_val
        else:
            value = 0.0
        self._do_backup(leaf.path, value)

    def backup(self, leaf, value):
        self._do_backup(leaf.path, value)

    def _do_backup(self, path, value):
        for nd, ci in reversed(path):
            nd.backup_child(ci, value)
            value = -value

    def extract_policy(self, n_actions):
        visits = np.zeros(n_actions, dtype=np.float32)
        for i in range(self.root.n_children):
            if self.root.child_moves[i] < n_actions:
                visits[self.root.child_moves[i]] = self.root.child_n[i]
        total = visits.sum()
        if total > 0:
            visits /= total
        return visits


class Glicko2Rating:
    """Glicko-2 rating for a single player."""

    def __init__(self, mu=1500.0, phi=350.0, sigma=0.06):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def to_dict(self):
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    @staticmethod
    def from_dict(d):
        return Glicko2Rating(d["mu"], d["phi"], d["sigma"])


class Glicko2System:
    """Glicko-2 rating system with deflation protection."""

    TAU = 0.5

    def __init__(self, path=None):
        self.ratings = {}
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    def ensure(self, name, mu=1500.0, phi=350.0):
        if name not in self.ratings:
            self.ratings[name] = Glicko2Rating(mu, phi)
        return self.ratings[name]

    def _g(self, phi):
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / (math.pi**2))

    def _E(self, mu, muj, phij):
        return 1.0 / (1.0 + math.exp(-self._g(phij) * (mu - muj)))

    def update(self, name, opponents_results):
        r = self.ensure(name)
        if not opponents_results:
            r.phi = min(350.0, math.sqrt(r.phi**2 + r.sigma**2))
            return

        mu = (r.mu - 1500.0) / 173.7178
        phi = r.phi / 173.7178
        v_inv = 0.0
        delta_sum = 0.0
        for opp_name, score in opponents_results:
            opp = self.ensure(opp_name)
            muj = (opp.mu - 1500.0) / 173.7178
            phij = opp.phi / 173.7178
            g_val = self._g(phij)
            e_val = self._E(mu, muj, phij)
            v_inv += g_val**2 * e_val * (1 - e_val)
            delta_sum += g_val * (score - e_val)

        if v_inv < 1e-12:
            return
        v = 1.0 / v_inv
        delta = v * delta_sum
        a = math.log(r.sigma**2)
        tau2 = self.TAU**2
        phi2 = phi**2

        def f(x):
            ex = math.exp(x)
            d2 = delta**2
            num1 = ex * (d2 - phi2 - v - ex)
            den1 = 2.0 * (phi2 + v + ex) ** 2
            return num1 / den1 - (x - a) / tau2

        A = a
        if delta**2 > phi2 + v:
            B = math.log(delta**2 - phi2 - v)
        else:
            k = 1
            while f(a - k * self.TAU) < 0:
                k += 1
                if k > 100:
                    break
            B = a - k * self.TAU

        for _ in range(50):
            C = (A + B) / 2.0
            if abs(B - A) < 1e-6:
                break
            if f(C) * f(A) < 0:
                B = C
            else:
                A = C

        sigma_new = math.exp(C / 2.0)
        phi_star = math.sqrt(phi2 + sigma_new**2)
        phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        mu_new = mu + phi_new**2 * delta_sum

        r.mu = mu_new * 173.7178 + 1500.0
        r.phi = phi_new * 173.7178
        r.sigma = sigma_new

    def deflation_adjust(self, anchor_name="random_rollout", anchor_target=1000.0):
        if anchor_name not in self.ratings:
            return
        drift = self.ratings[anchor_name].mu - anchor_target
        if abs(drift) > 1.0:
            for r in self.ratings.values():
                r.mu -= drift

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1].mu)

    def save(self, path=None):
        p = path or self.path
        if p:
            with open(p, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.ratings.items()}, f, indent=2
                )

    def load(self, path=None):
        p = path or self.path
        if p and os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            self.ratings = {k: Glicko2Rating.from_dict(v) for k, v in data.items()}


class RandomRolloutAgent:
    """Anchor agent: plays random legal moves."""

    def choose_move(self, board, player, board_size):
        del player
        n2 = board_size**2
        legal = [i for i in range(n2) if board[i] == 0]
        return random.choice(legal) if legal else -1


def arena_3agent(
    model_current_path,
    model_best_path,
    cfg,
    device,
    games_per_pair=20,
    rust_binary="./target/release/mcts_demo",
    use_rust_nn=False,
    rating_path=None,
):
    del rust_binary, use_rust_nn
    alphazero_net_cls = runtime_support.AlphaZeroNet
    load_torch_state_dict = runtime_support.load_torch_state_dict
    torch_module = runtime_support.torch
    tree_mcts_cls = TreeMCTS
    encode_board = runtime_support.encode_board

    glicko = Glicko2System(rating_path)
    glicko.ensure("random_rollout", mu=1000.0, phi=100.0)
    glicko.ensure("current", mu=1500.0, phi=200.0)
    glicko.ensure("best", mu=1500.0, phi=200.0)

    board_size = cfg["board"]
    n2 = board_size**2
    win_len = cfg["win"]
    rand_agent = RandomRolloutAgent()

    def play_game(agent_a_fn, agent_b_fn, swap=False):
        board = np.zeros(n2, dtype=np.int8)
        player = 1
        for _move_n in range(n2):
            fn = agent_a_fn if (player == 1) != swap else agent_b_fn
            move = fn(board, player)
            if move < 0 or move >= n2 or board[move] != 0:
                return (0.0, 1.0) if ((player == 1) != swap) else (1.0, 0.0)
            board[move] = player
            if win_len > 0:
                r0, c0 = divmod(move, board_size)
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cnt = 1
                    for sign in [1, -1]:
                        nr, nc = r0 + sign * dr, c0 + sign * dc
                        while (
                            0 <= nr < board_size
                            and 0 <= nc < board_size
                            and board[nr * board_size + nc] == player
                        ):
                            cnt += 1
                            nr += sign * dr
                            nc += sign * dc
                    if cnt >= win_len:
                        if (player == 1) != swap:
                            return (1.0, 0.0)
                        return (0.0, 1.0)
            if not [i for i in range(n2) if board[i] == 0]:
                return (0.5, 0.5)
            player = -player
        return (0.5, 0.5)

    model_curr = alphazero_net_cls(cfg).to(device)
    model_curr.load_state_dict(
        load_torch_state_dict(model_current_path, torch_module, map_location=device)
    )
    model_curr.eval()

    if model_best_path and os.path.exists(model_best_path):
        model_best = alphazero_net_cls(cfg).to(device)
        model_best.load_state_dict(
            load_torch_state_dict(model_best_path, torch_module, map_location=device)
        )
        model_best.eval()
    else:
        model_best = model_curr

    mcts_curr = tree_mcts_cls(cfg, model_curr, device)
    mcts_best = tree_mcts_cls(cfg, model_best, device)
    iters = cfg["iters"] // 4

    def nn_move(mcts, board, player):
        enc = encode_board(
            cfg,
            np.array(board, dtype=np.int8)
            if not isinstance(board, np.ndarray)
            else board,
            player,
        )
        legal_mask = np.array(
            [1.0 if board[i] == 0 else 0.0 for i in range(min(n2, cfg["actions"]))]
        )
        if cfg["actions"] > n2:
            legal_mask = np.concatenate([legal_mask, np.zeros(cfg["actions"] - n2)])
        pol = mcts.search(enc, player, legal_mask, iters)
        legal = [i for i in range(n2) if board[i] == 0]
        if not legal:
            return -1
        return max(legal, key=lambda a: pol[a] if a < len(pol) else 0)

    curr_fn = lambda b, p: nn_move(mcts_curr, b, p)
    best_fn = lambda b, p: nn_move(mcts_best, b, p)
    rand_fn = lambda b, p: rand_agent.choose_move(b, p, board_size)

    pairs = [
        ("current", "random_rollout", curr_fn, rand_fn),
        ("best", "random_rollout", best_fn, rand_fn),
        ("current", "best", curr_fn, best_fn),
    ]
    results = {name: [] for name in ["current", "best", "random_rollout"]}

    print("  3-Agent Round-Robin Arena:")
    for name_a, name_b, fn_a, fn_b in pairs:
        wa, wb, d = 0, 0, 0
        for gi in range(games_per_pair):
            swap = gi % 2 == 1
            sa, sb = play_game(fn_a, fn_b, swap)
            if sa > sb:
                wa += 1
            elif sb > sa:
                wb += 1
            else:
                d += 1
            results[name_a].append((name_b, sa))
            results[name_b].append((name_a, sb))
        print(f"    {name_a} vs {name_b}: {wa}-{wb}-{d}")

    for name, opp_results in results.items():
        glicko.update(name, opp_results)

    glicko.deflation_adjust("random_rollout", 1000.0)
    glicko.save()

    print("  Ratings (Glicko-2, deflation-adjusted):")
    for name, r in glicko.leaderboard():
        print(f"    {name:20s}  {r.mu:7.1f} ± {r.phi:.1f}")

    curr_r = glicko.ratings.get("current", Glicko2Rating())
    best_r = glicko.ratings.get("best", Glicko2Rating())
    promoted = curr_r.mu > best_r.mu + 30
    if promoted:
        print(f"  → PROMOTED: current ({curr_r.mu:.0f}) > best ({best_r.mu:.0f})")
    else:
        print(f"  → NOT promoted: current ({curr_r.mu:.0f}) vs best ({best_r.mu:.0f})")

    return {k: v.to_dict() for k, v in glicko.ratings.items()}, promoted


class TreeMCTSEngine:
    """Wraps TreeMCTS to conform to calibration/eval engine protocol."""

    def __init__(self, engine_name, cfg, model, device, tree_mcts_cls=None):
        self._name = engine_name
        tree_cls = tree_mcts_cls or TreeMCTS
        self._mcts = tree_cls(cfg, model, device)
        self._cfg = cfg
        self._eval_iters = self._cfg["iters"] // 4

    def select_move(self, state):
        board_enc = state._encode()
        raw_player = state.current_player()
        player = 1 if raw_player == 0 else -1
        legal_mask = np.zeros(self._cfg["actions"], dtype=np.float32)
        for action in state.legal_moves():
            if 0 <= action < self._cfg["actions"]:
                legal_mask[action] = 1.0
        policy = self._mcts.search(board_enc, player, legal_mask, self._eval_iters)
        legal = state.legal_moves()
        if legal:
            chosen = max(legal, key=lambda a: policy[a] if a < len(policy) else 0)
        else:
            chosen = 0
        return chosen, {"time_used_ms": 0, "simulations": self._eval_iters}

    def reset(self):
        return None

    def name(self):
        return self._name
