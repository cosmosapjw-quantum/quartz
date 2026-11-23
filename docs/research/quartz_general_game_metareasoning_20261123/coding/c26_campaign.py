#!/usr/bin/env python3
"""Frozen C26 actual-game campaign and independently implemented Python oracle.

The native decision process receives only legal histories, public seeds, budget,
and the TRAIN-fitted model. Exact evaluation labels live in a separate artifact.
Python memoization accelerates offline recurrence evaluation: returned node counts
sum every full-tree occurrence, including duplicate boards and terminal entries.
The native solver itself performs every charged traversal without memoization.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import select
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
DOC = HERE.parent
ROOT = HERE.parents[3]
BINARY = ROOT / "target/release/quartz_c26_game_metareasoning"
DATA = HERE / "data"
RESULTS = HERE / "results"
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
POLICY_XOR = 0xD1B54A32D192ED03
LAMBDA = 2**-10
TAU = 2**-5
WALL_PRICE = 1e-6
POLICIES = (
    "INITIAL_STOP",
    "ALL_ACTIONS",
    "RANDOM_ORDER",
    "GREEDY_VOC",
    "PANDORA",
    "GIBBS_VOC",
)
LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
CAPS = (1, 2, 9)
SEEDS = (0, 1, 2, 3)
TIMING_FIELDS = ("elapsed_us", "cue_us", "initial_us", "query_us", "controller_us")
LOGICAL_FIELDS = (
    "initial_move",
    "initial_solve_move",
    "initial_u",
    "selected_move",
    "final_u",
    "cue_nodes",
    "initial_nodes",
    "additional_nodes",
    "charged_nodes",
    "additional_queries",
    "query_moves",
    "predicted_additional_nodes",
    "stop_reason",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def json_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json_lines(path, values):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json_text(value) + "\n")


def provenance():
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "utc": utc(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "git_head": git.stdout.strip(),
        "git_status": status.stdout.splitlines(),
        "harness_sha256": file_sha(__file__),
        "source_spec_sha256": file_sha(HERE / "SOURCE_SPECS.json"),
        "contract_sha256": file_sha(HERE / "CODE_CONTRACT.md"),
    }


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def terminal(board):
    return winner(board) != 0 or 0 not in board


def current_player(board):
    return 1 if sum(x != 0 for x in board) % 2 == 0 else -1


def legal_moves(board):
    return [] if terminal(board) else [i for i, x in enumerate(board) if x == 0]


def apply(board, move):
    if type(move) is not int or move not in legal_moves(board):
        raise ValueError("illegal move or terminal continuation")
    return board[:move] + (current_player(board),) + board[move + 1 :]


def replay(history, require_nonterminal=False):
    if not isinstance(history, (list, tuple)) or len(history) > 9:
        raise ValueError("history must be a legal move list of length <=9")
    board = (0,) * 9
    for move in history:
        board = apply(board, move)
    if require_nonterminal and terminal(board):
        raise ValueError("terminal root")
    return board


@lru_cache(maxsize=None)
def oracle(board, root_player):
    """Absolute winner signs and explicit max/min; logical uncached tree count."""
    if root_player not in (-1, 1):
        raise ValueError("root_player must be +1 or -1")
    if terminal(board):
        return winner(board) * root_player, 1
    children = [oracle(apply(board, move), root_player) for move in legal_moves(board)]
    values = [value for value, _ in children]
    value = max(values) if current_player(board) == root_player else min(values)
    return value, 1 + sum(nodes for _, nodes in children)


def enumerate_reachable():
    histories = {}

    def visit(board, history):
        if board in histories:
            return
        histories[board] = history
        for move in legal_moves(board):
            visit(apply(board, move), history + (move,))

    visit((0,) * 9, ())
    return histories


def d4_maps():
    maps = []
    for reflected in (False, True):
        for rotation in range(4):
            mapping = []
            for i in range(9):
                r, c = divmod(i, 3)
                if reflected:
                    c = 2 - c
                for _ in range(rotation):
                    r, c = c, 2 - r
                mapping.append(3 * r + c)
            maps.append(tuple(mapping))
    return maps


D4 = d4_maps()


def canonicalize(board, history):
    versions = []
    for mapping in D4:
        transformed = [0] * 9
        for i, value in enumerate(board):
            transformed[mapping[i]] = value
        versions.append((tuple(transformed), tuple(mapping[x] for x in history)))
    return min(versions)


def board_text(board):
    return ",".join(map(str, board))


def truth_record(source):
    board = replay(source["history"], require_nonterminal=True)
    root_player = current_player(board)
    children = []
    for move in legal_moves(board):
        child = apply(board, move)
        value, nodes = oracle(child, root_player)
        children.append(
            {
                "move": move,
                "u": (value + 1) / 2,
                "solve_nodes": nodes,
                "terminal_known": terminal(child),
                "remaining": len(legal_moves(child)),
            }
        )
    value, nodes = oracle(board, root_player)
    return {
        "id": source["id"],
        "board": list(board),
        "root_player": root_player,
        "root_u": (value + 1) / 2,
        "root_tree_nodes": nodes,
        "candidates": children,
    }


def build_sources():
    reachable = enumerate_reachable()
    if len(reachable) != 5478:
        raise AssertionError("reachable population changed")
    canonical = {ply: {} for ply in (4, 5, 6)}
    for board, history in reachable.items():
        if len(history) in canonical and not terminal(board):
            cboard, chistory = canonicalize(board, history)
            old = canonical[len(history)].get(cboard)
            if old is None or chistory < old:
                canonical[len(history)][cboard] = chistory
    population = {str(ply): len(group) for ply, group in canonical.items()}
    if population != {"4": 108, "5": 153, "6": 183}:
        raise AssertionError("canonical population changed: " + str(population))
    public, offline = [], []
    for ply, group in sorted(canonical.items()):
        ordered = sorted(
            (sha_bytes(("C26-source-v1|" + board_text(board)).encode()), board, history)
            for board, history in group.items()
        )
        for rank, (source_hash, board, history) in enumerate(ordered[:100]):
            split = "TRAIN" if rank < 30 else "CAL" if rank < 50 else "TEST"
            seed_hash = sha_bytes(("C26-cue-v1|" + board_text(board)).encode())
            source = {
                "id": f"C26-p{ply}-{source_hash[:16]}",
                "ply": ply,
                "source_hash": source_hash,
                "split": split,
                "history": list(history),
                "cue_seed": int(seed_hash[:16], 16),
            }
            if replay(history) != board:
                raise AssertionError(
                    "transformed history does not produce canonical board"
                )
            public.append(source)
            offline.append(truth_record(source))
    return (
        public,
        offline,
        {
            "reachable_states": len(reachable),
            "canonical_nonterminal_population": population,
            "nonterminal_reachable_states": sum(not terminal(b) for b in reachable),
        },
    )


class SplitMix64:
    def __init__(self, seed):
        self.state = int(seed) & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def index(self, n):
        if n <= 0:
            raise ValueError("uniform index needs n>0")
        threshold = (1 << 64) % n
        while True:
            value = self.next_u64()
            if value >= threshold:
                return value % n

    def uniform(self):
        return (self.next_u64() >> 11) * (2**-53)


def cue_candidates(board, cue_seed):
    root_player = current_player(board)
    candidates = []
    for move in legal_moves(board):
        child = apply(board, move)
        counts = [0, 0, 0]
        cue_nodes = 0
        if terminal(child):
            counts[winner(child) * root_player + 1] = 4
            cue_nodes = 1
        else:
            rng = SplitMix64(cue_seed ^ (((move + 1) * GOLDEN) & MASK64))
            for _ in range(4):
                state = child
                cue_nodes += 1
                while not terminal(state):
                    moves = legal_moves(state)
                    state = apply(state, moves[rng.index(len(moves))])
                    cue_nodes += 1
                counts[winner(state) * root_player + 1] += 1
        value, nodes = oracle(child, root_player)
        candidates.append(
            {
                "move": move,
                "terminal_known": terminal(child),
                "remaining": len(legal_moves(child)),
                "counts": counts,
                "rollout_mean": (0.5 * counts[1] + counts[2]) / 4,
                "cue_nodes": cue_nodes,
                "u": (value + 1) / 2,
                "solve_nodes": nodes,
            }
        )
    return candidates


def feature_key(remaining, counts):
    return str(remaining) + ":" + ":".join(map(str, counts))


def fit_model(rows):
    features, remaining = {}, {}
    total = {"counts": [0, 0, 0], "n": 0, "cost_sum": 0}
    source_ids = []

    def add(record, label, nodes):
        record["counts"][label] += 1
        record["n"] += 1
        record["cost_sum"] += nodes

    for row in rows:
        source_ids.append({"id": row["id"], "source_hash": row.get("source_hash")})
        for candidate in row["candidates"]:
            if candidate["terminal_known"]:
                continue
            key = feature_key(candidate["remaining"], candidate["counts"])
            rem = str(candidate["remaining"])
            label = {0.0: 0, 0.5: 1, 1.0: 2}[candidate["u"]]
            nodes = candidate["solve_nodes"]
            for table, k in ((features, key), (remaining, rem)):
                if k not in table:
                    table[k] = {"counts": [0, 0, 0], "n": 0, "cost_sum": 0}
                add(table[k], label, nodes)
            add(total, label, nodes)
    if not total["n"]:
        raise ValueError("no TRAIN nonterminal candidates")
    smooth = lambda record: [(n + 1) / (record["n"] + 3) for n in record["counts"]]
    model = {
        "schema": "c26-categorical-v1",
        "feature_probs": {k: smooth(v) for k, v in sorted(features.items())},
        "remaining_probs": {k: smooth(v) for k, v in sorted(remaining.items())},
        "global_probs": smooth(total),
        "remaining_costs": {
            k: v["cost_sum"] / v["n"] for k, v in sorted(remaining.items())
        },
        "global_cost": total["cost_sum"] / total["n"],
        "lambda_nodes": LAMBDA,
        "tau": TAU,
    }
    return model, {
        "features": features,
        "remaining": remaining,
        "global": total,
        "sources": source_ids,
        "nonterminal_candidates": total["n"],
        "smoothing": "add one in each of three utility classes",
    }


def model_prediction(model, remaining, counts):
    key = feature_key(remaining, counts)
    probabilities = model["feature_probs"].get(
        key, model["remaining_probs"].get(str(remaining), model["global_probs"])
    )
    normalizer = sum(probabilities)
    probabilities = [p / normalizer for p in probabilities]
    cost = model["remaining_costs"].get(str(remaining), model["global_cost"])
    return probabilities, cost


def expected_gain(probabilities, b):
    return probabilities[1] * max(0.5 - b, 0.0) + probabilities[2] * max(1.0 - b, 0.0)


def reservation_index(probabilities, charge):
    _, ph, p1 = probabilities
    mu = ph / 2 + p1
    if p1 > 0 and charge <= p1 / 2:
        return 1 - charge / p1
    if ph + p1 > 0 and charge <= mu:
        return (mu - charge) / (ph + p1)
    return mu - charge


def choice_probabilities(scores, tau):
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("positive finite temperature required")
    maximum = max(scores)
    weights = [math.exp((score - maximum) / tau) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def sample_probability(probabilities, draw):
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if draw < cumulative:
            return i
    return len(probabilities) - 1


def public_request(
    source,
    *,
    mode="run",
    cap=9,
    policy_seed=0,
    trace=False,
    repeats=None,
    warmup=None,
    model=None,
):
    survey = mode == "survey"
    return {
        "id": source["id"],
        "history": list(source["history"]),
        "cue_seed": source["cue_seed"],
        "policy_seed": 0 if survey else policy_seed,
        "mode": mode,
        "additional": 9 if survey else cap,
        "repeats": 1
        if survey
        else (repeats if repeats is not None else (1 if trace else 9)),
        "warmup": 0
        if survey
        else (warmup if warmup is not None else (0 if trace else 1)),
        "trace": False if survey else trace,
        "model": None if survey else model,
    }


def expected_run(history, cue_seed, policy_seed, cap, policy, model):
    """Independent decision replay from offline truth; never sent to the native API."""
    board = replay(history, require_nonterminal=True)
    candidates = cue_candidates(board, cue_seed)
    by_move = {c["move"]: c for c in candidates}
    known = {c["move"]: c["u"] for c in candidates if c["terminal_known"]}
    nonterminal = [c for c in candidates if not c["terminal_known"]]
    initial_solve_move = None
    initial_nodes = 0
    if not any(value == 1 for value in known.values()) and nonterminal:
        initial = max(nonterminal, key=lambda c: (c["rollout_mean"], -c["move"]))
        initial_solve_move = initial["move"]
        initial_nodes = initial["solve_nodes"]
        known[initial_solve_move] = initial["u"]
    incumbent = max(known, key=lambda move: (known[move], -move))
    initial_move = incumbent
    initial_u = known[incumbent]
    b = initial_u
    remaining = [c["move"] for c in nonterminal if c["move"] not in known]
    predictions = []
    for candidate in candidates:
        is_terminal = candidate["terminal_known"]
        probs, cost = (
            (None, None)
            if is_terminal
            else model_prediction(model, candidate["remaining"], candidate["counts"])
        )
        predictions.append(
            {
                "move": candidate["move"],
                "remaining": candidate["remaining"],
                "counts": candidate["counts"],
                "terminal_known": is_terminal,
                "probabilities": probs,
                "predicted_nodes": cost,
                "observed_u": candidate["u"] if is_terminal else None,
            }
        )
    prediction_by_move = {p["move"]: p for p in predictions}
    rng = SplitMix64(cue_seed ^ POLICY_XOR ^ policy_seed)
    decisions, query_moves = [], []
    additional_nodes = 0
    predicted_nodes = 0.0
    while True:
        scored = []
        for move in remaining:
            prediction = prediction_by_move[move]
            probs, cost = prediction["probabilities"], prediction["predicted_nodes"]
            gain = expected_gain(probs, b)
            scored.append(
                {
                    "move": move,
                    "g": gain,
                    "q": gain - LAMBDA * cost,
                    "z": reservation_index(probs, LAMBDA * cost),
                    "probabilities": probs,
                    "predicted_nodes": cost,
                    "probability": 0.0,
                }
            )
        chosen, draw, stop_probability = None, None, 1.0
        if policy == "INITIAL_STOP":
            reason = "initial_stop"
        elif len(query_moves) >= cap:
            reason = "cap_reached"
        elif not remaining:
            reason = "exhausted"
        elif b == 1 and policy != "ALL_ACTIONS":
            reason = "known_win"
        elif policy == "ALL_ACTIONS":
            chosen, reason = remaining[0], "all_actions"
        elif policy == "RANDOM_ORDER":
            chosen, reason = remaining[rng.index(len(remaining))], "random_order"
            for entry in scored:
                entry["probability"] = 1 / len(remaining)
        elif policy == "GREEDY_VOC":
            best = max(scored, key=lambda c: (c["q"], -c["move"]))
            if best["q"] > 0:
                chosen, reason = best["move"], "positive_voc"
            else:
                reason = "nonpositive_voc"
        elif policy == "PANDORA":
            best = max(scored, key=lambda c: (c["z"], -c["move"]))
            if best["z"] > b:
                chosen, reason = best["move"], "reservation_above_incumbent"
            else:
                reason = "reservation_below_incumbent"
        elif policy == "GIBBS_VOC":
            probabilities = choice_probabilities([0.0] + [c["q"] for c in scored], TAU)
            stop_probability = probabilities[0]
            for entry, probability in zip(scored, probabilities[1:]):
                entry["probability"] = probability
            draw = rng.uniform()
            selected = sample_probability(probabilities, draw)
            chosen = None if selected == 0 else remaining[selected - 1]
            reason = "sampled_stop" if chosen is None else "gibbs_sample"
        else:
            raise ValueError("unknown policy")
        if chosen is not None and policy != "GIBBS_VOC":
            stop_probability = 0.0
            if policy != "RANDOM_ORDER":
                for entry in scored:
                    entry["probability"] = float(entry["move"] == chosen)
        before = b
        queried_u, queried_nodes = None, 0
        if chosen is not None:
            child = by_move[chosen]
            queried_u, queried_nodes = child["u"], child["solve_nodes"]
            if queried_u > b:
                b, incumbent = queried_u, chosen
            known[chosen] = queried_u
            query_moves.append(chosen)
            additional_nodes += queried_nodes
            predicted_nodes += prediction_by_move[chosen]["predicted_nodes"]
        decisions.append(
            {
                "b_before": before,
                "remaining": list(remaining),
                "candidates": scored,
                "stop_probability": stop_probability,
                "chosen_move": chosen,
                "choice": "STOP" if chosen is None else "FOCUS",
                "uniform_draw": draw,
                "queried_u": queried_u,
                "queried_nodes": queried_nodes,
                "b_after": b,
                "reason": reason,
            }
        )
        if chosen is None:
            break
        remaining.remove(chosen)
    cue_nodes = sum(c["cue_nodes"] for c in candidates)
    return {
        "policy": policy,
        "initial_move": initial_move,
        "initial_u": initial_u,
        "initial_solve_move": initial_solve_move,
        "selected_move": incumbent,
        "final_u": b,
        "cue_nodes": cue_nodes,
        "initial_nodes": initial_nodes,
        "additional_nodes": additional_nodes,
        "charged_nodes": cue_nodes + initial_nodes + additional_nodes,
        "additional_queries": len(query_moves),
        "query_moves": query_moves,
        "predicted_additional_nodes": predicted_nodes,
        "stop_reason": reason,
        "predictions": predictions,
        "decisions": decisions,
    }


def assert_same(actual, expected, context=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{context}: expected an object")
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"{context}: missing {key}")
            assert_same(actual[key], value, context + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{context}: array length/type differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_same(a, b, f"{context}[{i}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isfinite(actual):
            raise AssertionError(f"{context}: nonfinite/non-numeric {actual!r}")
        if not math.isclose(actual, expected, abs_tol=2e-11, rel_tol=2e-12):
            raise AssertionError(f"{context}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def validate_survey(source, response):
    assert_same(
        response, {"schema": "c26-game-v1", "mode": "survey", "id": source["id"]}
    )
    expected = cue_candidates(
        replay(source["history"], require_nonterminal=True), source["cue_seed"]
    )
    assert_same(response["candidates"], expected, source["id"] + ".survey")
    assert_same(
        response["cue_nodes"], sum(c["cue_nodes"] for c in expected), "survey.cue_nodes"
    )


def validate_response(source, request, response, model, require_trace=False):
    assert_same(response, {"schema": "c26-game-v1", "mode": "run", "id": source["id"]})
    runs = response["runs"]
    if len(runs) != len(POLICIES) * request["repeats"]:
        raise AssertionError("wrong number of native runs")
    keys = {(run["policy"], run["repeat"]) for run in runs}
    expected_keys = set(itertools.product(POLICIES, range(request["repeats"])))
    if keys != expected_keys or len(keys) != len(runs):
        raise AssertionError("duplicate/missing policy-repetition cell")
    expected = {
        policy: expected_run(
            source["history"],
            source["cue_seed"],
            request["policy_seed"],
            request["additional"],
            policy,
            model,
        )
        for policy in POLICIES
    }
    for run in runs:
        truth = expected[run["policy"]]
        identity = f"{source['id']}/{request['additional']}/{request['policy_seed']}/{run['policy']}"
        assert_same(run, {field: truth[field] for field in LOGICAL_FIELDS}, identity)
        if run["selected_move"] not in legal_moves(replay(source["history"])):
            raise AssertionError(identity + ": illegal deployment")
        if run["final_u"] < run["initial_u"]:
            raise AssertionError(identity + ": known-incumbent safety violation")
        for field in TIMING_FIELDS:
            if (
                not isinstance(run[field], (int, float))
                or not math.isfinite(run[field])
                or run[field] < 0
            ):
                raise AssertionError(identity + ": invalid timing " + field)
        component_sum = sum(run[field] for field in TIMING_FIELDS[1:])
        if not math.isclose(
            component_sum, run["elapsed_us"], rel_tol=2e-9, abs_tol=2e-6
        ):
            raise AssertionError(identity + ": timing components do not sum to elapsed")
        if require_trace:
            assert_same(
                run["predictions"], truth["predictions"], identity + ".predictions"
            )
            assert_same(run["decisions"], truth["decisions"], identity + ".decisions")
            for prediction in run["predictions"]:
                if "u" in prediction or "solve_nodes" in prediction:
                    raise AssertionError(
                        identity + ": oracle leaked into run predictions"
                    )
                if (
                    not prediction["terminal_known"]
                    and prediction.get("observed_u") is not None
                ):
                    raise AssertionError(
                        identity + ": unrevealed prediction contains oracle truth"
                    )
    return {
        policy: {field: expected[policy][field] for field in LOGICAL_FIELDS}
        for policy in POLICIES
    }


class NativeClient:
    """One persistent native JSON-lines process, with per-roundtrip transport timing."""

    def __init__(self, binary):
        self.binary = Path(binary).resolve()
        self.binary_sha256 = file_sha(self.binary)
        self.stderr_file = tempfile.TemporaryFile(mode="w+t")
        started = time.perf_counter_ns()
        self.process = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
        )
        self.launch_us = (time.perf_counter_ns() - started) / 1000
        self.closed = False
        self.requests = 0
        self.roundtrip_us = 0.0

    def send_raw(self, line):
        if self.closed:
            raise RuntimeError("native process already closed")
        started = time.perf_counter_ns()
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 60.0)
        if not ready:
            raise TimeoutError("native JSON-lines response exceeded 60 seconds")
        text = self.process.stdout.readline()
        elapsed = (time.perf_counter_ns() - started) / 1000
        if not text:
            raise RuntimeError(
                f"native process ended early, status={self.process.poll()}"
            )
        response = json.loads(text)
        if not isinstance(response, dict):
            raise AssertionError("native response must be a JSON object")
        self.requests += 1
        self.roundtrip_us += elapsed
        return response, {
            "roundtrip_us": elapsed,
            "request_bytes": len(line.encode()) + 1,
            "response_bytes": len(text.encode()),
        }

    def request(self, request):
        response, metadata = self.send_raw(json_text(request))
        if "error" in response:
            raise RuntimeError(
                "native rejected valid request: " + str(response["error"])
            )
        return response, metadata

    def close(self):
        if self.closed:
            return
        self.process.stdin.close()
        try:
            status = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            status = self.process.wait()
            raise RuntimeError("native process did not finish after stdin EOF")
        finally:
            self.closed = True
            self.process.stdout.close()
            self.stderr_file.seek(0)
            self.stderr = self.stderr_file.read()
            self.stderr_file.close()
        self.exit_code = status
        if status != 0:
            raise RuntimeError(f"native exited {status}: {self.stderr[-2000:]}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def audit_all_reachable(client):
    """Exhaustive nonterminal-root cross-language check, separate from cohorts."""
    started = time.perf_counter()
    roots, children = 0, 0
    signs, players, by_ply = set(), set(), Counter()
    digest = hashlib.sha256()
    max_nodes = 0
    for board, history in sorted(
        enumerate_reachable().items(), key=lambda x: (len(x[1]), x[0])
    ):
        if terminal(board):
            continue
        identity = sha_bytes(board_text(board).encode())
        source = {
            "id": "verify-" + identity[:16],
            "history": list(history),
            "cue_seed": int(identity[:16], 16),
        }
        response, _ = client.request(public_request(source, mode="survey"))
        validate_survey(source, response)
        value, _ = oracle(board, current_player(board))
        signs.add((value + 1) / 2)
        players.add(current_player(board))
        roots += 1
        children += len(response["candidates"])
        by_ply[len(history)] += 1
        max_nodes = max(max_nodes, *(c["solve_nodes"] for c in response["candidates"]))
        # Exclude volatile wall measurement from the reproducible logical digest.
        digest.update(
            (
                json_text(
                    {
                        "id": response["id"],
                        "candidates": response["candidates"],
                        "cue_nodes": response["cue_nodes"],
                    }
                )
                + "\n"
            ).encode()
        )
    return {
        "schema": "c26-all-reachable-audit-v1",
        "utc": utc(),
        "roots": roots,
        "candidates": children,
        "by_ply": dict(sorted(by_ply.items())),
        "root_players": sorted(players),
        "root_utilities": sorted(signs),
        "maximum_child_solve_nodes": max_nodes,
        "logical_response_sha256": digest.hexdigest(),
        "binary_sha256": client.binary_sha256,
        "harness_sha256": file_sha(__file__),
        "elapsed_seconds": time.perf_counter() - started,
        "assertion_tested": "Every reachable nonterminal root: every child root-frame minimax utility, full-tree node multiplicity, SplitMix64 cue counts, cue cost and terminal semantics agree.",
        "scope": "Finite TicTacToe state graph, 4520 nonterminal roots; not a policy performance cohort.",
    }


def paired_bootstrap(values_by_ply, draws=2000, seed=2600923):
    """Fixed TRAIN; resample source parents within ply, never timing/seed rows."""
    import numpy as np

    groups = [
        np.asarray(values_by_ply[ply], dtype=float) for ply in sorted(values_by_ply)
    ]
    if not groups or any(
        len(values) == 0 or not np.isfinite(values).all() for values in groups
    ):
        raise ValueError("bootstrap needs finite, nonempty ply strata")
    rng = np.random.Generator(np.random.PCG64(seed))
    total = sum(len(values) for values in groups)
    boot = np.zeros(draws, dtype=float)
    for values in groups:
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        boot += values[indices].sum(axis=1) / total
    return {
        "mean": float(sum(values.sum() for values in groups) / total),
        "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975], method="linear")],
        "parents": total,
        "strata": {str(ply): len(values_by_ply[ply]) for ply in sorted(values_by_ply)},
        "draws": draws,
        "seed": seed,
        "rng": "numpy.PCG64",
        "numpy_version": np.__version__,
        "interval": "stratified paired source-parent percentile bootstrap",
    }


def summarize_seed_runs(flat_runs):
    groups = defaultdict(list)
    for run in flat_runs:
        groups[(run["id"], run["ply"], run["cap"], run["seed"], run["policy"])].append(
            run
        )
    summaries = []
    for (identity, ply, cap, seed, policy), runs in sorted(groups.items()):
        if len(runs) != 9 or {r["repeat"] for r in runs} != set(range(9)):
            raise AssertionError(
                "nine recorded repetitions required in every seed cell"
            )
        reference = runs[0]
        for run in runs[1:]:
            assert_same(
                run,
                {field: reference[field] for field in LOGICAL_FIELDS},
                "timing determinism",
            )
        metrics = {
            "utility": reference["final_u"],
            "initial_utility": reference["initial_u"],
            "utility_gain": reference["final_u"] - reference["initial_u"],
            "regret": reference["oracle_u"] - reference["final_u"],
            "charged_nodes": reference["charged_nodes"],
            "cue_nodes": reference["cue_nodes"],
            "initial_nodes": reference["initial_nodes"],
            "additional_nodes": reference["additional_nodes"],
            "additional_queries": reference["additional_queries"],
            "predicted_additional_nodes": reference["predicted_additional_nodes"],
        }
        for field in TIMING_FIELDS:
            metrics[field] = statistics.median(r[field] for r in runs)
        metrics["net_nodes"] = metrics["utility"] - LAMBDA * metrics["charged_nodes"]
        metrics["net_wall"] = metrics["utility"] - WALL_PRICE * metrics["elapsed_us"]
        summaries.append(
            {
                "id": identity,
                "ply": ply,
                "cap": cap,
                "seed": seed,
                "policy": policy,
                "metrics": metrics,
            }
        )
    return summaries


def summarize_parents(seed_rows):
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["id"], row["ply"], row["cap"], row["policy"])].append(row)
    result = []
    for (identity, ply, cap, policy), rows in sorted(groups.items()):
        if len(rows) != 4 or {r["seed"] for r in rows} != set(SEEDS):
            raise AssertionError(
                "four policy seeds required in every source-parent cell"
            )
        metrics = {
            key: statistics.mean(r["metrics"][key] for r in rows)
            for key in rows[0]["metrics"]
        }
        result.append(
            {
                "id": identity,
                "ply": ply,
                "cap": cap,
                "policy": policy,
                "metrics": metrics,
            }
        )
    return result


def comparison(parent_rows, cap, policy, baseline):
    lookup = {
        (row["id"], row["policy"]): row for row in parent_rows if row["cap"] == cap
    }
    identities = sorted({identity for identity, name in lookup if name == policy})
    metrics = (
        "utility",
        "utility_gain",
        "regret",
        "net_nodes",
        "net_wall",
        "charged_nodes",
        "additional_nodes",
        "elapsed_us",
        "additional_queries",
    )
    contrasts = {}
    for metric in metrics:
        strata = defaultdict(list)
        for identity in identities:
            a, b = lookup[(identity, policy)], lookup[(identity, baseline)]
            strata[a["ply"]].append(a["metrics"][metric] - b["metrics"][metric])
        contrasts[metric] = paired_bootstrap(strata)
    return {
        "cap": cap,
        "policy": policy,
        "baseline": baseline,
        "contrast": "policy minus baseline",
        "metrics": contrasts,
    }


def calibration(sources, model):
    rows, parent_residuals = [], defaultdict(list)
    for source in sources:
        candidates = cue_candidates(replay(source["history"]), source["cue_seed"])
        for candidate in candidates:
            if candidate["terminal_known"]:
                continue
            probs, predicted = model_prediction(
                model, candidate["remaining"], candidate["counts"]
            )
            label = int(2 * candidate["u"])
            error = [p - float(i == label) for i, p in enumerate(probs)]
            residual = candidate["u"] - (probs[1] / 2 + probs[2])
            row = {
                "id": source["id"],
                "ply": source["ply"],
                "move": candidate["move"],
                "remaining": candidate["remaining"],
                "counts": candidate["counts"],
                "probabilities": probs,
                "u": candidate["u"],
                "predicted_nodes": predicted,
                "actual_nodes": candidate["solve_nodes"],
                "cost_error": predicted - candidate["solve_nodes"],
                "brier": sum(e * e for e in error),
                "utility_residual": residual,
            }
            rows.append(row)
            parent_residuals[source["id"]].append(residual)
    if not rows:
        raise ValueError("no nonterminal calibration candidates")

    def summary(items):
        n = len(items)
        return {
            "candidates": n,
            "brier": statistics.mean(r["brier"] for r in items),
            "probability_mean": [
                statistics.mean(r["probabilities"][i] for r in items) for i in range(3)
            ],
            "observed_frequency": [
                sum(int(2 * r["u"]) == i for r in items) / n for i in range(3)
            ],
            "predicted_nodes_mean": statistics.mean(
                r["predicted_nodes"] for r in items
            ),
            "actual_nodes_mean": statistics.mean(r["actual_nodes"] for r in items),
            "cost_bias": statistics.mean(r["cost_error"] for r in items),
            "cost_mae": statistics.mean(abs(r["cost_error"]) for r in items),
            "cost_rmse": math.sqrt(
                statistics.mean(r["cost_error"] ** 2 for r in items)
            ),
            "cost_mape": statistics.mean(
                abs(r["cost_error"]) / r["actual_nodes"] for r in items
            ),
        }

    groups, by_parent, feature_groups = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for row in rows:
        groups[row["remaining"]].append(row)
        by_parent[row["id"]].append(row)
        feature_groups[feature_key(row["remaining"], row["counts"])].append(row)
    parent_brier = [
        statistics.mean(r["brier"] for r in group) for group in by_parent.values()
    ]
    # Symmetrized ordered pairs, equal weighting for each source that has >=2 candidates.
    pair_moments = []
    pair_count = 0
    for residuals in parent_residuals.values():
        if len(residuals) < 2:
            continue
        pairs = list(itertools.combinations(residuals, 2))
        pair_count += len(pairs)
        pair_moments.append(
            (
                statistics.mean((a + b) / 2 for a, b in pairs),
                statistics.mean((a * a + b * b) / 2 for a, b in pairs),
                statistics.mean(a * b for a, b in pairs),
            )
        )
    if pair_moments:
        mean = statistics.mean(x[0] for x in pair_moments)
        variance = statistics.mean(x[1] for x in pair_moments) - mean * mean
        covariance = statistics.mean(x[2] for x in pair_moments) - mean * mean
        dependence = {
            "source_parents": len(pair_moments),
            "unordered_pairs": pair_count,
            "paired_residual_mean": mean,
            "paired_residual_variance": variance,
            "within_root_covariance": covariance,
            "within_root_correlation": covariance / variance if variance > 0 else None,
        }
    else:
        dependence = {
            "source_parents": 0,
            "unordered_pairs": 0,
            "within_root_correlation": None,
        }
    dependence["interpretation"] = (
        "Descriptive symmetrized pair moments, equal parent weighting. "
        "No independence test or population-general claim; TRAIN is fixed."
    )
    report = {
        "schema": "c26-calibration-v1",
        "source_parents": len(sources),
        "nonterminal_source_parents": len(by_parent),
        "pooled_candidates": summary(rows),
        "equal_parent_brier": statistics.mean(parent_brier),
        "by_remaining": {str(k): summary(v) for k, v in sorted(groups.items())},
        "feature_bins": {k: summary(v) for k, v in sorted(feature_groups.items())},
        "within_root_residual_dependence": dependence,
        "minimum_model_probability": min(min(r["probabilities"]) for r in rows),
        "maximum_model_probability": max(max(r["probabilities"]) for r in rows),
        "predicted_nodes_range": [
            min(r["predicted_nodes"] for r in rows),
            max(r["predicted_nodes"] for r in rows),
        ],
    }
    return report, rows


def prepare(data_dir):
    data_dir = Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(
            "prepared data directory must be empty; never overwrite a frozen cohort"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    public, offline, population = build_sources()
    write_json_lines(data_dir / "public_sources.jsonl", public)
    write_json_lines(data_dir / "offline_oracle.jsonl.gz", offline)
    manifest = {
        "schema": "c26-prepared-data-v1",
        "provenance": provenance(),
        "population": population,
        "source_parents": len(public),
        "split_counts": dict(Counter(s["split"] for s in public)),
        "files": {
            name: file_sha(data_dir / name)
            for name in ("public_sources.jsonl", "offline_oracle.jsonl.gz")
        },
        "public_source_payload_sha256": sha_bytes(
            ("\n".join(json_text(s) for s in public) + "\n").encode()
        ),
        "oracle_logical_payload_sha256": sha_bytes(
            ("\n".join(json_text(s) for s in offline) + "\n").encode()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_blinding": "Requests explicitly select only id, history, cue_seed, policy controls and TRAIN model.",
    }
    write_json(data_dir / "source_manifest.json", manifest)
    return manifest


def load_inputs(data_dir):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "source_manifest.json")
    for name, expected in manifest["files"].items():
        if file_sha(data_dir / name) != expected:
            raise AssertionError("frozen source artifact hash changed: " + name)
    if manifest["provenance"]["source_spec_sha256"] != file_sha(
        HERE / "SOURCE_SPECS.json"
    ):
        raise AssertionError("source specification changed after preparation")
    if manifest["provenance"]["contract_sha256"] != file_sha(HERE / "CODE_CONTRACT.md"):
        raise AssertionError("contract changed after preparation")
    sources = list(json_lines(data_dir / "public_sources.jsonl"))
    if Counter(s["split"] for s in sources) != {"TRAIN": 90, "CAL": 60, "TEST": 150}:
        raise AssertionError("frozen split cardinalities changed")
    if len({s["id"] for s in sources}) != 300:
        raise AssertionError("duplicate source parents")
    offline = {
        row["id"]: row for row in json_lines(data_dir / "offline_oracle.jsonl.gz")
    }
    if set(offline) != {s["id"] for s in sources}:
        raise AssertionError("oracle and public source identities differ")
    return sources, offline, manifest


def audit_frozen_model(data_dir):
    data_dir = Path(data_dir)
    model = read_json(data_dir / "model.json")
    record = read_json(data_dir / "model_provenance.json")
    if file_sha(data_dir / "model.json") != record["model_sha256"]:
        raise AssertionError("frozen TRAIN model has changed")
    if file_sha(data_dir / "training_raw.jsonl.gz") != record["training_raw_sha256"]:
        raise AssertionError("raw TRAIN measurements have changed")
    raw = list(json_lines(data_dir / "training_raw.jsonl.gz"))
    rows = [
        {
            "id": r["id"],
            "source_hash": r["source_hash"],
            "candidates": r["response"]["candidates"],
        }
        for r in raw
    ]
    refit, counts = fit_model(rows)
    assert_same(model, refit, "TRAIN parameter reproduction")
    assert_same(record["raw_counts"], counts, "TRAIN sufficient statistics")
    source_ids = {s["id"] for s in load_inputs(data_dir)[0] if s["split"] == "TRAIN"}
    if set(r["id"] for r in raw) != source_ids or len(raw) != 90:
        raise AssertionError(
            "TRAIN provenance does not contain exactly the frozen90 parents"
        )
    return model, record


def train(data_dir, binary):
    data_dir = Path(data_dir)
    if (data_dir / "model.json").exists() or (
        data_dir / "training_raw.jsonl.gz"
    ).exists():
        raise FileExistsError(
            "TRAIN outputs already exist; preserve them instead of refitting in place"
        )
    sources, _, _ = load_inputs(data_dir)
    selected = [s for s in sources if s["split"] == "TRAIN"]
    started = time.perf_counter()
    records = []
    with NativeClient(binary) as client:
        with gzip.open(
            data_dir / "training_raw.jsonl.gz", "wt", encoding="utf-8"
        ) as handle:
            for source in selected:
                request = public_request(source, mode="survey")
                response, timing = client.request(request)
                validate_survey(source, response)
                row = {
                    "id": source["id"],
                    "source_hash": source["source_hash"],
                    "request": request,
                    "request_sha256": sha_bytes(json_text(request).encode()),
                    "response": response,
                    "transport": timing,
                }
                records.append(row)
                handle.write(json_text(row) + "\n")
    fitted, raw_counts = fit_model(
        [
            {
                "id": r["id"],
                "source_hash": r["source_hash"],
                "candidates": r["response"]["candidates"],
            }
            for r in records
        ]
    )
    write_json(data_dir / "model.json", fitted)
    record = {
        "schema": "c26-training-provenance-v1",
        "provenance": provenance(),
        "training_source_ids": [s["id"] for s in selected],
        "training_source_hashes": [s["source_hash"] for s in selected],
        "source_manifest_sha256": file_sha(data_dir / "source_manifest.json"),
        "training_raw_sha256": file_sha(data_dir / "training_raw.jsonl.gz"),
        "model_sha256": file_sha(data_dir / "model.json"),
        "raw_counts": raw_counts,
        "binary_sha256": client.binary_sha256,
        "native_exit_code": client.exit_code,
        "process_launch_us": client.launch_us,
        "native_roundtrip_us": client.roundtrip_us,
        "survey_measurement_us": sum(
            r["response"].get("measurement_elapsed_us", 0) for r in records
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "policy": "Frozen before CAL; CAL diagnoses only; TEST does not tune any parameter.",
    }
    write_json(data_dir / "model_provenance.json", record)
    audit_frozen_model(data_dir)
    diagnostic, rows = calibration(selected, fitted)
    write_json(data_dir / "training_calibration.json", diagnostic)
    write_json_lines(data_dir / "training_calibration_rows.jsonl.gz", rows)
    return record


def run_campaign(split, data_dir, results_dir, binary):
    if split not in ("CAL", "TEST"):
        raise ValueError("campaign split must be CAL or TEST")
    data_dir, output = Path(data_dir), Path(results_dir) / split
    sources, _, _ = load_inputs(data_dir)
    selected = [s for s in sources if s["split"] == split]
    model, model_record = audit_frozen_model(data_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "schema": "c26-campaign-v1",
        "split": split,
        "status": "running",
        "provenance": provenance(),
        "source_parents": len(selected),
        "model_sha256": model_record["model_sha256"],
        "source_manifest_sha256": file_sha(data_dir / "source_manifest.json"),
        "training_provenance_sha256": file_sha(data_dir / "model_provenance.json"),
        "caps": list(CAPS),
        "policy_seeds": list(SEEDS),
        "repeats": 9,
        "warmup": 1,
        "trace_repeats": 1,
        "source_ids": [s["id"] for s in selected],
    }
    write_json(output / "run_manifest.json", manifest)
    counts = Counter()
    native_execution_us = native_recorded_us = native_warmup_us = native_trace_us = 0.0
    roundtrip_us = 0.0
    try:
        with NativeClient(binary) as client:
            with (
                gzip.open(
                    output / "timing_raw.jsonl.gz", "wt", encoding="utf-8"
                ) as timing_file,
                gzip.open(
                    output / "trace_raw.jsonl.gz", "wt", encoding="utf-8"
                ) as trace_file,
            ):
                for source_index, source in enumerate(selected):
                    for cap, seed in itertools.product(CAPS, SEEDS):
                        timing_signature = None
                        for is_trace, handle in (
                            (False, timing_file),
                            (True, trace_file),
                        ):
                            request = public_request(
                                source,
                                cap=cap,
                                policy_seed=seed,
                                trace=is_trace,
                                model=model,
                            )
                            response, transport = client.request(request)
                            signature = validate_response(
                                source, request, response, model, require_trace=is_trace
                            )
                            if is_trace:
                                assert_same(
                                    signature,
                                    timing_signature,
                                    "trace/timing logical identity",
                                )
                            else:
                                timing_signature = signature
                            public = {k: v for k, v in request.items() if k != "model"}
                            record = {
                                "schema": "c26-native-record-v1",
                                "id": source["id"],
                                "ply": source["ply"],
                                "split": split,
                                "cap": cap,
                                "seed": seed,
                                "kind": "trace" if is_trace else "timing",
                                "source_hash": source["source_hash"],
                                "request": public,
                                "request_sha256": sha_bytes(
                                    json_text(request).encode()
                                ),
                                "model_sha256": model_record["model_sha256"],
                                "transport": transport,
                                "response": response,
                            }
                            handle.write(json_text(record) + "\n")
                            counts[
                                "trace_requests" if is_trace else "timing_requests"
                            ] += 1
                            counts["trace_runs" if is_trace else "timing_runs"] += len(
                                response["runs"]
                            )
                            roundtrip_us += transport["roundtrip_us"]
                            native_execution_us += response.get(
                                "execution_elapsed_us", 0.0
                            )
                            native_warmup_us += response.get("warmup_elapsed_us", 0.0)
                            decision_us = sum(
                                run["elapsed_us"] for run in response["runs"]
                            )
                            if is_trace:
                                native_trace_us += decision_us
                            else:
                                native_recorded_us += decision_us
                    if (source_index + 1) % 25 == 0:
                        print(
                            json_text(
                                {
                                    "split": split,
                                    "verified_source_parents": source_index + 1,
                                    "total": len(selected),
                                }
                            ),
                            flush=True,
                        )
        if file_sha(binary) != client.binary_sha256:
            raise AssertionError("native binary changed during campaign")
        audit_frozen_model(data_dir)
        manifest.update(
            {
                "status": "completed",
                "counts": dict(counts),
                "binary_sha256": client.binary_sha256,
                "native_exit_code": client.exit_code,
                "elapsed_seconds": time.perf_counter() - started,
                "process_launch_us": client.launch_us,
                "roundtrip_us": roundtrip_us,
                "native_execution_us": native_execution_us,
                "recorded_decision_us": native_recorded_us,
                "discarded_warmup_decision_us": native_warmup_us,
                "separate_trace_decision_us": native_trace_us,
                "native_scheduling_output_aggregation_us": native_execution_us
                - native_recorded_us
                - native_warmup_us
                - native_trace_us,
                "transport_parse_validation_serialization_residual_us": roundtrip_us
                - native_execution_us,
                "python_oracle_io_orchestration_residual_us": (
                    time.perf_counter() - started
                )
                * 1e6
                - roundtrip_us,
                "files": {
                    name: file_sha(output / name)
                    for name in ("timing_raw.jsonl.gz", "trace_raw.jsonl.gz")
                },
                "independent_validation": "Every recorded logical signature, candidate prediction, "
                "trace choice, random draw, exact reveal and charged node count checked.",
                "timing": "Per-seed median of nine recorded native decisions; one warmup discarded; "
                "trace pass excluded; mean four seeds; equal source-parent weight.",
            }
        )
        write_json(output / "run_manifest.json", manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(output / "run_manifest.json", manifest)
        raise
    return analyze_campaign(split, data_dir, results_dir, already_validated=True)


def validate_campaign(split, data_dir, results_dir):
    data_dir, output = Path(data_dir), Path(results_dir) / split
    sources, _, _ = load_inputs(data_dir)
    sources = {s["id"]: s for s in sources if s["split"] == split}
    model, model_record = audit_frozen_model(data_dir)
    manifest = read_json(output / "run_manifest.json")
    if manifest["status"] != "completed":
        raise AssertionError("campaign is not completed")
    if manifest["model_sha256"] != model_record["model_sha256"]:
        raise AssertionError("campaign used a different frozen model")
    for name, expected in manifest["files"].items():
        if file_sha(output / name) != expected:
            raise AssertionError("raw campaign file changed: " + name)
    expected_cells = set(itertools.product(sources, CAPS, SEEDS))
    timing_signatures = {}
    result = {"split": split, "source_parents": len(sources)}
    for is_trace, filename in (
        (False, "timing_raw.jsonl.gz"),
        (True, "trace_raw.jsonl.gz"),
    ):
        cells = set()
        runs = 0
        for record in json_lines(output / filename):
            cell = (record["id"], record["cap"], record["seed"])
            if cell in cells or cell not in expected_cells:
                raise AssertionError("duplicate or unexpected raw request cell")
            cells.add(cell)
            source = sources[record["id"]]
            request = public_request(
                source,
                cap=record["cap"],
                policy_seed=record["seed"],
                trace=is_trace,
                model=model,
            )
            public = {k: v for k, v in request.items() if k != "model"}
            if set(record["request"]) != set(public):
                raise AssertionError("raw public request has unapproved fields")
            assert_same(record["request"], public, "public request reconstruction")
            if record["request_sha256"] != sha_bytes(json_text(request).encode()):
                raise AssertionError("public request/frozen model digest mismatch")
            if record["source_hash"] != source["source_hash"]:
                raise AssertionError("source identity mismatch")
            signature = validate_response(
                source, request, record["response"], model, require_trace=is_trace
            )
            if is_trace:
                assert_same(
                    signature, timing_signatures[cell], "stored trace/timing signature"
                )
            else:
                timing_signatures[cell] = signature
            runs += len(record["response"]["runs"])
        if cells != expected_cells:
            raise AssertionError("missing raw campaign cells")
        result["trace_runs" if is_trace else "timing_runs"] = runs
        result["trace_requests" if is_trace else "timing_requests"] = len(cells)
    write_json(output / "validation.json", result)
    return result


def analyze_campaign(split, data_dir, results_dir, already_validated=False):
    data_dir, output = Path(data_dir), Path(results_dir) / split
    if not already_validated:
        validate_campaign(split, data_dir, results_dir)
    sources, offline, _ = load_inputs(data_dir)
    selected = [s for s in sources if s["split"] == split]
    model, model_record = audit_frozen_model(data_dir)
    manifest = read_json(output / "run_manifest.json")
    flat = []
    for record in json_lines(output / "timing_raw.jsonl.gz"):
        for run in record["response"]["runs"]:
            flat.append(
                dict(
                    run,
                    id=record["id"],
                    ply=record["ply"],
                    cap=record["cap"],
                    seed=record["seed"],
                    oracle_u=offline[record["id"]]["root_u"],
                )
            )
    seed_rows = summarize_seed_runs(flat)
    parent_rows = summarize_parents(seed_rows)
    write_json_lines(output / "seed_medians.jsonl.gz", seed_rows)
    write_json_lines(output / "parent_metrics.jsonl.gz", parent_rows)
    by_policy = {}
    for cap, policy in itertools.product(CAPS, POLICIES):
        rows = [
            row for row in parent_rows if row["cap"] == cap and row["policy"] == policy
        ]
        if len(rows) != len(selected):
            raise AssertionError("parent aggregate lost or duplicated a source")
        by_policy[f"{cap}:{policy}"] = {
            "source_parents": len(rows),
            "mean": {
                key: statistics.mean(r["metrics"][key] for r in rows)
                for key in rows[0]["metrics"]
            },
        }
    comparisons = [
        comparison(parent_rows, cap, policy, baseline)
        for cap in CAPS
        for baseline_index, baseline in enumerate(POLICIES)
        for policy in POLICIES[baseline_index + 1 :]
    ]
    primary_comparison = next(
        c
        for c in comparisons
        if c["cap"] == 9
        and c["policy"] == "PANDORA"
        and c["baseline"] == "INITIAL_STOP"
    )
    primary = dict(
        primary_comparison["metrics"]["net_nodes"],
        cap=9,
        policy="PANDORA",
        baseline="INITIAL_STOP",
        metric="delta final_u - 2^-10 * delta charged_nodes",
        positive_gain_gate=primary_comparison["metrics"]["net_nodes"]["ci95"][0] > 0,
    )
    efficiencies = []
    for policy in POLICIES:
        c = comparison(parent_rows, 9, "ALL_ACTIONS", policy)
        efficiencies.append(
            {
                "policy": policy,
                "reference": "ALL_ACTIONS",
                "node_savings": c["metrics"]["charged_nodes"],
                "time_savings_us": c["metrics"]["elapsed_us"],
            }
        )
    diagnostic, calibration_rows = calibration(selected, model)
    write_json_lines(output / "calibration_rows.jsonl.gz", calibration_rows)
    diagnostic["rows_sha256"] = file_sha(output / "calibration_rows.jsonl.gz")
    write_json(output / "calibration.json", diagnostic)
    gibbs_probabilities, q_values, z_values = [], [], []
    draw_decisions = 0
    for record in json_lines(output / "trace_raw.jsonl.gz"):
        for run in record["response"]["runs"]:
            if run["policy"] != "GIBBS_VOC":
                continue
            for decision in run["decisions"]:
                if decision["uniform_draw"] is None:
                    continue
                draw_decisions += 1
                gibbs_probabilities.append(decision["stop_probability"])
                for candidate in decision["candidates"]:
                    gibbs_probabilities.append(candidate["probability"])
                    q_values.append(candidate["q"])
                    z_values.append(candidate["z"])
    positive = [p for p in gibbs_probabilities if p > 0]
    finite_parameters = {
        "tau": TAU,
        "lambda_nodes": LAMBDA,
        "sampled_decisions": draw_decisions,
        "minimum_positive_choice_probability": min(positive) if positive else None,
        "maximum_choice_probability": max(gibbs_probabilities)
        if gibbs_probabilities
        else None,
        "underflow_zero_probabilities_on_sampled_steps": sum(
            p == 0 for p in gibbs_probabilities
        ),
        "q_range": [min(q_values), max(q_values)] if q_values else None,
        "z_range": [min(z_values), max(z_values)] if z_values else None,
        "forced_stop_probabilities_excluded": True,
    }
    report = {
        "schema": "c26-analysis-v1",
        "split": split,
        "model_sha256": model_record["model_sha256"],
        "run_manifest_sha256": file_sha(output / "run_manifest.json"),
        "source_parents": len(selected),
        "primary": primary,
        "primary_status": "frozen TEST confirmatory contrast"
        if split == "TEST"
        else "CAL diagnostic only",
        "policy_means": by_policy,
        "paired_comparisons": comparisons,
        "full_horizon_efficiency_against_all_actions": efficiencies,
        "gibbs_actual_finite_parameters": finite_parameters,
        "calibration": diagnostic,
        "runtime_overhead": {
            k: v
            for k, v in manifest.items()
            if k.endswith("_us") or k == "elapsed_seconds"
        },
        "aggregation": "median nine timing repetitions within source/cap/seed/policy; "
        "mean four seeds; equal source-parent mean; fixed TRAIN model",
        "inference_scope": "Paired source-parent bootstrap stratified by ply, conditional on fixed TRAIN. "
        "Shared game descendants can couple source parents; this is a finite held-out cohort "
        "comparison and does not establish universal iid game performance.",
        "prices": {"utility_per_node": LAMBDA, "utility_per_microsecond": WALL_PRICE},
        "no_test_retuning": True,
        "files": {
            name: file_sha(output / name)
            for name in (
                "seed_medians.jsonl.gz",
                "parent_metrics.jsonl.gz",
                "calibration_rows.jsonl.gz",
            )
        },
    }
    write_json(output / "analysis.json", report)
    return {
        "split": split,
        "source_parents": len(selected),
        "primary": primary,
        "analysis": str(output / "analysis.json"),
        "calibration": str(output / "calibration.json"),
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "train",
            "cal",
            "test",
            "analyze",
            "validate",
            "audit-model",
            "audit-all",
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--split", choices=("CAL", "TEST"), default="TEST")
    args = parser.parse_args()
    execution = {
        "schema": "c26-command-execution-v1",
        "started_utc": utc(),
        "command": [sys.executable] + sys.argv,
        "harness_sha256": file_sha(__file__),
        "status": "running",
    }
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    record_path = DOC / "evidence" / f"DATA_EXEC_{args.command}_{stamp}.json"
    try:
        if args.command == "prepare":
            result = prepare(args.data_dir)
        elif args.command == "train":
            trained = train(args.data_dir, args.binary)
            result = {
                "model_sha256": trained["model_sha256"],
                "source_parents": len(trained["training_source_ids"]),
                "nonterminal_candidates": trained["raw_counts"][
                    "nonterminal_candidates"
                ],
                "native_exit_code": trained["native_exit_code"],
                "elapsed_seconds": trained["elapsed_seconds"],
                "provenance": str(args.data_dir / "model_provenance.json"),
            }
        elif args.command in ("cal", "test"):
            result = run_campaign(
                args.command.upper(), args.data_dir, args.results_dir, args.binary
            )
        elif args.command == "analyze":
            result = analyze_campaign(args.split, args.data_dir, args.results_dir)
        elif args.command == "validate":
            result = validate_campaign(args.split, args.data_dir, args.results_dir)
        elif args.command == "audit-model":
            _, record = audit_frozen_model(args.data_dir)
            result = {
                "model_sha256": record["model_sha256"],
                "training_source_parents": 90,
                "parameter_reproduction": "verified",
            }
        else:
            with NativeClient(args.binary) as client:
                result = audit_all_reachable(client)
            result["native_exit_code"] = client.exit_code
            write_json(DOC / "evidence/DATA_ALL_REACHABLE.json", result)
        execution.update(
            {
                "status": "completed",
                "exit_code": 0,
                "result": result,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_utc": utc(),
            }
        )
        write_json(record_path, execution)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        execution.update(
            {
                "status": "failed",
                "exit_code": 1,
                "error_class": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_utc": utc(),
            }
        )
        write_json(record_path, execution)
        print(json_text(execution), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
