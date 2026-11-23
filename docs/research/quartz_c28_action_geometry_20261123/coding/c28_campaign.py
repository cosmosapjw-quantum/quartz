#!/usr/bin/env python3
"""Reproduce C28 scientific verification and the frozen native cost comparison.

Verification and benchmarks use only the stdlib.  Figures additionally require
matplotlib.  The complete-world Fraction oracle is deliberately independent of
both native planner formulas.  Game oracle labels are constructed only in this
offline checker, never sent to the controller.

Examples (output directories must be new and inside this C28 document tree):
  python3 coding/c28_campaign.py verify --binary /path/to/binary --suite all \
      --output evidence/verify_01
  python3 coding/c28_campaign.py benchmark --binary /path/to/binary \
      --output evidence/benchmark_01
  python3 coding/c28_campaign.py figures --benchmark evidence/benchmark_01 \
      --output evidence/figures_01

Run from the repository or use absolute output paths. Relative paths are resolved
against the current working directory, not against this script.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import selectors
import shlex
import statistics
import subprocess
import sys
import time
import traceback

DOC = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
INPUTS = ROOT / "docs/research/quartz_general_game_metareasoning_20261123/coding/data"
SCHEMA = "c28-geometry-v1"
ABS_TOL = 5e-12
TIE = F(1, 10**12)
ORACLE_SHA = "58c09680fca14ac5b05a5fb8e20d40d397202ba36762f6a54406cdffc29fc35e"
SOURCE_SHA = "9c9f52a0344a4b6c7c25cde6d519479784a0695fcb8f4681486403fb21fe15c1"
MODEL_SHA = "0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126"
MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")


def close_float(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool),
            f"{label}: not a real numeric field: {actual!r}")
    residual = abs(float(actual) - float(expected))
    require(math.isfinite(actual) and residual <= ABS_TOL,
            f"{label}: actual={actual!r}, expected={expected}, residual={residual}; "
            f"fixed abs_tol={ABS_TOL}, rel_tol=0")
    return residual


def epsilon_choice(scores):
    """Fixed native tie rule applied to exact rational gains keyed by action id."""
    if not scores or max(scores.values()) <= TIE:
        return None
    best = max(scores.values())
    return min(i for i, score in scores.items() if score >= best - TIE)


@lru_cache(maxsize=1)
def oracle():
    path = DOC / "theory/exact_lifted_h2.py"
    require(sha256(path) == ORACLE_SHA, "Frozen complete-world oracle byte identity changed")
    spec = importlib.util.spec_from_file_location("c28_exact_world_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frozen_inputs():
    contract = json.loads((DOC / "coding/CONTRACT.json").read_text())
    source_path, model_path = INPUTS / "public_sources.jsonl", INPUTS / "model.json"
    require(contract["game_campaign"]["source_sha256"] == SOURCE_SHA,
            "Source hash lock differs from C28 contract")
    require(contract["game_campaign"]["model_sha256"] == MODEL_SHA,
            "Model hash lock differs from C28 contract")
    require(sha256(source_path) == SOURCE_SHA, "Frozen source bytes changed")
    require(sha256(model_path) == MODEL_SHA, "Frozen model bytes changed")
    sources = [json.loads(line) for line in source_path.read_text().splitlines()]
    require(len(sources) == 300 and len({r["id"] for r in sources}) == 300,
            "Frozen cohort must contain exactly 300 unique public roots")
    require(all(type(r["cue_seed"]) is int and 0 <= r["cue_seed"] <= MASK64
                for r in sources), "Invalid recorded cue seed")
    return contract, sources, json.loads(model_path.read_text())


class RawRecorder:
    """Write the exact request/response lines and independent evidence as gzip."""
    def __init__(self, output):
        self.output = Path(output)
        self.stack = ExitStack()
        self.handles = {}
        self.digests = {}
        self.counts = Counter()

    def append(self, kind, raw):
        if kind not in self.handles:
            file = self.stack.enter_context((self.output / (kind + ".jsonl.gz")).open("wb"))
            self.handles[kind] = self.stack.enter_context(
                gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0))
            self.digests[kind] = hashlib.sha256()
        data = raw.encode("utf-8")
        if not data.endswith(b"\n"):
            data += b"\n"
        self.handles[kind].write(data)
        self.digests[kind].update(data)
        self.counts[kind] += 1

    def record(self, kind, value):
        self.append(kind, encode(value))

    def close(self):
        self.stack.close()
        return {kind: {"lines": self.counts[kind],
                       "uncompressed_sha256": digest.hexdigest(),
                       "gzip_sha256": sha256(self.output / (kind + ".jsonl.gz"))}
                for kind, digest in self.digests.items()}


class NativeClient:
    """One resident JSONL process; a malformed line must not end the stream."""
    def __init__(self, binary, *, cpu=None, recorder=None, stderr_path=None, timeout=120):
        self.binary = Path(binary).resolve()
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise RuntimeError(f"NOT_RUN: C28 executable missing or not executable: {self.binary}")
        self.command = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [str(self.binary)]
        self.recorder, self.timeout, self.exit_code = recorder, timeout, None
        self.stderr_file = open(stderr_path, "w", encoding="utf-8") if stderr_path else None
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=self.stderr_file or subprocess.DEVNULL,
                                            encoding="utf-8", text=True, bufsize=1, env=env)
        except BaseException:
            if self.stderr_file:
                self.stderr_file.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request_raw(self, line):
        require("\n" not in line, "A request must be exactly one JSON line")
        if self.recorder:
            self.recorder.append("requests", line)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        if not self.selector.select(self.timeout):
            raise TimeoutError(f"Native response timed out after {self.timeout}s")
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"Native stream ended: exit={self.process.poll()}")
        if self.recorder:
            self.recorder.append("responses", response_line)
        response = json.loads(response_line, parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON output: {x}")))
        require(isinstance(response, dict) and response.get("schema") == SCHEMA,
                f"Unexpected response schema: {response!r}")
        return response

    def request(self, request):
        return self.request_raw(encode(request))

    def close(self):
        if self.exit_code is not None:
            return self.exit_code
        try:
            self.process.stdin.close()
            self.exit_code = self.process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.kill()
            self.exit_code = self.process.wait(timeout=10)
        finally:
            self.selector.close()
            self.process.stdout.close()
            if self.stderr_file:
                self.stderr_file.close()
        return self.exit_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        code = self.close()
        if exc_type is None:
            require(code == 0, f"Native process exit code {code}")


def synthetic_request(model, m=F(0), b=F(1, 2), *, ident="case", horizon=2, **timing):
    return {"mode": "synthetic", "id": ident,
            "probabilities": [[float(q) for q in row] for row in model.p],
            "loadings": list(map(float, model.r)), "costs": list(map(float, model.cost)),
            "m": float(m), "b": float(b), "horizon": horizon, **timing}


def exact_synthetic(model, m, b, horizon=2):
    ref = oracle()
    q = ref.world_scores(model, ref.worlds(model, m), b, tuple(range(model.n)), horizon)
    candidates = {i: q[i] for i in range(model.n)}
    return candidates, b + max(q.values()), epsilon_choice(candidates)


def check_timings(response, repeats, iterations):
    timings = response.get("timings")
    require(isinstance(timings, list) and len(timings) == repeats, "Wrong timing repeat count")
    previous = None
    for repeat, row in enumerate(timings):
        require(row["repeat"] == repeat and row["iterations"] == iterations,
                "Wrong timing index or iteration count")
        require(row["order"] in ("direct_first", "lifted_first"), "Unknown timing order")
        require(previous is None or row["order"] != previous, "Timing order did not alternate")
        previous = row["order"]
        for method in ("direct", "lifted"):
            require(type(row[method + "_ns"]) is int and row[method + "_ns"] > 0,
                    "Timing must be positive integer total nanoseconds")


def check_synthetic(response, model, m, b, horizon=2, *, full_world=True):
    require("error" not in response and response.get("mode") == "synthetic",
            f"Synthetic execution failed: {response}")
    exact = exact_synthetic(model, m, b, horizon) if full_world else None
    residuals = []
    for method in ("direct", "lifted"):
        result = response[method]
        require(len(result["scores"]) == model.n, "Score vector has wrong length")
        for i, q in enumerate(result["scores"]):
            require(math.isfinite(q), "Nonfinite planner score")
            if exact:
                residuals.append(close_float(q, exact[0][i], f"{method}.score[{i}]"))
        if exact:
            residuals.append(close_float(result["value"], exact[1], method + ".value"))
            require(result["choice"] == exact[2],
                    f"{method} choice={result['choice']} != exact epsilon choice={exact[2]}")
        close_float(result["value"], float(b) + max([0.0] + result["scores"]),
                    method + ".raw_optimum_value")
    require(response["direct"]["choice"] == response["lifted"]["choice"],
            "Direct and lifted choices differ")
    differences = [close_float(a, z, f"direct/lifted score[{i}]")
                   for i, (a, z) in enumerate(zip(response["direct"]["scores"],
                                                 response["lifted"]["scores"]))]
    close_float(response["max_score_abs_difference"], max(differences, default=0),
                "reported maximum score difference")
    return {"max_oracle_abs_residual": max(residuals, default=0),
            "max_method_abs_difference": max(differences, default=0),
            "full_world_oracle": full_world,
            "exact_scores": {str(i): str(q) for i, q in exact[0].items()} if exact else None,
            "exact_value": str(exact[1]) if exact else None,
            "epsilon_choice": exact[2] if exact else None}


def synthetic_cases(random_count=48):
    """Frozen small-domain correctness controls, seed 2808; not fitted data."""
    ref = oracle()
    cases = []
    for sign, name in ((-1, "signed"), (1, "common")):
        for cost in (F(1, 100), F(2, 25)):
            cases.append((f"fixture-{name}-c{cost}", ref.family(c=cost, sign=sign), F(0), F(1, 2), 2))
    cases.append(("fixture-product", ref.family(r=F(0)), F(0), F(1, 2), 2))
    for sign in (-1, 1):
        for shift in (F(-1, 10**8), F(-1, 10**13), F(0), F(1, 10**13), F(1, 10**8)):
            m = sign * F(19, 24) + shift
            cases.append((f"boundary-{m}", ref.family(), m, F(1, 2), 2))
    edge_models = [
        ref.Model((), (), ()),
        ref.Model(((F(0), F(1), F(0)),), (F(9, 10),), (F(0),)),
        ref.Model(((F(1), F(0), F(0)), (F(0), F(0), F(1))),
                  (F(-99, 100), F(99, 100)), (F(0), F(0))),
        ref.Model(((F(1, 3),) * 3,) * 3, (F(0),) * 3, (F(0),) * 3),
        ref.Model(((F(1, 2), F(0), F(1, 2)),) * 2,
                  (F(2**20 - 1, 2**20), -F(2**20 - 1, 2**20)), (F(0), F(1, 10))),
    ]
    for k, model in enumerate(edge_models):
        for m in (F(-1), F(0), F(1)):
            for b, horizon in ((F(0), 1), (F(1, 2), 2), (F(1), 2)):
                cases.append((f"edge-{k}-m{m}-b{b}-h{horizon}", model, m, b, horizon))
    rng = random.Random(2808)
    for k in range(random_count):
        n = 1 + k % 5
        rows = []
        for _ in range(n):
            weights = [rng.randrange(8) for _ in range(3)]
            if not sum(weights):
                weights[0] = 1
            rows.append(tuple(F(w, sum(weights)) for w in weights))
        model = ref.Model(tuple(rows),
                          tuple(rng.choice((F(-99, 100), F(-4, 5), F(0), F(1, 2), F(99, 100)))
                                for _ in range(n)),
                          tuple(rng.choice((F(0), F(1, 100), F(3, 10), F(1))) for _ in range(n)))
        cases.append((f"rational-{k:03}", model, rng.choice((F(-1), F(-3, 7), F(0), F(4, 5), F(1))),
                      rng.choice((F(0), F(1, 3), F(1, 2), F(1))), 1 + k % 2))
    return cases


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def terminal(board):
    return bool(winner(board)) or 0 not in board


def play(board, move, player):
    require(type(move) is int and 0 <= move < 9 and board[move] == 0,
            f"Illegal move {move}")
    return board[:move] + (player,) + board[move + 1:]


def replay(history):
    board, player = (0,) * 9, 1
    for move in history:
        require(not terminal(board), "History continues after terminal")
        board = play(board, move, player)
        player = -player
    return board, player


@lru_cache(maxsize=None)
def minimax(board, turn, root_player):
    """Independent root-frame game solution and full (unpruned) tree node count.

    Memoization avoids repeated Python work; the returned logical node count sums
    every subtree occurrence, matching a traversal without caching or pruning.
    """
    if terminal(board):
        return winner(board) * root_player, 1
    values, nodes = [], 1
    for move, value in enumerate(board):
        if value == 0:
            score, count = minimax(play(board, move, turn), -turn, root_player)
            values.append(score)
            nodes += count
    return (max(values) if turn == root_player else min(values)), nodes


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def index(self, n):
        threshold = ((-n) & MASK64) % n
        while True:
            value = self.next()
            if value >= threshold:
                return value % n


def game_candidates(history, cue_seed, model):
    board, root_player = replay(history)
    require(not terminal(board), "Game oracle requires nonterminal root")
    result = []
    for move, value in enumerate(board):
        if value:
            continue
        child = play(board, move, root_player)
        known = terminal(child)
        counts, cue_nodes = [0, 0, 0], 0
        if known:
            counts[winner(child) * root_player + 1] = 4
            cue_nodes = 1
        else:
            rng = SplitMix64(cue_seed ^ (((move + 1) * GAMMA) & MASK64))
            for _ in range(4):
                rollout_board, turn, nodes = child, -root_player, 1
                while not terminal(rollout_board):
                    moves = [i for i, mark in enumerate(rollout_board) if mark == 0]
                    rollout_board = play(rollout_board, moves[rng.index(len(moves))], turn)
                    turn = -turn
                    nodes += 1
                counts[winner(rollout_board) * root_player + 1] += 1
                cue_nodes += nodes
        # labels are offline reference evidence, not fields in a native request.
        outcome, solve_nodes = minimax(child, -root_player, root_player)
        utility = F(outcome + 1, 2)
        remaining = str(child.count(0))
        key = remaining + ":" + ":".join(map(str, counts))
        probabilities = model["feature_probs"].get(key,
                           model["remaining_probs"].get(remaining, model["global_probs"]))
        p = tuple(F(str(q)) for q in probabilities)
        p = tuple(q / sum(p) for q in p)
        if known:
            p = tuple(F(int(i == outcome + 1)) for i in range(3))
        predicted_nodes = model["remaining_costs"].get(remaining, model["global_cost"])
        result.append({"move": move, "utility": utility, "solve_nodes": solve_nodes,
                       "terminal_known": known, "counts": counts, "cue_nodes": cue_nodes,
                       "p": p, "cost": F(0) if known else F(str(predicted_nodes)) / 1024})
    return result


def posterior_latent_mean(model, known):
    """Posterior odds from component likelihood products, without recursive Bayes."""
    features = model.features()
    plus, minus = F(1), F(1)
    for i, value in known.items():
        k = model.values.index(value)
        p, t = model.p[i][k], model.r[i] * features[i][k]
        plus *= p * (1 + t)
        minus *= p * (1 - t)
    require(plus + minus > 0, "Unsupported independently observed outcome")
    return (plus - minus) / (plus + minus)


def game_request(source, model, *, trace=True, repeats=0, warmup=0):
    return {"mode": "game", "id": source["id"], "history": source["history"],
            "cue_seed": source["cue_seed"], "model": model, "rho": 0.75,
            "trace": trace, "repeats": repeats, "warmup": warmup}


def compare_game_methods(response):
    require("error" not in response and response.get("mode") == "game",
            f"Game execution failed: {response}")
    discrete = ("initial_move", "final_move", "initial_utility", "utility", "initial_nodes",
                "additional_nodes", "charged_nodes", "additional_queries", "query_moves")
    direct, lifted = response["direct"], response["lifted"]
    for field in discrete:
        require(direct[field] == lifted[field], f"Game direct/lifted {field} differs")
    close_float(direct["final_m"], lifted["final_m"], "game final_m direct/lifted")
    require(len(direct["decisions"]) == len(lifted["decisions"]), "Decision count differs")
    for a, b in zip(direct["decisions"], lifted["decisions"]):
        for field in ("remaining_moves", "horizon", "choice"):
            require(a[field] == b[field], f"Game decision {field} differs")
        for field in ("b", "m"):
            close_float(a[field], b[field], "game decision " + field)
        require(len(a["scores"]) == len(b["scores"]), "Decision score lengths differ")
        for x, y in zip(a["scores"], b["scores"]):
            close_float(x, y, "game decision score direct/lifted")


def check_game(response, request, *, check_scores=True):
    """Independent cues, outcomes, logical cost and optional joint-world decisions.

    The public 300 roots have at most five legal candidates. Large ad hoc roots
    still receive independent game and charge checks; exact world enumeration is
    explicitly reported as not run when there are more than five candidates.
    """
    compare_game_methods(response)
    candidates = game_candidates(request["history"], request["cue_seed"], request["model"])
    by_move = {c["move"]: i for i, c in enumerate(candidates)}
    known = {i: c["utility"] for i, c in enumerate(candidates) if c["terminal_known"]}
    initial_nodes = 0
    initial_solve = None
    if F(1) not in known.values():
        unknown = [i for i in range(len(candidates)) if i not in known]
        if unknown:
            chosen = max(unknown, key=lambda i: (candidates[i]["counts"][1] +
                                                2 * candidates[i]["counts"][2], -i))
            known[chosen] = candidates[chosen]["utility"]
            initial_nodes = candidates[chosen]["solve_nodes"]
            initial_solve = candidates[chosen]["move"]
    initial = min(known, key=lambda i: (-known[i], i))
    initial_utility = known[initial]
    ref = oracle()
    model = ref.Model(tuple(c["p"] for c in candidates),
                      (F(str(request.get("rho", 0.75))),) * len(candidates),
                      tuple(c["cost"] for c in candidates))
    full_world = bool(check_scores and request.get("trace") and len(candidates) <= 5)
    distribution = ref.worlds(model, F(0)) if full_world else None
    if distribution is not None:
        for i, outcome in known.items():
            _, distribution = ref.observe(distribution, i, outcome)
    for method in ("direct", "lifted"):
        summary = response[method]
        require(summary["initial_move"] == candidates[initial]["move"],
                "Initial incumbent differs from independent cue/verification reconstruction")
        require(summary["initial_utility"] == float(initial_utility), "Initial utility/root sign differs")
        require(summary["initial_nodes"] == initial_nodes, "Initial solve node count differs")
        require(summary["additional_queries"] == len(summary["query_moves"]) <= 2,
                "Additional query count/cap mismatch")
    summary = response["direct"]
    queries = summary["query_moves"]
    require(len(set(queries)) == len(queries), "Candidate queried more than once")
    incumbent, additional_nodes = initial, 0
    observed = dict(known)
    max_residual, decision_checks = 0.0, 0
    decisions = summary["decisions"]
    require(not request.get("trace") or len(decisions) in (len(queries), len(queries) + 1),
            "Trace does not contain every query decision")
    require(len(decisions) <= 2, "Decision trace exceeded the two-query horizon")
    for k, decision in enumerate(decisions):
        remaining = tuple(i for i in range(len(candidates)) if i not in observed)
        require(observed[incumbent] < 1 and remaining,
                "Decision recorded after a winning incumbent or exhausted candidates")
        require(decision["remaining_moves"] == [candidates[i]["move"] for i in remaining],
                "Decision remaining set differs from actual known values")
        require(decision["horizon"] == 2 - k, "Decision horizon does not decrease from H2 to H1")
        require(decision["b"] == float(observed[incumbent]), "Decision known incumbent differs")
        close_float(decision["m"], posterior_latent_mean(model, observed), "game posterior mean")
        require(len(decision["scores"]) == len(remaining), "Decision scores not aligned to remaining moves")
        if full_world:
            q = ref.world_scores(model, distribution, observed[incumbent], remaining, 2 - k)
            expected = epsilon_choice({candidates[i]["move"]: q[i] for i in remaining})
            require(decision["choice"] == expected, "Game choice differs from complete-world oracle")
            for method in ("direct", "lifted"):
                actual_scores = response[method]["decisions"][k]["scores"]
                for i, actual in zip(remaining, actual_scores):
                    max_residual = max(max_residual, close_float(actual, q[i], method + " game complete-world score"))
            decision_checks += 1
        expected_choice = queries[k] if k < len(queries) else None
        require(decision["choice"] == expected_choice, "Trace choice differs from actual query sequence")
        if expected_choice is None:
            require(k == len(decisions) - 1, "Continued after STOP")
            continue
        i = by_move[expected_choice]
        require(i not in observed, "Queried already-known candidate")
        observed[i] = candidates[i]["utility"]
        additional_nodes += candidates[i]["solve_nodes"]
        if observed[i] > observed[incumbent]:
            incumbent = i
        if distribution is not None:
            _, distribution = ref.observe(distribution, i, observed[i])
    if request.get("trace"):
        stopped_explicitly = bool(decisions and decisions[-1]["choice"] is None)
        still_active = (len(queries) < 2 and observed[incumbent] < 1 and
                        len(observed) < len(candidates))
        require(not still_active or stopped_explicitly,
                "Incomplete decision trace: active horizon requires a decision or explicit STOP")
    if not request.get("trace"):
        for move in queries:
            require(move in by_move, "Queried illegal move")
            i = by_move[move]
            require(i not in observed, "Queried already-known candidate")
            observed[i] = candidates[i]["utility"]
            additional_nodes += candidates[i]["solve_nodes"]
            if observed[i] > observed[incumbent]:
                incumbent = i
    cue_nodes = sum(c["cue_nodes"] for c in candidates)
    require(summary["final_move"] == candidates[incumbent]["move"], "Final known-only incumbent/tie retention differs")
    require(summary["utility"] == float(observed[incumbent]), "Final utility/root-player sign differs")
    require(summary["additional_nodes"] == additional_nodes, "Additional exact-solve node count differs")
    require(summary["charged_nodes"] == cue_nodes + initial_nodes + additional_nodes,
            "Charged nodes do not equal all cue + initial + additional logical nodes")
    close_float(summary["final_m"], posterior_latent_mean(model, observed), "final latent posterior")
    return {"root_player": replay(request["history"])[1], "cue_nodes": cue_nodes,
            "initial_solve_move": initial_solve, "initial_nodes": initial_nodes,
            "additional_nodes": additional_nodes,
            "observed_queries": [{"move": move, "utility": str(candidates[by_move[move]]["utility"]),
                                  "nodes": candidates[by_move[move]]["solve_nodes"]} for move in queries],
            "independent_minimax_and_charge_check": True,
            "complete_world_decisions_checked": decision_checks,
            "complete_world_status": "CHECKED" if full_world else "NOT_RUN_LARGE_OR_NO_TRACE",
            "max_oracle_abs_residual": max_residual}


def benchmark_synthetic(contract):
    settings = contract["synthetic_campaign"]
    rng = random.Random(settings["seed"])
    ref = oracle()
    for n in settings["candidate_counts"]:
        rows, magnitudes, costs = [], [], []
        for _ in range(n):
            weights = [rng.randrange(1, 10) for _ in range(3)]
            rows.append(tuple(F(w, sum(weights)) for w in weights))
            magnitudes.append(rng.choice((F(1, 4), F(1, 2), F(4, 5), F(19, 20), F(99, 100))))
            costs.append(rng.choice((F(0), F(1, 100), F(1, 20), F(1, 5), F(3, 10), F(1, 2))))
        for pattern in settings["loading_patterns"]:
            r = tuple(x if pattern == "common" or i % 2 == 0 else -x for i, x in enumerate(magnitudes))
            model = ref.Model(tuple(rows), r, tuple(costs))
            for m0 in settings["latent_means"]:
                for b0 in settings["incumbents"]:
                    m, b = F(str(m0)), F(str(b0))
                    iterations = settings["iterations_per_sample_small_n" if n < 16
                                          else "iterations_per_sample_n_ge_16"]
                    ident = f"n{n}-m{m}-b{b}-{pattern.replace(' ', '_')}"
                    request = synthetic_request(model, m, b, ident=ident,
                                                repeats=settings["timing_repeats"],
                                                warmup=settings["warmup_pairs"], iterations=iterations)
                    yield request, model, m, b, {"n": n, "m": float(m), "b": float(b), "pattern": pattern}


def timing_row(request, response, tags=None):
    check_timings(response, request["repeats"], request.get("iterations", 1))
    # Each native time is a TOTAL for 'iterations'; divide exactly once here.
    direct = statistics.median(row["direct_ns"] / row["iterations"] for row in response["timings"])
    lifted = statistics.median(row["lifted_ns"] / row["iterations"] for row in response["timings"])
    return {"id": request["id"], "mode": request["mode"], **(tags or {}),
            "direct_median_ns": direct, "lifted_median_ns": lifted,
            "direct_minus_lifted_ns": direct - lifted, "lifted_over_direct_ratio": lifted / direct,
            "repeats": len(response["timings"]), "iterations": request.get("iterations", 1)}


def aggregate(rows):
    if not rows:
        return {"roots_or_configurations": 0}
    direct = statistics.fmean(r["direct_median_ns"] for r in rows)
    lifted = statistics.fmean(r["lifted_median_ns"] for r in rows)
    return {"roots_or_configurations": len(rows),
            "mean_of_within_root_direct_medians_ns": direct,
            "mean_of_within_root_lifted_medians_ns": lifted,
            "paired_mean_direct_minus_lifted_ns": statistics.fmean(r["direct_minus_lifted_ns"] for r in rows),
            "ratio_of_paired_means_lifted_over_direct": lifted / direct,
            "mean_of_paired_lifted_over_direct_ratios": statistics.fmean(r["lifted_over_direct_ratio"] for r in rows),
            "median_of_paired_lifted_over_direct_ratios": statistics.median(r["lifted_over_direct_ratio"] for r in rows),
            "inference_scope": "Descriptive fixed-cohort engineering costs; no new TEST/generalization claim"}


def git_output(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return {"command": ["git", "-C", str(ROOT), *args], "exit_code": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def identities(binary=None):
    paths = {"campaign": Path(__file__), "contract": DOC / "coding/CONTRACT.json",
             "interface": DOC / "coding/NATIVE_INTERFACE.md", "oracle": DOC / "theory/exact_lifted_h2.py",
             "native_source": ROOT / "src/bin/quartz_c28_action_geometry.rs",
             "python_tests": ROOT / "tests/test_c28_action_geometry.py",
             "frozen_sources": INPUTS / "public_sources.jsonl", "frozen_model": INPUTS / "model.json"}
    if binary:
        paths["native_binary"] = Path(binary).resolve()
    return {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}


def new_output(path):
    output = Path(path).resolve()
    require(DOC.resolve() in output.parents, f"Output must be inside the C28 tree: {DOC}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_campaign(args):
    contract, sources, model = frozen_inputs()
    binary = args.binary or os.environ.get("C28_BINARY")
    if not binary:
        raise RuntimeError("NOT_RUN: provide --binary or C28_BINARY; native verification cannot be skipped")
    binary = str(Path(binary).resolve())
    require(Path(binary).is_file(), f"NOT_RUN: native binary missing: {binary}")
    output = new_output(args.output)
    initial_identities = identities(binary)
    cpu = contract["resources"]["cpu_affinity"]
    metadata = {"schema": "quartz-c28-campaign-v1", "status": "RUNNING",
                "started_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
                "arguments": vars(args), "identity_relation": "BYTE_IDENTITY for all frozen inputs and code",
                "identities": initial_identities, "head": git_output("rev-parse", "HEAD"),
                "git_status_before": git_output("status", "--porcelain"),
                "python": sys.version, "platform": platform.platform(),
                "cpu_affinity_native": cpu, "native_threads": 1,
                "native_command": ["taskset", "-c", str(cpu), binary],
                "parent_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")), "unknown") if Path("/proc/cpuinfo").exists() else platform.processor(),
                "accuracy": {"absolute_score_tolerance": ABS_TOL, "relative_tolerance": 0,
                             "choice_tolerance": str(TIE)},
                "game_cohort_status": contract["game_campaign"]["source_status"],
                "timed_exclusions": contract["game_campaign"]["outside_timing"],
                "timed_synthetic_boundary": contract["synthetic_campaign"]["timed_boundary"],
                "timed_game_boundary": contract["game_campaign"]["timed_boundary"],
                "speedup_required": False, "exclusive_cpu_claim": False}
    write_json(output / "metadata.json", metadata)
    recorder = RawRecorder(output)
    client = None
    started = time.perf_counter()
    try:
        client = NativeClient(binary, cpu=cpu, recorder=recorder,
                              stderr_path=output / "native_stderr.log")
        if args.subcommand == "verify":
            checks = []
            if args.suite in ("synthetic", "all"):
                cases = synthetic_cases()
                if args.limit:
                    cases = cases[:args.limit]
                for ident, law, m, b, horizon in cases:
                    request = synthetic_request(law, m, b, ident=ident, horizon=horizon)
                    response = client.request(request)
                    check_timings(response, 0, 1)
                    check = {"id": ident, "mode": "synthetic", **check_synthetic(response, law, m, b, horizon)}
                    recorder.record("verification", check)
                    checks.append(check)
            if args.suite in ("game", "all"):
                for source in sources[:args.limit or len(sources)]:
                    request = game_request(source, model)
                    response = client.request(request)
                    check_timings(response, 0, 1)
                    check = {"id": source["id"], "mode": "game", **check_game(response, request)}
                    recorder.record("verification", check)
                    checks.append(check)
            summary = {"status": "SMOKE_VERIFIED" if args.limit else "SCOPED_IMPLEMENTATION_VERIFIED",
                       "suite": args.suite, "limit": args.limit,
                       "counts": dict(Counter(row["mode"] for row in checks)),
                       "max_oracle_abs_residual": max((x["max_oracle_abs_residual"] for x in checks), default=0),
                       "game_complete_world_decisions": sum(x.get("complete_world_decisions_checked", 0) for x in checks),
                       "game_observed_additional_queries": sum(len(x.get("observed_queries", [])) for x in checks),
                       "game_status": contract["game_campaign"]["source_status"],
                       "checks": checks}
        else:
            medians = []
            for request, law, m, b, tags in benchmark_synthetic(contract):
                response = client.request(request)
                # Exact world enumeration is confined to small configurations.
                check_synthetic(response, law, m, b, full_world=law.n <= 5)
                row = timing_row(request, response, tags)
                recorder.record("root_medians", row)
                medians.append(row)
            settings = contract["game_campaign"]
            for source in sources:
                request = game_request(source, model, trace=False,
                                       repeats=settings["timing_repeats"], warmup=settings["warmup_pairs"])
                response = client.request(request)
                check_game(response, request, check_scores=False)
                row = timing_row(request, response, {"ply": source["ply"], "historical_split": source["split"]})
                recorder.record("root_medians", row)
                medians.append(row)
            groups = defaultdict(list)
            for row in medians:
                groups[row["mode"]].append(row)
                if row["mode"] == "synthetic":
                    groups[f"synthetic_n{row['n']}_{row['pattern'].replace(' ', '_')}"] .append(row)
            summary = {"status": "FROZEN_FIXED_COHORT_BENCHMARK_COMPLETE", "counts": dict(Counter(x["mode"] for x in medians)),
                       "aggregation": {key: aggregate(rows) for key, rows in groups.items()},
                       "median_rows": medians, "bootstrap": "NOT_RUN; descriptive paired summaries only",
                       "conclusion_rule": "Report measured sign/magnitude; neither speedup nor new utility improvement is required"}
        exit_code = client.close()
        require(exit_code == 0, f"Native process exited {exit_code}")
        require(identities(binary) == initial_identities, "Source/input/binary changed during campaign")
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(output / "summary.json", summary)
        metadata.update(status=summary["status"], native_exit_code=exit_code, command_exit_code=0,
                        ended_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=summary["wall_seconds"], raw_artifacts=recorder.close(),
                        final_identity_check="BYTE_IDENTICAL")
        write_json(output / "metadata.json", metadata)
        print(json.dumps({"status": summary["status"], "output": str(output),
                          "counts": summary["counts"], "exit_code": 0}, sort_keys=True))
        return 0
    except BaseException as error:
        exit_code = client.close() if client else None
        metadata.update(status="ERROR", error=repr(error), traceback=traceback.format_exc(),
                        native_exit_code=exit_code, command_exit_code=1,
                        ended_utc=datetime.now(timezone.utc).isoformat(), raw_artifacts=recorder.close())
        write_json(output / "metadata.json", metadata)
        print(f"ERROR: {error}; preserved partial evidence in {output}", file=sys.stderr)
        return 1


def run_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = new_output(args.output)
    ref = oracle()
    grid = sorted(set([F(k, 240) for k in range(-240, 241)] +
                      [F(-19, 24), F(19, 24), F(-11, 16), F(11, 16), F(-7, 8), F(7, 8)]))
    data = []
    for cost in (F(1, 100), F(2, 25)):
        for sign, name in ((-1, "signed"), (1, "common")):
            law = ref.family(c=cost, sign=sign)
            # Exhaustive contingent plans provide an exact piecewise-affine
            # representation, independently of evaluating the lifted formula.
            lines = ref.plan_lines(law, F(1, 2), (0, 1, 2))
            for m in grid:
                q = {i: max(a + m * slope for a, slope, first, _ in lines if first == i)
                     for i in (-1, 0, 1, 2)}
                chosen = min(q, key=lambda i: (-q[i], i))
                data.append({"cost_A": str(cost), "pattern": name, "m": str(m),
                             "value": str(F(1, 2) + max(q.values())), "exact_choice": chosen})
    write_json(output / "analytic_data.json", data)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, constrained_layout=True)
    for column, cost in enumerate((F(1, 100), F(2, 25))):
        for name, color in (("signed", "#137c8b"), ("common", "#b56a1b")):
            rows = [row for row in data if row["cost_A"] == str(cost) and row["pattern"] == name]
            x = [float(F(row["m"])) for row in rows]
            axes[0, column].plot(x, [float(F(row["value"])) for row in rows], color=color, label=name)
            axes[1, column].step(x, [row["exact_choice"] for row in rows], where="mid", color=color,
                                 label=name, alpha=.88, linewidth=1.7)
        axes[0, column].set_title(f"A cost = {float(cost):.2f}; b = 0.5")
        axes[0, column].set_ylabel("Optimal expected utility")
        axes[1, column].set_ylabel("Exact first action")
        axes[1, column].set_yticks([-1, 0, 1, 2], ["STOP", "A", "B", "C"])
        axes[1, column].set_xlabel("Latent posterior mean m")
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
            axis.set_xlim(-1, 1)
        axes[0, column].legend()
    for boundary in (-19 / 24, 19 / 24):
        axes[0, 0].axvline(boundary, color="0.45", linestyle=":", linewidth=.9)
        axes[1, 0].axvline(boundary, color="0.45", linestyle=":", linewidth=.9)
    fig.suptitle("C28 exact signed/common decision geometry\n"
                 "rA = rB = 0.8; rC = −0.8 (signed) or +0.8 (common); costs B,C = 0.3")
    for suffix in ("png", "svg"):
        fig.savefig(output / ("signed_common_geometry." + suffix), dpi=180)
    plt.close(fig)
    manifest = {"schema": "quartz-c28-figures-v1", "command": shlex.join(sys.argv),
                "oracle_sha256": sha256(DOC / "theory/exact_lifted_h2.py"),
                "campaign_sha256": sha256(__file__), "status": "GENERATED_PENDING_VISUAL_REVIEW",
                "analytic_basis": "Exact rational enumeration of contingent two-query plans",
                "analytic_rows": len(data), "benchmark": None}
    if args.benchmark:
        benchmark = Path(args.benchmark).resolve()
        summary = json.loads((benchmark / "summary.json").read_text())
        metadata = json.loads((benchmark / "metadata.json").read_text())
        require(summary["status"] == "FROZEN_FIXED_COHORT_BENCHMARK_COMPLETE" and
                metadata["command_exit_code"] == 0, "Benchmark evidence incomplete")
        rows = summary["median_rows"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
        ns = sorted({row["n"] for row in rows if row["mode"] == "synthetic"})
        for method, color in (("direct", "#b56a1b"), ("lifted", "#137c8b")):
            values = [statistics.fmean(row[method + "_median_ns"] for row in rows
                                      if row["mode"] == "synthetic" and row["n"] == n) / 1000 for n in ns]
            axes[0].plot(ns, values, "o-", color=color, label=method)
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Candidates n")
        axes[0].set_ylabel("Mean of configuration medians (μs / planner call)")
        axes[0].set_title("Synthetic: preparation + H2 planner")
        axes[0].legend()
        games = [row for row in rows if row["mode"] == "game"]
        axes[1].scatter([row["direct_median_ns"] / 1000 for row in games],
                        [row["lifted_median_ns"] / 1000 for row in games], s=11, alpha=.55, color="#137c8b")
        maximum = max(max(row["direct_median_ns"], row["lifted_median_ns"]) for row in games) / 1000
        axes[1].plot([0, maximum], [0, maximum], ":", color="0.3", label="equal cost")
        axes[1].set_xlabel("Direct root median (μs)")
        axes[1].set_ylabel("Lifted root median (μs)")
        axes[1].set_title("300 fixed roots: whole resident controller")
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=.2)
        fig.suptitle("Native paired costs; descriptive retrospective cohort")
        for suffix in ("png", "svg"):
            fig.savefig(output / ("native_costs." + suffix), dpi=180)
        plt.close(fig)
        manifest["benchmark"] = {"path": str(benchmark), "summary_sha256": sha256(benchmark / "summary.json"),
                                 "metadata_sha256": sha256(benchmark / "metadata.json"),
                                 "native_binary": metadata["identities"]["native_binary"]}
    manifest["files"] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output), "files": list(manifest["files"])}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="subcommand", required=True)
    for name in ("verify", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("--binary", help="Actual C28 executable; otherwise C28_BINARY")
        sub.add_argument("--output", required=True, help="New directory inside the C28 document tree")
        if name == "verify":
            sub.add_argument("--suite", choices=("synthetic", "game", "all"), default="all")
            sub.add_argument("--limit", type=int, help="Smoke only: first N cases per selected suite; labels partial result")
    figures = commands.add_parser("figures")
    figures.add_argument("--output", required=True)
    figures.add_argument("--benchmark", help="Completed benchmark directory for empirical cost figure")
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive; omit for the complete frozen suite")
    try:
        return run_figures(args) if args.subcommand == "figures" else run_campaign(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
