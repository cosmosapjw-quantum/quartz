#!/usr/bin/env python3
"""Frozen C27 data/model/controller audit and campaign harness.

All offline game values stay in this Python analyst process. Native run requests
contain public history/cues and the frozen TRAIN model only. No command runs at
import. Paths derive from this file so a restored checkout remains executable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise, product
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DOC = HERE.parent
ROOT = HERE.parents[3]
OLD = ROOT / "docs/research/quartz_general_game_metareasoning_20261123/coding"
C26_PATH = OLD / "c26_campaign.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C26 = load_module(C26_PATH, "c27_reused_c26")
CONTRACT = json.loads((HERE / "EXPERIMENT_CONTRACT.json").read_text())
POLICIES = tuple(CONTRACT["policies"])
U = (0.0, 0.5, 1.0)
TOL = 1e-12
LAMBDA = 1 / 1024
EXPECTED_OLD = {
    "public_sources.jsonl": "9c9f52a0344a4b6c7c25cde6d519479784a0695fcb8f4681486403fb21fe15c1",
    "training_raw.jsonl.gz": "486a50f34867e382cbff040e339dcc5e051349d2418dbc93a074ec25acdb6659",
    "model.json": "0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126",
}
TIMES = ("elapsed_ns", "cue_ns", "initial_ns", "query_ns", "controller_ns")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "utc": datetime.now(UTC).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain=v1").splitlines(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "harness_sha256": sha(Path(__file__)),
        "contract_sha256": sha(HERE / "CODE_CONTRACT.md"),
        "experiment_contract_sha256": sha(HERE / "EXPERIMENT_CONTRACT.json"),
    }


def old_artifacts():
    for name, expected in EXPECTED_OLD.items():
        if sha(OLD / "data" / name) != expected:
            raise AssertionError("changed C26 evidence: " + name)
    sources = list(read_rows(OLD / "data/public_sources.jsonl"))
    raw = list(read_rows(OLD / "data/training_raw.jsonl.gz"))
    train = [s for s in sources if s["split"] == "TRAIN"]
    if (
        len(train) != 90
        or len(raw) != 90
        or {r["id"] for r in raw} != {s["id"] for s in train}
    ):
        raise AssertionError("TRAIN must be exactly old90 root parents")
    lookup = {s["id"]: s for s in train}
    for row in raw:
        source = lookup[row["id"]]
        if row["source_hash"] != source["source_hash"]:
            raise AssertionError("TRAIN source hash mismatch")
        request = row["request"]
        for key in ("id", "history", "cue_seed"):
            if request[key] != source[key]:
                raise AssertionError("TRAIN cue/source mismatch")
        if object_sha(request) != row["request_sha256"]:
            raise AssertionError("TRAIN request payload changed")
    return sources, raw, read_json(OLD / "data/model.json")


def contrast_scores(law):
    if len(law) != 3 or any(not math.isfinite(p) or p < 0 for p in law):
        raise ValueError("three finite nonnegative probabilities required")
    if abs(math.fsum(law) - 1) > TOL:
        raise ValueError("probabilities must normalize")
    mean = math.fsum(u * p for u, p in zip(U, law))
    support = [abs(u - mean) for u, p in zip(U, law) if p > 0]
    scale = max(support)
    return tuple((u - mean) / scale if scale else 0.0 for u in U)


def joint_worlds(laws, rho):
    """Enumerate explicit joint worlds; no tanh posterior/closed H2 formula."""
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must be finite in [0,1)")
    parts = []
    for law in laws:
        score = contrast_scores(law)
        parts.append(
            [
                (u, p * (1 + rho * f), p * (1 - rho * f))
                for u, p, f in zip(U, law, score)
                if p > 0
            ]
        )
    worlds = []
    for choices in product(*parts):
        plus = math.prod(item[1] for item in choices)
        minus = math.prod(item[2] for item in choices)
        weight = (plus + minus) / 2
        if weight:
            worlds.append((tuple(item[0] for item in choices), weight))
    total = math.fsum(weight for _, weight in worlds)
    if not math.isfinite(total) or abs(total - 1) > 2e-12:
        raise ValueError("invalid joint normalization")
    return tuple((world, weight / total) for world, weight in worlds)


def observe_worlds(dist, candidate, value):
    matches = [(world, weight) for world, weight in dist if world[candidate] == value]
    total = math.fsum(weight for _, weight in matches)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("unsupported exact observation")
    return tuple((world, weight / total) for world, weight in matches)


def world_marginal(dist, candidate):
    return tuple(math.fsum(p for world, p in dist if world[candidate] == u) for u in U)


def gross_gain(law, b):
    return math.fsum(p * max(0.0, u - b) for p, u in zip(law, U))


def world_scores(dist, costs, b, remaining, horizon):
    entries = []
    for move in remaining:
        probs = world_marginal(dist, move)
        gain = gross_gain(probs, b)
        k1 = gain - costs[move]
        k2 = None
        if horizon == 2:
            continuation = 0.0
            for u, probability in zip(U, probs):
                if not probability:
                    continue
                posterior = observe_worlds(dist, move, u)
                after = max(b, u)
                continuation += probability * max(
                    [0.0]
                    + [
                        gross_gain(world_marginal(posterior, j), after) - costs[j]
                        for j in remaining
                        if j != move
                    ]
                )
            k2 = k1 + continuation
        entries.append(
            {
                "move": move,
                "probabilities": list(probs),
                "g": gain,
                "k1": k1,
                "k2": k2,
                "q": k2 if horizon == 2 else k1,
            }
        )
    return entries


def pick_score(entries, horizon):
    field = "k2" if horizon == 2 else "k1"
    if not entries:
        return None
    maximum = max(row[field] for row in entries)
    if maximum <= TOL:
        return None
    return min(row["move"] for row in entries if row[field] >= maximum - TOL)


def log_ratio(laws, outcomes, rho):
    if len(laws) != len(outcomes) or not 0 <= rho < 1:
        raise ValueError("invalid joint likelihood inputs")
    if rho == 0 or not laws:
        return 0.0
    positive = negative = 0.0
    for law, u in zip(laws, outcomes):
        if u not in U or law[U.index(u)] <= 0:
            raise ValueError("outcome outside fitted support")
        f = contrast_scores(law)[U.index(u)]
        positive += math.log1p(rho * f)
        negative += math.log1p(-rho * f)
    top = max(positive, negative)
    return top + math.log((math.exp(positive - top) + math.exp(negative - top)) / 2)


def assign_folds(sources):
    if len(sources) != 90 or len({s["id"] for s in sources}) != 90:
        raise ValueError("CV requires exactly90 unique TRAIN roots")
    if any(s["split"] != "TRAIN" for s in sources):
        raise ValueError("CAL/TEST cannot enter CV")
    folds = {}
    for ply in (4, 5, 6):
        group = sorted(
            (s for s in sources if s["ply"] == ply), key=lambda s: s["source_hash"]
        )
        if len(group) != 30:
            raise ValueError("CV requires30 TRAIN roots per ply")
        for index, source in enumerate(group):
            folds[source["id"]] = index // 10
    if len(folds) != 90:
        raise ValueError("unexpected TRAIN ply")
    return folds


def fit_fold(raw, folds, heldout):
    selected = [
        {
            "id": row["id"],
            "source_hash": row["source_hash"],
            "candidates": row["response"]["candidates"],
        }
        for row in raw
        if folds[row["id"]] != heldout
    ]
    if len(selected) != 60:
        raise ValueError("fold must fit60 whole roots")
    return C26.fit_model(selected)


def price_from_measurements(measurements):
    elapsed = sum(m["elapsed_ns"] for m in measurements)
    nodes = sum(m["solve_nodes"] for m in measurements)
    if not measurements or elapsed <= 0 or nodes <= 0:
        raise ValueError("price requires positive elapsed and nodes")
    for row in measurements:
        if (
            type(row["elapsed_ns"]) is not int
            or row["elapsed_ns"] < 0
            or type(row["solve_nodes"]) is not int
            or row["solve_nodes"] <= 0
        ):
            raise ValueError("invalid instrumented query measurement")
    scale = elapsed / nodes
    return {
        "exact_ns_per_node": scale,
        "lambda_nodes": LAMBDA,
        "lambda_time_ns": LAMBDA / scale,
        "sum_elapsed_ns": elapsed,
        "sum_solve_nodes": nodes,
        "queries": len(measurements),
        "scope": "instrumented nonterminal exact-query amortized wall ns/node",
    }


def run_request(source, bundle, seed, repeats=5, warmup=1):
    return {
        "id": source["id"],
        "history": list(source["history"]),
        "cue_seed": source["cue_seed"],
        "policy_seed": seed,
        "mode": "run",
        "additional": 2,
        "repeats": repeats,
        "warmup": warmup,
        "trace": True,
        "model": bundle["model"],
        "rho": bundle["rho"],
    }


def survey_request(source):
    return {
        "id": source["id"],
        "history": list(source["history"]),
        "cue_seed": source["cue_seed"],
        "policy_seed": 0,
        "mode": "survey",
        "additional": 2,
        "repeats": 5,
        "warmup": 0,
        "trace": False,
        "model": None,
        "rho": 0.0,
    }


def semantic_projection(run):
    return {
        key: value
        for key, value in run.items()
        if key not in TIMES and key not in ("repeat", "execution_slot")
    }


def same(actual, expected, where="root", tolerance=2e-12):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(where + ": expected mapping")  # noqa: TRY004
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(where + ": missing " + key)
            same(actual[key], value, where + "." + key, tolerance)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(where + ": sequence length mismatch")
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, where + f"[{index}]", tolerance)
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or abs(actual - expected) > tolerance
        ):
            raise AssertionError(f"{where}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{where}: {actual!r} != {expected!r}")


def assert_repeat_identity(runs):
    if not runs:
        raise AssertionError("missing repeated runs")
    first = semantic_projection(runs[0])
    for run in runs[1:]:
        other = semantic_projection(run)
        if set(other) != set(first):
            raise AssertionError("repeat semantic keys differ")
        same(other, first, "repeat identity")


def root_bootstrap(values, draws=10000, seed=2709):
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty root values required")
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(
        axis=1
    )
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(x) for x in np.quantile(bootstrap, [0.025, 0.975], method="linear")
        ],
        "roots": len(values),
        "draws": draws,
        "seed": seed,
        "rng": "numpy.PCG64",
        "numpy_version": np.__version__,
        "scope": "descriptive unstratified paired source-root resampling; no independent-tree coverage",
    }


def latent_field(laws, rho, observations):
    """Independent product-of-latent-component likelihood diagnostic."""
    plus = minus = 1.0
    seen = set()
    for move, u in observations:
        if move in seen:
            raise ValueError("duplicate exact observation")
        seen.add(move)
        law = laws[move]
        k = U.index(u)
        if law[k] <= 0:
            raise ValueError("unsupported observation")
        f = contrast_scores(law)[k]
        plus *= law[k] * (1 + rho * f)
        minus *= law[k] * (1 - rho * f)
    if plus <= 0 or minus <= 0:
        raise ValueError("unsupported latent component")
    return math.log(plus / minus)


def reference_context(source, model):
    board = C26.replay(source["history"], require_nonterminal=True)
    candidates = C26.cue_candidates(board, source["cue_seed"])
    by_move = {c["move"]: c for c in candidates}
    laws = [(1.0, 0.0, 0.0)] * 9
    costs, predictions = {}, []
    for c in candidates:
        if c["terminal_known"]:
            probabilities = [float(u == c["u"]) for u in U]
            predicted_nodes = None
        else:
            probabilities, predicted_nodes = C26.model_prediction(
                model, c["remaining"], c["counts"]
            )
            costs[c["move"]] = LAMBDA * predicted_nodes
        laws[c["move"]] = tuple(probabilities)
        predictions.append(
            {
                "move": c["move"],
                "remaining": c["remaining"],
                "counts": c["counts"],
                "terminal_known": c["terminal_known"],
                "probabilities": probabilities,
                "predicted_nodes": predicted_nodes,
                "observed_u": c["u"] if c["terminal_known"] else None,
            }
        )
    known = {c["move"]: c["u"] for c in candidates if c["terminal_known"]}
    observed = [
        {"move": move, "u": value, "kind": "terminal"}
        for move, value in sorted(known.items())
    ]
    nonterminal = [c for c in candidates if not c["terminal_known"]]
    initial_solve = None
    initial_nodes = 0
    if 1 not in known.values() and nonterminal:
        initial = max(nonterminal, key=lambda c: (c["rollout_mean"], -c["move"]))
        initial_solve = initial["move"]
        initial_nodes = initial["solve_nodes"]
        known[initial_solve] = initial["u"]
        observed.append({"move": initial_solve, "u": initial["u"], "kind": "initial"})
    if not known:
        raise AssertionError("initialization must leave an exact-known alternative")
    incumbent = max(known, key=lambda move: (known[move], -move))
    return {
        "candidates": candidates,
        "by_move": by_move,
        "laws": laws,
        "costs": costs,
        "predictions": predictions,
        "known": known,
        "known_observations": observed,
        "initial_solve_move": initial_solve,
        "initial_nodes": initial_nodes,
        "initial_move": incumbent,
        "initial_u": known[incumbent],
    }


def condition_history(dist, observations):
    seen = set()
    for move, value in observations:
        if move in seen:
            raise ValueError("duplicate history observation")
        dist = observe_worlds(dist, move, value)
        seen.add(move)
    return dist


def reference_run(source, model, rho, seed, policy):
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    ctx = reference_context(source, model)
    known = dict(ctx["known"])
    initial_obs = [(x["move"], x["u"]) for x in ctx["known_observations"]]
    observations = list(initial_obs)
    joint = condition_history(joint_worlds(ctx["laws"], rho), initial_obs)
    independent = condition_history(joint_worlds(ctx["laws"], 0), initial_obs)
    remaining = sorted(move for move in ctx["by_move"] if move not in known)
    initial_field = latent_field(ctx["laws"], rho, initial_obs)
    projection = None
    if policy == "JOINT_H2":
        dist = joint
    elif policy == "INITIAL_PROJECTED_H2":
        projected_laws = [world_marginal(joint, move) for move in range(9)]
        dist = joint_worlds(projected_laws, 0)
        projection = [
            {"move": move, "probabilities": list(projected_laws[move])}
            for move in remaining
        ]
    else:
        dist = independent
    b, incumbent = ctx["initial_u"], ctx["initial_move"]
    rng = C26.SplitMix64(source["cue_seed"] ^ C26.POLICY_XOR ^ seed)
    queries, decisions = [], []
    additional_nodes = 0
    predicted_nodes = 0.0
    while True:
        horizon = 2 - len(queries)
        before = b
        field = (
            latent_field(ctx["laws"], rho, observations)
            if policy == "JOINT_H2"
            else 0.0
        )
        forced = policy == "INITIAL_STOP" or horizon == 0 or not remaining or b == 1
        scores = (
            [] if forced else world_scores(dist, ctx["costs"], b, remaining, horizon)
        )
        for entry in scores:
            entry["predicted_nodes"] = ctx["costs"][entry["move"]] / LAMBDA
        chosen = None
        if not forced:
            chosen = (
                remaining[rng.index(len(remaining))]
                if policy == "RANDOM_TWO"
                else pick_score(scores, horizon)
            )
        queried_u, queried_nodes = None, 0
        if chosen is not None:
            child = ctx["by_move"][chosen]
            queried_u, queried_nodes = child["u"], child["solve_nodes"]
            if queried_u > b:
                b, incumbent = queried_u, chosen
            known[chosen] = queried_u
            queries.append(chosen)
            additional_nodes += queried_nodes
            predicted_nodes += ctx["costs"][chosen] / LAMBDA
            observations.append((chosen, queried_u))
            dist = observe_worlds(dist, chosen, queried_u)
        after_field = (
            latent_field(ctx["laws"], rho, observations)
            if policy == "JOINT_H2"
            else 0.0
        )
        decisions.append(
            {
                "b_before": before,
                "b_after": b,
                "remaining": list(remaining),
                "horizon": horizon,
                "log_odds_before": field,
                "log_odds_after": after_field,
                "candidates": scores,
                "chosen_move": chosen,
                "choice": "STOP" if chosen is None else "FOCUS",
                "queried_u": queried_u,
                "queried_nodes": queried_nodes,
            }
        )
        if chosen is None:
            break
        remaining.remove(chosen)
    cue_nodes = sum(c["cue_nodes"] for c in ctx["candidates"])
    return {
        "policy": policy,
        "initial_move": ctx["initial_move"],
        "initial_u": ctx["initial_u"],
        "initial_solve_move": ctx["initial_solve_move"],
        "selected_move": incumbent,
        "final_u": b,
        "cue_nodes": cue_nodes,
        "initial_nodes": ctx["initial_nodes"],
        "additional_nodes": additional_nodes,
        "charged_nodes": cue_nodes + ctx["initial_nodes"] + additional_nodes,
        "additional_queries": len(queries),
        "query_moves": queries,
        "predicted_additional_nodes": predicted_nodes,
        "predictions": ctx["predictions"],
        "known_observations": ctx["known_observations"],
        "initial_log_odds": initial_field
        if policy in ("JOINT_H2", "INITIAL_PROJECTED_H2")
        else 0.0,
        "projection": projection,
        "final_log_odds": latent_field(ctx["laws"], rho, observations)
        if policy == "JOINT_H2"
        else 0.0,
        "decisions": decisions,
    }


def validate_run_response(source, request, response, bundle, cache=None):
    same(response, {"schema": "c27-joint-v1", "mode": "run", "id": source["id"]})
    runs = response.get("runs", [])
    repeats = request["repeats"]
    if len(runs) != repeats * len(POLICIES):
        raise AssertionError("wrong policy/repeat count")
    expected_order = [
        (repeat, policy) for repeat in range(repeats) for policy in POLICIES
    ]
    if [(run["repeat"], run["policy"]) for run in runs] != expected_order:
        raise AssertionError("run serialization order or duplicate cell")
    groups = defaultdict(list)
    for run in runs:
        policy, repeat = run["policy"], run["repeat"]
        key = (source["id"], request["policy_seed"], policy, bundle["rho"])
        if cache is not None and key in cache:
            expected = cache[key]
        else:
            expected = reference_run(
                source, bundle["model"], bundle["rho"], request["policy_seed"], policy
            )
            if cache is not None:
                cache[key] = expected
        same(run, expected, source["id"] + "/" + policy)
        for field in TIMES:
            if type(run.get(field)) is not int or run[field] < 0:
                raise AssertionError(
                    "native timing must be nonnegative integer ns: " + field
                )
        if run["elapsed_ns"] <= 0:
            raise AssertionError("nonpositive native elapsed")
        phase_sum = sum(run[field] for field in TIMES[1:])
        if phase_sum != run["elapsed_ns"]:
            raise AssertionError("phase times do not partition full native elapsed")
        offset = (request["policy_seed"] + repeat) % len(POLICIES)
        expected_slot = (POLICIES.index(policy) - offset) % len(POLICIES)
        if run.get("execution_slot") != expected_slot:
            raise AssertionError("policy execution order was not rotated")
        if (
            len(run["query_moves"]) != len(set(run["query_moves"]))
            or len(run["query_moves"]) > 2
        ):
            raise AssertionError("duplicate/excess additional query")
        if run["final_u"] < run["initial_u"]:
            raise AssertionError("exact-known incumbent was lost")
        for pred in run["predictions"]:
            if not pred["terminal_known"] and pred["observed_u"] is not None:
                raise AssertionError("unqueried value in initial runtime predictions")
        groups[policy].append(run)
    for group in groups.values():
        assert_repeat_identity(group)
    return {
        "validated_runs": len(runs),
        "reference": "explicit joint-world conditioning and H2 enumeration",
    }


def validate_survey_response(source, response):
    same(response, {"schema": "c27-joint-v1", "mode": "survey", "id": source["id"]})
    expected = C26.cue_candidates(
        C26.replay(source["history"], require_nonterminal=True), source["cue_seed"]
    )
    if len(response.get("candidates", [])) != len(expected):
        raise AssertionError("survey candidate count changed")
    measurements = []
    for actual, candidate in zip(response["candidates"], expected):
        same(actual, candidate, source["id"] + "/survey")
        timing = actual["timings"]
        if candidate["terminal_known"]:
            if timing:
                raise AssertionError("public terminal cannot enter timing price")
        else:
            if len(timing) != 5 or [x["repeat"] for x in timing] != list(range(5)):
                raise AssertionError("five exact-query repeats required")
            for row in timing:
                if row["solve_nodes"] != candidate["solve_nodes"]:
                    raise AssertionError("exact solve node count changed")
                measurements.append(
                    {"id": source["id"], "move": candidate["move"], **row}
                )
    return measurements


def diagnostic_records(source, model, rho, seed, run):
    """Evaluate all three predictors on identical observed-history exposures."""
    ctx = reference_context(source, model)
    observations = [(x["move"], x["u"]) for x in ctx["known_observations"]]
    joint = condition_history(joint_worlds(ctx["laws"], rho), observations)
    base = joint_worlds(ctx["laws"], 0)
    projected = joint_worlds([world_marginal(joint, move) for move in range(9)], 0)
    records = []
    for index, decision in enumerate(run["decisions"]):
        stage = "initial" if index == 0 else "later"
        distributions = {
            "joint": joint,
            "product": base,
            "initial_projected": projected,
        }
        for move in decision["remaining"]:
            actual = ctx["by_move"][move]["u"]
            selected = decision["chosen_move"] == move
            realized = max(0.0, actual - decision["b_before"])
            for predictor, dist in distributions.items():
                probs = world_marginal(dist, move)
                probability = probs[U.index(actual)]
                if probability <= 0:
                    raise AssertionError(
                        "diagnostic true outcome outside model support"
                    )
                gain = gross_gain(probs, decision["b_before"])
                records.append(
                    {
                        "id": source["id"],
                        "ply": source["ply"],
                        "seed": seed,
                        "generating_policy": run["policy"],
                        "history_index": index,
                        "stage": stage,
                        "history": [{"move": m, "u": u} for m, u in observations],
                        "move": move,
                        "predictor": predictor,
                        "probabilities": list(probs),
                        "observed_offline_u": actual,
                        "b_before": decision["b_before"],
                        "log_loss": -math.log(probability),
                        "brier": math.fsum(
                            (p - float(u == actual)) ** 2 for p, u in zip(probs, U)
                        ),
                        "selected": selected,
                        "predicted_direct_gain": gain,
                        "realized_direct_gain": realized if selected else None,
                        "selected_gain_residual": gain - realized if selected else None,
                    }
                )
        chosen = decision["chosen_move"]
        if chosen is not None:
            value = decision["queried_u"]
            if any(m == chosen for m, _ in observations):
                raise AssertionError("duplicate diagnostic observation")
            observations.append((chosen, value))
            joint = observe_worlds(joint, chosen, value)
            base = observe_worlds(base, chosen, value)
            projected = observe_worlds(projected, chosen, value)
    return records


def complete_vector_diagnostic(source, model, rho):
    ctx = reference_context(source, model)
    pairs = [
        (ctx["laws"][c["move"]], c["u"])
        for c in ctx["candidates"]
        if not c["terminal_known"]
    ]
    laws, outcomes = zip(*pairs) if pairs else ((), ())
    product_log = math.fsum(math.log(law[U.index(u)]) for law, u in pairs)
    improvement = log_ratio(laws, outcomes, rho)
    return {
        "id": source["id"],
        "ply": source["ply"],
        "nonterminal_candidates": len(pairs),
        "product_nll": -product_log,
        "joint_nll": -product_log - improvement,
        "log_likelihood_improvement": improvement,
    }


def prepare(data_dir):
    """Publish public assignments only; no oracle invocation or new labels."""
    data_dir = Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError("prepare requires a new empty data directory")
    old_path = OLD / "data/public_sources.jsonl"
    if sha(old_path) != EXPECTED_OLD["public_sources.jsonl"]:
        raise AssertionError("old source exclusions changed")
    old_sources = list(read_rows(old_path))
    excluded = {
        C26.canonicalize(C26.replay(s["history"]), s["history"])[0] for s in old_sources
    }
    if len(excluded) != 300:
        raise AssertionError("must exclude all300 old roots")
    groups = {4: {}, 5: {}, 6: {}}
    for board, history in C26.enumerate_reachable().items():
        if len(history) not in groups or C26.terminal(board):
            continue
        board, history = C26.canonicalize(board, history)
        if board in excluded:
            continue
        previous = groups[len(history)].get(board)
        if previous is None or history < previous:
            groups[len(history)][board] = history
    if {k: len(v) for k, v in groups.items()} != {4: 8, 5: 53, 6: 83}:
        raise AssertionError("new public source frame changed")
    sources = []
    for ply, group in sorted(groups.items()):
        ranked = sorted(
            (
                hashlib.sha256(
                    (CONTRACT["split_salt"] + C26.board_text(board)).encode()
                ).hexdigest(),
                board,
                history,
            )
            for board, history in group.items()
        )
        for index, (digest, board, history) in enumerate(ranked):
            seed = hashlib.sha256(
                (CONTRACT["cue_salt"] + C26.board_text(board)).encode()
            ).hexdigest()
            sources.append(
                {
                    "id": f"C27-p{ply}-{digest[:16]}",
                    "source_hash": digest,
                    "ply": ply,
                    "board": list(board),
                    "history": list(history),
                    "cue_seed": int(seed[:16], 16),
                    "split": "CAL"
                    if index < CONTRACT["new_cal_by_ply"][str(ply)]
                    else "TEST",
                }
            )
    if Counter(s["split"] for s in sources) != {"CAL": 24, "TEST": 120}:
        raise AssertionError("wrong frozen split counts")
    write_rows(data_dir / "public_sources.jsonl", sources)
    manifest = {
        "schema": "c27-public-source-v1",
        "provenance": provenance(),
        "public_sources_sha256": sha(data_dir / "public_sources.jsonl"),
        "excluded_c26_public_sha256": EXPECTED_OLD["public_sources.jsonl"],
        "counts_by_split_ply": {
            split: dict(Counter(s["ply"] for s in sources if s["split"] == split))
            for split in ("CAL", "TEST")
        },
        "source_count": len(sources),
        "public_assignment_before_new_labels": True,
        "newness_scope": "new campaign roots excluding all300 C26 roots; descendants and semantic oracle may overlap",
    }
    write_json(data_dir / "source_manifest.json", manifest)
    return manifest


def load_public(data_dir):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "source_manifest.json")
    if sha(data_dir / "public_sources.jsonl") != manifest["public_sources_sha256"]:
        raise AssertionError("public assignment changed")
    if manifest["provenance"]["experiment_contract_sha256"] != sha(
        HERE / "EXPERIMENT_CONTRACT.json"
    ):
        raise AssertionError("source contract changed")
    sources = list(read_rows(data_dir / "public_sources.jsonl"))
    if len(sources) != 144 or len({s["id"] for s in sources}) != 144:
        raise AssertionError("duplicate or missing new source")
    for split, expected in (
        ("CAL", {4: 2, 5: 8, 6: 14}),
        ("TEST", {4: 6, 5: 45, 6: 69}),
    ):
        if dict(Counter(s["ply"] for s in sources if s["split"] == split)) != expected:
            raise AssertionError("frozen source strata mismatch")
    return sources, manifest


def fit(data_dir):
    data_dir = Path(data_dir)
    load_public(data_dir)
    sources, raw, _ = old_artifacts()
    training = [s for s in sources if s["split"] == "TRAIN"]
    folds = assign_folds(training)
    scores = []
    fold_evidence = []
    for fold in range(3):
        model, counts = fit_fold(raw, folds, fold)
        fold_evidence.append(
            {
                "fold": fold,
                "training_sources": counts["sources"],
                "training_sources_sha256": object_sha(counts["sources"]),
                "model": model,
                "model_payload_sha256": object_sha(model),
                "nonterminal_candidates": counts["nonterminal_candidates"],
            }
        )
        training_ids = {s["id"] for s in counts["sources"]}
        for row in raw:
            if folds[row["id"]] != fold:
                continue
            if row["id"] in training_ids:
                raise AssertionError("held-out root entered fitting")
            candidates = [
                c for c in row["response"]["candidates"] if not c["terminal_known"]
            ]
            laws = [
                C26.model_prediction(model, c["remaining"], c["counts"])[0]
                for c in candidates
            ]
            outcomes = [c["u"] for c in candidates]
            scores.append(
                {
                    "id": row["id"],
                    "fold": fold,
                    "source_hash": row["source_hash"],
                    "nonterminal_candidates": len(candidates),
                    "scores": [
                        {
                            "rho": rho,
                            "log_likelihood_improvement": log_ratio(
                                laws, outcomes, rho
                            ),
                        }
                        for rho in CONTRACT["rho_grid"]
                    ],
                }
            )
    if len(scores) != 90 or len({r["id"] for r in scores}) != 90:
        raise AssertionError("CV did not score each TRAIN root exactly once")
    means = [
        {
            "rho": rho,
            "mean_log_likelihood_improvement": statistics.mean(
                row["scores"][i]["log_likelihood_improvement"] for row in scores
            ),
        }
        for i, rho in enumerate(CONTRACT["rho_grid"])
    ]
    maximum = max(row["mean_log_likelihood_improvement"] for row in means)
    selected = next(
        row for row in means if row["mean_log_likelihood_improvement"] >= maximum - TOL
    )
    selection = {
        "schema": "c27-rho-selection-v1",
        "provenance": provenance(),
        "rho": selected["rho"],
        "grid_scores": means,
        "fold_assignment": folds,
        "fold_evidence": fold_evidence,
        "root_scores": scores,
        "base_model_sha256": EXPECTED_OLD["model.json"],
        "training_raw_sha256": EXPECTED_OLD["training_raw.jsonl.gz"],
        "selection": "three-fold heldout-root mean logR; each fold independently refits60; smallest-rho tie1e-12",
        "final_base": "unchanged original full90 C26 model",
        "scope": "TRAIN-only selection, not CAL/TEST evidence",
    }
    write_json(data_dir / "model_selection.json", selection)
    return {"rho": selection["rho"], "grid_scores": means, "scored_roots": len(scores)}


def price(data_dir, binary):
    data_dir = Path(data_dir)
    load_public(data_dir)
    read_json(data_dir / "model_selection.json")
    sources, _, _ = old_artifacts()
    selected = [s for s in sources if s["split"] == "TRAIN"]
    records, measurements = [], []
    raw_path = data_dir / "price_raw.jsonl.gz"
    if raw_path.exists() or (data_dir / "price.json").exists():
        raise FileExistsError("price evidence already exists")
    with (
        C26.NativeClient(binary) as client,
        gzip.open(raw_path, "xt", encoding="utf-8") as handle,
    ):
        for source in selected:
            request = survey_request(source)
            response, transport = client.request(request)
            measurements.extend(validate_survey_response(source, response))
            record = {
                "id": source["id"],
                "source_hash": source["source_hash"],
                "request": request,
                "request_sha256": object_sha(request),
                "response": response,
                "transport": transport,
            }
            records.append(record)
            handle.write(canonical(record) + "\n")
    if len(measurements) != CONTRACT["price_query_count"]:
        raise AssertionError("TRAIN price must contain exactly1480 queries")
    result = price_from_measurements(measurements)
    result.update(
        {
            "schema": "c27-train-price-v1",
            "provenance": provenance(),
            "binary_sha256": client.binary_sha256,
            "native_exit_code": client.exit_code,
            "training_roots": len(records),
            "price_repeats": 5,
            "price_raw_sha256": sha(raw_path),
            "source_ids_sha256": object_sha([r["id"] for r in records]),
            "native_roundtrip_us": client.roundtrip_us,
            "process_launch_us": client.launch_us,
        }
    )
    write_json(data_dir / "price.json", result)
    return result


def frozen_source_paths():
    return [
        Path(__file__).resolve(),
        ROOT / "src/bin/quartz_c27_joint_posterior.rs",
        ROOT / "tests/test_c27_joint_posterior.py",
        HERE / "CODE_CONTRACT.md",
        HERE / "EXPERIMENT_CONTRACT.json",
        HERE / "NUMERICAL_DESIGN.md",
        HERE / "IMPLEMENTATION_PLAN.md",
        DOC / "theory/exact_joint_oracle.py",
        C26_PATH,
        ROOT / "src/game.rs",
        ROOT / "src/games/tictactoe.rs",
    ]


def freeze(data_dir, binary):
    data_dir = Path(data_dir)
    load_public(data_dir)
    _, _, model = old_artifacts()
    selected = read_json(data_dir / "model_selection.json")
    pricing = read_json(data_dir / "price.json")
    if selected["base_model_sha256"] != EXPECTED_OLD["model.json"]:
        raise AssertionError("deployment base model changed")
    if sha(data_dir / "price_raw.jsonl.gz") != pricing["price_raw_sha256"]:
        raise AssertionError("price raw changed")
    if sha(binary) != pricing["binary_sha256"]:
        raise AssertionError("native binary changed after price")
    data_names = [
        "public_sources.jsonl",
        "source_manifest.json",
        "model_selection.json",
        "price_raw.jsonl.gz",
        "price.json",
    ]
    bundle = {
        "schema": "c27-frozen-bundle-v1",
        "provenance": provenance(),
        "rho": selected["rho"],
        "model": model,
        "base_model_sha256": EXPECTED_OLD["model.json"],
        "lambda_nodes": LAMBDA,
        "lambda_time_ns": pricing["lambda_time_ns"],
        "exact_ns_per_node": pricing["exact_ns_per_node"],
        "binary_sha256": sha(binary),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in frozen_source_paths()
        },
        "data_sha256": {name: sha(data_dir / name) for name in data_names},
        "contract": CONTRACT,
    }
    write_json(data_dir / "frozen_bundle.json", bundle)
    return {
        "rho": bundle["rho"],
        "lambda_time_ns": bundle["lambda_time_ns"],
        "bundle_sha256": sha(data_dir / "frozen_bundle.json"),
        "binary_sha256": bundle["binary_sha256"],
    }


def load_frozen(data_dir, binary=None):
    data_dir = Path(data_dir)
    bundle = read_json(data_dir / "frozen_bundle.json")
    for path, expected in bundle["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise AssertionError("frozen source changed: " + path)
    for name, expected in bundle["data_sha256"].items():
        if sha(data_dir / name) != expected:
            raise AssertionError("frozen data changed: " + name)
    if binary is not None and sha(binary) != bundle["binary_sha256"]:
        raise AssertionError("frozen binary changed")
    old_artifacts()
    if bundle["base_model_sha256"] != EXPECTED_OLD["model.json"]:
        raise AssertionError("wrong deployment model identity")
    same(
        bundle["model"],
        read_json(OLD / "data/model.json"),
        "unchanged deployment model",
    )
    if bundle["contract"] != CONTRACT:
        raise AssertionError("experiment contract drift")
    return bundle


def campaign(data_dir, binary, split, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("campaign output directory must be new and empty")
    bundle = load_frozen(data_dir, binary)
    sources, manifest = load_public(data_dir)
    selected = [s for s in sources if s["split"] == split]
    if len(selected) != (24 if split == "CAL" else 120):
        raise AssertionError("wrong complete campaign source count")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl.gz"
    oracle_path = output_dir / "offline_oracle.jsonl.gz"
    request_count = validated = 0
    try:
        with (
            C26.NativeClient(binary) as client,
            gzip.open(raw_path, "xt", encoding="utf-8") as raw_handle,
            gzip.open(oracle_path, "xt", encoding="utf-8") as truth_handle,
        ):
            for source in selected:
                # Labels are created only here, after the bundle is frozen.
                truth_handle.write(canonical(C26.truth_record(source)) + "\n")
                cache = {}
                for seed in CONTRACT["policy_seeds"]:
                    request = run_request(source, bundle, seed)
                    response, transport = client.request(request)
                    check = validate_run_response(
                        source, request, response, bundle, cache
                    )
                    record = {
                        "id": source["id"],
                        "ply": source["ply"],
                        "split": split,
                        "seed": seed,
                        "request": request,
                        "request_sha256": object_sha(request),
                        "response": response,
                        "transport": transport,
                    }
                    raw_handle.write(canonical(record) + "\n")
                    request_count += 1
                    validated += check["validated_runs"]
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                "provenance": provenance(),
                "type": type(error).__name__,
                "message": str(error),
                "completed_requests": request_count,
                "status": "INVALID_INCOMPLETE; no scientific admission",
            },
        )
        raise
    result = {
        "schema": "c27-campaign-v1",
        "provenance": provenance(),
        "split": split,
        "source_count": len(selected),
        "requests": request_count,
        "validated_runs": validated,
        "policy_seeds": CONTRACT["policy_seeds"],
        "repeats": 5,
        "warmup": 1,
        "source_ids": [s["id"] for s in selected],
        "public_sources_sha256": manifest["public_sources_sha256"],
        "frozen_bundle_sha256": sha(data_dir / "frozen_bundle.json"),
        "binary_sha256": client.binary_sha256,
        "native_exit_code": client.exit_code,
        "native_roundtrip_us": client.roundtrip_us,
        "process_launch_us": client.launch_us,
        "files": {
            "raw.jsonl.gz": sha(raw_path),
            "offline_oracle.jsonl.gz": sha(oracle_path),
        },
        "timing_scope": "full native controller including trace; decoding/serialization/IPC/startup excluded",
        "labels_runtime_input": False,
    }
    write_json(output_dir / "campaign_manifest.json", result)
    return result


def load_campaign(data_dir, output_dir):
    output_dir = Path(output_dir)
    bundle = load_frozen(data_dir)
    manifest = read_json(output_dir / "campaign_manifest.json")
    if sha(Path(data_dir) / "frozen_bundle.json") != manifest["frozen_bundle_sha256"]:
        raise AssertionError("campaign used a different bundle")
    if (
        manifest["binary_sha256"] != bundle["binary_sha256"]
        or manifest["native_exit_code"] != 0
    ):
        raise AssertionError("native identity/exit mismatch")
    for name, expected in manifest["files"].items():
        if sha(output_dir / name) != expected:
            raise AssertionError("campaign evidence changed: " + name)
    sources = [s for s in load_public(data_dir)[0] if s["split"] == manifest["split"]]
    if manifest["source_ids"] != [s["id"] for s in sources]:
        raise AssertionError("campaign source membership changed")
    records = list(read_rows(output_dir / "raw.jsonl.gz"))
    expected = {(s["id"], seed) for s in sources for seed in CONTRACT["policy_seeds"]}
    actual = [(r["id"], r["seed"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise AssertionError("campaign missing or duplicate root/seed request")
    truth = list(read_rows(output_dir / "offline_oracle.jsonl.gz"))
    if len(truth) != len(sources) or {r["id"] for r in truth} != {
        s["id"] for s in sources
    }:
        raise AssertionError("oracle source cardinality mismatch")
    return bundle, manifest, sources, records, {r["id"]: r for r in truth}


def verify(data_dir, output_dir):
    bundle, manifest, sources, records, truth = load_campaign(data_dir, output_dir)
    by_id = {s["id"]: s for s in sources}
    checked = 0
    cache = {}
    for record in records:
        source = by_id[record["id"]]
        expected_request = run_request(source, bundle, record["seed"])
        if record["request"] != expected_request or record[
            "request_sha256"
        ] != object_sha(expected_request):
            raise AssertionError(
                "runtime request is not the frozen public-only request"
            )
        if record["ply"] != source["ply"] or record["split"] != source["split"]:
            raise AssertionError("runtime source metadata changed")
        checked += validate_run_response(
            source, expected_request, record["response"], bundle, cache
        )["validated_runs"]
    for source in sources:
        same(truth[source["id"]], C26.truth_record(source), source["id"] + "/offline")
    return {
        "schema": "c27-independent-verification-v1",
        "split": manifest["split"],
        "roots": len(sources),
        "requests": len(records),
        "validated_runs": checked,
        "source_and_bundle_identities": "verified",
        "method": "independent explicit joint worlds, legal replay, exact minimax/sign/node checks",
        "invalid_support_or_semantic_failures": 0,
        "scope": "execution semantics; does not establish empirical superiority",
    }


def aggregate_records(records, truth, bundle):
    seed_rows, dispersion = [], []
    for record in records:
        by_policy = defaultdict(list)
        for run in record["response"]["runs"]:
            by_policy[run["policy"]].append(run)
        if set(by_policy) != set(POLICIES):
            raise AssertionError("missing policy cell")
        for policy in POLICIES:
            runs = sorted(by_policy[policy], key=lambda row: row["repeat"])
            if len(runs) != 5 or [r["repeat"] for r in runs] != list(range(5)):
                raise AssertionError("exactly five measured repeats required")
            assert_repeat_identity(runs)
            first = runs[0]
            metrics = {
                "utility": first["final_u"],
                "initial_utility": first["initial_u"],
                "utility_gain": first["final_u"] - first["initial_u"],
                "regret": truth[record["id"]]["root_u"] - first["final_u"],
                **{
                    key: first[key]
                    for key in (
                        "charged_nodes",
                        "cue_nodes",
                        "initial_nodes",
                        "additional_nodes",
                        "additional_queries",
                        "predicted_additional_nodes",
                    )
                },
                **{key: statistics.median(run[key] for run in runs) for key in TIMES},
            }
            metrics["net_nodes"] = (
                metrics["utility"] - LAMBDA * metrics["charged_nodes"]
            )
            metrics["net_wall"] = (
                metrics["utility"] - bundle["lambda_time_ns"] * metrics["elapsed_ns"]
            )
            metadata = {
                "id": record["id"],
                "ply": record["ply"],
                "seed": record["seed"],
                "policy": policy,
            }
            seed_rows.append({**metadata, "metrics": metrics})
            dispersion.append(
                {
                    **metadata,
                    "timing_ns": {
                        field: {
                            "samples": [r[field] for r in runs],
                            "min": min(r[field] for r in runs),
                            "median": statistics.median(r[field] for r in runs),
                            "max": max(r[field] for r in runs),
                            "population_stdev": statistics.pstdev(
                                r[field] for r in runs
                            ),
                        }
                        for field in TIMES
                    },
                    "semantic_repeat_identity": True,
                }
            )
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["id"], row["policy"])].append(row)
    roots = []
    for (identity, policy), rows in sorted(grouped.items()):
        if len(rows) != 4 or {r["seed"] for r in rows} != set(CONTRACT["policy_seeds"]):
            raise AssertionError("exactly four policy seeds per root required")
        roots.append(
            {
                "id": identity,
                "ply": rows[0]["ply"],
                "policy": policy,
                "metrics": {
                    key: statistics.mean(row["metrics"][key] for row in rows)
                    for key in rows[0]["metrics"]
                },
            }
        )
    return seed_rows, roots, dispersion


def comparison(root_rows, policy, baseline):
    lookup = {(row["id"], row["policy"]): row for row in root_rows}
    identities = sorted({row["id"] for row in root_rows if row["policy"] == policy})
    metrics = (
        "net_wall",
        "utility",
        "elapsed_ns",
        "net_nodes",
        "charged_nodes",
        "additional_nodes",
        "additional_queries",
        "regret",
    )
    differences = []
    for identity in identities:
        a, b = lookup[(identity, policy)], lookup[(identity, baseline)]
        differences.append(
            {
                "id": identity,
                "ply": a["ply"],
                "metrics": {
                    key: a["metrics"][key] - b["metrics"][key] for key in metrics
                },
            }
        )
    summaries = {
        key: root_bootstrap([row["metrics"][key] for row in differences])
        for key in metrics
    }
    signs = Counter(
        "positive"
        if r["metrics"]["net_wall"] > 0
        else "negative"
        if r["metrics"]["net_wall"] < 0
        else "zero"
        for r in differences
    )
    strata = {}
    for ply in sorted({row["ply"] for row in differences}):
        group = [row for row in differences if row["ply"] == ply]
        strata[str(ply)] = {
            "roots": len(group),
            "mean_difference": {
                key: statistics.mean(row["metrics"][key] for row in group)
                for key in metrics
            },
        }
    return {
        "policy": policy,
        "baseline": baseline,
        "direction": "policy minus baseline",
        "metrics": summaries,
        "root_signs_net_wall": dict(signs),
        "ply_strata": strata,
        "per_root": differences,
    }


def summarize_diagnostics(records):
    groups = defaultdict(list)
    matched = defaultdict(dict)
    for row in records:
        groups[(row["generating_policy"], row["stage"], row["predictor"])].append(row)
        key = (
            row["id"],
            row["seed"],
            row["generating_policy"],
            row["history_index"],
            row["move"],
        )
        if row["predictor"] in matched[key]:
            raise AssertionError("duplicate diagnostic predictor exposure")
        matched[key][row["predictor"]] = row
    summaries = []
    bins = (0, 0.05, 0.10, 0.20, 0.50, 1.01)
    for (policy, stage, predictor), rows in sorted(groups.items()):
        chosen = [row for row in rows if row["selected"]]
        reliability = []
        for low, high in pairwise(bins):
            group = [
                row for row in chosen if low <= row["predicted_direct_gain"] < high
            ]
            reliability.append(
                {
                    "bin": [low, high],
                    "selected_queries": len(group),
                    "mean_predicted_gain": statistics.mean(
                        r["predicted_direct_gain"] for r in group
                    )
                    if group
                    else None,
                    "mean_realized_direct_gain": statistics.mean(
                        r["realized_direct_gain"] for r in group
                    )
                    if group
                    else None,
                }
            )
        summaries.append(
            {
                "generating_policy": policy,
                "stage": stage,
                "predictor": predictor,
                "candidate_history_exposures": len(rows),
                "roots": len({r["id"] for r in rows}),
                "histories": len(
                    {(r["id"], r["seed"], r["history_index"]) for r in rows}
                ),
                "mean_log_loss": statistics.mean(r["log_loss"] for r in rows),
                "mean_brier": statistics.mean(r["brier"] for r in rows),
                "selected_queries": len(chosen),
                "selected_mean_signed_residual": statistics.mean(
                    r["selected_gain_residual"] for r in chosen
                )
                if chosen
                else None,
                "selected_gain_mae": statistics.mean(
                    abs(r["selected_gain_residual"]) for r in chosen
                )
                if chosen
                else None,
                "selected_gain_rmse": math.sqrt(
                    statistics.mean(r["selected_gain_residual"] ** 2 for r in chosen)
                )
                if chosen
                else None,
                "gain_reliability": reliability,
            }
        )
    pair_groups = defaultdict(list)
    for predictors in matched.values():
        if set(predictors) != {"joint", "product", "initial_projected"}:
            raise AssertionError("unmatched conditional prediction comparison")
        joint, base, projected = (
            predictors[key] for key in ("joint", "product", "initial_projected")
        )
        pair_groups[(joint["generating_policy"], joint["stage"])].append(
            {
                "id": joint["id"],
                "joint_minus_product_log_loss": joint["log_loss"] - base["log_loss"],
                "joint_minus_product_brier": joint["brier"] - base["brier"],
                "joint_minus_projected_log_loss": joint["log_loss"]
                - projected["log_loss"],
                "joint_minus_projected_brier": joint["brier"] - projected["brier"],
            }
        )
    paired = []
    for (policy, stage), rows in sorted(pair_groups.items()):
        paired.append(
            {
                "generating_policy": policy,
                "stage": stage,
                "exposures": len(rows),
                "roots": len({r["id"] for r in rows}),
                **{
                    key: statistics.mean(row[key] for row in rows)
                    for key in rows[0]
                    if key != "id"
                },
            }
        )
    return {
        "by_generating_policy_stage_predictor": summaries,
        "matched_history_differences": paired,
        "scope": "each predictor evaluated on the same candidate/history exposures; lower log loss/Brier is better",
        "selection_limit": "different generating-policy exposure means are not a causal predictive-improvement contrast",
        "gain_target": "observed direct incumbent improvement only; excludes subsequent-query information value and is not a full K2 target",
        "sampling_limit": "exposure counts include repeated policy seeds; they are not independent root counts",
    }


def analyze(data_dir, output_dir):
    output_dir = Path(output_dir)
    bundle, manifest, sources, records, truth = load_campaign(data_dir, output_dir)
    seed_rows, root_rows, dispersion = aggregate_records(records, truth, bundle)
    by_id = {source["id"]: source for source in sources}
    diagnostics = []
    for record in records:
        source = by_id[record["id"]]
        for run in record["response"]["runs"]:
            if run["repeat"] == 0:
                diagnostics.extend(
                    diagnostic_records(
                        source, bundle["model"], bundle["rho"], record["seed"], run
                    )
                )
    vector_rows = [
        complete_vector_diagnostic(source, bundle["model"], bundle["rho"])
        for source in sources
    ]
    contrasts = [
        comparison(root_rows, "JOINT_H2", "PRODUCT_H2"),
        comparison(root_rows, "JOINT_H2", "INITIAL_PROJECTED_H2"),
        comparison(root_rows, "JOINT_H2", "INITIAL_STOP"),
        comparison(root_rows, "PRODUCT_H2", "INITIAL_STOP"),
        comparison(root_rows, "RANDOM_TWO", "INITIAL_STOP"),
        comparison(root_rows, "JOINT_H2", "RANDOM_TWO"),
    ]
    primary = contrasts[0]
    lower = primary["metrics"]["net_wall"]["ci95"][0]
    quality_lower = primary["metrics"]["utility"]["ci95"][0]
    semantic_count = len(sources) * 4 * 5 * 5
    if manifest["validated_runs"] != semantic_count:
        raise AssertionError("campaign did not independently validate every run")
    policy_summaries = []
    for policy in POLICIES:
        rows = [row for row in root_rows if row["policy"] == policy]
        policy_summaries.append(
            {
                "policy": policy,
                "roots": len(rows),
                "mean_metrics": {
                    key: statistics.mean(row["metrics"][key] for row in rows)
                    for key in rows[0]["metrics"]
                },
            }
        )
    summary = {
        "schema": "c27-analysis-v1",
        "provenance": provenance(),
        "split": manifest["split"],
        "source_count": len(sources),
        "source_ply_counts": dict(Counter(s["ply"] for s in sources)),
        "rho": bundle["rho"],
        "lambda_time_ns": bundle["lambda_time_ns"],
        "primary": primary,
        "secondary_contrasts": contrasts[1:],
        "policy_summaries": policy_summaries,
        "conditional_diagnostics": summarize_diagnostics(diagnostics),
        "joint_vector_log_likelihood": {
            "roots": len(vector_rows),
            "mean_product_nll": statistics.mean(r["product_nll"] for r in vector_rows),
            "mean_joint_nll": statistics.mean(r["joint_nll"] for r in vector_rows),
            "mean_log_likelihood_improvement": statistics.mean(
                r["log_likelihood_improvement"] for r in vector_rows
            ),
        },
        "gate": {
            "evaluated_for_admission": manifest["split"] == "TEST",
            "primary_descriptive_lower": lower,
            "primary_lower_gt_zero": lower > 0,
            "quality_descriptive_lower": quality_lower,
            "quality_lower_ge_minus_002": quality_lower >= -0.02,
            "invalid_support_leakage_semantic_failures": 0,
            "empirical_admission": manifest["split"] == "TEST"
            and lower > 0
            and quality_lower >= -0.02,
            "meaning": "frozen finite-source operational admission; not independent-tree/population-wide inference",
            "calibration_cannot_rescue_primary": True,
        },
        "native_timing_scope": manifest["timing_scope"],
        "uncertainty_scope": "10000 paired root resamples seed2709, equal root weighting over actual source mixture",
        "known_dependence": "shared descendants and earlier all-state semantic oracle; new campaign roots only",
        "root_aggregation": "median5 timings per root/seed/policy then mean4 seeds",
        "semantic_validation": {
            "runs": semantic_count,
            "reference": "explicit full joint worlds and offline exact game oracle",
        },
        "files": {},
    }
    outputs = [
        ("seed_summaries.jsonl.gz", seed_rows),
        ("root_summaries.jsonl.gz", root_rows),
        ("paired_root_differences.jsonl.gz", primary["per_root"]),
        ("timing_dispersion.jsonl.gz", dispersion),
        ("conditional_diagnostics.jsonl.gz", diagnostics),
        ("joint_vector_diagnostics.jsonl.gz", vector_rows),
    ]
    for name, rows in outputs:
        write_rows(output_dir / name, rows)
        summary["files"][name] = sha(output_dir / name)
    write_json(output_dir / "analysis.json", summary)
    return {
        "split": manifest["split"],
        "roots": len(sources),
        "primary": {
            key: primary["metrics"][key]
            for key in ("net_wall", "utility", "elapsed_ns")
        },
        "gate": summary["gate"],
        "analysis_sha256": sha(output_dir / "analysis.json"),
    }


def integration(binary, limit=3):
    """Bounded old-TRAIN integration only; no fitting, new cohorts or prices."""
    if not 1 <= limit <= 6:
        raise ValueError("integration limit must be1..6 old TRAIN roots")
    sources, _, model = old_artifacts()
    train = {
        ply: [s for s in sources if s["split"] == "TRAIN" and s["ply"] == ply]
        for ply in (4, 5, 6)
    }
    selected = [train[ply][rank] for rank in range(2) for ply in (4, 5, 6)][:limit]
    checked, transport_rows = 0, []
    with C26.NativeClient(binary) as client:
        invalid = [
            "{broken",
            canonical(
                {
                    **run_request(selected[0], {"model": model, "rho": 0.8}, 0, 1, 0),
                    "offline_u": [1],
                }
            ),
            canonical(
                {
                    **run_request(selected[0], {"model": model, "rho": 0.8}, 0, 1, 0),
                    "rho": 1,
                }
            ),
            canonical(
                {
                    **run_request(selected[0], {"model": model, "rho": 0.8}, 0, 1, 0),
                    "additional": 3,
                }
            ),
            canonical(
                {
                    **run_request(selected[0], {"model": model, "rho": 0.8}, 0, 1, 0),
                    "history": [0, 0],
                }
            ),
        ]
        for line in invalid:
            response, _ = client.send_raw(line)
            if "error" not in response:
                raise AssertionError("native accepted malformed/offline/invalid input")
        for source in selected:
            for rho in (0.0, 0.8, 0.9):
                bundle = {"model": model, "rho": rho}
                request = run_request(source, bundle, 3, repeats=5, warmup=1)
                response, transport = client.request(request)
                result = validate_run_response(source, request, response, bundle)
                checked += result["validated_runs"]
                transport_rows.append({"id": source["id"], "rho": rho, **transport})
    return {
        "schema": "c27-targeted-integration-v1",
        "provenance": provenance(),
        "old_train_roots": [s["id"] for s in selected],
        "validated_runs": checked,
        "invalid_inputs_rejected": len(invalid),
        "recovery": True,
        "native_exit_code": client.exit_code,
        "binary_sha256": client.binary_sha256,
        "native_roundtrip_us": client.roundtrip_us,
        "transport": transport_rows,
        "scope": "fixed old TRAIN subset and predefined rho0/.8/.9; no rho fit/price/new-source execution",
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fit", "price", "freeze", "campaign", "verify", "analyze"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data-dir", type=Path, default=HERE / "data")
        if name in ("price", "freeze", "campaign"):
            cmd.add_argument("--binary", type=Path, required=True)
        if name in ("campaign", "verify", "analyze"):
            cmd.add_argument("--output-dir", type=Path, required=True)
        if name == "campaign":
            cmd.add_argument("--split", choices=("CAL", "TEST"), required=True)
        cmd.add_argument(
            "--report", type=Path, help="optional new immutable JSON execution receipt"
        )
    cmd = sub.add_parser("integration")
    cmd.add_argument("--binary", type=Path, required=True)
    cmd.add_argument("--limit", type=int, default=3)
    cmd.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.data_dir)
    elif args.command == "fit":
        result = fit(args.data_dir)
    elif args.command == "price":
        result = price(args.data_dir, args.binary)
    elif args.command == "freeze":
        result = freeze(args.data_dir, args.binary)
    elif args.command == "campaign":
        result = campaign(args.data_dir, args.binary, args.split, args.output_dir)
    elif args.command == "verify":
        result = verify(args.data_dir, args.output_dir)
    elif args.command == "analyze":
        result = analyze(args.data_dir, args.output_dir)
    else:
        result = integration(args.binary, args.limit)
    if args.report:
        write_json(
            args.report,
            {"command": args.command, "provenance": provenance(), "result": result},
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    cli()
