#!/usr/bin/env python3
"""Exact sampled game-outcome utility of real gravity-Connect4 QUARTZ MCTS.

Consumes immutable external Pascal Pons exact action scores only *after*
MCTS evaluation. All computed regrets are sign(WDL)-utility comparisons,
not Bayesian expected value of computation, model strength or Elo.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


BUDGETS = [8, 16, 32, 64]


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def score_sign(score: int) -> int:
    if type(score) is not int or score == -1000 or not -21 <= score <= 21:
        raise ValueError("invalid/sentinel Connect4 solver score")
    return (score > 0) - (score < 0)


def board_from_columns(history: str) -> list[int]:
    """Independent simple row-major-bottom-up gravity encoder."""
    board=[0]*42
    height=[0]*7
    for ply, char in enumerate(history):
        if char not in "1234567":
            raise ValueError("invalid Connect4 history character")
        col=int(char)-1
        if height[col]>=6:
            raise ValueError("overfull Connect4 column")
        board[height[col]*7+col]=1 if ply%2==0 else -1
        height[col]+=1
    return board


def join_root(exact: dict, observed: dict, *, expected_board: list[int]) -> dict:
    """Join actual QUARTZ action with external exact 7-column WDL scores.

    `observed` must never carry oracle scores into Rust MCTS evaluation.
    """
    if exact.get("status")!="EXACT_EXTERNAL_SOLVER":
        raise ValueError("oracle action-value row not admitted")
    history=str(exact.get("sequence",""))
    if observed.get("history")!=history:
        raise ValueError("root history/solver input mismatch")
    scores=exact.get("scores")
    if not isinstance(scores,list) or len(scores)!=7:
        raise ValueError("oracle seven-action vector missing")
    legal=[i for i in range(7) if expected_board[35+i]==0]
    for i,s in enumerate(scores):
        if (i in legal) != (s!=-1000):
            raise ValueError("solver legality does not match board")
        if i in legal:
            score_sign(s)
    if len(expected_board)!=42 or observed.get("board_bottom_up")!=expected_board:
        raise ValueError("Rust Connect4 board differs from independent gravity replay")
    if observed.get("side_to_move")!=(1 if len(history)%2==0 else -1):
        raise ValueError("wrong current player parity")
    if observed.get("legal_columns_0based")!=legal:
        raise ValueError("legal gravity columns do not match")
    if observed.get("eval_mode")!="uniform_no_neural_network":
        raise ValueError("unexpected non-frozen search evaluator")
    steps=observed.get("steps")
    if not isinstance(steps,list) or [s.get("budget") for s in steps]!=BUDGETS:
        raise ValueError("wrong resident visit schedule")
    initial_root=observed.get("root_hash")
    if not isinstance(initial_root,str) or not initial_root:
        raise ValueError("missing resident root identity")
    prev_calls=int(observed.get("root_initial_evaluator_calls",1))
    actions=[]
    utilities=[]
    times=[]
    counts=[]
    for step in steps:
        if step.get("root_hash")!=initial_root:
            raise ValueError("root hash changed during incremental search")
        if step.get("root_visits")!=step["budget"]:
            raise ValueError("resident MCTS visits not exhausted")
        a=step.get("best_action")
        if type(a) is not int or a not in legal:
            raise ValueError("illegal selected column")
        cumul=step.get("evaluator_calls")
        if type(cumul) is not int or cumul<prev_calls:
            raise ValueError("invalid cumulative evaluator call count")
        delta=cumul-prev_calls
        if delta!=step.get("additional_evaluator_calls"):
            raise ValueError("additional evaluator calls mismatched")
        prev_calls=cumul
        actions.append(a)
        utilities.append(score_sign(scores[a]))
        counts.append(delta)
        ms=step.get("step_wall_us")
        if type(ms) is not int or ms<0:
            raise ValueError("invalid wall-clock measurement")
        times.append(ms)
    fresh=observed.get("fresh16",{})
    if fresh.get("resident_match") is not True:
        raise ValueError("fresh16 failed independent restart identity")
    if fresh.get("best_action")!=actions[1] or fresh.get("evaluator_calls")!=steps[1]["evaluator_calls"]:
        raise ValueError("fresh16 does not match resident 16")
    if score_sign(exact["best_score"])!=max(score_sign(s) for s in scores if s!=-1000):
        raise ValueError("root optimal WDL inconsistent")
    optimal=max(utilities) if not legal else score_sign(exact["best_score"])
    deltas=[later-earlier for earlier,later in zip(utilities,utilities[1:])]
    regrets=[optimal-u for u in utilities]
    return {
        "position_id":exact.get("position_id"),
        "history":history,
        "ply":len(history),
        "root_hash":initial_root,
        "search_seed":observed.get("seed"),
        "status":"EXACT_EXTERNAL_WDL_JOINED_TO_REAL_QUARTZ_MCTS",
        "best_actions_zero_based":actions,
        "best_columns_1based":[a+1 for a in actions],
        "action_scores_ordinal":[scores[a] for a in actions],
        "action_utility_by_budget":utilities,
        "utility_delta_by_transition":deltas,
        "game_regret_by_budget":regrets,
        "oracle_optimal_WDL":optimal,
        "optimal_WDL_columns_zero_based":[i for i,s in enumerate(scores) if s!=-1000 and score_sign(s)==optimal],
        "oracle_fastest_win_or_slowest_loss_score":exact["best_score"],
        "oracle_best_score_columns_zero_based":[i for i,s in enumerate(scores) if s==exact["best_score"]],
        "cumulative_eval_calls":[s["evaluator_calls"] for s in steps],
        "additional_eval_calls_all_budgets":counts,
        "extra_eval_calls":counts[1:],
        "step_wall_us":times,
        "fresh16_match":True,
        "policy_kind":"uniform_prior_zero_value_no_NN",
        "nonclaims":["not_Bayesian_expected_VOC","not_trained_policy","not_iid_or_population"],
    }


def dump_new(path: Path,value) -> None:
    if path.exists():
        raise FileExistsError(f"output exists: {path}")
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def run(executable: Path, *, seeds: list[int], positions: list[dict], timeout_s: float=45.0) -> tuple[list[dict],str]:
    requests=[]
    for seed in seeds:
        for pos in positions:
            requests.append({"cmd":"research_connect4_cpu","history":pos["sequence"],"seed":seed,"budgets":BUDGETS})
    text="\n".join(json.dumps(req,separators=(",",":"),sort_keys=True) for req in requests)
    text+='\n'+json.dumps({"cmd":"quit"})+'\n'
    proc=subprocess.run(
        [str(executable),"--server"],input=text.encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=timeout_s,check=False,
    )
    if proc.returncode!=0:
        raise RuntimeError(f"Rust server failed: exit={proc.returncode}, stderr={proc.stderr[-1600:]!r}")
    results=[]
    for line in proc.stdout.decode("utf-8","replace").splitlines():
        if line.strip():
            results.append(json.loads(line))
    if len(results)!=len(requests):
        raise RuntimeError(f"wrong Rust reply count: {len(results)} vs {len(requests)}; stderr={proc.stderr.decode()[-900:]}")
    for req,result in zip(requests,results):
        if "error" in result:
            raise RuntimeError(f"Connect4 server rejected {req}: {result}")
        if result.get("schema")!="quartz_connect4_resident_mcts_v1":
            raise RuntimeError(f"unexpected Connect4 schema: {result}")
        if result.get("history")!=req["history"] or result.get("seed")!=req["seed"]:
            raise RuntimeError("request-response identity mismatch")
    return results,proc.stderr.decode("utf-8","replace")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=17)
    args=parser.parse_args()
    if args.seed!=17:
        raise ValueError("contract fixes one independent seed 17")
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/".git").exists())
    if git(repo,"status","--porcelain"):
        raise RuntimeError("source tree must be clean before evidence execution")
    lock=json.loads((d/"INPUT_LOCK.json").read_text())
    if lock["schema"]!="quartz_connect4_gamestate05_inputs_v1":
        raise RuntimeError("source input lock mismatch")
    for item in lock["source_files"]:
        if item["relative_path"] not in [
            "Cargo.toml","Cargo.lock","src/game.rs",
            "docs/research/cognitive_metareasoning_20261120/standard_external_04/POSITIONS_FIXED.json",
            "docs/research/cognitive_metareasoning_20261120/standard_external_04/evidence/20261120_standard_cpu/c4_oracle_results.json"
        ]:
            continue  # server and games source legitimately changed after baseline
        file=repo/item["relative_path"]
        if file.stat().st_size!=item["size"] or sha(file)!=item["sha256"]:
            raise RuntimeError("locked scientific input changed: "+item["relative_path"])
    for item in lock["external_files"]:
        f=Path(item["path"])
        if f.stat().st_size!=item["size"] or sha(f)!=item["sha256"]:
            raise RuntimeError("external exact oracle binary/book changed")
    expected_dir=repo/"docs/research/cognitive_metareasoning_20261120/standard_external_04"
    frozen=json.loads((expected_dir/"POSITIONS_FIXED.json").read_text())
    positions=frozen["connect4"]
    oracle=json.loads((expected_dir/"evidence/20261120_standard_cpu/c4_oracle_results.json").read_text())
    keyed={row["position_id"]:row for row in oracle}
    if len(positions)!=12 or len(keyed)!=12:
        raise RuntimeError("external solver corpus cardinality mismatch")
    for pos in positions:
        if keyed[pos["id"]]["sequence"]!=pos["sequence"]:
            raise RuntimeError("positions and oracle rows differ")
    exe=args.rust_binary.resolve()
    if not exe.is_file():
        raise FileNotFoundError(exe)
    started=time.perf_counter()
    raw,stderr=run(exe,seeds=[17],positions=positions)
    joined=[
        join_root(keyed[pos["id"]],observed,expected_board=board_from_columns(pos["sequence"]))
        for pos,observed in zip(positions,raw)
    ]
    # Frozen first 12 cases, no post-result subsetting or tuning.
    delta=Counter(str(k) for row in joined for k in row["utility_delta_by_transition"])
    budgets_summary={}
    for i,budget in enumerate(BUDGETS):
        util=[row["action_utility_by_budget"][i] for row in joined]
        regrets=[row["game_regret_by_budget"][i] for row in joined]
        budgets_summary[str(budget)]={
            "n":len(joined),
            "wdl_counts":dict(Counter(str(x) for x in util)),
            "mean_utility":sum(util)/len(util),
            "mean_regret":sum(regrets)/len(regrets),
            "optimal_WDL_count":sum(x==0 for x in regrets),
            "mean_evaluator_calls":sum(row["cumulative_eval_calls"][i] for row in joined)/len(joined),
            "mean_step_wall_us":sum(row["step_wall_us"][i] for row in joined)/len(joined),
        }
    by_transition={}
    for i, pair in enumerate(zip(BUDGETS,BUDGETS[1:])):
        by_transition[f"{pair[0]}->{pair[1]}"]={
            "signed_gain_counts":dict(Counter(str(row["utility_delta_by_transition"][i]) for row in joined)),
            "total_signed_gain":sum(row["utility_delta_by_transition"][i] for row in joined),
            "mean_additional_eval_calls":sum(row["extra_eval_calls"][i] for row in joined)/len(joined),
            "mean_step_wall_us":sum(row["step_wall_us"][i+1] for row in joined)/len(joined),
        }
    summary={
        "schema":"quartz_connect4_game_utility_summary_v1",
        "n_distinct_roots":len(joined),
        "seed":17,
        "budgets":budgets_summary,
        "transitions":by_transition,
        "all_signed_delta_counts":dict(delta),
        "fresh16_mismatches":sum(not row["fresh16_match"] for row in joined),
        "negative_cases":[row for row in joined if any(x<0 for x in row["utility_delta_by_transition"])],
        "changed_action_positions":[row["position_id"] for row in joined if len(set(row["best_actions_zero_based"]))>1],
        "whole_run_s":time.perf_counter()-started,
        "evaluator":"Rust_UniformEval_no_pretrained_NN",
        "oracle":"external_PascalPons_Connect4_exact_signed_scores",
        "nonclaims":["not_Bayesian_expected_VOC","not_learnt_recognition","not_iid","no_GPU"],
    }
    for item in lock["external_files"]:
        if sha(Path(item["path"]))!=item["sha256"]:
            raise RuntimeError("external input changed during run")
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    dump_new(target/"raw_mcts_responses.json",raw)
    dump_new(target/"joined_action_utility.json",joined)
    dump_new(target/"summary.json",summary)
    manifest={
        "schema":"quartz_connect4_game_utility_run_manifest_v1",
        "source_head":git(repo,"rev-parse","HEAD"),
        "source_tree":git(repo,"rev-parse","HEAD^{tree}"),
        "input_lock_sha256":sha(d/"INPUT_LOCK.json"),
        "binary_sha256":sha(exe),
        "binary_path":str(exe),
        "rust_server_stderr":stderr[:1000],
        "source_sha256":{
            "src/games/connect4.rs":sha(repo/"src/games/connect4.rs"),
            "src/mcts_server.rs":sha(repo/"src/mcts_server.rs"),
            "run_connect4_utility.py":sha(Path(__file__)),
        },
        "result_sha256":{p.name:sha(p) for p in target.iterdir()},
        "python":sys.version.split()[0],
        "seed":17,
        "sample_unit":"12_fixed_non_iid_root_histories",
        "not_verified":"the mathematical correctness of the external solver beyond prior independent metamorphic checks",
    }
    dump_new(target/"execution_manifest.json",manifest)
    print(json.dumps({
        "status":"completed","source":manifest["source_head"],
        "n":len(joined),"budgets":budgets_summary,
        "transitions":by_transition,
        "output":str(target),
    },ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
