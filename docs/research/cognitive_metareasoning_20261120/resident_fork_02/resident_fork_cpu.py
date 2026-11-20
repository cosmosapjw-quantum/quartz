"""Strict CPU resident-root paired MCTS search contract and grouped summary.

The signed reference alignment delta is a realized search outcome, not
posterior expected VOC, game-theoretic regret, or an Elo comparison.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics


def _policy(entry, board):
    if not isinstance(entry, (tuple, list)) or not entry:
        raise ValueError("missing sparse policy")
    seen=set()
    dense=[0.0]*len(board)
    for item in entry:
        if not isinstance(item,(list,tuple)) or len(item)!=2:
            raise ValueError("malformed policy tuple")
        a,score=item
        if not isinstance(a,int) or a in seen or not 0<=a<len(board):
            raise ValueError("malformed action index or duplicate policy")
        if board[a]!=0:
            raise ValueError("illegal action in sparse policy")
        weight=float(score)
        if not math.isfinite(weight) or weight<0:
            raise ValueError("nonfinite or negative policy weight")
        dense[a]=weight
        seen.add(a)
    total=sum(dense)
    if abs(total-1.0)>1e-4:
        raise ValueError("policy total must be one")
    return dense


def _check_action(row, board):
    a=row["best_move"]
    if not isinstance(a,int) or not 0<=a<len(board) or board[a]!=0:
        raise ValueError("illegal best_move")
    return int(a)


def _same_policy(left,right,tol=1e-7):
    return len(left)==len(right) and all(
        math.isclose(x,y,abs_tol=tol,rel_tol=0)
        for x,y in zip(left,right)
    )


def validate_pair(position,first,second,fresh, *,
                  python_calls,wall_ms,checkpoint,seed):
    """Validate 8 resident +8 same-root versus fresh16 with actual CPU NN calls.

    The output preserves any fresh16-vs-resident mismatch as a negative finding.
    """
    board=position["board"]
    if len(board)!=49 or all(x!=0 for x in board):
        raise ValueError("invalid seven-by-seven root")
    if not (int(first["root_visits"])==8 and
            int(second["root_visits"])==16 and
            int(fresh["root_visits"])==16 and
            int(first["additional_root_visits"])==8 and
            int(second["additional_root_visits"])==8 and
            int(fresh["additional_root_visits"])==16):
        raise ValueError("root visit delta or budget mismatch")
    root=first.get("root_hash")
    if not isinstance(root,str) or not root or root!=second.get("root_hash"):
        raise ValueError("root hash changed on resident continuation")
    if root!=fresh.get("root_hash"):
        raise ValueError("root hash changed in fresh16 control")
    a0,a1,af=(_check_action(o,board) for o in (first,second,fresh))
    p0,p1,pf=(_policy(o["policy"],board) for o in (first,second,fresh))
    eval0=int(first["evaluator_calls"])
    eval1=int(second["evaluator_calls"])
    evalf=int(fresh["evaluator_calls"])
    delta=int(second["additional_evaluator_calls"])
    if not (1<=eval0<=9 and eval0<=eval1<=eval0+8 and
            0<=delta<=8 and delta==eval1-eval0 and
            1<=evalf<=17):
        raise ValueError("evaluator call accounting outside budget")
    n0,nd,nf=(int(x) for x in python_calls)
    if (n0,nd,nf)!=(eval0,delta,evalf):
        raise ValueError(f"CPU NN forward item calls disagree with Rust evaluator calls: model={(n0,nd,nf)} engine={(eval0,delta,evalf)}")
    if any(not math.isfinite(float(t)) or float(t)<0 for t in wall_ms):
        raise ValueError("nonfinite or negative wall timing")
    ref=int(position["reference_best"]);oracle=int(position["oracle_best"])
    if not (0<=ref<49 and 0<=oracle<49):
        raise ValueError("reference/oracle out of board")
    first_ref=int(a0==ref);second_ref=int(a1==ref)
    first_oracle=int(a0==oracle);second_oracle=int(a1==oracle)
    fresh_match=bool(a1==af and _same_policy(p1,pf) and eval1==evalf)
    sig={
        "root":root,"best_stop":a0,"best_continue":a1,
        "policy_stop":p0,"policy_continue":p1,
        "calls_stop":eval0,"calls_continue":eval1,
    }
    trajectory_hash=hashlib.sha256(json.dumps(sig,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {
        "position_id":str(position["id"]),
        "checkpoint_id":str(checkpoint),
        "search_seed":int(seed),
        "root_hash":root,
        "root_visits":[8,16],
        "best_stop":a0,"best_continue":a1,"best_fresh16":af,
        "reference_best":ref,"oracle_best":oracle,
        "stop_reference_correct":first_ref,
        "continue_reference_correct":second_ref,
        "delta_ref":second_ref-first_ref,
        "delta_oracle":second_oracle-first_oracle,
        "NN_call_identity":"exact",
        "nn_calls_stop":eval0,
        "nn_calls_continue_total":eval1,
        "nn_calls_fresh16":evalf,
        "delta_eval_calls":delta,
        "wall_ms_stop":float(wall_ms[0]),
        "wall_ms_additional":float(wall_ms[1]),
        "wall_ms_fresh16":float(wall_ms[2]),
        "fresh16_match":fresh_match,
        "stop_policy_tied":sum(abs(v-max(p0))<1e-7 for v in p0)>1,
        "continue_policy_tied":sum(abs(v-max(p1))<1e-7 for v in p1)>1,
        "policy_stop":p0,
        "policy_continue":p1,
        "policy_fresh16":pf,
        "trajectory_sha256":trajectory_hash,
        "interpretation":"same_root_resident_evaluation_true_cpu_calls_reference_proxy_agreement",
    }


def summarize_rows(rows):
    if not rows:
        raise ValueError("no experiments")
    signed=Counter(str(r["delta_ref"]) for r in rows)
    signed_oracle=Counter(str(r["delta_oracle"]) for r in rows)
    for k in ("-1","0","1"):
        signed.setdefault(k,0);signed_oracle.setdefault(k,0)
    group=defaultdict(list)
    for row in rows:
        group[(row["position_id"],row["checkpoint_id"])].append(row)
    unique={f"{p}/{ck}":len({r["trajectory_sha256"] for r in group[(p,ck)]}) for p,ck in sorted(group)}
    by_checkpoint={}
    for ck in sorted(set(r["checkpoint_id"] for r in rows)):
        rr=[r for r in rows if r["checkpoint_id"]==ck]
        by_checkpoint[ck]={
            "n":len(rr),
            "reference_signed":dict(Counter(str(r["delta_ref"]) for r in rr)),
            "mean_reference_signed":sum(r["delta_ref"] for r in rr)/len(rr),
            "mean_extra_evals":sum(r["delta_eval_calls"] for r in rr)/len(rr),
        }
    return {
        "schema":"quartz_resident_fork02_cpu_summary_v1",
        "claim":"signed_reference_alignment_for_existing_trained_NN_with_resident_MCTS",
        "n_trials":len(rows),
        "n_positions":len({r["position_id"] for r in rows}),
        "n_checkpoints":len({r["checkpoint_id"] for r in rows}),
        "n_search_seeds":len({r["search_seed"] for r in rows}),
        "root_position_checkpoint_pairs":len(group),
        "signed_ref":dict(sorted(signed.items())),
        "signed_oracle":dict(sorted(signed_oracle.items())),
        "mean_signed_ref":sum(r["delta_ref"] for r in rows)/len(rows),
        "delta_eval_calls_total":sum(r["delta_eval_calls"] for r in rows),
        "mean_delta_eval_calls":sum(r["delta_eval_calls"] for r in rows)/len(rows),
        "mean_step_wall_ms":sum(r["wall_ms_additional"] for r in rows)/len(rows),
        "median_step_wall_ms":statistics.median(r["wall_ms_additional"] for r in rows),
        "fresh16_mismatches":sum(not r["fresh16_match"] for r in rows),
        "stop_ties":sum(r["stop_policy_tied"] for r in rows),
        "continue_ties":sum(r["continue_policy_tied"] for r in rows),
        "unique_trajectory_counts_by_root":unique,
        "all_repeated_seed_trajectories_identical":all(v==1 for v in unique.values()),
        "by_checkpoint":by_checkpoint,
        "no_intervals_because_independence_not_established":True,
        "interpretation_caveat":[
            "mined positions not iid",
            "three seeds may be algorithmically identical trajectories",
            "reference and oracle are stronger model proxies",
            "alignment is not objective regret or full Bayesian VOC",
            "baseline_strict MCTS not the QUARTZ adaptive controller",
            "timings are CPU only and exclude GPU inference behavior",
        ],
    }
