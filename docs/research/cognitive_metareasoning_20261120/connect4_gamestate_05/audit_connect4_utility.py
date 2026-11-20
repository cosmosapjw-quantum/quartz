#!/usr/bin/env python3
"""Independent score/rules/raw-evidence readback of QUARTZ Connect4 CPU results.

Use a standalone Pascal Pons oracle requery, distinct replay and sign evaluator.
No trained net, no claims of statistically independent positions.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def geometry_win(board: list[int], player: int) -> bool:
    for y in range(6):
        for x in range(7):
            if board[y * 7 + x] != player:
                continue
            for dx, dy in [(1,0),(0,1),(1,1),(-1,1)]:
                if all(
                    0 <= x+dx*i < 7 and 0 <= y+dy*i < 6
                    and board[(y+dy*i)*7 + (x+dx*i)] == player
                    for i in range(4)
                ):
                    return True
    return False


def reconstruct(history: str) -> tuple[list[int],list[int]]:
    board=[0]*42
    heights=[0]*7
    for step, x in enumerate(history):
        if x not in "1234567":
            raise AssertionError("invalid historical column")
        if geometry_win(board,1) or geometry_win(board,-1):
            raise AssertionError("already won before terminal")
        col=ord(x)-ord('1')
        if heights[col]>=6:
            raise AssertionError("full column")
        board[heights[col]*7+col]=1 if step%2==0 else -1
        heights[col]+=1
    if geometry_win(board,1) or geometry_win(board,-1):
        raise AssertionError("nonterminal root contains four")
    return board,heights


def query(exe:Path,book:Path,history:str) -> list[int]:
    proc=subprocess.run(
        [str(exe),"-a","-b",str(book)],
        input=(history+"\n").encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=12.0,check=True,
        cwd=str(exe.parent),
    )
    t=proc.stdout.decode().split()
    if len(t)!=8 or t[0]!=history:
        raise AssertionError(f"unexpected PascalPons reply for {history}: {t}")
    return [int(x) for x in t[1:]]


def main() -> None:
    root=Path(__file__).resolve().parent
    ev=root/"evidence/20261120_connect4_exact_12"
    if not ev.is_dir():
        raise RuntimeError("source fixed raw evidence absent")
    man=json.loads((ev/"execution_manifest.json").read_text())
    lock=json.loads((root/"INPUT_LOCK.json").read_text())
    for name,expected in man["result_sha256"].items():
        if digest(ev/name)!=expected:
            raise AssertionError("origin run manifest output changed: "+name)
    for item in lock["external_files"]:
        file=Path(item["path"])
        if file.stat().st_size!=item["size"] or digest(file)!=item["sha256"]:
            raise AssertionError("external game-theoretic solver changed")
    oracle_exec=Path(lock["external_files"][0]["path"])
    oracle_book=Path(lock["external_files"][1]["path"])
    # Do not rely on lock list ordering for exe/book.
    if oracle_exec.name!="c4solver":
        oracle_exec,oracle_book=oracle_book,oracle_exec
    if oracle_exec.name!="c4solver" or oracle_book.name!="7x6_small.book":
        raise AssertionError("external oracle identities ambiguous")
    origin=Path(__file__).resolve().parents[3] / "standard_external_04"
    # research/decision-utility directory's sibling is under same root
    origin=root.parent/"standard_external_04"
    samples=json.loads((origin/"POSITIONS_FIXED.json").read_text())["connect4"]
    exact_rows=json.loads((origin/"evidence/20261120_standard_cpu/c4_oracle_results.json").read_text())
    byid={x["position_id"]:x for x in exact_rows}
    raw=json.loads((ev/"raw_mcts_responses.json").read_text())
    joined=json.loads((ev/"joined_action_utility.json").read_text())
    summary=json.loads((ev/"summary.json").read_text())
    if not len(samples)==len(raw)==len(joined)==12:
        raise AssertionError("sample cardinality changed")
    ulists=[]
    signed=Counter()
    changed=[]
    matched_mirror=0
    costs=[]
    for pos,ob,row in zip(samples,raw,joined):
        expected_exact=byid[pos["id"]]
        history=pos["sequence"]
        if ob["history"]!=history or row["history"]!=history or row["position_id"]!=pos["id"]:
            raise AssertionError("join ordering/source mismatch")
        board,heights=reconstruct(history)
        if board!=pos["board"] or board!=ob["board_bottom_up"]:
            raise AssertionError("Rust/Python board replay mismatch")
        moves=[i for i,h in enumerate(heights) if h<6]
        if ob["legal_columns_0based"]!=moves:
            raise AssertionError("Rust legal columns mismatch")
        if ob["root_hash"]!=row["root_hash"] or len(ob["steps"])!=4:
            raise AssertionError("root/step mismatch")
        oracle=query(oracle_exec,oracle_book,history)
        if oracle!=expected_exact["scores"]:
            raise AssertionError("fresh external solver action scores differ from frozen reference")
        mirror="".join(str(8-int(d)) for d in history)
        if list(reversed(oracle))!=query(oracle_exec,oracle_book,mirror):
            raise AssertionError("oracle mirror symmetry violated")
        matched_mirror+=1
        star=max((v>0)-(v<0) for v in oracle if v!=-1000)
        assert star==(expected_exact["best_score"]>0)-(expected_exact["best_score"]<0)
        utilities=[]
        counts=[]
        prior_calls=ob["root_initial_evaluator_calls"]
        hashes=set()
        for budget,step in zip((8,16,32,64),ob["steps"]):
            if step["budget"]!=budget or step["root_visits"]!=budget:
                raise AssertionError("budget drift")
            action=step["best_action"]
            if action not in moves:
                raise AssertionError("illegal MCTS column")
            if step["root_hash"]!=ob["root_hash"]:
                raise AssertionError("root identity drift")
            val=oracle[action]
            if val==-1000:raise AssertionError("selected external invalid-column sentinel")
            utilities.append((val>0)-(val<0))
            if step["evaluator_calls"]<prior_calls or step["additional_evaluator_calls"]!=step["evaluator_calls"]-prior_calls:
                raise AssertionError("inconsistent evaluator call accounting")
            counts.append(step["additional_evaluator_calls"])
            prior_calls=step["evaluator_calls"]
        if not ob["fresh16"]["resident_match"]:
            raise AssertionError("fresh16 replay mismatch")
        if utilities != row["action_utility_by_budget"]:
            raise AssertionError("stored per-budget utility disagrees with oracle requery")
        if [star-x for x in utilities]!=row["game_regret_by_budget"]:
            raise AssertionError("stored regret disagrees with game-theoretic WDL")
        deltas=[right-left for left,right in zip(utilities,utilities[1:])]
        if deltas!=row["utility_delta_by_transition"]:
            raise AssertionError("stored signed incremental utility mismatch")
        signed.update(map(str,deltas))
        if len(set(row["best_actions_zero_based"]))>1:
            changed.append(pos["id"])
        ulists.append(utilities)
        costs.append(counts)
    for i,budget in enumerate((8,16,32,64)):
        vals=[a[i] for a in ulists]
        expect=summary["budgets"][str(budget)]
        if sum(vals)/len(vals)!=expect["mean_utility"]:
            raise AssertionError("summary average utility mismatch")
        if Counter(str(v) for v in vals)!=Counter(expect["wdl_counts"]):
            raise AssertionError("summary WDL counts mismatch")
    for i,transition in enumerate(("8->16","16->32","32->64")):
        vals=[u[i+1]-u[i] for u in ulists]
        stored=summary["transitions"][transition]
        if sum(vals)!=stored["total_signed_gain"] or Counter(map(str,vals))!=Counter(stored["signed_gain_counts"]):
            raise AssertionError("summary signed-gain counts mismatch")
    if summary["fresh16_mismatches"]!=0:
        raise AssertionError("frozen restart mismatch")
    independent={
        "schema":"quartz_connect4_game_utility_independent_audit_v1",
        "n_distinct_roots":len(samples),
        "external_action_score_requeries":len(samples),
        "external_mirror_requeries":matched_mirror,
        "board_replay_checks":len(samples),
        "fresh16_matches":len(samples),
        "exact_wdl_by_budget":{
            str(b):dict(Counter(str(row[i]) for row in ulists))
            for i,b in enumerate((8,16,32,64))
        },
        "total_signed_changes_by_transition":{
            key:sum(a[i+1]-a[i] for a in ulists)
            for i,key in enumerate(("8->16","16->32","32->64"))
        },
        "signed_change_counts":dict(signed),
        "changed_action_positions":changed,
        "additional_evaluator_calls_total":sum(sum(c) for c in costs),
        "science_status":"computed exact utility from externally reported game-theoretic WDL, finite sample",
        "not_claimed":["no trained Connect4 NN","not Bayesian expected VOC","not iid","not proof of external solver internal correctness"],
    }
    dest=ev/"INDEPENDENT_READBACK.json"
    if dest.exists():
        raise RuntimeError("refusing to overwrite earlier review")
    dest.write_text(json.dumps(independent,indent=2,sort_keys=True)+"\n")
    print(json.dumps(independent,sort_keys=True))


if __name__=="__main__":
    main()
