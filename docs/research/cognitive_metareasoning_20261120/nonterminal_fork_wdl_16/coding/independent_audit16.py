#!/usr/bin/env python3
"""C9 independent actual native MCTS and complete-game rule/WDL audit.
No imports from C04/C05 C07/C08 runner, scorer or Rust feature implementation.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math
repo=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
home=Path(__file__).resolve().parent.parent
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def legal_winning(b,side,a):
    # Independent axes plus explicit board bounds; does not use original runner's checker.
    if not (0<=a<225) or b[a]:return False
    x,y=divmod(a,15)[1],a//15
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
      n=1
      for sign in (-1,1):
        i=1
        while 0<=x+sign*i*dx<15 and 0<=y+sign*i*dy<15:
          xx=x+sign*i*dx;yy=y+sign*i*dy
          if b[yy*15+xx]!=side:break
          n+=1;i+=1
      if n>=5:return True
    return False
def replay_prefix(h):
    board=[0]*225
    for ply,a in enumerate(h):
       if not isinstance(a,int) or not (0<=a<225) or board[a]:raise AssertionError("illegal source move")
       p=1 if ply%2==0 else 2
       if legal_winning(board,p,a):raise AssertionError("source history contains terminal move")
       board[a]=p
    return board
def immediate(b,p):
    return [a for a in range(225) if not b[a] and legal_winning(b,p,a)]
def full_certificates(b,side):
    # Complete independent oracle, not 32-neighbor shortcut; correctness is finite.
    if immediate(b,side) or immediate(b,3-side):return {}
    out={}
    for a in range(225):
       if b[a] or legal_winning(b,side,a):continue
       c=b.copy();c[a]=side
       wins=immediate(c,side)
       if len(wins)>=2:out[a]=wins
    return out
def mask_of(actions):
    words=[0]*4
    for a in actions:
       assert 0<=a<225
       words[a//64]|=1<<(a%64)
    return words

def main():
    cohortfile=home/"COHORT_40_FROZEN.json"
    assert sha(cohortfile)=="8349441f247f720028f8534ac4abf656be56f8ac58778191de900722b4af1049"
    model=repo/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/evidence/model/MODEL_TRAINED.json"
    assert sha(model)=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
    src=json.loads(cohortfile.read_text())
    source_by_id={r["id"]:r for group in ("constructed_fork_positive","independent_MCTS_played_roots")
                  for r in src["data_splits"][group]}
    rfile=home/"evidence/real_root16_blind/RAW_ROOT_195.json"
    roots=json.loads(rfile.read_text())
    scored=json.loads((home/"evidence/real_root16_blind/C08_ROOT_RULE_SCORE.json").read_text())
    assert roots["n_actual_native_root_requests"]==195
    assert roots["native_release_sha256"]=="1168f2979936a283a9e7e613b869fd30ec9aba826e15ac228c6ee9d7e7e4c4d3"
    assert scored["source_real_Rust_sha256"]==sha(rfile)
    arms=("learned_all_legal","learned_fork_mask","learned_fork_proof_choice",
           "learned_sham_mask","learned_top16")
    per_st={"constructed":[],"MCTS_played":[]}
    ncert=0
    for row in roots["per_source"]:
       srcrow=source_by_id[row["source_id"]]
       h=row["history"]
       assert h==srcrow["history"]
       b=replay_prefix(h)
       side=1 if len(h)%2==0 else 2
       legal=[a for a,v in enumerate(b) if v==0]
       cert=full_certificates(b,side)
       ncert+=len(cert)
       rank=row["arms"]["learned_all_legal"]["initial_ranked_actions"]
       assert len(rank)==len(legal) and set(rank)==set(legal)
       k16=set(rank[:16])
       cert_out=set(cert)-k16
       typ="constructed" if "engineered" in row["source_type"] else "MCTS_played"
       values={}
       for arm in arms:
           x=row["arms"][arm]
           assert x["arm"]==arm and x["history"]==h
           assert x["source_prior"]=="C15_frozen_trained_policy13_value0"
           assert x["original_full_root_legal_count"]==len(legal)
           assert x["legal_actions"]==legal
           assert x["same_root_N_W_M2_P_preserved_on_activation"]
           assert x["same_root_ptr_hash"] and x["teacher_oracle_never_online"]
           assert x["first8_root_visits"]==8
           assert x["final_root_visits"]==x["root_edge_visit_sum"]
           assert x["target_cpu_budget_us"]==20000
           assert x["total_actual_cpu_wall_us"]>0
           assert x["final_legal_move"] in legal
           assert x["final_mcts_move"] in legal
           assert len(x["first8_edge_stats"])==len(legal)
           assert sum(int(edge[1]) for edge in x["first8_edge_stats"])==8
           assert {int(edge[0]) for edge in x["first8_edge_stats"]}==set(legal)
           assert len(x["original_full_prior"])==len(legal)
           prior={int(a):float(p) for a,p in x["original_full_prior"]}
           assert set(prior)==set(legal)
           assert abs(sum(prior.values())-1)<.0002
           active=x["after_active_actions"]
           assert len(active)==len(set(active)) and set(active)<=set(legal)
           assert x["final_legal_move"] in active
           if arm=="learned_all_legal":
               assert set(active)==set(legal) and not x["added_actions"]
               assert x["root_mask_after"] in ([0]*4,mask_of(active))
           else:
               assert len(x["initial_candidates"])==16
               assert set(x["initial_candidates"])==k16
               assert x["root_mask_after"]==mask_of(active)
               assert set(active)==k16|set(x["added_actions"])
           assert len(x["added_actions"])<=4 or arm=="learned_all_legal"
           proofs=x["certificates_omitted"]
           assert len(proofs)<=4
           for certrow in proofs:
               a=int(certrow["action"])
               ep=sorted(map(int,certrow["endpoints"]))
               assert a in cert_out
               assert a in cert
               assert len(set(ep))==2 and all(y in cert[a] for y in ep)
           if arm=="learned_fork_mask":
               assert set(x["added_actions"])=={int(v["action"]) for v in proofs}
           if arm=="learned_fork_proof_choice":
               if proofs:
                   assert x["explicit_proof_choice_override"]
                   assert x["final_legal_move"]==proofs[0]["action"]
               else:assert not x["explicit_proof_choice_override"]
           else:assert not x["explicit_proof_choice_override"]
           if arm=="learned_sham_mask":
               assert not set(x["added_actions"])&set(cert)
           values[arm]={"selected":int(x["final_legal_move"]),
                   "is_certified":int(x["final_legal_move"] in cert),
                   "wall_us":int(x["total_actual_cpu_wall_us"])}
       per_st[typ].append({"id":row["source_id"],"n_full_cert":len(cert),
             "n_top16_cert":len(set(cert)&k16),
             "n_omitted_cert":len(cert_out),"arms":values})
    summary={}
    for k,rows in per_st.items():
       assert len(rows)==(23 if k=="constructed" else 16)
       n=len(rows)
       def mean(arm):return sum(x["arms"][arm]["wall_us"] for x in rows)/n
       aa=mean("learned_fork_mask");bb=mean("learned_all_legal")
       summary[k]={"n":n,
        "n_strict_reachability_miss":sum(x["n_top16_cert"]==0 and x["n_omitted_cert"]>0 for x in rows),
        "n_any_omitted_cert":sum(x["n_omitted_cert"]>0 for x in rows),
        "n_top16_has_cert":sum(x["n_top16_cert"]>0 for x in rows),
        "n_mask_final_choice_cert":sum(x["arms"]["learned_fork_mask"]["is_certified"] for x in rows),
        "n_full_final_choice_cert":sum(x["arms"]["learned_all_legal"]["is_certified"] for x in rows),
        "n_full_vs_mask_diff":sum(x["arms"]["learned_fork_mask"]["selected"]!=x["arms"]["learned_all_legal"]["selected"] for x in rows),
        "mean_cpu_mask":aa,"mean_cpu_full":bb,
        "cost_mismatch_abs_rel":abs(aa-bb)/aa,
        "cost_matched_10pct":abs(aa-bb)/aa<=.1}
       original=scored["strata"][k]
       assert math.isclose(original["mean_CPU_wall_mask_us"],aa,abs_tol=1e-8)
       assert math.isclose(original["mean_CPU_wall_full_us"],bb,abs_tol=1e-8)
       assert original["n"]==n and original["predeclared_10pct_CPU_cost_gate"]==summary[k]["cost_matched_10pct"]
       assert original["per_arm"]["learned_all_legal"]["n_certified_final_moves"]==summary[k]["n_full_final_choice_cert"]
    gamefile=home/"evidence/real_games16/RAW_32_ACTUAL_COMPLETE_GAMES.json"
    game=json.loads(gamefile.read_text())
    ledger=home/"evidence/real_games16/COMPLETED_GAMES.jsonl"
    assert game["completed_source_progress_sha256"]==sha(ledger)
    lines=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(lines)==len(game["games"])==32
    byid={g["game_id"]:g for g in lines}
    assert len(byid)==32
    assert all(byid[g["game_id"]]==g for g in game["games"])
    assert game["n_source_opening_clusters"]==8 and game["n_legal_complete_games"]==32
    histories=src["data_splits"]["paired_fullgame_openings"]
    names=[x["id"] for x in src["data_splits"]["independent_MCTS_played_roots"]]
    assert len(set(histories))==8 and set(histories)<=set(names)
    active_arms=("learned_fork_mask","learned_fork_proof_choice")
    tallies={arm:Counter() for arm in active_arms}
    pair_scores={arm:{} for arm in active_arms}
    n_played_moves=0
    n_proof_events=0
    for g in game["games"]:
       assert g["opening_id"] in histories and g["active_policy"] in active_arms
       assert g["active_color"] in (1,2)
       assert {g["black_policy"],g["white_policy"]}=={g["active_policy"],"learned_all_legal"}
       assert len(g["original_opening_history"])==12
       assert g["n_moves_after_opening"]==len(g["per_move_real_native"])
       h=g["original_opening_history"].copy()
       prior_open=source_by_id[g["opening_id"]]
       assert prior_open["history"]==h
       assert prior_open["parent_game_id"]==g["source_parent_game_id"]
       win_player=None
       for turn,move in enumerate(g["per_move_real_native"]):
           b=replay_prefix(h)
           p=1 if len(h)%2==0 else 2
           assert move["ply"]==len(h) and move["side"]==p
           assert move["source_history_length"]==len(h)
           chosen=int(move["flat_action"])
           assert chosen>=0 and chosen<225 and b[chosen]==0
           assert move["policy"]==(g["black_policy"] if p==1 else g["white_policy"])
           outcome=legal_winning(b,p,chosen)
           assert bool(move["terminal_win_by_full_rules"])==outcome
           assert move["native_active_wall_us"]>0 and move["native_mcts_visits"]>=8
           assert not move["explicit_proof_override"] or move["actual_certificate_selected"]
           n_proof_events+=int(move["reopened_count"]>0)+int(move["explicit_proof_override"])
           h.append(chosen)
           n_played_moves+=1
           if outcome:
               assert turn==len(g["per_move_real_native"])-1
               win_player=p
       assert h==g["final_history"]
       if win_player is not None:
          assert g["winner"]==win_player and g["terminal_reason"]=="FIVE_IN_ROW"
       else:
          assert len(h)==225 and g["winner"] is None and g["terminal_reason"]=="FULL_BOARD_DRAW"
       score=0.5 if g["winner"] is None else float(g["winner"]==g["active_color"])
       assert g["active_score"]==score
       key=g["active_policy"]
       tallies[key][score]+=1
       pair_scores[key].setdefault(g["opening_id"],[]).append(score)
    wdl_summary=json.loads((home/"evidence/real_games16/C08_WDL_SUMMARY.json").read_text())
    assert wdl_summary["source_raw_sha256"]==sha(gamefile)
    for arm in active_arms:
       assert len(pair_scores[arm])==8
       assert all(len(v)==2 for v in pair_scores[arm].values())
       row=wdl_summary["arms"][arm]
       assert row["W"]==tallies[arm][1.] and row["L"]==tallies[arm][0.]
       assert row["D"]==tallies[arm][0.5]
       assert math.isclose(row["mean_game_score"],(tallies[arm][1.]+0.5*tallies[arm][0.5])/16)
    assert n_played_moves==sum(len(g["per_move_real_native"]) for g in game["games"])
    assert n_proof_events==0
    result={"schema":"quartz_cog06_C09_external_independent_replay_v1",
       "status":"PASS","root_evidence_sha256":sha(rfile),
       "source40_sha256":sha(cohortfile),"model_sha256":sha(model),
       "actual_full_game_raw_sha256":sha(gamefile),
       "actual_original_game_ledger_sha256":sha(ledger),
       "n_actual_source_roots":39,"n_real_Rust_root_requests":195,
       "n_native_verified_completed_games":32,
       "n_actual_native_game_moves":n_played_moves,
       "n_reopening_or_override_game_events":n_proof_events,
       "total_fork_certificates_at_source_roots":ncert,
       "independent_root_rules_mcts_mask_prior_NW_M2_checked":True,
       "independent_actual_complete_game_WDL_rule_replay_checked":True,
       "both_cost_strata_10percent_gate":all(r["cost_matched_10pct"] for r in summary.values()),
       "separate_science_claim":"0 strict effective missing source candidates, zero general heldout reactivation, no identifiable cognitive mechanism gain",
       "strata":summary,
       "game_WDL":{a:{"W":tallies[a][1.],"D":tallies[a][0.5],"L":tallies[a][0.]} for a in active_arms},
       "no_Elo_or_general_population_claim":True}
    out=home/"evidence/C09_INDEPENDENT_REPLAY.json"
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("C09_INDEPENDENT_REPLAY",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":main()
