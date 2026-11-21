#!/usr/bin/env python3
"""C16 theory: proof-certificate witness versus independent exhaustive action check."""
import hashlib,json,random
from pathlib import Path
D=((1,0),(0,1),(1,1),(1,-1))
def w(b,side,a):
    assert b[a]==0
    x,y=a%15,a//15
    for dx,dy in D:
        run=1
        for k in (-1,1):
            xx,yy=x+k*dx,y+k*dy
            while 0<=xx<15 and 0<=yy<15 and b[yy*15+xx]==side:
                run+=1;xx+=k*dx;yy+=k*dy
        if run>=5:return True
    return False
def history(h):
    b=[0]*225
    for j,a in enumerate(h):
        assert 0<=a<225 and b[a]==0
        side=1 if j%2==0 else 2
        assert not w(b,side,a),"preterminal original history"
        b[a]=side
    return b
def wins(b,side):
    return {a for a in range(225) if b[a]==0 and w(b,side,a)}
def local_wins_after(b,side,a):
    t=b.copy()
    assert not w(b,side,a)
    t[a]=side
    x,y=a%15,a//15
    cand=set()
    for dx,dy in D:
      for sign in (-1,1):
        for dist in range(1,5):
          xx,yy=x+sign*dist*dx,y+sign*dist*dy
          if 0<=xx<15 and 0<=yy<15:
            v=yy*15+xx
            if t[v]==0:cand.add(v)
    return {c for c in cand if w(t,side,c)}
def certificate(b,side,a):
    if wins(b,side) or wins(b,3-side) or w(b,side,a):return None
    ends=sorted(local_wins_after(b,side,a))
    return ends[:2] if len(ends)>=2 else None
def main():
    original=[31,160,32,177,33,190]
    b=history(original)
    assert wins(b,1)==set() and wins(b,2)==set()
    a=34
    cert=certificate(b,1,a)
    assert cert==[30,35],cert
    s=b.copy();s[a]=1
    replies=0
    for opp in range(225):
        if s[opp]:continue
        assert not w(s,2,opp),"unsound opponent immediate response"
        after=s.copy();after[opp]=2
        survivors=[c for c in cert if after[c]==0 and w(after,1,c)]
        assert survivors,("counterexample",opp)
        replies+=1
    assert replies==218
    # 100 independent candidate/source observations compare global and local win endpoints.
    prior=Path(__file__).resolve().parents[1].parent/"trained_prior_fair_cost_15/COHORT_224_FROZEN.json"
    groups=json.loads(prior.read_text())["splits"]["train"]
    observations=0
    for row in groups[:10]:
        h=row["history"]; state=history(h)
        side=1 if len(h)%2==0 else 2
        if wins(state,side) or wins(state,3-side):continue
        rng=random.Random(row["source_seed"])
        chosen=rng.sample([a for a in range(225) if state[a]==0],10)
        for a in chosen:
            if w(state,side,a):continue
            t=state.copy();t[a]=side
            exact=wins(t,side)
            local=local_wins_after(state,side,a)
            assert exact==local,(row["id"],a,sorted(exact^local))
            observations+=1
    assert observations>=50,observations
    report={"run_status":"PASS","independent_win_implementation":"pure Python four-axis enumerator",
        "complete_opponent_replies_checked":replies,
        "certificate_action":34,"certificate_example_history":original,
        "certificate_endpoints":[30,35],
        "local_vs_full_winning_action_candidates_verified":observations,
        "Wolfram_disjoint_endpoints_replies":"separate CAS statement for independent proof",
        "theorem_status":"derived conditional proof and checked finite representative",
        "not_exact_general_game_solution":True,
        "source_C15_train_sha256":hashlib.sha256(prior.read_bytes()).hexdigest()}
    out=Path(__file__).with_name("R07_EXACT_CHECKS.json")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
