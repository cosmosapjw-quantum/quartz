#!/usr/bin/env python3
"""C09 independent contingent-plan reconstruction from retained data; no native launch."""
import gzip, hashlib, itertools, json, pathlib, subprocess, sys
from fractions import Fraction as F
from datetime import datetime, timezone
BASE=pathlib.Path("/home/cosmosapjw/quantum/research/quartz_c28_action_geometry_20261123_486e")
P="docs/research/quartz_c28_action_geometry_20261123"
C="f791e96e651d1849df601dcd7b98c18bcc3fa082"
def gb(path):return subprocess.check_output(["git","-C",str(BASE),"show",C+":"+path])
def data(stage,kind):return [json.loads(x) for x in gzip.decompress(gb(P+"/evidence/"+stage+"/"+kind+".jsonl.gz")).splitlines()]
VALUES=(F(0),F(1,2),F(1))
EPS=F(1,10**12)
def tilted(rows,loads):
    out=[]
    for p,r in zip(rows,loads):
        mu=sum(q*u for q,u in zip(p,VALUES))
        scale=max(abs(u-mu) for q,u in zip(p,VALUES) if q)
        out.append(tuple(r*(u-mu)/scale if q and scale else F(0) for q,u in zip(p,VALUES)))
    return out
def joint(rows,loads,m,known):
    tilt=tilted(rows,loads);worlds=[]
    for state in itertools.product(range(3),repeat=len(rows)):
        if any(VALUES[state[i]]!=u for i,u in known.items()):continue
        weight=F(0)
        for z in [-1,1]:
            term=(1+z*m)/2
            for i,k in enumerate(state):term*=rows[i][k]*(1+z*tilt[i][k])
            weight+=term
        if weight:worlds.append((tuple(VALUES[k] for k in state),weight))
    normalizer=sum(q for _,q in worlds)
    return [(w,q/normalizer) for w,q in worlds]
def plan_scores(rows,loads,costs,m,b,remaining,known,horizon):
    """Enumerate policies before taking expectations; uses no Bellman max interchange."""
    w=joint(rows,loads,m,known);out={};counts={}
    for first in remaining:
        options=(None,)+tuple(j for j in remaining if j!=first) if horizon==2 else (None,)
        observations=tuple(u for u in VALUES if any(state[first]==u for state,_ in w))
        best=None; count=0
        for policy in itertools.product(options,repeat=len(observations)):
            follow=dict(zip(observations,policy));value=F(0)
            for state,prob in w:
                second=follow[state[first]]
                gain=max(b,state[first]) if second is None else max(b,state[first],state[second])-costs[second]
                value+=prob*(gain-costs[first]-b)
            best=value if best is None else max(best,value);count+=1
        out[first]=best;counts[first]=count
    return out,counts
def choose(scores,ids=None):
    if not scores or max(scores.values())<=EPS:return None
    return min((i if ids is None else ids[i]) for i,v in scores.items() if v>=max(scores.values())-EPS)
def compare(q,row,positions):
    residual=0.
    for method in ["direct","lifted"]:
        for pos,i in enumerate(positions):
            diff=abs(row[method]["scores"][pos]-float(q[i])); assert diff<=5e-12;residual=max(residual,diff)
    return residual
started=datetime.now(timezone.utc).isoformat()
report={"schema":"quartz-c28-c09-independent-reconstruction-v1","review_commit":C,"native_rerun":False,"argv":sys.argv,"started_utc":started,"synthetic":[]}
sr,so=data("c06_synthetic","requests"),data("c06_synthetic","responses")
for req,res in zip(sr,so):
    if not (req["id"].startswith("fixture-") or req["id"] in ["boundary--19/24","boundary-19/24","edge-4-m-1-b1/2-h2","edge-4-m1-b1/2-h2"]):continue
    rows=[tuple(F(str(x)) for x in p) for p in req["probabilities"]]
    rows=[tuple(x/sum(p) for x in p) for p in rows]
    loads=list(map(lambda x:F(str(x)),req["loadings"]));costs=list(map(lambda x:F(str(x)),req["costs"]))
    m=F(req["id"][9:]) if req["id"].startswith("boundary-") else F(str(req["m"]))
    q,counts=plan_scores(rows,loads,costs,m,F(str(req["b"])),tuple(range(len(rows))),{},req["horizon"])
    residual=compare(q,res,tuple(q))
    selected=choose(q)
    for method in ["direct","lifted"]:
        assert res[method]["choice"]==selected
        assert abs(res[method]["value"]-float(F(str(req["b"]))+max([F(0)]+list(q.values()))))<=5e-12
    report["synthetic"].append({"id":req["id"],"exact_scores":{str(i):str(x) for i,x in q.items()},"plan_counts":counts,"epsilon_choice":selected,"max_residual":residual})
# Reconstruct one actually observed root from board winners, tree traversal and seeded cues.
rq,rs=data("c07_game","requests"),data("c07_game","responses")
request,response=rq[0],rs[0]
assert request["id"]=="C26-p4-0059adf99c0886cf"
LINES=[(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def win(b):
    for a,c,d in LINES:
        if b[a] and b[a]==b[c]==b[d]:return b[a]
    return 0
def term(b):return win(b)!=0 or 0 not in b
def child(b,i,t):
    x=list(b);assert x[i]==0;x[i]=t;return tuple(x)
def solve(b,t,root):
    if term(b):return win(b)*root,1
    values=[];nodes=1
    for i,x in enumerate(b):
        if not x:
            v,n=solve(child(b,i,t),-t,root);values.append(v);nodes+=n
    return (max(values) if t==root else min(values)),nodes
MASK=2**64-1;G=0x9e3779b97f4a7c15
class RNG:
    def __init__(self,s):self.s=s
    def draw(self,n):
        threshold=((2**64)-n)%n
        while True:
            self.s=(self.s+G)&MASK;z=self.s
            z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK
            z=((z^(z>>27))*0x94d049bb133111eb)&MASK
            z=z^(z>>31)
            if z>=threshold:return z%n
board=(0,)*9;root=1
for move in request["history"]:
    assert not term(board);board=child(board,move,root);root=-root
cand=[]
for move,v in enumerate(board):
    if v:continue
    cb=child(board,move,root);known=term(cb);counts=[0,0,0];cn=0
    if known:counts[win(cb)*root+1]=4;cn=1
    else:
        rng=RNG(request["cue_seed"]^(((move+1)*G)&MASK))
        for _ in range(4):
            b,t=cb,-root;cn+=1
            while not term(b):
                moves=[i for i,x in enumerate(b) if x==0]
                b=child(b,moves[rng.draw(len(moves))],t);t=-t;cn+=1
            counts[win(b)*root+1]+=1
    val,nodes=solve(cb,-root,root)
    rem=str(cb.count(0));key=rem+":"+":".join(map(str,counts));mod=request["model"]
    p=mod["feature_probs"].get(key,mod["remaining_probs"].get(rem,mod["global_probs"]))
    p=tuple(F(str(x)) for x in p);p=tuple(x/sum(p) for x in p)
    if known:p=tuple(F(int(k==val+1)) for k in range(3))
    cost=F(str(mod["remaining_costs"].get(rem,mod["global_cost"])))/1024
    cand.append({"move":move,"utility":F(val+1,2),"nodes":nodes,"cues":cn,"counts":counts,"known":known,"p":p,"cost":cost})
known={i:c["utility"] for i,c in enumerate(cand) if c["known"]}
initial_solve=None;initial_nodes=0
if F(1) not in known.values():
    unknown=[i for i in range(len(cand)) if i not in known]
    if unknown:
        initial_solve=min(unknown,key=lambda i:(-(cand[i]["counts"][1]+2*cand[i]["counts"][2]),cand[i]["move"]))
        known[initial_solve]=cand[initial_solve]["utility"];initial_nodes=cand[initial_solve]["nodes"]
incumbent=min(known,key=lambda i:(-known[i],cand[i]["move"]))
rows=[c["p"] for c in cand];loads=[F(3,4)]*len(cand);costs=[c["cost"] for c in cand];tilt=tilted(rows,loads)
def posterior(obs):
    component=[]
    for z in [-1,1]:
        likelihood=F(1)
        for i,u in obs.items():
            k=VALUES.index(u);likelihood*=rows[i][k]*(1+z*tilt[i][k])
        component.append(likelihood)
    return (component[1]-component[0])/sum(component)
details=[];querymoves=[];additionalnodes=0
for d in response["direct"]["decisions"]:
    remaining=tuple(i for i in range(len(cand)) if i not in known)
    assert d["remaining_moves"]==[cand[i]["move"] for i in remaining]
    assert abs(d["m"]-float(posterior(known)))<=5e-12
    q,counts=plan_scores(rows,loads,costs,F(0),known[incumbent],remaining,known,d["horizon"])
    choice=choose(q,[c["move"] for c in cand])
    assert choice==d["choice"]
    maxerr=max(abs(x-float(q[i])) for i,x in zip(remaining,d["scores"]));assert maxerr<=5e-12
    details.append({"horizon":d["horizon"],"exact_m":str(posterior(known)),"exact_scores":{str(cand[i]["move"]):str(x) for i,x in q.items()},"plan_counts":counts,"choice":choice,"max_residual":maxerr})
    if choice is None:break
    i=next(i for i,c in enumerate(cand) if c["move"]==choice);known[i]=cand[i]["utility"]
    querymoves.append(choice);additionalnodes+=cand[i]["nodes"]
    if known[i]>known[incumbent]:incumbent=i
totalcues=sum(c["cues"] for c in cand)
assert response["direct"]["query_moves"]==querymoves
assert response["direct"]["final_move"]==cand[incumbent]["move"]
assert response["direct"]["initial_nodes"]==initial_nodes
assert response["direct"]["additional_nodes"]==additionalnodes
assert response["direct"]["charged_nodes"]==totalcues+initial_nodes+additionalnodes
assert abs(response["direct"]["final_m"]-float(posterior(known)))<=5e-12
report["game"]={"id":request["id"],"root_player":root,"initial_solve_move":cand[initial_solve]["move"],"initial_nodes":initial_nodes,"cue_nodes":totalcues,"query_moves":querymoves,"additional_nodes":additionalnodes,"charged_nodes":totalcues+initial_nodes+additionalnodes,"final_move":cand[incumbent]["move"],"utility":str(known[incumbent]),"exact_final_m":str(posterior(known)),"decisions":details}
report["status"]="PASS_SCOPED_INDEPENDENT_RECONSTRUCTION";report["ended_utc"]=datetime.now(timezone.utc).isoformat()
report["script_sha256"]=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
(BASE/P/"evidence/C09_REVIEW_RECONSTRUCTION.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,sort_keys=True))
