#!/usr/bin/env python3
"""Verified 13-feature Gomoku15 finite-teacher policy-only model. No value-head."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from freeze_cohorts import replay,win
N=15
DIRECTIONS=((1,0),(0,1),(1,1),(1,-1))
FEATURE_ORDER=("bias","center_cheb","center_manhattan","own_r1","opp_r1",
 "own_r2","opp_r2","own_run5","opp_run5","own_window5","opp_window5",
 "own_immediate5","opp_immediate5")
def one_feature(board,side,a):
    if len(board)!=225 or side not in (1,2) or a<0 or a>=225 or board[a]!=0:
        raise ValueError("invalid feature board / side / action")
    x,y=a%15,a//15
    opp=3-side
    r1=[0,0];r2=[0,0]
    for dy in range(-2,3):
      for dx in range(-2,3):
        if not dx and not dy:continue
        xx,yy=x+dx,y+dy
        if not(0<=xx<15 and 0<=yy<15):continue
        stone=board[yy*15+xx]
        if not stone:continue
        target=0 if stone==side else 1
        if max(abs(dx),abs(dy))==1:r1[target]+=1
        elif max(abs(dx),abs(dy))==2:r2[target]+=1
    out=[1.,1.-max(abs(x-7),abs(y-7))/7.,
         1.-(abs(x-7)+abs(y-7))/14.,
         r1[0]/8.,r1[1]/8.,r2[0]/16.,r2[1]/16.]
    for target_side in (side,opp):
      best_run=1
      best_window=0
      for dx,dy in DIRECTIONS:
        run=1
        for sign in (-1,1):
          xx,yy=x+sign*dx,y+sign*dy
          while 0<=xx<15 and 0<=yy<15 and board[yy*15+xx]==target_side:
            run+=1;xx+=sign*dx;yy+=sign*dy
        best_run=max(best_run,run)
        for start in range(-4,1):
          coords=[(x+(start+j)*dx,y+(start+j)*dy) for j in range(5)]
          if any(not(0<=xx<15 and 0<=yy<15) for xx,yy in coords):continue
          vals=[board[yy*15+xx] for xx,yy in coords]
          if (3-target_side) in vals:continue
          count=1+sum(v==target_side for (xx,yy),v in zip(coords,vals) if (xx,yy)!=(x,y))
          best_window=max(best_window,count)
      out.extend((min(best_run,5)/5.,best_window/5.))
    # Order contract is runs(side,opp), then windows(side,opp), then winning(side,opp).
    # Rearrange the just appended fields.
    ownrun,ownwin,opprun,oppwin=out[-4:]
    out=out[:-4]+[ownrun,opprun,ownwin,oppwin,
                     float(win(board,side,a)),float(win(board,opp,a))]
    assert len(out)==13
    assert all(0<=q<=1 for q in out),out
    return np.asarray(out,dtype=np.float64)

def position_features(history):
    b=replay(history)
    side=1 if len(history)%2==0 else 2
    legal=[a for a in range(225) if b[a]==0]
    return np.asarray(legal,dtype=np.int64),np.asarray(
        [one_feature(b,side,a) for a in legal],dtype=np.float64)

def predict_logits(theta,history):
    actions,features=position_features(history)
    return actions,features@np.asarray(theta,dtype=np.float64)

def stable_prob(logits):
    z=np.asarray(logits,dtype=np.float64)
    if z.size==0 or not np.all(np.isfinite(z)):raise ValueError("bad logits")
    ex=np.exp(z-z.max())
    return ex/ex.sum()

def nll_grad_hessian(theta,features,targets,l2=0.01):
    t=np.asarray(theta,dtype=np.float64)
    loss=0.5*l2*float(t@t)
    grad=l2*t.copy()
    hess=np.eye(len(t))*l2
    n=len(features)
    assert n>0 and len(targets)==n
    for x,yi in zip(features,targets):
        z=x@t
        zm=z.max()
        ez=np.exp(z-zm)
        p=ez/ez.sum()
        loss+=(float(np.log(ez.sum())+zm-z[yi]))/n
        mu=p@x
        grad+=(mu-x[yi])/n
        xc=x-mu
        hess+=(xc.T@(xc*p[:,None]))/n
    return loss,grad,hess

def score(theta,rows,targets):
    losses=[];hit=0;top16=0
    for history,y in zip(rows,targets):
        actions,z=predict_logits(theta,history)
        if int(y) not in actions:raise ValueError("teacher illegal move")
        pi=int(np.flatnonzero(actions==int(y))[0])
        zz=z-z.max()
        losses.append(float(np.log(np.exp(zz).sum())-zz[pi]))
        ranked=sorted(zip(actions.tolist(),z.tolist()),key=lambda x:(-x[1],x[0]))
        hit+=int(int(ranked[0][0])==int(y))
        top16+=int(int(y) in [int(x[0]) for x in ranked[:16]])
    return {"n":len(rows),"mean_NLL":float(np.mean(losses)),
            "top1_accuracy":hit/len(rows),"top16_coverage":top16/len(rows)}

def fit(training,validation,l2=0.01,max_iter=70):
    train_history=[x["history"] for x in training]
    train_labels=[int(x["label"]) for x in training]
    feats=[];idx=[]
    for h,y in zip(train_history,train_labels):
        actions,f=position_features(h)
        if y not in actions:raise ValueError("illegal teacher label")
        feats.append(f);idx.append(int(np.flatnonzero(actions==y)[0]))
    theta=np.zeros(13,dtype=np.float64)
    trace=[]
    for it in range(max_iter):
        val,g,H=nll_grad_hessian(theta,feats,idx,l2=l2)
        gnorm=float(np.max(np.abs(g)))
        trace.append({"iteration":it,"train_obj":val,"gradient_inf":gnorm})
        if gnorm<1e-9:break
        delta=np.linalg.solve(H,g)
        step=1.
        descent=float(g@delta)
        while step>1e-10:
            candidate=theta-step*delta
            next_obj=nll_grad_hessian(candidate,feats,idx,l2=l2)[0]
            if next_obj<=val-0.0001*step*descent:break
            step*=0.5
        if step<=1e-10:raise RuntimeError("Newton Armijo search failed")
        theta=candidate
    train_sc=score(theta,train_history,train_labels)
    val_sc=score(theta,[x["history"] for x in validation],[x["label"] for x in validation])
    return theta,trace,train_sc,val_sc

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train_from_teacher(train_label_file,model_output):
    label=json.loads(Path(train_label_file).read_text())
    parent=Path(__file__).resolve().parent.parent
    cohort=json.loads((parent/"COHORT_224_FROZEN.json").read_text())
    assert label["cohort_sha256"]==sha(parent/"COHORT_224_FROZEN.json")
    for split in ("train","validation"):
      expect=[r["id"] for r in cohort["splits"][split]]
      assert sorted(x["id"] for x in label["splits"][split])==sorted(expect)
      lookup={r["id"]:r for r in cohort["splits"][split]}
      for row in label["splits"][split]:
        assert row["history"]==lookup[row["id"]]["history"]
        assert row["status"]=="OK"
    theta,trace,train_sc,val_sc=fit(label["splits"]["train"],label["splits"]["validation"])
    obj={"schema":"quartz_cog05_loglinear_policy13_v1",
         "model_type":"finite_teacher_supervised_legal_action_policy_only",
         "trained_value_head":False,"value":0.0,
         "feature_order":list(FEATURE_ORDER),"theta":theta.tolist(),
         "l2":0.01,"training_algorithm":"deterministic_full_batch_damped_newton",
         "iterations":len(trace),"final_grad_inf":trace[-1]["gradient_inf"],
         "training_metrics":train_sc,"validation_metrics":val_sc,
         "cohort_sha256":label["cohort_sha256"],
         "train_validation_labels_sha256":sha(train_label_file),
         "teacher_is_finite_search_not_minimax":True,
         "C14_roots_used_in_training":False,"heldout_test_teacher_labels_used":False,
         "frozen_after_training_before_test":True}
    Path(model_output).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    return obj

if __name__=="__main__":
 import argparse
 ap=argparse.ArgumentParser()
 ap.add_argument("--teacher-labels",type=Path,required=True)
 ap.add_argument("--out",type=Path,required=True)
 args=ap.parse_args()
 print(json.dumps(train_from_teacher(args.teacher_labels,args.out),sort_keys=True))
