#!/usr/bin/env python3
"""Fit signed next-action gain and RELEASE CPU cost for bounded MCTS computation.

Online policy and root feature vector mirror Rust dynamic10.rs exactly.
This is not a Bayesian posterior or optimal metalevel policy. Target labels
are ex-post external exact WDL gains and can be negative.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

STAGES=(8,16,32)
BUDGETS=(8,16,32,64)
FEATURES='root_visits_v1'
SCHEMA='quartz_cog02_dynamic_two_step_v1'
COST_FLOOR_MS=1e-3
ALPHA=1.0


def train_ridge_with_intercept(X,y,alpha=ALPHA):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if X.ndim!=2 or X.shape[0]==0 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('invalid supervised ridge data, shape, or nonfinite values')
    if not np.isfinite(alpha) or alpha<=0:
        raise ValueError('Ridge regularization must be finite positive')
    if np.max(np.abs(X[:,0]-1.0))>1e-12:
        raise ValueError('intercept feature must equal one for every row')
    penalty=np.eye(X.shape[1],dtype=np.float64)*alpha
    penalty[0,0]=0.0
    beta=np.linalg.solve(X.T@X+penalty,X.T@y)
    if not np.isfinite(beta).all():
        raise RuntimeError('nonfinite fitted Ridge coefficient')
    return beta


def fit_positive_cost(X,y,alpha=ALPHA):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if y.shape!=(len(X),) or not np.isfinite(y).all() or np.any(y<0):
        raise ValueError('invalid actual release CPU batch costs')
    clipped=np.maximum(COST_FLOOR_MS,y)
    betas=train_ridge_with_intercept(X,np.log(clipped),alpha)
    fitted=np.exp(np.clip(X@betas,-20.0,10.0))
    scale=float(np.mean(clipped)/np.mean(fitted))
    if not np.isfinite(scale) or not 0<scale<1e5:
        raise RuntimeError('invalid fitted release CPU cost scale')
    return betas,scale


def model_cost(model,stage,features,joint=False):
    name='two_step' if joint else 'one_step'
    row=model[f'{name}_cost_log_coefficients'][stage]
    scale=model[f'{name}_cost_scale'][stage]
    z=sum(a*x for a,x in zip(row,features))
    if not math.isfinite(z) or not 0<scale<1e5:
        raise ValueError('nonfinite predictor cost parameters')
    cost=max(1e-6,math.exp(max(-20.,min(10.,z)))*scale)
    if not math.isfinite(cost):
        raise ValueError('nonfinite CPU time forecast')
    return cost


def predict_option(model,stage,features,variant='dynamic'):
    if model.get('schema')!=SCHEMA or model.get('features')!=FEATURES:
        raise ValueError('unrecognized two-step surrogate identity')
    if stage not in (0,1,2) or len(features)!=9 or not all(math.isfinite(x) for x in features):
        raise ValueError('malformed current MCTS root feature vector')
    if variant not in ('dynamic','myopic'):
        raise ValueError('invalid metacomputation policy variant')
    lam=model['lambda_wdl_per_ms']
    if not math.isfinite(lam) or not 0<=lam<=10:
        raise ValueError('invalid lambda')
    gs=sum(a*x for a,x in zip(model['one_step_gain_coefficients'][stage],features))
    cs=model_cost(model,stage,features,joint=False)
    ns=gs-lam*cs
    if stage<2:
        gj=sum(a*x for a,x in zip(model['two_step_gain_coefficients'][stage],features))
        cj=model_cost(model,stage,features,joint=True)
        nj=gj-lam*cj
    else:
        gj=None;cj=None;nj=None
    if not math.isfinite(gs) or not math.isfinite(ns) or (gj is not None and not math.isfinite(gj)):
        raise ValueError('nonfinite root gain forecast')
    if variant=='dynamic' and stage<2 and nj>0 and nj>max(ns,0):
        choice='commit_two'
    elif ns>0:
        choice='single'
    else:
        choice='stop'
    return {
        'action':choice,'at_visits':STAGES[stage],
        'predicted_single_gain':gs,'predicted_joint_gain':gj,
        'predicted_single_cost_ms':cs,'predicted_joint_cost_ms':cj,
        'net_single':ns,'net_joint':nj,
    }


def simulate_bounded_policy(model,observed,variant='dynamic'):
    features=observed.get('features')
    utility=observed.get('utilities',observed.get('utility'))
    walls=observed['search_stage_wall_ms']
    if (not isinstance(features,list) or len(features)!=3 or
        not isinstance(utility,list) or len(utility)!=4 or
        not isinstance(walls,list) or len(walls)!=4 or
        any(not math.isfinite(x) or x<0 for x in walls)):
        raise ValueError('full four-stage actual CPU trace required')
    stage=0
    decisions=[]
    skipped=[]
    while stage<3:
        p=predict_option(model,stage,features[stage],variant=variant)
        decisions.append(p)
        if p['action']=='stop':
            break
        if p['action']=='commit_two':
            skipped.append(STAGES[stage+1])
            stage+=2
        else:
            stage+=1
    if stage>3:
        raise RuntimeError('invalid two-stage horizon')
    return {
        'visits':BUDGETS[stage],
        'selected_index':stage,
        'utility':utility[stage],
        'search_ms':sum(walls[:stage+1]),
        'decisions':decisions,'skipped_stages':skipped,
        'choice_count':dict(Counter(x['action'] for x in decisions)),
    }


def fit_dynamic_from_train(rows,alpha=ALPHA):
    if not rows:raise ValueError('nonempty train-family sourced trajectories required')
    n=len(rows)
    one_g=[];two_g=[];one_cost=[];two_cost=[]
    one_scale=[];two_scale=[]
    diag={'training_rows':n,'one_step':[],'two_step':[]}
    for stage in range(3):
        X=np.asarray([r['features'][stage] for r in rows],dtype=np.float64)
        if X.shape!=(n,9) or not np.isfinite(X).all():
            raise ValueError('train features malformed')
        util=np.asarray([r['utility'] for r in rows],dtype=np.float64)
        wall=np.asarray([r['search_stage_wall_ms'] for r in rows],dtype=np.float64)
        if util.shape!=(n,4) or wall.shape!=(n,4) or not np.isfinite(util).all() or not np.isfinite(wall).all() or np.any(wall<0):
            raise ValueError('complete train utility and actual CPU stage costs needed')
        gain=util[:,stage+1]-util[:,stage]
        cost=wall[:,stage+1]
        beta=train_ridge_with_intercept(X,gain,alpha)
        cost_beta,scale=fit_positive_cost(X,cost,alpha)
        one_g.append(beta.tolist())
        one_cost.append(cost_beta.tolist())
        one_scale.append(scale)
        diag['one_step'].append({
            'start_visits':STAGES[stage],
            'mean_signed_gain':float(gain.mean()),
            'label_counts':dict(Counter(str(int(x)) for x in gain)),
            'gain_training_MSE':float(np.mean((X@beta-gain)**2)),
            'mean_cpu_release_ms':float(np.mean(cost)),
            'cost_floor_applied_count':int(np.sum(cost<COST_FLOOR_MS)),
        })
        if stage<2:
            gj=util[:,stage+2]-util[:,stage]
            cj=wall[:,stage+1]+wall[:,stage+2]
            beta2=train_ridge_with_intercept(X,gj,alpha)
            cost_beta2,scale2=fit_positive_cost(X,cj,alpha)
            two_g.append(beta2.tolist())
            two_cost.append(cost_beta2.tolist())
            two_scale.append(scale2)
            diag['two_step'].append({
                'start_visits':STAGES[stage],
                'target_visits':BUDGETS[stage+2],
                'mean_signed_joint_gain':float(gj.mean()),
                'joint_label_counts':dict(Counter(str(int(x)) for x in gj)),
                'joint_gain_training_MSE':float(np.mean((X@beta2-gj)**2)),
                'mean_actual_joint_cpu_ms':float(np.mean(cj)),
                'cost_floor_applied_count':int(np.sum(cj<COST_FLOOR_MS)),
            })
    model={
        'schema':SCHEMA,'features':FEATURES,
        'one_step_gain_coefficients':one_g,
        'two_step_gain_coefficients':two_g,
        'one_step_cost_log_coefficients':one_cost,
        'two_step_cost_log_coefficients':two_cost,
        'one_step_cost_scale':one_scale,
        'two_step_cost_scale':two_scale,
        'lambda_wdl_per_ms':0.,
        'train_diagnostics':diag,
    }
    for stage in range(3):
        _=predict_option(model,stage,rows[0]['features'][stage])
    return model
