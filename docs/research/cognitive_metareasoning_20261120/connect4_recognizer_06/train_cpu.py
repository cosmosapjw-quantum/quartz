#!/usr/bin/env python3
"""CPU-only distillation of exact Connect4 WDL-optimal action *sets*.

Model 126 -> 64 ReLU -> (7 policy logits + tanh value), fixed 80 epochs.
The 12 historical experiment roots remain completely out of training/model
selection. Validation decides the epoch; test is read after that decision.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import torch
from torch import nn
import torch.nn.functional as F

from teacher_cpu import mirror_encoding, mirror_target

SEED=20261120
EPOCHS=80
BATCH=64
LEARNING_RATE=0.003
WEIGHT_DECAY=1e-4


class TinyPolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(126,64)
        self.fc2=nn.Linear(64,8)
    def forward(self,x):
        out=self.fc2(F.relu(self.fc1(x)))
        return out[:,:7],out[:,7].tanh()


def model_to_json(model:TinyPolicyValue,metadata:dict)->dict:
    return {
        "format":"quartz_connect4_tiny_mlp_v1",
        "input_dim":126,"hidden_dim":64,"output_dim":8,
        "layers":"Linear126x64 ReLU Linear64x8; policy softmax legal columns; value tanh",
        "w1":model.fc1.weight.detach().cpu().float().tolist(),
        "b1":model.fc1.bias.detach().cpu().float().tolist(),
        "w2":model.fc2.weight.detach().cpu().float().tolist(),
        "b2":model.fc2.bias.detach().cpu().float().tolist(),
        "metadata":metadata
    }


def predict_json(m:dict,features:list[float],legal:list[bool])->tuple[list[float],float]:
    """Dependency-free independent f64 arithmetic on exported f32 weights."""
    if len(features)!=126 or len(legal)!=7 or not any(legal):raise ValueError("wrong shapes")
    hidden=[max(0.0,sum(w*x for w,x in zip(row,features))+bias) for row,bias in zip(m['w1'],m['b1'])]
    out=[sum(w*x for w,x in zip(row,hidden))+bias for row,bias in zip(m['w2'],m['b2'])]
    masked=[a if good else -1e12 for a,good in zip(out[:7],legal)]
    max_logit=max(v for v,good in zip(masked,legal) if good)
    e=[math.exp(v-max_logit) if good else 0.0 for v,good in zip(masked,legal)]
    normalizer=sum(e)
    return [z/normalizer for z in e],math.tanh(out[7])


def expand_train_mirror(rows:list[dict])->list[dict]:
    augmented=[]
    for row in rows:
        if row['split']!='train':raise ValueError('mirror augmentation must be train-only')
        augmented.append(dict(row))
        reflected=dict(row)
        reflected['id']=row['id']+'_M'
        reflected['features']=mirror_encoding(row['features'])
        reflected['target_policy']=mirror_target(row['target_policy'])
        reflected['legal_columns']=[6-i for i in row['legal_columns']][::-1]
        augmented.append(reflected)
    return augmented


def tensor_batch(rows):
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32)
    p=torch.tensor([r['target_policy'] for r in rows],dtype=torch.float32)
    v=torch.tensor([r['root_wdl'] for r in rows],dtype=torch.float32)
    legal=torch.tensor([[c in r['legal_columns'] for c in range(7)] for r in rows],dtype=torch.bool)
    assert x.shape[1:]==(126,) and p.shape[1:]==(7,) and legal.shape==p.shape
    assert torch.allclose(p.sum(dim=1),torch.ones(len(rows)),atol=1e-6)
    assert not torch.any((p>0)&(~legal))
    return x,p,v,legal


def calc_loss(model,inputs):
    x,p,v,legal=inputs
    logits,value=model(x)
    masked=logits.masked_fill(~legal,-1.0e9)
    policy_loss=-(p*F.log_softmax(masked,dim=1)).sum(dim=1).mean()
    value_loss=((value-v)**2).mean()
    return policy_loss + 0.5*value_loss,policy_loss,value_loss


def metrics(model,rows):
    if not rows:
        return {"n":0,"loss":None}
    inputs=tensor_batch(rows)
    with torch.no_grad():
        total,loss_policy,loss_value=calc_loss(model,inputs)
        scores,value=model(inputs[0])
        predicted=scores.masked_fill(~inputs[3],-1e9).argmax(dim=1).tolist()
        correct=sum(int(a in row['optimal_actions']) for a,row in zip(predicted,rows))
        realized=[(row['exact_action_scores'][a]>0)-(row['exact_action_scores'][a]<0) for row,a in zip(rows,predicted)]
        regrets=[row['root_wdl']-observed for row,observed in zip(rows,realized)]
        return {
            'n':len(rows),
            'loss':float(total),
            'policy_CE':float(loss_policy),
            'value_MSE':float(loss_value),
            'top1_WDL_optimal_fraction':correct/len(rows),
            'mean_game_theoretic_regret':sum(regrets)/len(rows),
            'predicted_WDL_counts':dict(Counter(str(x) for x in realized)),
            'mean_predicted_value':float(value.mean()),
        }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('train only on committed clean source')
    labels_dir=d/'evidence/teacher_20261120'
    origin=json.loads((labels_dir/'labels.json').read_text())
    label_manifest=json.loads((labels_dir/'label_manifest.json').read_text())
    def sha(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    if sha(labels_dir/'labels.json')!=label_manifest['labels_sha256']:
        raise RuntimeError('frozen teacher labels changed')
    if origin['schema']!='quartz_connect4_oracle_labels_v1':
        raise RuntimeError('unrecognized source labels')
    by_split={s:[] for s in ('train','val','test')}
    for row in origin['full_rows']:
        if row['status']=='OK':
            by_split[row['split']].append(row)
    if any(len(by_split[s])!=n for s,n in (('train',128),('val',32),('test',32))):
        raise RuntimeError('training cohort not full 192, stop and re-scope explicitly')
    families={s:{r['family'] for r in rows} for s,rows in by_split.items()}
    assert not families['train']&families['val'] and not families['train']&families['test'] and not families['val']&families['test']

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    train=expand_train_mirror(by_split['train'])
    x,p,v,legal=tensor_batch(train)
    model=TinyPolicyValue().train()
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    records=[]
    best=None
    best_validation=float('inf')
    best_epoch=None
    for epoch in range(1,EPOCHS+1):
        permutation=torch.randperm(len(train))
        model.train()
        for subset in permutation.split(BATCH):
            batch=(x[subset],p[subset],v[subset],legal[subset])
            opt.zero_grad(set_to_none=True)
            loss,_,_=calc_loss(model,batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            train_report=metrics(model,train)
            val_report=metrics(model,by_split['val'])
        val_loss=val_report['loss']
        if val_loss < best_validation:
            best_validation=val_loss
            best_epoch=epoch
            best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        records.append({
            'epoch':epoch,
            'train_loss':train_report['loss'],
            'train_optimal_frac':train_report['top1_WDL_optimal_fraction'],
            'val_loss':val_report['loss'],
            'val_optimal_frac':val_report['top1_WDL_optimal_fraction'],
            'val_regret':val_report['mean_game_theoretic_regret'],
        })
    assert best is not None and best_epoch is not None
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        final_metrics={
            split:metrics(model,rows)
            for split,rows in by_split.items()
        }
    info={
        'selected_epoch':best_epoch,'epochs':EPOCHS,'batch':BATCH,
        'optimizer':'Adam','lr':LEARNING_RATE,'weight_decay':WEIGHT_DECAY,
        'hidden_dim':64,'input_dim':126,'output_dim':8,'seed':SEED,
        'teacher_label_sha256':label_manifest['labels_sha256'],
        'source_git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'metric_for_epoch_selection':'validation loss; ties earliest',
        'training_mirror_augmentation':True,
        'trained_on_historical_12_eval_roots':False,
    }
    exported=model_to_json(model,metadata=info)
    summary={
        'schema':'quartz_connect4_tiny_mlp_training_v1',
        'best_epoch':best_epoch,'best_val_loss':best_validation,
        'metrics':final_metrics,'epoch_curve':records,
        'teacher_input_sha256':label_manifest['labels_sha256'],
        'training_examples':len(train),'validation_examples':len(by_split['val']),
        'test_examples':len(by_split['test']),
        'scientific_nonclaims':['not human cognition validation','not universal strength','not Bayesian VOC'],
    }
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    weights=target/'tiny_mlp_weights.json'
    weights.write_text(json.dumps(exported,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    sums=target/'train_result.json'
    sums.write_text(json.dumps(summary,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    manifest={
        'schema':'quartz_connect4_tiny_mlp_export_manifest_v1',
        'source_commit':info['source_git_head'],
        'labels_sha256':info['teacher_label_sha256'],
        'model_json_sha256':sha(weights),
        'summary_json_sha256':sha(sums),
        'torch_version':torch.__version__,
        'torch_cuda_version':torch.version.cuda,
        'cpu_threads':torch.get_num_threads(),
        'epoch_selected':best_epoch,
        'model_architecture':'126->64(ReLU)->8(policy[7],value[tanh])',
    }
    (target/'train_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'best_epoch':best_epoch,'weights_sha':manifest['model_json_sha256'],
        'metrics':final_metrics
    },sort_keys=True))


if __name__=='__main__':main()
