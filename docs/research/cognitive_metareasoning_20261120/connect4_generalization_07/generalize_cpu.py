#!/usr/bin/env python3
"""Fixed-parameter CPU comparison of flat MLP vs local 3x3 convolution.

The exact solver is an offline teacher only; no solver call in model runtime.
Train, validation, and test split families are strictly separate. Validation
loss fixes each epoch before test is read. No hyperparameter search.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys

import torch
from torch import nn
import torch.nn.functional as F

from label_1024 import verify_cohort


SEED=20261121
EPOCHS=80
BATCH=64
LR=0.003
WEIGHT_DECAY=1e-4


def prior_modules():
    p=Path(__file__).resolve().parent.parent/'connect4_recognizer_06'
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))
    from train_cpu import (
        TinyPolicyValue, tensor_batch, calc_loss, metrics, expand_train_mirror,
        model_to_json, predict_json,
    )
    return TinyPolicyValue,tensor_batch,calc_loss,metrics,expand_train_mirror,model_to_json,predict_json


class ConvPolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.head=nn.Linear(16*6*7,8)
    def forward(self,features):
        if features.ndim!=2 or features.shape[1]!=126:
            raise ValueError('Connect4 ConvNet input must be Nx126')
        x=features.reshape(-1,3,6,7)
        z=self.head(F.relu(self.conv(x)).flatten(start_dim=1))
        return z[:,:7],z[:,7].tanh()


def validate_splits(cohort):
    r=verify_cohort(cohort)
    by={s:set() for s in ('train','val','test')}
    for row in cohort['rows']:
        by[row['split']].add(row['family'])
    disjoint=not(by['train']&by['val'] or by['train']&by['test'] or by['val']&by['test'])
    if not disjoint:
        raise RuntimeError('train/val/test share family')
    if r['colors']!={'black':512,'white':512}:
        raise RuntimeError('color imbalance')
    return {'families':r['families'],'splits':r['splits'],
            'balanced_colors':True,'no_shared_family':True}


def model_to_portable(net,kind:str,metadata:dict)->dict:
    Tiny,_,_,_,_,from_old,_=prior_modules()
    if kind=='mlp':
        return from_old(net,metadata)
    if kind!='conv' or not isinstance(net,ConvPolicyValue):
        raise ValueError('unrecognized frozen architecture')
    assert list(net.conv.weight.shape)==[16,3,3,3]
    assert list(net.head.weight.shape)==[8,672]
    return {
        'format':'quartz_connect4_conv3x3_v1',
        'input_planes':3,'board_height':6,'board_width':7,
        'conv_out_channels':16,'kernel_size':3,'padding':1,
        'head_dim':8,
        'layers':'Conv2d 3->16 kernel3 pad1 ReLU Flatten(672) Linear672->8',
        'conv_weight':net.conv.weight.detach().cpu().float().tolist(),
        'conv_bias':net.conv.bias.detach().cpu().float().tolist(),
        'head_weight':net.head.weight.detach().cpu().float().tolist(),
        'head_bias':net.head.bias.detach().cpu().float().tolist(),
        'metadata':metadata,
    }


def legal_masked_scores(logits:list[float],legal:list[bool])->list[float]:
    if len(logits)!=7 or len(legal)!=7 or not any(legal):
        raise ValueError('legal action mask or logits malformed')
    if any(not math.isfinite(x) for x in logits):
        raise ValueError('non-finite model logits')
    mx=max(x for x,good in zip(logits,legal) if good)
    scores=[math.exp(x-mx) if good else 0.0 for x,good in zip(logits,legal)]
    norm=sum(scores)
    if not math.isfinite(norm) or norm<=0:
        raise ValueError('invalid policy normalizer')
    return [x/norm for x in scores]


def compare_portable(model:dict,features:list[float],legal:list[bool]):
    if len(features)!=126 or any(not math.isfinite(x) for x in features):
        raise ValueError('features must be 126 finite numeric values')
    fmt=model.get('format')
    if fmt=='quartz_connect4_tiny_mlp_v1':
        *_,old_predict=prior_modules()
        return old_predict(model,features,legal)
    if fmt!='quartz_connect4_conv3x3_v1':
        raise ValueError('unexpected model format')
    cw=model['conv_weight']
    cb=model['conv_bias']
    hw=model['head_weight']
    hb=model['head_bias']
    if (len(cw)!=16 or len(cb)!=16 or len(hw)!=8 or len(hb)!=8
        or any(len(x)!=672 for x in hw)):
        raise ValueError('spatial model malformed')
    hidden=[0.0]*(16*42)
    for oc in range(16):
        for y in range(6):
            for x in range(7):
                acc=float(cb[oc])
                for ic in range(3):
                    for dy in range(3):
                        yp=y+dy-1
                        if not 0<=yp<6:continue
                        for dx in range(3):
                            xp=x+dx-1
                            if 0<=xp<7:
                                acc+=float(cw[oc][ic][dy][dx])*features[ic*42+yp*7+xp]
                hidden[oc*42+y*7+x]=max(acc,0.0)
    out=[sum(float(a)*b for a,b in zip(row,hidden))+float(bias)
         for row,bias in zip(hw,hb)]
    return legal_masked_scores(out[:7],legal),math.tanh(out[7])


def sha(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def set_determinism():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    torch.manual_seed(SEED)


def input_parts(rows):
    _,tensor_batch,*_=prior_modules()
    return tensor_batch(rows)


def train_architecture(kind,train,validation):
    Tiny,_,calc_loss,metrics,*_=prior_modules()
    torch.manual_seed(SEED)
    model=Tiny() if kind=='mlp' else ConvPolicyValue()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    train_tensor=input_parts(train)
    best_loss=float('inf')
    best_state=None
    best_epoch=None
    history=[]
    perm_rng=torch.Generator().manual_seed(SEED+137)
    # Identical data order between architectures, independent of parameter count.
    perms=[torch.randperm(len(train),generator=perm_rng) for _ in range(EPOCHS)]
    for epoch,perm in enumerate(perms,1):
        model.train()
        for inds in perm.split(BATCH):
            batch=tuple(t[inds] for t in train_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=calc_loss(model,batch)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            tr=metrics(model,train)
            va=metrics(model,validation)
        vloss=va['loss']
        if not math.isfinite(vloss):
            raise RuntimeError('nonfinite validation loss')
        if vloss<best_loss:
            best_loss=vloss;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({
            'epoch':epoch,'train_loss':tr['loss'],'train_regret':tr['mean_game_theoretic_regret'],
            'train_optimal':tr['top1_WDL_optimal_fraction'],
            'validation_loss':vloss,'validation_regret':va['mean_game_theoretic_regret'],
            'validation_optimal':va['top1_WDL_optimal_fraction'],
        })
    assert best_state is not None and best_epoch is not None
    model.load_state_dict(best_state)
    model.eval()
    return model,{'selected_epoch':best_epoch,'min_validation_loss':best_loss,'curve':history}


def evaluate_rows(model,rows):
    if not rows:return []
    x,p,v,legal=input_parts(rows)
    with torch.inference_mode():
        logit,pred_value=model(x)
        mask_logits=logit.masked_fill(~legal,-1.0e9)
        predicted_prob=torch.softmax(mask_logits,dim=1)
        chosen=mask_logits.argmax(dim=1).tolist()
        values=pred_value.tolist()
        policies=predicted_prob.tolist()
    out=[]
    for row,a,value,policy in zip(rows,chosen,values,policies):
        assert a in row['legal_columns']
        score=row['exact_action_scores'][a]
        assert score!=-1000
        wdl=(score>0)-(score<0)
        regret=row['root_wdl']-wdl
        r={
            'id':row['id'],'family':row['family'],'split':row['split'],
            'plies':row['plies'],'current_player':row['current_player'],
            'root_wdl':row['root_wdl'],'action':a,'chosen_WDL':wdl,
            'regret':regret,
            'model_value':value,'optimal_action_set':row['optimal_actions'],
            'prob_on_WDL_optimal_set':sum(policy[c] for c in row['optimal_actions']),
            'legal_random_expected_regret':row['root_wdl']-sum(
                (s>0)-(s<0) for s in row['exact_action_scores'] if s!=-1000
            )/len(row['legal_columns']),
        }
        out.append(r)
    return out


def summarise_rows(rows):
    if not rows:return {}
    return {
        'n':len(rows),
        'families':len({r['family'] for r in rows}),
        'root_WDL_counts':dict(Counter(str(r['root_wdl']) for r in rows)),
        'mean_regret':sum(r['regret'] for r in rows)/len(rows),
        'mean_random_legal_regret':sum(r['legal_random_expected_regret'] for r in rows)/len(rows),
        'top1_WDL_optimal_fraction':sum(r['regret']==0 for r in rows)/len(rows),
        'mean_value_MSE':sum((r['model_value']-r['root_wdl'])**2 for r in rows)/len(rows),
        'mean_prob_optimal_set':sum(r['prob_on_WDL_optimal_set'] for r in rows)/len(rows),
        'by_true_WDL':{
            str(k):{
                'n':sum(r['root_wdl']==k for r in rows),
                'mean_regret':sum(r['regret'] for r in rows if r['root_wdl']==k)/
                              max(1,sum(r['root_wdl']==k for r in rows))
            } for k in (-1,0,1)},
        'by_current_player':{
            str(k):{
                'n':sum(r['current_player']==k for r in rows),
                'mean_regret':sum(r['regret'] for r in rows if r['current_player']==k)/
                              max(1,sum(r['current_player']==k for r in rows))
            } for k in (-1,1)},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('require clean committed code for generalization experiment')
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    # Runtime & input lock authority: unchanged frozen sources, excludes the
    # novel generated training and Rust integration files.
    for item in lock['source_sha256']:
        p=Path(item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise RuntimeError('PREVIOUS_SOURCE_FILE_CHANGED: '+str(p))
    cohort=json.loads((d/'COHORT_1024_FROZEN.json').read_text())
    validate_splits(cohort)
    evidence=d/'evidence/teacher_1024_20261121'
    manifest=json.loads((evidence/'teacher_manifest.json').read_text())
    if sha(evidence/'labels.json')!=manifest['labels_sha256']:
        raise RuntimeError('1024 oracle label identity changed')
    labelled=json.loads((evidence/'labels.json').read_text())
    if labelled.get('status_counts')!={'OK':1024}:
        raise RuntimeError('incomplete teacher labels; no imputation allowed')
    train_rows=[r for r in labelled['rows'] if r['split']=='train']
    val_rows=[r for r in labelled['rows'] if r['split']=='val']
    test_rows=[r for r in labelled['rows'] if r['split']=='test']
    assert (len(train_rows),len(val_rows),len(test_rows))==(640,192,192)
    fam={s:{r['family'] for r in rows} for s,rows in
         [('train',train_rows),('val',val_rows),('test',test_rows)]}
    assert not fam['train']&fam['val'] and not fam['train']&fam['test'] and not fam['val']&fam['test']
    Tiny,_,calc_loss,metrics,expand_train_mirror,_,_=prior_modules()
    set_determinism()
    train_aug=expand_train_mirror(train_rows)
    assert len(train_aug)==1280
    frozen_sha=manifest['labels_sha256']
    saved_models={}
    learning={}
    predictions={}
    for kind in ('mlp','conv'):
        model,selection=train_architecture(kind,train_aug,val_rows)
        meta={
            'study_id':'QUARTZ-COG-01B-RECOGNIZER-GENERALIZATION-07',
            'architecture':kind,'training_source_commit':subprocess.check_output(
                ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'exact_teacher_labels_sha256':frozen_sha,
            'selected_epoch':selection['selected_epoch'],
            'fixed_epochs':EPOCHS,'batch':BATCH,'Adam_lr':LR,
            'Adam_weight_decay':WEIGHT_DECAY,'seed':SEED,
            'torch_cpu_threads':2,'train_mirror_augmentation':True,
            'test_labels_used_for_selection':False,
        }
        portable=model_to_portable(model,kind,meta)
        saved_models[kind]=portable
        parts={}
        for split,rows in [('train',train_rows),('val',val_rows),('test',test_rows)]:
            arr=evaluate_rows(model,rows)
            parts[split]=summarise_rows(arr)
            predictions.setdefault(split,{})[kind]=arr
        learning[kind]={'selection':selection,'eval':parts}
        print('MODEL_TRAIN_DONE',kind,'epoch',selection['selected_epoch'],
              'val_regret',parts['val']['mean_regret'],
              'test_regret',parts['test']['mean_regret'],flush=True)
    # PRE-EXISTING R06 model on new data, always evaluated only after
    # selection; never used for new-model training or hyperparameter choice.
    old_file=d.parent/'connect4_recognizer_06/evidence/nn_train_20261120/tiny_mlp_weights.json'
    older=json.loads(old_file.read_text())
    old_net=Tiny().eval()
    with torch.no_grad():
        for attr in ('fc1','fc2'):
            layer=getattr(old_net,attr)
            field='1' if attr=='fc1' else '2'
            layer.weight.copy_(torch.tensor(older['w'+field],dtype=torch.float32))
            layer.bias.copy_(torch.tensor(older['b'+field],dtype=torch.float32))
    learning['old_mlp_cog06']={'frozen_checkpoint_sha256':sha(old_file),
                              'eval':{}}
    for split,rows in [('train',train_rows),('val',val_rows),('test',test_rows)]:
        actual=evaluate_rows(old_net,rows)
        predictions.setdefault(split,{})['old_mlp_cog06']=actual
        learning['old_mlp_cog06']['eval'][split]=summarise_rows(actual)
    # Save fixed, reproducible records and production weights.
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for kind,portable in saved_models.items():
        (out/(kind+'_model.json')).write_text(
            json.dumps(portable,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    records={
        'schema':'quartz_connect4_generalization07_training_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'label_source_sha256':frozen_sha,
        'training_samples':len(train_aug),'train_roots':len(train_rows),
        'validation_roots':len(val_rows),'test_roots':len(test_rows),
        'architecture_comparison':['mlp','conv'],
        'training':learning,
        'status':'two_fixed_models_completed_no_test_tuning',
        'stats_status':'clusters=12 validation and 12 test; no iid per-row interval',
    }
    (out/'training_results.json').write_text(
        json.dumps(records,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    (out/'predictions.json').write_text(
        json.dumps(predictions,sort_keys=True,ensure_ascii=False)+'\n')
    # Verify label/source locked throughout run.
    for it in lock['source_sha256']:
        if sha(Path(it['path']))!=it['sha256']:
            raise RuntimeError('input source changed during CPU training')
    meta={
        'schema':'quartz_connect4_generalization07_training_manifest_v1',
        'source_commit':records['source_commit'],
        'label_source_sha256':frozen_sha,
        'locked_cohort_sha256':sha(d/'COHORT_1024_FROZEN.json'),
        'python':sys.version.split()[0],
        'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'cpu_threads':torch.get_num_threads(),
        'fixed_hyperparameters':{'seed':SEED,'epochs':EPOCHS,'batch':BATCH,
                                 'lr':LR,'weight_decay':WEIGHT_DECAY},
        'raw_output_sha256':{p.name:sha(p) for p in out.iterdir()},
        'not_claimed':['model trained on gold standard after validation selection',
                       'full-game strength','Bayesian VOC controller'],
    }
    (out/'execution_manifest.json').write_text(
        json.dumps(meta,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    print('TRAINING_END',json.dumps({
        'source':meta['source_commit'],
        'model_sha256':{k:sha(out/(k+'_model.json')) for k in ('mlp','conv')},
        'val':{k:v['eval']['val']['mean_regret'] for k,v in learning.items()},
        'test':{k:v['eval']['test']['mean_regret'] for k,v in learning.items()},
    },sort_keys=True),flush=True)


if __name__=='__main__':main()
