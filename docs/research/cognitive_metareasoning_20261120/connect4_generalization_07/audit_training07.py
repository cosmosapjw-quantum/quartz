#!/usr/bin/env python3
"""Independent readback and fixed-output paired family-cluster uncertainty.

Does NOT refit the 80-epoch models. The bootstrap conditions on fixed
training and val selection; sample unit is 5-ply opening family (n=12 test).
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import sys

def sha(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def quantile95(values):
    v=sorted(values)
    return [v[int(0.025*(len(v)-1))],v[int(0.975*(len(v)-1))]]


def main():
    d=Path(__file__).resolve().parent
    out=d/'evidence/mlp_conv_train_20261121'
    raw=json.loads((out/'training_results.json').read_text())
    predictions=json.loads((out/'predictions.json').read_text())
    manifest=json.loads((out/'execution_manifest.json').read_text())
    labels=json.loads((d/'evidence/teacher_1024_20261121/labels.json').read_text())
    frozen=json.loads((d/'COHORT_1024_FROZEN.json').read_text())
    assert manifest['source_commit']=='588495c705093b8618f9cc41aa4303677df23f8e'
    assert raw['source_commit']==manifest['source_commit']
    assert manifest['label_source_sha256']==sha(d/'evidence/teacher_1024_20261121/labels.json')
    assert manifest['locked_cohort_sha256']==sha(d/'COHORT_1024_FROZEN.json')
    for name,hashval in manifest['raw_output_sha256'].items():
        assert sha(out/name)==hashval,name
    assert raw['training_samples']==1280
    assert len(frozen['rows'])==len(labels['rows'])==1024
    models=['old_mlp_cog06','mlp','conv']
    for kind in ('mlp','conv'):
        chosen=raw['training'][kind]['selection']['selected_epoch']
        curve=raw['training'][kind]['selection']['curve']
        assert len(curve)==80
        assert chosen==min(curve,key=lambda row:row['validation_loss'])['epoch']
        assert raw['training'][kind]['eval']['test']['n']==192
    for model in models:
        for split,count in [('train',640),('val',192),('test',192)]:
            items=predictions[split][model]
            labels_split=[r for r in labels['rows'] if r['split']==split]
            assert len(items)==len(labels_split)==count
            for score,truth in zip(items,labels_split):
                assert score['id']==truth['id'] and score['family']==truth['family']
                assert score['root_wdl']==truth['root_wdl']
                col=score['action']
                assert col in truth['legal_columns']
                outcome=truth['exact_action_scores'][col]
                assert outcome!=-1000
                sign=(outcome>0)-(outcome<0)
                assert score['chosen_WDL']==sign
                assert score['regret']==score['root_wdl']-sign
                assert 0<=score['prob_on_WDL_optimal_set']<=1+1e-6
            report=raw['training'][model]['eval'][split]
            assert abs(report['mean_regret']-
                       sum(item['regret'] for item in items)/count)<1e-12
    # Independent f64 exported model inference parity against frozen Torch
    # prediction rows on old train, val and heldout *before MCTS integration*.
    sys.path.insert(0,str(d))
    from generalize_cpu import compare_portable
    for kind in ('mlp','conv'):
        model=json.loads((out/(kind+'_model.json')).read_text())
        byid={row['id']:row for row in labels['rows']}
        sample=predictions['test'][kind][:12]+predictions['val'][kind][:12]
        for row in sample:
            original=byid[row['id']]
            mask=[c in original['legal_columns'] for c in range(7)]
            policy,value=compare_portable(model,original['features'],mask)
            best=max(range(7),key=lambda a:policy[a])
            assert best==row['action']
            assert abs(value-row['model_value'])<=3e-5,(kind,row['id'])
            assert abs(sum(policy[c] for c in original['optimal_actions'])-
                       row['prob_on_WDL_optimal_set'])<=3e-5
    def cluster_diff(split,first,second):
        a={r['id']:r for r in predictions[split][first]}
        b={r['id']:r for r in predictions[split][second]}
        byfam=defaultdict(list)
        for row in predictions[split][first]:
            byfam[row['family']].append((a[row['id']]['regret']-b[row['id']]['regret']))
        families=sorted(byfam)
        assert len(families)==12 and all(len(byfam[f])==16 for f in families)
        familymeans=[sum(byfam[f])/len(byfam[f]) for f in families]
        point=sum(familymeans)/len(familymeans)
        rng=random.Random(20261121)
        bootstrap=[sum(rng.choice(familymeans) for _ in familymeans)/len(familymeans)
                   for _ in range(3000)]
        return {'first_minus_second_mean_regret':point,
                'conditional_95_interval_family_bootstrap':quantile95(bootstrap),
                'family_mean_differences':dict(zip([str(x) for x in families],familymeans))}
    comparison={
        split:{
            'conv_minus_mlp':cluster_diff(split,'conv','mlp'),
            'conv_minus_old_mlp':cluster_diff(split,'conv','old_mlp_cog06'),
            'mlp_minus_old_mlp':cluster_diff(split,'mlp','old_mlp_cog06'),
        } for split in ('val','test')
    }
    conclusion={
        'schema':'quartz_connect4_generalization07_training_audit_v1',
        'source_commit':manifest['source_commit'],
        'old_model_source':sha(d.parent/'connect4_recognizer_06/evidence/nn_train_20261120/tiny_mlp_weights.json'),
        'output_source_sha256':manifest['raw_output_sha256'],
        'n_train_family':40,'n_val_family':12,'n_test_family':12,
        'n_train_val_test_rows':[640,192,192],
        'mirror_exact_board_classes_unique':True,
        'independent_portable_predictions_checked':{x:24 for x in ('mlp','conv')},
        'conditional_group_bootstrap_resamples':3000,
        'conditional_group_bootstrap_seed':20261121,
        'paired_regret_comparisons':comparison,
        'nonclaims':['bootstrap ignores retraining/validation stochasticity',
                     'no iid root generalization across unknown opening distributions',
                     'not same-CPU-time training/inference benchmark',
                     'not MCTS performance or VOC evidence'],
    }
    path=out/'INDEPENDENT_TRAINING_AUDIT.json'
    assert not path.exists()
    path.write_text(json.dumps(conclusion,indent=2,sort_keys=True)+'\n')
    print('TRAINING_INDEPENDENT_AUDIT_PASS',json.dumps({
        'test_conv_minus_mlp':comparison['test']['conv_minus_mlp'],
        'val_conv_minus_mlp':comparison['val']['conv_minus_mlp'],
        'test_conv_minus_old':comparison['test']['conv_minus_old_mlp'],
    },sort_keys=True))

if __name__=='__main__':main()
