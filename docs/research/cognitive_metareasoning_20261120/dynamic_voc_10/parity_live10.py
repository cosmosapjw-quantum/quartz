#!/usr/bin/env python3
"""Frozen model COG10 dynamic vs Rust release online 24 old VAL roots parity.

This validates the Rust two-step commitment against Python deterministic
full-trace replay BEFORE accessing completely new heldout C10 oracle labels.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def old_modules():
    d=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(d) not in sys.path:sys.path.insert(0,str(d))
    from evaluate_live08 import model_request,run_batch
    from train_gain08 import model_identity,current_root_features
    return model_request,run_batch,model_identity,current_root_features


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('parity test must run from committed clean source')
    from dynamic_model10 import predict_option,simulate_bounded_policy
    model_request,run_batch,model_identity,current_root_features=old_modules()
    g=d.parent/'connect4_generalization_07'
    c=json.loads((g/'COHORT_1024_FROZEN.json').read_text())
    rows=[x for x in c['rows'] if x['split']=='val' and x['id'].endswith(('_R00','_R01'))]
    assert len(rows)==24 and len({r['family'] for r in rows})==12
    src=d/'evidence/release_train_val_2496_20261121'
    original=json.loads((src/'JOINED_TRAIN_VAL_RELEASE.json').read_text())
    previous={(x['id'],x['mode']):x for x in original if x['split']=='val'}
    fitted_path=d/'evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json'
    frozen=json.loads(fitted_path.read_text())
    models=model_identity(repo)
    digests={key:sha(p) for key,p in models.items()}
    queries=[];keys=[]
    for case in rows:
        for mode in ('uniform','new_mlp','conv'):
            for arm in ('full','dynamic','myopic','myopic_time_match'):
                q=model_request(case,mode,'full',models,digests,None,None)
                if arm!='full':
                    q['dynamic_policy']=frozen['modes'][mode]['frozen_policies'][arm]
                queries.append(q)
                keys.append((case['id'],mode,arm))
    exe=args.binary.resolve()
    assert sha(exe)=='4db989b0ede0c56077de77a820a99cd777adcf87c98d7faedc62611491eb8a03'
    raw=run_batch(exe,queries)
    assert len(raw)==len(queries)==288
    results=[]
    found={}
    for (id,mode,arm),response in zip(keys,raw):
        found[(id,mode,arm)]=response
        if arm=='full':
            assert response['stopped_at_visits']==64
            continue
        model=frozen['modes'][mode]['frozen_policies'][arm]
        record=previous[(id,mode)]
        expected=simulate_bounded_policy(model,record,variant='dynamic' if arm=='dynamic' else 'myopic')
        assert response['stopped_at_visits']==expected['visits'],(id,mode,arm,expected,response['stopped_at_visits'])
        actual_skips=response['dynamic_skipped_checkpoints']
        assert actual_skips==expected['skipped_stages'],(id,mode,arm,actual_skips,expected['skipped_stages'])
        assert len(response['dynamic_decisions'])==len(expected['decisions'])
        for j,dec in enumerate(response['dynamic_decisions']):
            stage={8:0,16:1,32:2}[dec['at_visits']]
            x=current_root_features(response,stage)
            py=predict_option(model,stage,x,variant='dynamic' if arm=='dynamic' else 'myopic')
            assert dec['action']==py['action'],(id,mode,arm,dec,py)
            assert max(abs(a-b) for a,b in zip(x,dec['root_features']))<3e-9
            for key in ('predicted_single_gain','predicted_single_cost_ms','net_single'):
                assert abs(dec[key]-py[key])<3e-9,(id,mode,arm,key,dec[key],py[key])
            for key in ('predicted_joint_gain','predicted_joint_cost_ms','net_joint'):
                if dec[key] is None:
                    assert py[key] is None
                else:
                    assert abs(dec[key]-py[key])<3e-9
        full=found[(id,mode,'full')]
        for a,b in zip(full['steps'],response['steps']):
            assert a['best_action']==b['best_action'] and a['evaluator_calls']==b['evaluator_calls']
            assert a['visit_policy']==b['visit_policy']
        assert len(response['steps'])==list((8,16,32,64)).index(response['stopped_at_visits'])+1
        results.append({'id':id,'mode':mode,'arm':arm,'stop_visits':response['stopped_at_visits'],
                        'decisions':[d['action'] for d in response['dynamic_decisions']],
                        'committed_skipped_stage':actual_skips})
    for key,model in models.items():
        assert sha(model)==digests[key]
    receipt={
        'schema':'quartz_cog02_dynamic10_actual_Rust_vs_Python_parity_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'release_binary_sha256':sha(exe),
        'frozen_policy_SHA256':sha(fitted_path),
        'G07_val_position_count':len(rows),'val_families':12,
        'actual_release_requests':288,
        'tested_new_C10_holdout':False,
        'all_actual_Rust_dynamic_stops_and_two_stage_skips_agree_with_Python':True,
        'root_prefix_action_eval_and_visit_distribution_reproduced':True,
        'results':results,
        'dynamic_option_counts':dict(Counter(str((r['mode'],tuple(r['decisions']))) for r in results if r['arm']=='dynamic')),
    }
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    p=out/'PARITY_RESULTS.json'
    p.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    (out/'PARITY_MANIFEST.json').write_text(json.dumps({
        'source_commit':receipt['source_commit'],
        'release_binary_sha256':receipt['release_binary_sha256'],
        'frozen_policy_SHA256':receipt['frozen_policy_SHA256'],
        'source_validation_cohort_SHA256':sha(g/'COHORT_1024_FROZEN.json'),
        'results_sha256':sha(p),
        'n_actual_Rust_requests':288,
    },sort_keys=True,indent=2)+'\n')
    print('C10_FROZEN_RUST_PYTHON_DYNAMIC_PARITY_PASS',json.dumps({
        'n':288,'dynamic_counts':receipt['dynamic_option_counts'],
    },sort_keys=True))


if __name__=='__main__':main()
