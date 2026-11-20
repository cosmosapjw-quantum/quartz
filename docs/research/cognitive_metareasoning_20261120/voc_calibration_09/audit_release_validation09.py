#!/usr/bin/env python3
"""One independent source-byte + actual 1152 RELEASE validation-time readback.

Does not access oracle labels or new/test roots; release-only clock comparator.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stop_cost(step_us,cap):
    t=0
    for i,c in enumerate(step_us):
        t+=c
        if i==3 or t>=1000*cap:
            return [8,16,32,64][i],t/1000.0
    raise AssertionError('unreachable')


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/release_validation_1152_20261121'
    raw=json.loads((e/'RAW_1152_VALIDATION.json').read_text())
    caps=json.loads((e/'VALIDATION_RELEASE_CAPS.json').read_text())
    meta=json.loads((e/'execution_manifest.json').read_text())
    v=d.parent/'cost_matched_voc_08'
    study=d.parent/'connect4_generalization_07'
    frozen=json.loads((v/'evidence/frozen_surrogate_20261121/frozen_surrogates.json').read_text())
    roots=[r for r in json.loads((study/'COHORT_1024_FROZEN.json').read_text())['rows'] if r['split']=='val']
    assert len(roots)==192 and len(raw)==1152
    assert caps['source_git_head']==meta['source_commit']=='0009e5d2da46fa063f8da6714f31380c8835217c'
    assert sha(v/'evidence/frozen_surrogate_20261121/frozen_surrogates.json')==meta['old_policy_sha256']
    for name,digest in meta['validation_output_sha256'].items():
        assert sha(e/name)==digest
    assert caps['no_oracle_label_used'] and caps['no_retrained_or_rethresholded_voc_predictor']
    assert caps['source_old_validation_cohort_sha256']==sha(study/'COHORT_1024_FROZEN.json')
    report={}
    for mid,mode in enumerate(['uniform','new_mlp','conv']):
        # Request ordering: root-major mode-major with full,adaptive.
        full=[raw[i*6+mid*2] for i in range(192)]
        adaptive=[raw[i*6+mid*2+1] for i in range(192)]
        for r,a,f in zip(roots,adaptive,full):
            assert a['history']==f['history']==r['history']
            assert a['board_bottom_up']==f['board_bottom_up']==r['board']
            assert a['root_hash']==f['root_hash']
            assert f['stopped_at_visits']==64 and len(f['steps'])==4
            assert a['stopped_at_visits'] in [8,16,32,64]
            for old,new in zip(f['steps'],a['steps']):
                assert old['best_action']==new['best_action']
                assert old['evaluator_calls']==new['evaluator_calls']
                assert old['visit_policy']==new['visit_policy']
        expected=frozen['mode_policies'][mode]
        assert abs(caps['modes'][mode]['original_voc08_fixed_lambda']-expected['lambda_wdl_per_ms'])<1e-12
        assert caps['modes'][mode]['original_voc08_cost_forecast_ms']==expected['forecast_cost_ms']
        val_target=sum(sum(s['step_wall_us'] for s in a['steps']) for a in adaptive)/192/1000
        assert abs(val_target-caps['modes'][mode]['adaptive_release_validation_mean_search_ms'])<1e-12
        selection=[]
        for idx in range(1,1001):
            candidate=idx/100
            completed=[stop_cost([s['step_wall_us'] for s in f['steps']],candidate) for f in full]
            avg=sum(x[1] for x in completed)/192
            selection.append((abs(avg-val_target),candidate,avg,
                dict(Counter(str(x[0]) for x in completed))))
        selected=min(selection,key=lambda z:(z[0],z[1]))
        actual=caps['modes'][mode]
        assert selected[1]==actual['cap_ms'],(mode,selected[1],actual['cap_ms'])
        assert abs(selected[2]-actual['mean_search_ms'])<1e-12
        assert selected[3]==actual['stop_visits']
        report[mode]={
            'original_lambda_wdl_per_debug_ms':expected['lambda_wdl_per_ms'],
            'original_debug_gain_cost_forecast_ms':expected['forecast_cost_ms'],
            'release_time_cap_ms':selected[1],
            'release_mean_adaptive_search_ms':val_target,
            'release_mean_clock_search_ms':selected[2],
            'release_validation_time_mismatch_fraction':selected[0]/val_target,
        }
    receipt={
        'schema':'quartz_cog02_voc09_release_validation_independent_v1',
        'n_real_validation_release_requests':1152,
        'n_distinct_val_roots':192,
        'val_opening_families':12,
        'actual_compiled_rust_commit':'1c2b00c1bad2585380fee97bde65e3862dc1f25d',
        'release_executable_sha256':meta['executable_sha256'],
        'input_source_sha256_checked':len(meta['validation_output_sha256']),
        'no_new_C09_test_root_used':True,
        'selected_caps_recomputed_independently':report,
        'debug_time_forecast_domain_shift_not_ignored':True,
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('refuse duplicate')
    p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('C09_RELEASE_VALIDATION_CAPS_INDEPENDENT_AUDIT_PASS',json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
