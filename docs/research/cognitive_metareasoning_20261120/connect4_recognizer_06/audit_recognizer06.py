#!/usr/bin/env python3
"""Independent exact-score reread of a trained-vs-uniform heldout evaluation."""
from pathlib import Path
import hashlib
import json
import collections
import subprocess


def digest(f):
    with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def replay_legal(sequence):
    heights=[0]*7
    board=[0]*42
    for ply,mark in enumerate(sequence):
        col=int(mark)-1
        assert 0<=col<7 and heights[col]<6
        board[heights[col]*7+col]=1 if ply%2==0 else -1
        heights[col]+=1
    return [c for c in range(7) if heights[c]<6], board


def oracle_query(exe,book,history):
    completed=subprocess.run(
        [str(exe),"-a","-b",str(book)],input=(history+"\n").encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=12.0,check=True)
    split=completed.stdout.decode().split()
    assert len(split)==8 and split[0]==history
    return [int(x) for x in split[1:]]


def sign(value):
    assert value!=-1000 and -21<=value<=21
    return (value>0)-(value<0)


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/heldout12_nn_vs_uniform_20261120'
    study=json.loads((e/'summary.json').read_text())
    manifest=json.loads((e/'evaluation_manifest.json').read_text())
    result=json.loads((e/'paired_heldout.json').read_text())
    raw=json.loads((e/'rust_raw_responses.json').read_text())
    freeze=json.loads((d.parent/'standard_external_04/POSITIONS_FIXED.json').read_text())['connect4']
    teacher=json.loads((d/'evidence/teacher_20261120/labels.json').read_text())
    cohort=json.loads((d/'COHORT_FROZEN.json').read_text())
    binary=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4/c4solver')
    book=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4/7x6_small.book')
    mlp=d/'evidence/nn_train_20261120/tiny_mlp_weights.json'
    assert len(result)==len(freeze)==12 and len(raw)==24
    assert digest(mlp)==manifest['model_weights_sha256']
    for name,expected in manifest["outputs_sha256"].items():
        assert digest(e/name)==expected,name
    assert digest(d.parent/'standard_external_04/POSITIONS_FIXED.json')==manifest['frozen_heldout_position_sha256']
    assert digest(d/'evidence/teacher_20261120/labels.json')==manifest['teacher_labels_sha256']
    all_training={item['history'] for item in teacher['full_rows']}
    mirrors={''.join(str(8-int(x)) for x in item['history']) for item in teacher['full_rows']}
    assert all(f['sequence'] not in all_training|mirrors for f in freeze)
    forbidden={min(f['sequence'][:5],''.join(str(8-int(k)) for k in f['sequence'][:5])) for f in freeze}
    assert not forbidden & {min(c['history'][:5],''.join(str(8-int(k)) for k in c['history'][:5])) for c in cohort['rows']}
    signature={}
    row_metrics={m:[] for m in ('uniform','trained')}
    for idx,(pos,paired) in enumerate(zip(freeze,result)):
        sequence=pos['sequence']
        assert paired['history']==sequence and paired['id']==pos['id']
        legal,board=replay_legal(sequence)
        assert board==pos['board']
        expected=oracle_query(binary,book,sequence)
        assert expected==paired['exact_action_scores']
        reflected=''.join(str(8-int(x)) for x in sequence)
        assert oracle_query(binary,book,reflected)==list(reversed(expected))
        opt=max(sign(expected[c]) for c in legal)
        assert opt in (-1,0,1)
        for m,raw_index in [('uniform',idx),('trained',idx+12)]:
            rawrow=raw[raw_index]
            item=paired[m]
            assert rawrow['history']==sequence
            assert rawrow['eval_mode']==('trained_tiny_mlp_cpu' if m=='trained' else 'uniform_no_neural_network')
            assert len(rawrow['steps'])==4
            assert rawrow['fresh16']['resident_match']
            assert rawrow['root_hash']==item['root_hash']
            assert rawrow['board_bottom_up']==board
            prev=0
            utility=[]
            for i,step in enumerate(rawrow['steps']):
                assert step['budget']==[8,16,32,64][i]
                assert step['root_visits']==step['budget']
                assert step['root_hash']==rawrow['root_hash']
                action=step['best_action']
                assert action in legal
                assert step['evaluator_calls']>=prev
                assert step['evaluator_calls']==item['cumulative_eval_calls'][i]
                prev=step['evaluator_calls']
                utility.append(sign(expected[action]))
            assert utility==item['utility']
            assert [opt-u for u in utility]==item['regret']
            row_metrics[m].append(item)
    independent={}
    for i,budget in enumerate([8,16,32,64]):
        independent[str(budget)]={}
        for m,items in row_metrics.items():
            u=[item['utility'][i] for item in items]
            regret=[item['regret'][i] for item in items]
            calls=[item['cumulative_eval_calls'][i] for item in items]
            independent[str(budget)][m]={
                "average_WDL":sum(u)/len(u),
                "average_regret":sum(regret)/len(regret),
                "n_optimal":sum(z==0 for z in regret),
                "average_evaluator_calls":sum(calls)/len(calls),
            }
            z=study['by_budget'][str(budget)][m]
            for key,target in [
                ('mean_utility',sum(u)/len(u)),
                ('mean_regret',sum(regret)/len(regret)),
                ('optimal_WDL_count',sum(z==0 for z in regret)),
                ('mean_evaluator_calls',sum(calls)/len(calls)),
            ]:
                assert z[key]==target,(m,budget,key)
    paired_diff={str(b):collections.Counter(
        item['trained']['utility'][i]-item['uniform']['utility'][i] for item in result
    ) for i,b in enumerate([8,16,32,64])}
    outcome={
        "schema":"quartz_c4_recognizer06_independent_readback_v1",
        "n_old_heldout":12,
        "fresh_oracle_queries":12,
        "fresh_mirror_queries":12,
        "source_output_hashes_verified":len(manifest["outputs_sha256"]),
        "mlp_model_hash_verified":True,
        "no_training_heldout_family_overlap":True,
        "all_fresh16_reproduced":True,
        "independent_budget_metrics":independent,
        "paired_model_minus_uniform_WDL_counts":{b:{str(k):v for k,v in sorted(c.items())} for b,c in paired_diff.items()},
        "science_status":"numerically checked tiny learned recognizer + actual QUARTZ Rust MCTS on 12 non-iid roots, not statistical generalization",
        "warnings":[
            "no CPU cost-matched policy comparison: only matched MCTS root visits",
            "heldout 12 roots all black to move",
            "small sample/selected opponent positions",
            "model training family validation generalized poorly",
            "no Bayesian VOC metacontroller implemented",
        ],
    }
    target=e/'INDEPENDENT_READBACK.json'
    if target.exists():raise FileExistsError("refuse overwrite independent readback")
    target.write_text(json.dumps(outcome,indent=2,sort_keys=True)+'\n')
    print(json.dumps(outcome,sort_keys=True))


if __name__=='__main__':main()
