"""Derive durable K57 report only from actual locked records."""
import json,pathlib,hashlib,subprocess as sp
from k57_plan import D,T,ROOT,load,sha
B=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic57_early_backups'
def main():
    a=load('AUDIT.json');s=load('SUMMARY.json');f=load('FROZEN.json');p=load('PARITY.json')['counts'];rp=load('RAPFI_REFERENCE.json')['counts'];b=load('BINARY_LOCK.json')
    st=json.loads((T/'STATS_INDEPENDENT.json').read_text());assert st['predictor_fraction_checked']
    assert a['total']['games']==3776 and a['total']['episodes']==896 and p['roots']==56 and rp['roots']==72
    old=D.parent/'quartz_kinetic56_voc_20261128/NATIVE_IDENTITY.json';i=json.loads(old.read_text())
    for q,digest in b['source_sha256'].items():assert sha(ROOT/q)==digest==hashlib.sha256(sp.check_output(['git','show',b['source_commit']+':'+q],cwd=ROOT)).hexdigest()
    assert sha(T/'mcts_demo')==b['binary_sha256']==sha(B/'mcts_demo') and b['old_binary_sha256']==i['binary_sha256']
    assert sha(ROOT/'Cargo.lock')==b['cargo_lock_sha256']
    for q in s['locks']:assert load(q+'_LOCK.json')['binary']==b['binary_sha256']
    assert '663 passed; 0 failed; 89 ignored' in (T/'FULL_TEST.log').read_text()
    i.update(k54_binary_sha256=i['parent_parent_binary_sha256'],parent_parent_binary_sha256=i['parent_binary_sha256'],binary_sha256=b['binary_sha256'],binary_bytes=b['bytes'],source_commit=b['source_commit'],source_sha256=b['source_sha256'],cargo_lock_sha256=b['cargo_lock_sha256'],rustc=b['rustc'],parent_binary_sha256=b['old_binary_sha256'],parent_native_identity_sha256=sha(old),full_test_raw_sha256=sha(T/'FULL_TEST.log'),scope='K57 opt-in early root trajectory and actual early target; independent fallible epoch memo seam, no async integration')
    (D/'NATIVE_IDENTITY.json').write_text(json.dumps(i,indent=2,sort_keys=True)+'\n')
    n=s['nn']['all'];tc=a['total'];ct=a['counts']
    lines=['# K57 early root information and decision value','',
    '## Authority and theory',
    'Parent 76acccfeccbf587a0e588f0e50f47cf2772b3ce2 clean, remote equal,145-file restore verified. Reread 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS. Bayesian simple regret decrease equals conditional E[theta_new-theta_old], by tower, factorizes flip probability and signed changed-action benefit. Jensen nonnegative for posterior-mean maximizing decision only; arbitrary PUCT not guaranteed. Actual finite opponent paired gain is not minimax regret. Finite checks63 factor,625 Jensen,396 boundary enumeration,20 cache accounting; Wolfram4 True. R10 4479e56b04fe47b88658f168788a032a76f31dbe. R01/R03 whitespace failures repaired.',
    '## Implementation',
    f"Native source {b['source_commit']}, SHA {b['binary_sha256']}, {b['rustc']}. First noncert Q turn only, assigned h=4/8/12/16, common Q20 prefix. Pre-lock snapshots at reached horizons, no extra evaluator calls; if locked at h or before or no nondominated rival, all arms continue unchanged Q20. Otherwise actual STOP at h, COLD fresh child5, REUSE retained child5, Q20 reference continues. Exact proof intervals and physical cache preserved, cap h intentional stop not original lock. Native red-green and full regression663/0/89. Actual56 old/new Q20 parity,52 eligible; counters {p}. EpochMemo57 separate fallible synchronous cache seam, test verifies failure not cached, capacity and generation invalidation; no integration with real async or mutable weights. GPU/native subagent unavailable, no training/main merge. "+(T/'SCHEDULE.log').read_text().strip()+'.',
    '## Fresh NN',
    '64 families,32 TRAIN/16 CAL/16 TEST, each horizon balanced within split, seven fixed checkpoints and paired colors, four actual Q-v-B32 policies=3584 games. Symmetric exact cache. 896 paired episodes including all no-event. Family bootstrap8000. Early policy effect not same as late K52.',
    '|arm|W/D/L|score|NN saving|NN lower95|CPU saving|gate|','|---|---|---:|---:|---:|---:|---|']
    for arm,q in n['arms'].items():
        v=q['physical']['metrics'];w=q['physical']['counts'];l=q['physical']['lower95'];lines.append(f"|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{v['NN_reduction']:.9f}|{l['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{q['physical']['gate']}|")
    lines+=['','|contrast|reached|changed|gain points|local physical delta|target extra|base delta|net Q|net total|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,q in n['pairs'].items():
        v=q['counts'];lines.append(f"|{arm}|{v['reached']}|{v['changed']}|{v['delta2']/2:g}|{v['local_nn']}|{v['target_extra']}|{v['base_delta']}|{v['delta_Q']}|{v['delta_total']}|")
    for arm,q in n['pairs'].items():lines.append(f"{arm}: paired score {q['metrics']['tau']}, CI {q['CI']['tau']}, factor {q['factor']}; local gain/call {q['ratio']['local_nn']}.")
    lines+=['','|assigned h|reached|STOP vs Q20 gain|COLD vs STOP gain|REUSE vs STOP gain|COLD vs Q20 gain|REUSE vs Q20 gain|','|---|---:|---:|---:|---:|---:|---:|']
    for h in ('4','8','12','16'):
        q=s['nn'][h]['pairs'];lines.append(f"|{h}|{q['stop']['counts']['reached']}|{q['stop']['counts']['delta2']/2:g}|{q['cold']['counts']['delta2']/2:g}|{q['reuse']['counts']['delta2']/2:g}|{q['cold_vs20']['counts']['delta2']/2:g}|{q['reuse_vs20']['counts']['delta2']/2:g}|")
    lines.append(f"Actual first unresolved root reasons {s['reasons']}; snapshot count {s['snapshot_counts']}; support {s['support']}. Assigned h changes base computation and survivor population, not simply one common cohort followed to every h. No-event included; local early-vs-Q20 cost may be negative, ratio only positive denominator.")
    lines+=['','## Factorized conditional predictor',
    f"TRAIN-only model {f}, freeze before CAL/TEST analysis, P(flip) Beta(1,1), conditional changed signed mean zero pseudo-count4; exact fractions independently verified. Constant train mean baseline. Offline replay chooses largest strictly positive predicted gain, no physical budget claim and no live new policy.",
    '|split|arm|reached|Brier|constant Brier|gain MSE|constant gain MSE|','|---|---|---:|---:|---:|---:|---:|']
    for split,q in s['predictive'].items():
        for arm in ('cold','reuse'):
            z=q[arm];m=z['metrics'];lines.append(f"|{split}|{arm}|{z['counts']['n']}|{m['brier']:.9f}|{m['brier0']:.9f}|{m['mse']:.9f}|{m['mse0']:.9f}|")
    for split,q in s['predictive'].items():lines.append(f"{split} replay {q['replay']}.")
    lines+=['','## Rapfi',
    f"72 archived qualified K56 root offline reference, actual counts {rp}; Q128 is a finite budget comparator, not oracle. TARGET action agreement with Q128 is not decision quality. Additional fresh48 families24 compact24 outer, Q96 and TARGET each separately vs same B128=192 games.",
    '|stratum|arm|W/D/L|score|score lower95|NN saving|NN lower95|CPU saving|gate|','|---|---|---|---:|---:|---:|---:|---:|---|']
    for pan,z in s['rapfi'].items():
        for arm,q in z['arms'].items():
            v=q['metrics'];w=q['counts'];l=q['lower95'];lines.append(f"|{pan}|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{l['score']:.9f}|{v['NN_reduction']:.9f}|{l['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{q['gate']}|")
    for pan,z in s['rapfi'].items():lines.append(f"{pan}: paired score {z['paired']['metrics']['tau']} CI {z['paired']['CI']['tau']}, counts {z['paired']['counts']}. Small stratum diagnostic, not superiority confirmation.")
    physical=tc['physical']+tc['nn'];logical=tc['logical']+tc['nn'];extra=p['old_nn']+p['control_nn']+p['early_nn']+sum(rp[k] for k in ('nn96','nnT','nn128'))
    lines+=['','## Audit and next',
    f"Actual {tc['games']} games, {tc['moves']} moves, {physical} physical and {logical} logical game calls; {extra} separate parity/reference calls, total {physical+extra}. Game proof receipts {ct.get('proof_receipts',0)}, bounds {ct.get('independent_bound_cells',0)}, queries {ct.get('proof_queries',0)}, early snapshots {ct.get('early_snapshots',0)}, cold child receipts {ct.get('child_receipts',0)}, reuse child receipts {ct.get('reuse_child_receipts',0)}, Rapfi waves {ct.get('waves',0)}. Independent stdlib family8000 bootstrap and rational model verification. Exact late roots {a['natural_exact']}.",
    'K58: exploit early snapshot variation using actual matched score and action changes; choose future estimator only from frozen TRAIN and new cohort, not holdout retuning. Compare early targeted vs continue with physical cost and account for censoring and prior/backup gaps; use separate new actual matched games for selected rule. Do not interpret Q128 agreement as truth. Next integrate fallible epoch seam with strict real history-plane adapter including model transition and failure-recovery tests; generic async still open. Original posterior P1, A1 P2 and emergent P3 remain open. Dropbox sync NOT_VERIFIED.']
    (D/'REPORT.md').write_text('\n'.join(x.rstrip() for x in lines)+'\n')
    with (D/'ORIGIN_WATCHDOG.md').open('a') as w:w.write(f"\n## Outcome\nActual early feature support {s['support']}; paired effect COLD vs STOP {n['pairs']['cold']['metrics']['tau']} CI {n['pairs']['cold']['CI']['tau']}; REUSE {n['pairs']['reuse']['metrics']['tau']} CI {n['pairs']['reuse']['CI']['tau']}. Factorized predictor on CAL/TEST {[(k,{a:v['metrics'] for a,v in q.items() if a!='replay'}) for k,q in s['predictive'].items() if k!='TRAIN']}. Finite opponent gain not posterior regret; no A1/P3 substitution. Rapfi Q128 disagreement and transfer scoped. Epoch cache seam tested, no async integration.\n")
    print('FINAL_REPORT',tc,'physical',physical,'logical',logical,'extra',extra,'total',physical+extra,flush=True)
if __name__=='__main__':main()
