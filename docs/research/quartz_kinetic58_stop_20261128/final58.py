"""Durable K58 scientific report, actual locked metrics and status."""
import json,pathlib,hashlib,subprocess as sp
from k58_plan import D,T,ROOT,load,sha
B=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic58_stop_backups'
def main():
    a=load('AUDIT.json');s=load('SUMMARY.json');p=load('PARITY.json')['counts'];h=load('HISTORICAL.json')['counts'];g=load('GENERIC.json');st=load('STATS_INDEPENDENT.json');z=load('STANDARDIZED.json');hx=load('HISTORICAL_EFFECT.json');b=load('BINARY_LOCK.json')
    assert a['total']['games']==7104 and a['total']['episodes']==2240 and p['fixed_requests']==56 and p['roots']==28
    old=D.parent/'quartz_kinetic57_early_20261128/NATIVE_IDENTITY.json';i=json.loads(old.read_text())
    for q,digest in b['source_sha256'].items():assert sha(ROOT/q)==digest==hashlib.sha256(sp.check_output(['git','show',b['source_commit']+':'+q],cwd=ROOT)).hexdigest()
    assert sha(T/'mcts_demo')==b['binary_sha256']==sha(B/'mcts_demo') and b['old_binary_sha256']==i['binary_sha256']
    assert sha(ROOT/'Cargo.lock')==b['cargo_lock_sha256']
    for q in s['locks']:assert load(q+'_LOCK.json')['binary']==b['binary_sha256']
    assert '664 passed; 0 failed; 89 ignored' in (T/'FULL_TEST.log').read_text()
    i.update(k55_binary_sha256=i['parent_parent_binary_sha256'],parent_parent_binary_sha256=i['parent_binary_sha256'],binary_sha256=b['binary_sha256'],binary_bytes=b['bytes'],source_commit=b['source_commit'],source_sha256=b['source_sha256'],cargo_lock_sha256=b['cargo_lock_sha256'],rustc=b['rustc'],parent_binary_sha256=b['old_binary_sha256'],parent_native_identity_sha256=sha(old),full_test_raw_sha256=sha(T/'FULL_TEST.log'),scope='K58 opt-in native first stable snapshot STOP, K57 fixed behavior retained; generic model identity/restart test, EpochMemo57 not integrated')
    (D/'NATIVE_IDENTITY.json').write_text(json.dumps(i,indent=2,sort_keys=True)+'\n')
    n=s['nn']['REPLICATION'];tc=a['total'];ct=a['counts'];p0=n['comparison']['stop'];ni=all(p0['criteria'].values())
    q=st['nn']['REPLICATION']['stop'];ni2=q['delta_score']['lower95']>=-.02 and q['delta_saving']['lower95']>=.05 and q['score']['lower95']>=.45 and q['saving']['lower95']>=.30 and n['arms']['stop']['physical']['metrics']['CPU_reduction']>0
    lines=['# K58 independent early stopping and stable trajectory','',
    '## Authority and derivation',
    'Parent 65fe001f5348b966fc80eb2c746c03527f428e11 clean/remote equal,134-file restore verified. Reread 15f3a198 README/QUARTZ_THEORY,7241ad0c THESIS. P1 expected posterior simple regret per compute, P2 original A1 anchor, P3 emergent signatures. K57 generator confounds assigned h and central/whole distribution. K58 primary128 fresh families preserve joint distribution; separate32 reverse-generator families. Strictly prespecified paired NI margin2pp and relative NN saving increment5pp, both one-sided95 family percentile lower, plus absolute B32 gate. New design choices, not original A1. Stratified by h, fixed checkpoint panel. Exact finite checks1320 mean update,3969 cross multiplication,2401 ratio,4 geometry, Wolfram4 true. R10 c6d80c54e8b3368f90f02cd88c4110f84ae677bd. R03 whitespace fixed before publication. Stable criterion uses same raw/proof-safe incumbent/rival and backed-up mean order, nondecreasing vote distance, visited both, not locked20. First eligible 8/12/16, later Q20. No posterior guarantee; future bounded returns can reverse mean ordering.',
    '## Native implementation and real strict NN tests',
    f"Source {b['source_commit']}, SHA {b['binary_sha256']}, {b['rustc']}. Actual red-green test and release, full Rust664/0/89. Fixed K57 old/new exact parity {p}. Historical predecision applicability {h}; only actually played matched h historical outcome subgroup {hx['counts']}, no unplayed terminal counterfactual. Strict generic request-local model A/B/A and restart parity, counts {g['counts']}; malformed identity rejected with process exit {g['wrong_identity']['returncode']}. Not in-process recovery: EpochMemo57 seam not wired into strict endpoint; async generation invalidation and training integration remain open. No GPU/native subagents/new training/main merge. "+(T/'SCHEDULE.log').read_text().strip()+'.',
    '## Actual fresh games and primary NI',
    '160 NN families x7 fixed checkpoints x2 colors x3 separate Q-v-identical-B32=6720 games. Primary128 families (5376 games), crossover32 (1344 games). Both arms same cache. Primary bootstrap8000 stratified family resamples preserving horizon proportions; unstratified sensitivity separate. Every no-event retained. No Q-v-Q, B-v-B or cross-evaluator games.',
    '|split|policy|W/D/L|score|physical NN saving|CPU saving|absolute gate|','|---|---|---|---:|---:|---:|---|']
    for split,q in s['nn'].items():
        for arm,v in q['arms'].items():
            w=v['physical']['counts'];m=v['physical']['metrics'];lines.append(f"|{split}|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{m['score']:.9f}|{m['NN_reduction']:.9f}|{m['CPU_reduction']:.9f}|{v['physical']['gate']}|")
    lines+=['','|split|contrast vs Q20|score delta|score lower95|saving delta|saving lower95|criteria|','|---|---|---:|---:|---:|---:|---|']
    for split,q in s['nn'].items():
        for arm,v in q['comparison'].items():lines.append(f"|{split}|{arm}|{q['pairs'][arm]['metrics']['tau']:.9f}|{v['score_lower95']:.9f}|{v['delta_saving']:.9f}|{v['saving_lower95']:.9f}|{v['criteria']}|")
    lines.append(f"Primary fixed STOP NI and efficiency joint={ni}; independent stdlib resampling joint={ni2}, lower score/saving {st['nn']['REPLICATION']['stop']}; unstratified sensitivity {n['unstratified']['stop']}. Other cohorts/policies exploratory; do not promote on one sample or generalize to A1/human play.")
    for split,q in s['nn'].items():
        for arm,v in q['pairs'].items():lines.append(f"{split}/{arm}: paired counts {v['counts']}, score CI {v['CI']['tau']}, signed local and downstream cost CI {v['CI']['local_nn']}/{v['CI']['delta_total']}.")
    lines.append(f"Events and actual stop horizons {s['events']}; eight cells {s['cell_points']}.")
    lines+=['','## Equal-generator descriptive comparison',
      'Central and whole-board generators overlap in support; they are not disjoint geometry classes. Equal weighting of central/whole at each assigned horizon, independent family resampling within eight cells. Aggregate weighted calls/moves before dividing. This secondary standardization does not replace joint-distribution primary inference.',
      '|h|STOP score delta|STOP saving delta|STABLE score delta|STABLE saving delta|','|---|---:|---:|---:|---:|']
    for h0,q in z['standardized'].items():lines.append(f"|{h0}|{q['stop']['delta_score']:.9f}|{q['stop']['delta_saving']:.9f}|{q['stable']['delta_score']:.9f}|{q['stable']['delta_saving']:.9f}|")
    lines+=['','## Rapfi compact independent diagnostic',
    '96 new compact families xpaired colors xQ96/TARGET separately vs same B128=384 games. Same prior score>=.45, NN>=.30 lower95, CPU positive. No direct Q-v-Q games.',
    '|arm|W/D/L|score|score lower95|NN saving|NN lower95|CPU saving|gate|','|---|---|---:|---:|---:|---:|---:|---|']
    for arm,q in s['rapfi']['arms'].items():
        w=q['counts'];v=q['metrics'];l=q['lower95'];lines.append(f"|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{l['score']:.9f}|{v['NN_reduction']:.9f}|{l['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{q['gate']}|")
    lines.append(f"Paired TARGET-Q96 effect {s['rapfi']['paired']['metrics']['tau']} CI {s['rapfi']['paired']['CI']['tau']}, counts {s['rapfi']['paired']['counts']}; independent {st['rapfi']}. Prior K57 compact small-cohort failure retained, not overwritten.")
    physical=tc['physical']+tc['nn'];logical=tc['logical']+tc['nn'];extra=p['old_nn']+p['new_nn']+p['stable_nn']+g['counts']['physical_success']+g['counts']['physical_rejected']
    lines+=['','## Audit and next',
    f"Actual {tc['games']} games, {tc['moves']} moves, {physical} physical and {logical} logical game calls; {extra} separate actual parity and generic calls, total {physical+extra}. Independent game proof receipts {ct.get('proof_receipts',0)}, bounds {ct.get('independent_bound_cells',0)}, proof queries {ct.get('proof_queries',0)}, early snapshot checks {ct.get('early_snapshots',0)}, Rapfi waves {ct.get('waves',0)}. Natural exact {a['natural_exact']}. Independent low-memory streaming check during host I/O stall: all28 gzip records,2240 unique episodes, shard totals and final hashes equal locked files; proof and full statistics subsequently independent. Separate integer/fraction statistics and family8000 bootstrap; same-model fixed panel and complete family outcomes, no universal proof.",
    'K59: if fixed STOP joint gate passed, freeze first-event scoped candidate and test repeated use across turns under a new paired NI design; if not, diagnose separate geometry and horizon support before new policy. For STABLE, use actual conditional stopping loss and probability of *harm*, rather than merely action agreement; preserve scarce changed outcomes and use matched data without retuning test. Collect actual full-horizon Q20 and all allowed early STOP counterfactual games on same roots if stronger temporal identification needed. Rapfi keep Q96 and TARGET separate, no superiority claim. Generic actual generation-tagged fallible memo seam not integrated; require same-process typed error propagation and epoch-safe pending requests before async. Original P1 posterior regret, A1 P2, P3 open. Dropbox sync NOT_VERIFIED.']
    (D/'REPORT.md').write_text('\n'.join(x.rstrip() for x in lines)+'\n')
    with (D/'ORIGIN_WATCHDOG.md').open('a') as w:w.write(f"\n## Outcome\nPrimary STOP joint={ni}; independent joint={ni2}; paired score lower={p0['score_lower95']}, extra saving lower={p0['saving_lower95']}. STABLE {n['comparison']['stable']}. Crossed and standardized diagnostics separate. Scoped real generic model isolation/restart; no async. No original A1/P3 substitution or posterior optimality claim.\n")
    print('FINAL_REPORT',tc,'physical',physical,'logical',logical,'extra',extra,'total',physical+extra,'primary',ni,'independent',ni2,flush=True)
if __name__=='__main__':main()
