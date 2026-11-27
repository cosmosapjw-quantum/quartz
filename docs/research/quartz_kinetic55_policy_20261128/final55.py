"""Report assembled from actual locked results."""
import pathlib,json,hashlib,subprocess as sp
from fractions import Fraction as F
from k55_plan import D,T,ROOT,load,sha
B=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic55_policy_backups'
def main():
    a=load('AUDIT.json');s=load('SUMMARY.json');f=load('FROZEN.json');m=load('MIX_FROZEN.json');b=load('BINARY_LOCK.json')
    p=json.loads((T/'PARITY.json').read_text());st=json.loads((T/'STATS_INDEPENDENT.json').read_text());pt=p['counts']
    assert a['total']['games']==4992 and a['total']['episodes']==896 and pt['requests']==42
    assert pt['old_nn']==pt['new_nn']==pt['selector_nn']
    assert f['table']==[0,0,0,0] and m['low']==[0,0,0,0] and m['high']==[2,0,0,0]
    # Independent direct dual certificate for expected-budget solution: all other positive options have smaller gain/cost.
    cells=f['panels']['53']['cells'];lam=F(14,2830)
    for z in cells:
        if z.get('n',0)<20:continue
        assert all(F(z[k+'_gain2'])-lam*z[k+'_cost']<=0 for k in ('cold','reuse'))
    assert F(m['expected_gain2'])==lam*f['cap'] and F(m['prob_num'],m['prob_den'])*2830==f['cap']
    old=D.parent/'quartz_kinetic54_cache_20261127/NATIVE_IDENTITY.json';identity=json.loads(old.read_text())
    for q,digest in b['source_sha256'].items():assert sha(ROOT/q)==digest==hashlib.sha256(sp.check_output(['git','show',b['source_commit']+':'+q],cwd=ROOT)).hexdigest()
    assert sha(T/'mcts_demo')==b['binary_sha256']==sha(B/'mcts_demo') and b['old_binary_sha256']==identity['binary_sha256']
    assert sha(ROOT/'Cargo.lock')==b['cargo_lock_sha256']
    for q in s['locks']:assert load(q+'_LOCK.json')['binary']==b['binary_sha256']
    assert '660 passed; 0 failed; 89 ignored' in (T/'FULL_TEST.log').read_text()
    identity.update(k52_binary_sha256=identity['parent_parent_binary_sha256'],parent_parent_binary_sha256=identity['parent_binary_sha256'],binary_sha256=b['binary_sha256'],binary_bytes=b['bytes'],source_commit=b['source_commit'],source_sha256=b['source_sha256'],cargo_lock_sha256=b['cargo_lock_sha256'],rustc=b['rustc'],parent_binary_sha256=b['old_binary_sha256'],parent_native_identity_sha256=sha(old),full_test_raw_sha256=sha(T/'FULL_TEST.log'),scope='K55 opt-in first-event native table and explicit synchronous immutable exact key evaluator; no async integration')
    (D/'NATIVE_IDENTITY.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    n=s['nn']['all'];tc=a['total'];ct=a['counts']
    lines=['# K55 contextual computation selection and Rapfi distribution transfer','',
    '## Authority and derivation',
    'Parent 4127fdf097bd01e40b7642168db5254a1692fb8e clean, remote equal and161-file restore verified. Reread 15f3a198 README/QUARTZ_THEORY and7241ad0c THESIS. Original P1 posterior simple regret per compute, P2 A1 anchor, P3 emergent signatures. Complete paired finite opponent outcomes, four pre-event cells bin3/7 and ply<=8/>8. Expected gain vs STOP under local logical budget; finite table enumeration not equivalent to ratio maximization. Exact tests243 score telescoping,400 ratio,1296 feasibility; Wolfram three identities true. R10 997bd5c80f7bf1841e5921b1f730b2683f2edd72. R01/R03 whitespace check failed and corrected before publication.',
    '## Historical fitting',
    f"K51-54 historical3360 matched episodes, fit only K53 complete STOP/COLD/REUSE. Integer table {f['table']}; cap {f['cap']}, feasible{f['feasible']}. K53 positive cell0 cost3396(COLD) or2830(REUSE) exceeds cap; cell2 negative; cells1/3 sparse. No negative gain forced. Historical cold diagnostics differ by cycle, K54 second-event cohort only descriptive: {f['second_cells']}. FROZEN.json retains all aggregate locks.",
    f"Before fresh data, separately amend with MIX expected-budget LP; extreme point at most one mixed cell. Exact optimum {m['low']} and {m['high']} with probability {m['prob_num']}/{m['prob_den']}, expected score2 gain {m['expected_gain2']} and logical cost {m['expected_logical_cost']}. Independent dual certificate lambda14/2830: all eligible operations G-lambda C<=0, attained at mixture, cap saturated. This is optimal for fitted expected-cost LP, not realized cap, future physical efficiency or posterior regret. Independent hashed pre-event coin, fixed original families/model/color and original relative arm order; original integer SELECT actually retained. No fresh outcomes seen before C03A.",
    '## Implementation',
    f"Source {b['source_commit']}, native SHA {b['binary_sha256']}, {b['rustc']}. research_gomoku7_k55 dispatch at same low20 event, STOP consumes first event without child, COLD/REUSE unchanged. Actual red-green tests and release build, Rust660/0/89. Generic synchronous FrozenExactEval55 opt-in bounded exact key closure, immutable evaluator identity contract. Same-board different recent-history keys checked; capacity1 test(4,3,1,1); actual sync MctsEngine uniform evaluator first(21,21,0,21), second cumulative(42,21,21,21), root snapshot equal and logical engine21. Forwarded calls not necessarily physical NN rows. No batch/train fallback safety claim; generic real NN not enabled. Actual old/new/selector parity {pt}; no GPU/native subagent/new training/main merge. Scheduling "+(T/'SCHEDULE.log').read_text().strip()+'.',
    '## Fresh actual NN experiment',
    '64 fresh families x7 fixed checkpoints x2 colors x5 separate actual Q-v-B32=4480 games. Symmetric exact per-request cache. All896 SELECT and896 MIX histories, logical/physical counts and semantic receipts match chosen fixed arm; independently checked. No prohibited matchups. Family bootstrap8000; fresh policy not tuned.',
    '|arm|W/D/L|score|logical NN saving|physical NN saving|physical lower95|CPU saving|gate|','|---|---|---:|---:|---:|---:|---:|---|']
    for arm,q in n['arms'].items():
        v=q['physical']['metrics'];w=q['physical']['counts'];l=q['physical']['lower95'];lines.append(f"|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{q['logical']['metrics']['NN_reduction']:.9f}|{v['NN_reduction']:.9f}|{l['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{q['physical']['gate']}|")
    lines+=['','|contrast vs STOP|gain points|improved/worsened|active|physical local|logical local|net Q|net total|','|---|---:|---|---:|---:|---:|---:|---:|']
    for arm,q in n['pairs'].items():
        v=q['counts'];lines.append(f"|{arm}|{v['delta2']/2:g}|{v['improved']}/{v['worsened']}|{v['interventions']}|{v['local_nn']}|{q['logical_extra_total']}|{v['delta_Q']}|{v['delta_total']}|")
    for arm,q in n['pairs'].items():
        lines.append(f"{arm} paired ITT {q['metrics']['tau']} CI {q['CI']['tau']}; local gain/call {q['ratio']['local_nn']}; signed net total ratio {q['ratio']['delta_total']}.")
    lines.append(f"Coin {s['coin']}; conditional fresh cells {s['diagnostic']}; MIX vs COLD score {n['mix_vs_cold']['score']}; MIX vs REUSE score {n['mix_vs_reuse']['score']}; panel effects old/new MIX {s['nn']['old']['pairs']['mix']['CI']['tau']} / {s['nn']['new']['pairs']['mix']['CI']['tau']}. No automatic promotion from one sample; signed denominator guards preserved. Actual MIX cost may differ from expected training budget. No posterior optimality.")
    lines+=['','## Rapfi transfer',
    '64 compact and64 constructed tactical six-stone opening families, paired colors, Q96 and frozen K53 TARGET each separately vs B128=512 actual games. Tactical two parallel adjacent three-stone runs, not human population. Separate strata primary diagnostics.',
    '|stratum|arm|W/D/L|score|score lower95|NN saving|NN lower95|CPU saving|gate|','|---|---|---|---:|---:|---:|---:|---:|---|']
    for pan,z in s['rapfi'].items():
        for arm,q in z['arms'].items():
            v=q['metrics'];w=q['counts'];l=q['lower95'];lines.append(f"|{pan}|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{l['score']:.9f}|{v['NN_reduction']:.9f}|{l['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{q['gate']}|")
    for pan,z in s['rapfi'].items():
        q=z['paired'];lines.append(f"{pan}: paired score {q['metrics']['tau']} CI {q['CI']['tau']}, counts {q['counts']}; independent target score/saving lower {st['rapfi'][pan]['T']['score']['lower95']}/{st['rapfi'][pan]['T']['NN_reduction']['lower95']}. No universal claim or paired superiority without positive CI.")
    physical=tc['physical']+tc['nn'];logical=tc['logical']+tc['nn'];parity=pt['old_nn']+pt['new_nn']+pt['selector_nn']
    lines+=['','## Audit and next',
    f"Actual {tc['games']} games, {tc['moves']} moves, {physical} physical game calls, {logical} logical game calls, {parity} actual separate NN parity calls. Generic uniform evaluator tests not NN calls. Cache hits {ct.get('cache_hits',0)}, proof receipts {ct.get('proof_receipts',0)}, bounds {ct.get('independent_bound_cells',0)}, certificates {ct.get('independent_partial_cert',0)}, proof queries {ct.get('proof_queries',0)}, votes {ct.get('votes',0)}, cold child receipts {ct.get('child_receipts',0)}, reuse child receipts {ct.get('reuse_child_receipts',0)}, Rapfi child receipts {ct.get('rapfi_children',0)}, waves {ct.get('waves',0)}. Natural exact {a['natural_exact']}. Independent coordinate/chronological/physical audits and stdlib family8000 bootstrap.",
    'K56: replace coarse global bin budget by predictive conditional uncertainty/utility with independent prospective validation (retain STOP and frozen single COLD reference, no tuning on fresh test). Explore budgeted support and uncertainty-aware robust estimators; if MIX not positive, do not promote. Rapfi transfer gates per stratum, separate tactical proof-dominated scope and absolute calls; no automatic cap escalation. Generic exact wrapper is only synchronous API seam with explicit key; next real NN history-aware input adapter and successful-forward identity/status before async batching. Original P1 posterior optimization, P2 A1, P3 remain open. Dropbox sync NOT_VERIFIED.']
    (D/'REPORT.md').write_text('\n'.join(x.rstrip() for x in lines)+'\n')
    with (D/'ORIGIN_WATCHDOG.md').open('a') as w:w.write(f"\n## Outcome\nInteger table STOP, expected-budget mixture {m['prob_num']}/{m['prob_den']} REUSE cell0. Fresh MIX effect {n['pairs']['mix']['metrics']['tau']} CI {n['pairs']['mix']['CI']['tau']}; no posterior proof. Rapfi compact/tactical target gates {s['rapfi']['compact']['arms']['T']['gate']} / {s['rapfi']['tactical']['arms']['T']['gate']}. Explicit sync exact-key wrapper test, no async/training claim. Original A1 and P3 still open.\n")
    print('FINAL_REPORT',tc,'physical',physical,'logical',logical,'parity',parity,flush=True)
if __name__=='__main__':main()
