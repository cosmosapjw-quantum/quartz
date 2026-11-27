"""Generate final report from locked actual results and identities."""
import pathlib,json,hashlib,subprocess as sp,sys
from k53_plan import D,T,ROOT,load,sha
B=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic53_reuse_backups'
def main():
    a=load('AUDIT.json');s=load('SUMMARY.json');c=load('CACHE_LOCK.json')['counts'];b=load('BINARY_LOCK.json')
    rp=json.loads((T/'RAPFI_PARITY.json').read_text());rr=json.loads((T/'REUSE_REFERENCE.json').read_text());st=json.loads((T/'STATS_INDEPENDENT.json').read_text())
    assert a['total']['games']==2944 and a['total']['episodes']==672 and c['roots']==56 and c['hits']==117
    assert rr['counts']['roots']==7 and rp['counts']['target_requests']==8
    old=D.parent/'quartz_kinetic52_confirm_20261127/NATIVE_IDENTITY.json';identity=json.loads(old.read_text())
    for q,digest in b['source_sha256'].items():assert sha(ROOT/q)==digest==hashlib.sha256(sp.check_output(['git','show',b['source_commit']+':'+q],cwd=ROOT)).hexdigest()
    assert sha(T/'mcts_demo')==b['binary_sha256']==sha(B/'mcts_demo') and b['old_binary_sha256']==identity['binary_sha256']
    assert sha(ROOT/'Cargo.lock')==b['cargo_lock_sha256']
    for q in s['locks']:assert load(q+'_LOCK.json')['binary']==b['binary_sha256']
    assert '657 passed; 0 failed; 89 ignored' in (T/'FULL_TEST.log').read_text()
    identity.update(parent_parent_binary_sha256=identity['parent_binary_sha256'],binary_sha256=b['binary_sha256'],binary_bytes=b['bytes'],source_commit=b['source_commit'],source_sha256=b['source_sha256'],cargo_lock_sha256=b['cargo_lock_sha256'],rustc=b['rustc'],parent_binary_sha256=b['old_binary_sha256'],parent_native_identity_sha256=sha(old),full_test_raw_sha256=sha(T/'FULL_TEST.log'),scope='K53 native fresh-only retained child statistics and Rapfi cold target opt-in, K52 cold path preserved; no generic parallel/training integration')
    (D/'NATIVE_IDENTITY.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    n=s['nn']['all'];r=s['rapfi'];tc=a['total'];ct=a['counts']
    lines=['# K53 reuse, multi-intervention and Rapfi target','',
    '## Authority and theory',
    'Parent e7116cdfa44644df726b24583b17283f080c0761 clean, remote equal,149-file restore verified. Reread 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS. P1 posterior expected simple-regret decrease per cost, P2 original A1, P3 emergent signatures. K52 cold target confirmed; K53 distinguishes evaluator cache from subtree statistics reuse and multi-event policy. Exact per-request cache key (a,b,p,ordered legal), fixed model, no state symmetry; L=M+H. Fresh-only retained child mean -sum five new backups/5, equivalent to -(Wpost-Wpre)/5 in exact arithmetic; old visits alter sample policy, not cold equivalence. Score contrast telescopes, not independent local additivity. Rapfi same Q96 then optional two fresh child PUCT5, d<=8 and prior disagreement, proof-safe rival; at most12 extra physical NN. Literature SOURCES.md, no transfer of graph-search guarantees. Finite checks8100 reuse,6561 cache,27 telescope,144 cost, Wolfram three True. R10 fdf3d23074a3122f60df16bbd42f70ae5bf45650. R03 trailing whitespace corrected.',
    '## Implementation and failure repair',
    f"Native source {b['source_commit']}, SHA {b['binary_sha256']}, {b['rustc']}. K53 opt-in research_gomoku7_k53 and research_rapfi_k53; existing K52 unchanged. Actual red-green native mean test, release build, full Rust657/0/89. Initial mutable local hook compile error fixed and preserved. NN PLAN arm-order generation sampled3 rather than4, first matched record aborted, all28 partial shards verified empty; preserved logs/old plan, inserted missing fourth arm with separate RNG, same openings/colors/models/Rapfi seeds/relative old arm order. No complete outcomes inspected before repair. PLAN_REPAIR and C07R. No excluded complete games. Native subagent NOT_RUN; CPU8 NN/4 Rapfi, GPU unavailable, no new training, main merge or general parallel training integration.",
    '## Exact evaluator cache and native parity',
    f"56 fixed K52 historical target roots old/new/cached trace equality; logical1848, actual cached1731,117 hits (6.33117% of total;17.4107% of target672). All117 hits against base root1176 keys, no within-phase duplicates,52/56 roots hit. This is actual Torch forward saving for sampled root requests, not whole-game or CPU speed. Separate Python retained-child reconstruction {rr['counts']}; Rapfi old/new base and target parity {rp['counts']}. No baseline games cached.",
    '## Fresh NN experiment',
    '48 new D4 families x7 fixed models x2 colors x4 actual separate Q-v-B32 policies=2688 games. CONTROL fixed20, COLD first K52, REUSE first K53, MULTI every eligible K52. Family bootstrap8000, exploratory; no Q-v-Q or different evaluator games.',
    '|arm|W/D/L|score|NN/move saving|CPU/move saving|score lower95|NN lower95|gate|','|---|---|---:|---:|---:|---:|---:|---|']
    for arm,q in n['arms'].items():
        w=q['counts'];v=q['metrics'];l=q['lower95'];lines.append(f"|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{v['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{l['score']:.9f}|{l['NN_reduction']:.9f}|{q['gate']}|")
    lines+=['','|contrast|episodes|improved/worsened|gain points|local delta NN|net Q|net total|identical histories|','|---|---:|---|---:|---:|---:|---:|---:|']
    for arm,q in n['pairs'].items():
        v=q['counts'];lines.append(f"|{arm}|{v['n']}|{v['improved']}/{v['worsened']}|{v['delta2']/2:g}|{v['local_nn']}|{v['delta_Q']}|{v['delta_total']}|{v['same_history']}|")
    for arm,q in n['pairs'].items():
        v=q['counts'];lines.append(f"{arm} paired score effect {q['metrics']['tau']:.10f}, CI {q['CI']['tau']}; local ratio {q['ratio']['local_nn']}; net total ratio {q['ratio']['delta_total']}; downstream signed costs {v['downstream_Q']}/{v['downstream_B']}.")
    lines.append(f"Reuse receipts {s['reuse_receipts']}; multi intervention histogram {s['multi_event_hist']}. First event same across all arms; COLD/MULTI identical through first choice; subsequent eligibility policy-dependent. Reuse local delta negative may save cost, no positive-denominator efficiency ratio interpreted. Primary reference COLD remains verified K52 single operation. Model/panel/seed effects in SUMMARY.json. No automatic multi or reuse adoption on exploratory sample.")
    lines+=['','## Fresh Rapfi outer targeted operation','64 fresh outer families x2 colors x2 separate same Rapfi B128 arms=256 games, Q96 and first eligible target, no cap tuning. Base visit lock and physical NN recorded separately from child probes.',
      '|arm|W/D/L|score|NN/move saving|CPU/move saving|score lower95|NN lower95|gate|','|---|---|---:|---:|---:|---:|---:|---|']
    for arm,q in r['arms'].items():
        w=q['counts'];v=q['metrics'];l=q['lower95'];lines.append(f"|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{v['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{l['score']:.9f}|{l['NN_reduction']:.9f}|{q['gate']}|")
    lines.append(f"Rapfi pair counts {r['paired']['counts']}; score delta {r['paired']['metrics']['tau']} CI {r['paired']['CI']['tau']}; local gain/call {r['paired']['ratio']['local_nn']}; baseline-relative savings contrast {r['contrast']['NN_reduction']}. Fixed outer distribution artificial, no universal human-play inference; independent confirmation needed even if all gates pass.")
    parity=c['old_nn']+c['plain_nn']+c['cache_physical']+rr['counts']['native_nn']+rr['counts']['reference_nn']+rp['counts']['old_nn']+rp['counts']['new_base_nn']+rp['counts']['new_target_nn']
    lines+=['','## Audit and next',
      f"Actual {tc['games']} new recorded games, {tc['moves']} moves, {tc['nn']} physical game calls; {parity} separate actual parity calls (cache logical hits excluded). Proof receipts {ct.get('proof_receipts',0)}, bounds {ct.get('independent_bound_cells',0)}, certificates {ct.get('independent_partial_cert',0)}, proof queries {ct.get('proof_queries',0)}, votes {ct.get('votes',0)}, cold child receipts {ct.get('child_receipts',0)}, reuse child receipts {ct.get('reuse_child_receipts',0)}, Rapfi child receipts {ct.get('rapfi_children',0)}, wave receipts {ct.get('wave_receipts',0)}, waves {ct.get('waves',0)}. Natural exact {a['natural_exact']}. Independent standard-library resampling and integer ratios in STATS_INDEPENDENT.json. All finite receipts checked, not all neural outputs independently rerun. Initial aborted unrecorded attempts excluded from primary game/call totals; counts not instrumented for aborted episodes.",
      'K54: choose based on actual confidence and cost Pareto comparison; cache is semantics-preserving at tested scope, now investigate exact per-request native caching with physical ledger and matched cached baseline, then prospective larger replication of promising multi or reuse if supported. If target Rapfi positive and meets gates, confirm independent outer families; otherwise target selection/child estimator rather than blanket cap tuning. Preserve fixed Q20 and K52 cold as validated separate operating points. Original posterior P1, A1 P2 and GM P3 remain open. Dropbox sync NOT_VERIFIED.']
    (D/'REPORT.md').write_text('\n'.join(x.rstrip() for x in lines)+'\n')
    with (D/'ORIGIN_WATCHDOG.md').open('a') as w:
        w.write(f"\n## Outcome\nCache exact56 roots,117 physical forwards saved; distinct REUSE and MULTI paired effects {n['pairs']['reuse']['metrics']['tau']} and {n['pairs']['multi']['metrics']['tau']}, CI {n['pairs']['reuse']['CI']['tau']} and {n['pairs']['multi']['CI']['tau']}. Rapfi target effect {r['paired']['metrics']['tau']} CI {r['paired']['CI']['tau']}, gates {r['arms']['T']['gate']}. No inference of posterior optimum or signature tuning.\n")
    print('FINAL_REPORT',tc,'parity',parity,flush=True)
if __name__=='__main__':main()
