"""Finalize from actual locked results; all stage receipts checked separately on restore."""
import pathlib,json,hashlib,subprocess as sp,sys,math
from k52_plan import D,T,ROOT,load,sha
B=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic52_confirm_backups'
def main():
    a=load('AUDIT.json');s=load('SUMMARY.json');p=load('PARITY_LOCK.json');b=load('BINARY_LOCK.json');f=load('SELECTION.json')
    r=json.loads((T/'RATIO_DIAGNOSTIC.json').read_text());stat=json.loads((T/'STATS_INDEPENDENT.json').read_text());rp=json.loads((T/'RAPFI_PARITY.json').read_text());late=json.loads((T/'LATE_EXTENDED.json').read_text())
    assert a['total']['games']==4416 and a['total']['episodes']==1344 and p['counts']['events']==200 and p['counts']['child_receipts']==400
    assert rp['counts']['requests']==16 and rp['counts']['old_nn']==rp['counts']['new_nn']
    assert r['all']['target']['local_nn']['admit_ratio'] and r['all']['global']['local_nn']['admit_ratio']
    old=D.parent/'quartz_kinetic51_targeted_20261127/NATIVE_IDENTITY.json';identity=json.loads(old.read_text())
    for q,digest in b['source_sha256'].items():
        assert sha(ROOT/q)==digest==hashlib.sha256(sp.check_output(['git','show',b['source_commit']+':'+q],cwd=ROOT)).hexdigest()
    assert sha(T/'mcts_demo')==b['binary_sha256']==sha(B/'mcts_demo') and b['old_binary_sha256']==identity['binary_sha256']
    assert sha(ROOT/'Cargo.lock')==b['cargo_lock_sha256']
    for q in s['locks']:assert load(q+'_LOCK.json')['binary']==b['binary_sha256']
    log=(T/'FULL_TEST.log').read_text();assert '656 passed; 0 failed; 89 ignored' in log
    identity.update(binary_sha256=b['binary_sha256'],binary_bytes=b['bytes'],source_commit=b['source_commit'],source_sha256=b['source_sha256'],cargo_lock_sha256=b['cargo_lock_sha256'],rustc=b['rustc'],parent_binary_sha256=b['old_binary_sha256'],parent_native_identity_sha256=sha(old),build='CARGO_BUILD_JOBS=4 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 CARGO_PROFILE_RELEASE_LTO=false cargo build --offline --locked --release --bin mcts_demo',full_test_raw_sha256=sha(T/'FULL_TEST.log'),scope='K52 native cold-target and fixed/adaptive Rapfi cap endpoints, not generic parallel/training integration')
    for k in ('reused_binary_sha256','backup_hardlink_verified'):identity.pop(k,None)
    (D/'NATIVE_IDENTITY.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    n=s['nn']['all'];x=n['effects']['target'];g=n['effects']['global'];c=f['selected'];z=s['rapfi']['test'];tc=a['total'];ct=a['counts']
    lines=[
      '# K52 native targeted confirmation and Rapfi root cap recovery','',
      '## Authority and derivation',
      'Parent 9f7b877b8fbb87a28eb58b766d1af67006276525 clean, remote equality and 121-file restore verified. Reread 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS. P1 posterior expected regret reduction per compute, P2 original A1 anchor, P3 emergent signatures. K51 cold TARGET preserved, not subtree reuse. Reuse pooled mean correction n(q-z)/(n+k), bound 2n/(n+k), counterexample in DERIVATION.md. 8100 reuse,320 vote,36 cost cases and three Wolfram identities. R10 2beecafbb37ddd1b3c297fe9d47e0c1eb22f20f7. Root cap recovery is predeclared candidate not calibrated posterior rule. No iid transfer of BAI theorem.',
      '## Native implementation and parity',
      f"Source {b['source_commit']}; binary SHA {b['binary_sha256']}, Rust {b['rustc']}. Opt-in research_gomoku7_k52 exactly K51 first eligible bin3/7 and fresh PUCT5 child comparisons; native call counter total, wrapper records root-only physical_nn for K51-compatible audit plus native_physical_nn. No reuse, no generic training/parallel integration. New research_rapfi_k52 fixed104/112 or table [128]*4+[96]*4 at existing32..39 probe; old endpoints preserved. Actual focused red-green and release build, full Rust tests656/0/89. Archived200 K51 target events/400 child receipts match tolerance2e-15 and exact decisions/cost; target extra2398, new native parity calls18322, old11724; 400 control/global comparisons plus28 B32 roots. Additional16 Rapfi requests old/new match,1728 calls each. Native implementation verification distinct from scientific confirmation.",
      '## Fresh NN experiment',
      '96 new D4 opening families x7 frozen models x2 colors x3 separate actual Q-v-B32 policies =4032 games. Fixed first eligible event, no tuning or prohibited matchups. Family bootstrap8000, fixed model panel.',
      '|arm|W/D/L|score|NN/move saving|CPU/move saving|score lower95|NN lower95|','|---|---|---:|---:|---:|---:|---:|'
    ]
    for arm in ('control','global','target'):
        q=n['arms'][arm];v=q['metrics'];w=q['counts'];l=q['lower95'];lines.append(f"|{arm}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{v['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{l['score']:.9f}|{l['NN_reduction']:.9f}|")
    lines+=['','|operator|reached|changed|improved/worsened|gain points|local NN|downstream Q|downstream B|net Q|net total|','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for arm,y in [('global',g),('target',x)]:
        v=y['counts'];lines.append(f"|{arm}|{v['reached']}|{v['changed']}|{v['improved']}/{v['worsened']}|{v['delta2']/2:g}|{v['local_nn']}|{v['downstream_Q']}|{v['downstream_B']}|{v['delta_Q']}|{v['delta_total']}|")
    for arm,y in [('global',g),('target',x)]:
        lines.append(f"{arm} paired ITT {y['metrics']['tau']:.10f}, CI {y['CI']['tau']}; local gain/NN {r['all'][arm]['local_nn']}; signed total ratio guard {r['all'][arm]['delta_total']}.")
    lines.append(f"Target minus global separate baseline-relative score contrast {n['target_minus_global']['delta']['score']}. Prespecified confirmation {s['confirmation']}; confirmed={s['confirmed']}. Independent resampling target ITT {stat['nn']['all']['target_effect']['tau']}; model panel effects old/new {s['nn']['old']['effects']['target']['CI']['tau']} / {s['nn']['new']['effects']['target']['CI']['tau']}. No inference on true minimax/posterior; effect may be small and finite model-dependent. K51 favorable result is not pooled into primary confirmation.")
    lines+=['','## Rapfi outer CAL and independent TEST',
      '32 CAL families x2 colors x3 candidates=192 actual games, frozen selection before TEST48 families x2 colors x2 policies=192. Same Rapfi B128, same proof/tie/wave substrate, all outer. Cal gates score lower>=.45 and NN lower>=.30 and CPU positive.',
      '|CAL|W/D/L|score|NN saving|CPU saving|score lower95|NN lower95|gate|','|---|---|---:|---:|---:|---:|---:|---|']
    for cap,q in f['cal'].items():
        v=q['metrics'];w=q['counts'];l=q['lower95'];lines.append(f"|{cap}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{v['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{l['score']:.9f}|{l['NN_reduction']:.9f}|{q['gate']}|")
    lines.append(f"Selected {c}, CAL admitted={f['admitted']}; eligible {f['eligible']}. Adaptive CAL actual noncert root caps {f['cal']['AD']['cap_counts']}.")
    lines+=['|TEST|W/D/L|score|NN saving|CPU saving|score lower95|NN lower95|gate|','|---|---|---:|---:|---:|---:|---:|---|']
    for name,arm in [('96','a'),(c,'b')]:
        q=z[arm];v=q['metrics'];w=q['counts'];l=q['lower95'];lines.append(f"|{name}|{w['wins']}/{w['draws']}/{w['losses']}|{v['score']:.9f}|{v['NN_reduction']:.9f}|{v['CPU_reduction']:.9f}|{l['score']:.9f}|{l['NN_reduction']:.9f}|{q['gate']}|")
    lines.append(f"Independent TEST candidate admitted={s['rapfi']['admitted']}; paired score delta {z['delta']['score']}, NN delta {z['delta']['NN_reduction']}. Independent Python bootstrap in STATS_INDEPENDENT. Cap selection not universal; no tuning from TEST; if CAL fallback, diagnostic only.")
    lines+=['','## Audit and next',
      f"Actual {tc['games']} new games, {tc['moves']} moves, {tc['nn']} physical game evaluator calls (local already counted); parity separate {p['counts']['native_nn']+p['counts']['old_nn']+rp['counts']['old_nn']+rp['counts']['new_nn']}. Independent proof receipts {ct.get('proof_receipts',0)}, bound observations {ct.get('independent_bound_cells',0)}, certificates {ct.get('independent_partial_cert',0)}, proof queries {ct.get('proof_queries',0)}, votes {ct.get('votes',0)}, child receipts {ct.get('child_receipts',0)}, Rapfi wave receipts {ct.get('wave_receipts',0)}, waves {ct.get('waves',0)}. Natural exact original {a['natural_exact']}; extended {late['extended_attempted']} attempted/{late['extended_complete']} complete, {late['not_extended']} not attempted. Preserve any censor. Independent integer and family stats and signed denominator guard. No claim independent neural model architecture or exhaustive theorem.",
      'Native subagent NOT_RUN, GPU unavailable, no new training, generic async integration or main merge. CPU8 NN/4 Rapfi plus bounded build/audit. Existing Rust warnings retained. Cloud sync NOT_VERIFIED.',
      ('K53 NN: positive independent confirmation if all prespecified gates hold; then investigate reuse as a genuinely different policy, and exact evaluator memoization as semantics-preserving alternative; do not infer multi-intervention benefit. ' if s['confirmed'] else 'K53 NN: independent confirmation did not meet all primary gates; keep fixed20 as default, preserve K51 positive sample and K52 negative/inconclusive result. Diagnose family/checkpoint/action-comparison effect and candidate selection; no silent reuse or tuning on TEST. ')+('Rapfi selected policy meets independent outer gate after CAL; further distribution validation before promotion. ' if s['rapfi']['admitted'] else 'Rapfi outer quality/efficiency remains unresolved; preserve fixed Q96 scoped prior and Q128 quality diagnostic; next test computation operator at uncertain roots rather than blanket cap increases. ')+'Original P1 posterior optimization, P2 A1 comparison and P3 emergent signatures remain open.'
    ]
    (D/'REPORT.md').write_text('\n'.join(x.rstrip() for x in lines)+'\n')
    with (D/'ORIGIN_WATCHDOG.md').open('a') as w:w.write(f"\n## Outcome\nK52 native cold-target parity passed. Independent TARGET-CONTROL effect {x['metrics']['tau']} CI {x['CI']['tau']}; confirmed={s['confirmed']} under prespecified gate, not posterior optimality. Rapfi selection {c}, CAL admitted {f['admitted']}, TEST admitted {s['rapfi']['admitted']}. Keep negative results scoped, no A1 substitution/signature tuning or generic integration claim.\n")
    print('FINAL_REPORT',s['confirmed'],s['rapfi']['admitted'],tc,flush=True)
if __name__=='__main__':main()
