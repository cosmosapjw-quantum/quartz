from plan62 import D,T,read,sha,ARMS
from pathlib import Path
import gzip,json
def main():
    p=read(D/'PLAN.json');dev=read(T/'DEV_SUMMARY.json');main=read(T/'MAIN_SUMMARY.json');arch=read(T/'ARCHIVE_SUMMARY.json');rp=read(T/'RAPFI_SUMMARY.json')
    locks={m:read(T/(m+'_LOCK.json')) for m in ('DEV','MAIN','ARCHIVE','RAPFI')}
    total_games=sum(x['counts']['games'] for x in locks.values());total_moves=sum(x['counts']['moves'] for x in locks.values());total_nn=sum(x['counts'].get('physical',x['counts'].get('nn',0)) for x in locks.values())
    assert total_games==4640
    lines=['# K62: two-action allocation and regret decomposition','', '## Scientific verdict']
    r=main['contrasts']['TARGET-H4'];d=main['contrasts']['TARGET-H8']
    lines += [f"Prespecified main TARGET-H4 performance gate {r['performance']}, signed utility lower > 0 {r['gates']['signed_utility']}, combined benefit {r['benefit']}; separate TARGET-H8 direction signed utility gate {d['gates']['signed_utility']}. Do not retune on heldout. Parent K61 c675203fe0ce8a3c6354275b90ffd114a7f1730f; clean/remote equal and parent actual-restore receipt checked.",'',
    '## Restricted action-level continuation study',
    f"All 1073 eligible H4 root occurrences in K61, five cells {p['population']}; outcome-blind SRS 80 per cell, 400 roots, four real branch games each (H4, FORCE_B with actual H4 prefix and external rival action, H8, TARGET). Same subsequent H4 vs same-model B32; no Q-v-Q. Actual {locks['DEV']['counts']}. The restricted oracle is retrospective, never played as an oracle policy.",'',
    '|quantity|HT population mean|finite-population design SE|raw sample sum|','|---|---:|---:|---:|']
    for k,z in dev['weighted'].items():
        lines += [f"|{k}|{z['mean']:.9f}|{z['design_se']:.9f}|{dev['raw_counts'][k]}|"]
    lines+=['','Exact HT regret identity verified independently: TARGET gain = restricted H4 regret minus target residual. H8 outside-pair signed residual is not regret. Candidate generation and evaluator discrimination separated. No stationarity or Bayesian guarantee.']
    for panel,s in [('MAIN',main),('ARCHIVE',arch)]:
        lines+=['',f'## {panel} held-out panel',f"Actual {locks[panel]['counts']}. {'64 disjoint fresh families, 7 frozen models' if panel=='MAIN' else '16 additional disjoint families, three distinct pretrained phase8c models, exploratory separate inference'}; both colors, each Quartz arm vs same-model B32.",'','|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            q=s['arms'][arm];z=q['point'];c=q['counts']
            lines+=[f"|{arm}|{c['games']}|{c['wins']}/{c['draws']}/{c['losses']}|{z['score']:.9f}|{z['q_rate']:.9f}|{z['b_rate']:.9f}|{z['saving']:.9f}|{z['cpu_saving']:.9f}|{z['utility']:.9f}|"]
        for k,x in s['contrasts'].items():
            lines += [f"{k}: point score {x['point']['score']:.9f}, saving {x['point']['saving']:.9f}, signed utility {x['point']['utility']:.9f}; lower95 score {x['lower95']['score']:.9f}, saving {x['lower95']['saving']:.9f}, utility {x['lower95']['utility']:.9f}; two-sided utility CI {x['CI']['utility']}; gates {x['gates']}. {x.get('pairs','')}"]
        lines += [f"Separate exact family bootstrap lower bounds: {s['independent']}.","Common-prefix accounting (not causal attribution): "+str(read(T/(panel+'_MECHANISM.json'))['arms'])]
    lines+=['','## Rapfi separate diagnostic',f"16 compact families, both colors; Q96/Q104 each vs same Rapfi B128. Actual {locks['RAPFI']['counts']}; {rp['points']}; Q104-Q96 score {rp['paired_delta_score']}, CI {rp['score_CI']}; saving {rp['paired_delta_saving']}, CI {rp['saving_CI']}. Exploratory fixed cap diagnostic, not direct NN target transfer.",'','## Verification']
    audit={}
    for mode in ('DEV','MAIN','ARCHIVE'):
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:audit[mode]=json.load(f)['proof']
    lines += [f"Total main empirical games {total_games}, played moves {total_moves}, physical evaluator calls {total_nn}; NN/Rapfi distinct units, not a speed comparison. 10 separate smoke games and parity calls excluded. Rust release build, 667 passed, 0 failed, 89 ignored; existing warnings. Genuine red-green pooled test. 64 old/new parity, 16 H4 target prefix checks, 2 malformed rejected; three archive checkpoint strict load and inference, 10 distinct tensor-state hashes. Exact rational 899 cases and Wolfram 3 True; Wolfram context no specific result. NN proof/cache/target audit {audit}; independent natural exact solver record "+str({k:v for k,v in read(T/'EXACT_DEV_MAIN_ARCHIVE.json').items() if k!='rows'})+". Rapfi audit "+str(rp['proof'])+". Not all NN outputs independently re-inferred. CPU-only, 4 dev/6 main workers, no native subagents, no main merge.",
    '', '## Drift watchdog and next',
    'Original 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS reread. P1 posterior expected regret per unit computation remains the target; K62 tests fixed-suffix realized restricted action regret and pooled estimator, not posterior Bayes optimality. P2 original A1_legacy_base not substituted by B32. P3 emergent signatures open; no signature tuning. Fixed two-arm candidate and 2+2 attempts experimental. Preserve H4 if TARGET benefit gate fails. K63: separate candidate coverage from action-value inference, use development restricted oracle and H8 outside-pair evidence to decide whether to expand candidate set or improve discriminator. Do not tune heldout outcomes; validate new families and preserve the archived panel separately. Investigate zero-extra-cost H4 candidate choice against targeted computation and actual terminal utility, and use targeted proof/counterfactual continuation with same evaluator. Rapfi separate 15x15 risk/cost experiment.']
    (T/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',total_games,total_moves,total_nn,flush=True)
if __name__=='__main__':main()
