from plan65 import D,T,read,sha,ARMS,DEV
import gzip,json
def main():
    p=read(D/'PLAN.json');dev=read(T/'DEV_SUMMARY.json');panels={m:read(T/(m+'_SUMMARY.json')) for m in ('MAIN','ARCHIVE','NOVEL')};rp=read(T/'RAPFI_SUMMARY.json')
    locks={m:read(T/(m+'_LOCK.json')) for m in ('DEV','MAIN','ARCHIVE','NOVEL','RAPFI')}
    games=sum(locks[m]['counts']['games'] for m in ('MAIN','ARCHIVE','NOVEL','RAPFI'));moves=sum(locks[m]['counts']['moves'] for m in ('MAIN','ARCHIVE','NOVEL','RAPFI'))
    actual_nn=sum(locks[m]['counts']['physical'] for m in ('MAIN','ARCHIVE','NOVEL'));rapfi_nn=locks['RAPFI']['counts']['nn']
    assert games==3424 and locks['DEV']['counts']['requests']==2770
    promo={m:s['promotion'] for m,s in panels.items()}
    lines=['# Origin65: proof-aware prefix and candidate coverage','','## Scientific verdict',
    f"Main/represented archive/unseen-to-K64-fit checkpoint panel benefit gates {promo}. Combined criteria by candidate { {k:all(x[k] for x in promo.values()) for k in ('MA4-FREE','X4-MA4')} }. No post-holdout policy selection. Historical K64 parent 0d5d80326339de38454db48935a92805f6016a34 preserved.",
    '', '## Theory and historical diagnostic',
    'Fixed four root visits; exclude proven minimax losses from allocation where alternatives remain. This preserves an optimum but does not prove finite-opponent strength. Separate all-visited clipped Q readout (unknown is not zero) and distinct-first coverage. Original prior unchanged and all legal actions passed to evaluator. Interior PUCT unchanged. L4 and A4 identical search prefix; M4 and MA4 identical prefix. X4 distinct-first is not Gumbel SH or nested sampling theorem.',
    f"Retrospective root panel 554 K64 occurrences, same immutable source plan with stratified MAIN weights and finite ARCHIVE census, actual {locks['DEV']['counts']}. Actual fresh root probes, not played games or new terminal labels. No fitting. Original 3308 D4 families excluded, 48 new families, same 13 frozen weights.",
    '', '|domain|arm|mean visited feasible|mean loss visits|P(chosen unvisited)|prior mass visited|mean physical|raw all-loss roots|','|---|---|---:|---:|---:|---:|---:|---:|']
    for dom,q in dev.items():
        for arm in DEV:
            r=q['arms'][arm];lines.append(f"|{dom}|{arm}|{r['visited']:.6f}|{r['loss_visits']:.6f}|{r['unvisited_chosen']:.6f}|{r['mass']:.6f}|{r['physical']:.6f}|{r['raw_all_loss']}|")
        lines += [f"{dom} raw and weighted choice change: {q['contrasts']}."]
    for panel,s in panels.items():
        lines+=['',f'## {panel} heldout arena',f"Actual {locks[panel]['counts']}. {'24 fresh families, seven fixed models' if panel=='MAIN' else '12 fresh families, three fixed models'}; both colors, each arm vs same B32.",'','|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            q=s['arms'][arm];z=q['point'];c=q['counts']
            lines += [f"|{arm}|{c['games']}|{c['wins']}/{c['draws']}/{c['losses']}|{z['score']:.9f}|{z['q_rate']:.9f}|{z['b_rate']:.9f}|{z['saving']:.9f}|{z['cpu_saving']:.9f}|{z['utility']:.9f}|"]
        for k,x in s['contrasts'].items():
            lines += [f"{k}: point score {x['point']['score']:.9f}, saving {x['point']['saving']:.9f}, signed utility {x['point']['utility']:.9f}; lower95 score {x['lower95']['score']:.9f}, saving {x['lower95']['saving']:.9f}, utility {x['lower95']['utility']:.9f}; utility CI {x['CI']['utility']}; gates {x['gates']}."]
        lines += [f"Primary and incremental benefit gates {s['promotion']}; independent exact family bootstrap: {s['independent']}.","Mechanism and common-prefix accounting: "+str(read(T/(panel+'_MECHANISM.json')))]
    lines+=['','## Rapfi separate comparison',f"Actual {locks['RAPFI']['counts']}; Q96/Q104 each vs identical Rapfi B128. {rp['points']}; score difference {rp['paired_delta_score']}, CI {rp['score_CI']}; saving difference {rp['paired_delta_saving']}, CI {rp['saving_CI']}. No NN mask transfer.",'','## Verification']
    audit={}
    for mode in ('DEV','MAIN','ARCHIVE','NOVEL'):
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:audit[mode]=json.load(f)['proof']
    lines += [f"Actual empirical games {games}, moves {moves}, NN arena physical {actual_nn}, Rapfi calls {rapfi_nn}, separate historic probe physical {locks['DEV']['counts']['physical']}; NN/Rapfi distinct cost units. 14 smoke games, 5 smoke probes and parity excluded. Rust full suite 670 pass/0 fail/89 ignored; existing warnings. Genuine red on mask, unmasked fixture correction, focused green. 42 old/new comparisons, 30 new checks, 12 same-prefix comparisons and 30 malformed requests rejected; 13 strict-load/inference checkpoints, prior verified tensor identities. Exact finite 4011 cases and Wolfram 3 True. Independent proof/cache/selection audit {audit}; exact solver record "+str({k:v for k,v in read(T/'EXACT_ALL.json').items() if k!='rows'})+". Rapfi proof audit "+str(rp['proof'])+". No independent re-inference of every NN output; CPU workers 4/6/2, no GPU/native subagents, no main merge.",
    '', '## Drift and next',
    'Original Jan28 PRD and Jan29 paper, legacy corpus, init README/THEORY and July THESIS, early morphology and proof evidence: return to EXPAND and FOCUS at equal small horizon. Root loss masking and coverage are actual execution, not just a STOP table. Original mean-time matched baseline, generic async engine, true joint belief/VOC, MERGE across turns and human signature P3 not tested. Current fixed-reference experiment is not production deployment. Preserve all negative prior lanes and K64. Next research: measure omission vs ranking by actual matched finite-opponent root interventions outside the pair, distinguish additional breadth from more accurate readout at same physical resources; build proof-aware controller in generic engine before mean time reallocation. Do not tune on these validation panels.',
    'R10 first local unpublished commit omitted watchdog path, repaired before publish; first green attempt had incorrect unmasked fixture expectation, corrected only that assertion, original red failure retained.']
    (T/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',games,moves,actual_nn,rapfi_nn,locks['DEV']['counts']['physical'],flush=True)
if __name__=='__main__':main()
