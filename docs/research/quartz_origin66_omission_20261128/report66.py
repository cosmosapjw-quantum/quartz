from plan66 import D,T,read,sha,ARMS
import gzip,json
def main():
    p=read(D/'PLAN.json');dev=read(T/'DEV_SUMMARY.json');ind=read(T/'INDEPENDENT.json');panels={m:read(T/(m+'_SUMMARY.json')) for m in ('MAIN','ARCHIVE','NOVEL')};rp=read(T/'RAPFI_SUMMARY.json')
    locks={m:read(T/(m+'_LOCK.json')) for m in ('DEV','MAIN','ARCHIVE','NOVEL','RAPFI')}
    games=sum(x['counts']['games'] for x in locks.values());moves=sum(x['counts']['moves'] for x in locks.values())
    nn=sum(locks[m]['counts']['physical'] for m in ('DEV','MAIN','ARCHIVE','NOVEL'));rapfi_nn=locks['RAPFI']['counts']['nn']
    assert games==10091 and locks['DEV']['counts']['forced_games']==7285 and locks['DEV']['counts']['operation_games']==342
    promo={m:s['promotion'] for m,s in panels.items()}
    lines=['# Origin66: full legal continuation and one-step information allocation','','## Scientific verdict',
    f"Main/represented archive/previously archived checkpoint panel benefit gates {promo}. Combined criteria by candidate { {k:all(x[k] for x in promo.values()) for k in ('FOCUS-MA4','EXPAND-MA4')} }. No fitting, no post-holdout selection. Parent Origin65 9cb6db45f78904a538485f67591d0fb7d608a41f preserved.",
    '', '## Full legal atlas and actual information value',
    'All A legal moves, F excludes proven minimax loss where alternatives remain, L visited/singleton in F. Root actual MA4 search charged even for each forced move; thereafter identical FREE vs same-model B32. Each legal action actually played once in separate game, plus FOCUS and EXPAND games. The finite deterministic outcome oracle is not minimax nor a posterior mean. Exact regret decomposition includes mask loss, safe omission and ranking. FOCUS selects visited non-singleton rival or incumbent fallback; EXPAND selects highest-prior unvisited non-singleton safe candidate or no-op. Same four-visit MA4 prefix and at most one extra visit. Neither action is globally optimized; no fit.',
    f"Historical occupancy populations {p['population']}; 171 sampled roots (96/39/36); {locks['DEV']['counts']}. Independent exact Fraction aggregation and disjointness audit: {ind['roots']} roots, {ind['families']} new families, {ind['prior_locks']} prior locks.",
    '', '|domain|baseline score|oracle all|mask loss|omission loss|ranking loss|total regret|','|---|---:|---:|---:|---:|---:|---:|']
    for dom,q in dev.items():
        v=q['metrics'];lines.append('|'+dom+'|'+'|'.join(f"{v[k]['mean']:.8f}" for k in ('base','ma','mask','omission','ranking','regret'))+'|')
    for dom,q in dev.items():
        lines+=['',f"### {dom}",f"Raw positive regret counts {q['raw']}. Design SE per metric "+str({k:v['design_se'] for k,v in q['metrics'].items()})+'.',
        '|operation|gain|omission reduction|ranking change|extra physical|root signed gain|whole-game signed gain|','|---|---:|---:|---:|---:|---:|---:|']
        for arm,z in q['ops'].items():
            lines+=['|'+arm+'|'+'|'.join(f"{z[k]['mean']:.8f}" for k in ('gain','omission_reduction','ranking_change','extra','root_utility','whole_utility'))+'|']
            lines += [f"{arm} raw counts {z['raw']}; root signed gain design SE {z['root_utility']['design_se']:.8f}."]
        lines += ['Equal-paid descriptive subset '+str(q['equal_paid_descriptive'])+'.', 'Independent acquisition and oracle diagnostics '+str(ind['domains'][dom]['means'])+'.']
    for panel,s in panels.items():
        lines+=['',f'## {panel} independent opening arena',f"Actual {locks[panel]['counts']}. {'24 fresh families, seven fixed models' if panel=='MAIN' else '12 fresh families, three fixed models'}; both colors, each arm vs same B32.",'','|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            q=s['arms'][arm];z=q['point'];c=q['counts']
            lines += [f"|{arm}|{c['games']}|{c['wins']}/{c['draws']}/{c['losses']}|{z['score']:.9f}|{z['q_rate']:.9f}|{z['b_rate']:.9f}|{z['saving']:.9f}|{z['cpu_saving']:.9f}|{z['utility']:.9f}|"]
        for k,x in s['contrasts'].items():
            lines += [f"{k}: point score {x['point']['score']:.9f}, saving {x['point']['saving']:.9f}, signed utility {x['point']['utility']:.9f}; lower95 score {x['lower95']['score']:.9f}, saving {x['lower95']['saving']:.9f}, utility {x['lower95']['utility']:.9f}; utility CI {x['CI']['utility']}; gates {x['gates']}."]
        lines += [f"Primary benefit gates {s['promotion']}; independent exact family bootstrap: {s['independent']}.","Mechanism and common-prefix accounting: "+str(read(T/(panel+'_MECHANISM.json')))]
    lines+=['','## Rapfi separate comparison',f"Actual {locks['RAPFI']['counts']}; Q96/Q104 each vs identical Rapfi B128. {rp['points']}; score difference {rp['paired_delta_score']}, CI {rp['score_CI']}; saving difference {rp['paired_delta_saving']}, CI {rp['saving_CI']}. Additional check against 9 prior RAPFI_PLAN files (248 distinct historical family IDs) found zero intersection. Fixed-cap scope only.",'','## Verification']
    audit={}
    for mode in ('DEV','MAIN','ARCHIVE','NOVEL'):
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:audit[mode]=json.load(f)['proof']
    lines += [f"Actual empirical games {games}, moves {moves}, NN physical {nn}, Rapfi calls {rapfi_nn}; separate units. DEV 7627 games, arena 2400 NN games, Rapfi64; 42 smoke games and parity excluded. Rust 671 pass/0 fail/89 ignored; existing warnings. Genuine red-green disjoint operation targets. 54 old/new comparisons, 12 new operation and prefix checks, 36 malformed rejections, 13 strict-load/inference checkpoints. Exact finite 32181 cases and Wolfram 3 True. Independent proof/cache/selection audit {audit}; independent coordinate exact solver "+str({k:v for k,v in read(T/'EXACT_ALL.json').items() if k!='rows'})+". Rapfi audit "+str(rp['proof'])+". Not independent NN re-inference for all queries; fixed checkpoint histories not independent training repeats. CPU six workers DEV/arena and two Rapfi, no GPU/native subagents. No main merge.",
    '', '## Drift and next',
    'Original Jan28 PRD, Jan29 paper, init README/THEORY, frozen THESIS: actual computation alternatives rather than repeated halt tables. Origin66 supplies full finite-continuation labels to separate mask/omission/ranking and directly measures one purchased observation. Remain distinct from original P1 calibrated belief/GVOC, P2 same mean real time, P3 expert signatures and generic parallel MERGE. Future work should use the full atlas to distinguish candidate proposal probability from noisy final readout: conditional signed value and harmful switch estimates, exact proof/evidence structure, and cross-domain validation; do not select a new policy from heldout outcomes. Keep proof-safe and actual finite-opponent objectives separate.']
    (T/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',games,moves,nn,rapfi_nn,flush=True)
if __name__=='__main__':main()
