from plan64 import D,T,read,sha,ARMS
import gzip,json
def main():
    p=read(D/'PLAN.json');fit=read(T/'TABLE.json');panels={m:read(T/(m+'_SUMMARY.json')) for m in ('MAIN','ARCHIVE','NOVEL')};rp=read(T/'RAPFI_SUMMARY.json')
    locks={m:read(T/(m+'_LOCK.json')) for m in ('DEV','MAIN','ARCHIVE','NOVEL','RAPFI')}
    total_games=sum(x['counts']['games'] for x in locks.values());total_moves=sum(x['counts']['moves'] for x in locks.values());total_nn=sum(x['counts'].get('physical',x['counts'].get('nn',0)) for x in locks.values())
    assert total_games==6984
    lines=['# K64: marginal computation and domain robustness','','## Scientific verdict',
    f"ROBUST-FREE promotion gates: { {m:s['promotion'] for m,s in panels.items()} }; combined {all(s['promotion'] for s in panels.values())}. Parent K63 8e87b5e38b17e2ae7c3386fdefb39546bdaca47c verified. No post-test tuning.",
    '', '## Theory and development',
    f"Two frozen K63 FREE occupancy populations {p['population']}; 428 sampled MAIN roots and 126 ARCHIVE census roots. Actual {locks['DEV']['counts']}. Four root interventions FREE, T2 (1+1), T4 (2+2), OTHER (forced opposite of FREE); identical FREE suffix vs same-model B32. No Q-v-Q or cross-evaluator games. Oracle restricted to candidate pair; no oracle as deployable policy.",
    f"729 tables, {fit['feasible']} feasible, robust {fit['table']}, pooled diagnostic {fit['pooled_table']}. Worst empirical domain signed root utility with per-domain budget 1.5 and support; price 1/500. Independent integer enumeration checks all tables. Reachable decision certificates {fit['certificates']}. No posterior calibration or generalization proof.",
    '', '|policy|domain|local gain|local extra physical|local utility|whole-game utility difference|residual regret|','|---|---|---:|---:|---:|---:|---:|']
    for name,ds in fit['policies'].items():
        for domain,v in ds.items():lines.append('|'+name+'|'+domain+'|'+'|'.join(f"{v[k]['mean']:.9f}" for k in ('gain','k','utility','whole','residual'))+'|')
    lines+=['','Raw paired changes: '+str(fit['raw_counts'])+'. Design SE and cells in TABLE.json; selection optimism. ARCHIVE census is finite only; zero design SE does not imply no sampling uncertainty about future states.']
    for panel,s in panels.items():
        lines+=['',f'## {panel} held-out panel',f"Actual {locks[panel]['counts']}. {'32 fresh families, seven fixed models' if panel=='MAIN' else '32 fresh families, three represented archive models' if panel=='ARCHIVE' else '24 fresh families, three unseen trained tensor checkpoints; same original training-seed IDs, not independent seed replication'}; both colors, each arm vs same B32.",'','|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            q=s['arms'][arm];z=q['point'];c=q['counts']
            lines+=[f"|{arm}|{c['games']}|{c['wins']}/{c['draws']}/{c['losses']}|{z['score']:.9f}|{z['q_rate']:.9f}|{z['b_rate']:.9f}|{z['saving']:.9f}|{z['cpu_saving']:.9f}|{z['utility']:.9f}|"]
        for k,x in s['contrasts'].items():
            lines += [f"{k}: point score {x['point']['score']:.9f}, saving {x['point']['saving']:.9f}, signed utility {x['point']['utility']:.9f}; lower95 score {x['lower95']['score']:.9f}, saving {x['lower95']['saving']:.9f}, utility {x['lower95']['utility']:.9f}; utility CI {x['CI']['utility']}; gates {x['gates']}."]
        lines += [f"Promotion {s['promotion']}; independent exact family bootstrap: {s['independent']}.","Common-prefix and actual marginal selection accounting: "+str(read(T/(panel+'_MECHANISM.json'))['arms'])]
    lines+=['','## Rapfi separate diagnostic',f"Actual {locks['RAPFI']['counts']}; Q104/Q112 each vs identical Rapfi B128. {rp['points']}; score difference {rp['paired_delta_score']}, CI {rp['score_CI']}; saving difference {rp['paired_delta_saving']}, CI {rp['saving_CI']}. Exploratory fixed-cap diagnostic, no NN threshold transfer.",'','## Verification']
    audit={}
    for mode in ('DEV','MAIN','ARCHIVE','NOVEL'):
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:audit[mode]=json.load(f)['proof']
    lines += [f"Total empirical games {total_games}, moves {total_moves}, physical evaluator calls {total_nn}; NN/Rapfi separate units. 16 smoke games and parity excluded. Rust full suite 669 pass/0 fail/89 ignored; existing warnings. Genuine red-green T2 pooled test; 42 old/new requests, 6 per FREE/T2/T4 parity and 18 invalid rejected. Six archived strict inference, 13 distinct tensor states. Exact rational 2421 cases and Wolfram three True. NN independent proof/cache/decision audit {audit}; independent natural exact solver record "+str({k:v for k,v in read(T/'EXACT_ALL.json').items() if k!='rows'})+". Rapfi proof audit "+str(rp['proof'])+". Not all NN outputs independently re-inferred. CPU-only 4 dev/6 arena workers; native subagents unavailable, main not merged.",
    '', '## Drift watchdog and next',
    'Original 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS reread. P1 posterior expected regret reduction per compute remains open; K64 empirical marginal decision gain vs FREE is closer to terminal decision objective but no posterior model. Original P2 A1 comparator and P3 emergent signatures open; no signature tuning. K65 should preserve all negative gates and assess uncertainty in action-value differences directly rather than increase table granularity. New development data on underrepresented archive cells, fresh heldout families/checkpoints; consider stage-wise T2 then T4 only if further decision change justified. Keep FREE and K63 frozen and separate Rapfi scope.',
    'R01 whitespace preflight failure corrected before first commit. No claim of cloud sync or production promotion.']
    (T/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',total_games,total_moves,total_nn,flush=True)
if __name__=='__main__':main()
