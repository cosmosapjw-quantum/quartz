from plan63 import D,T,read,sha,ARMS
import gzip,json
def main():
    p=read(D/'PLAN.json');fit=read(T/'TABLE.json');main=read(T/'MAIN_SUMMARY.json');arch=read(T/'ARCHIVE_SUMMARY.json');rp=read(T/'RAPFI_SUMMARY.json')
    locks={m:read(T/(m+'_LOCK.json')) for m in ('DEV','MAIN','ARCHIVE','RAPFI')}
    total_games=sum(x['counts']['games'] for x in locks.values());total_moves=sum(x['counts']['moves'] for x in locks.values());total_nn=sum(x['counts'].get('physical',x['counts'].get('nn',0)) for x in locks.values())
    assert total_games==4856
    r=main['contrasts']['SELECT-H4'];d=main['contrasts']['SELECT-TARGET']
    lines=['# K63: constrained selective computation and free choice','','## Scientific verdict',
    f"Primary SELECT-H4 performance {r['performance']}, signed utility lower > 0 {r['gates']['signed_utility']}; TARGET retention {main['retention']}; combined promotion {main['promotion']}. H4 cost-first and K62 TARGET reference retained as comparisons. No post-test retune. Parent K62 2a918de10e0cae323907be3fece6d6ffce299ba5, clean/remote equal and actual restore verified.",
    '', '## Theory and development',
    f"Six-cell frozen H4 occupancy {p['population']}; 430 sampled occurrences, 80 per cell except census of 30. Actual {locks['DEV']['counts']}. STOP/H4, FREE from existing prefix values with unknown/tie retain incumbent, unchanged TARGET 2+2 and forced rival action diagnostic, all same H4 suffix vs identical B32. No cross evaluator or direct Quartz-v-Quartz games. Restricted oracle is retrospective only.",
    f"Exact 729-table search, {fit['feasible']} feasible, table {fit['table']}; price 1/500, local population mean extra cost budget 3/2. Selected using only newly generated development data, independent integer enumeration checks all 729. No guarantee under occupancy shift. No posterior calibration or P1 theorem.",
    '', '|policy|local gain|local extra physical|local utility|whole-game utility difference|restricted residual|','|---|---:|---:|---:|---:|---:|']
    for name,v in fit['policies'].items():
        lines.append('|'+name+'|'+'|'.join(f"{v[k]['mean']:.9f}" for k in ('gain','k','utility','whole','residual'))+'|')
    lines+=['','Design SE and individual cell/action exact values in TABLE.json; selection optimism applies to SELECT development values.']
    for panel,s in [('MAIN',main),('ARCHIVE',arch)]:
        lines+=['',f'## {panel} held-out panel',f"Actual {locks[panel]['counts']}. {'48 fresh families, seven fixed models' if panel=='MAIN' else '16 separate fresh families, three archived trained models, exploratory separate inference'}; both colors, all arms separately vs identical B32.",'','|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            q=s['arms'][arm];z=q['point'];c=q['counts']
            lines+=[f"|{arm}|{c['games']}|{c['wins']}/{c['draws']}/{c['losses']}|{z['score']:.9f}|{z['q_rate']:.9f}|{z['b_rate']:.9f}|{z['saving']:.9f}|{z['cpu_saving']:.9f}|{z['utility']:.9f}|"]
        for k,x in s['contrasts'].items():
            lines += [f"{k}: point score {x['point']['score']:.9f}, saving {x['point']['saving']:.9f}, signed utility {x['point']['utility']:.9f}; lower95 score {x['lower95']['score']:.9f}, saving {x['lower95']['saving']:.9f}, utility {x['lower95']['utility']:.9f}; two-sided utility CI {x['CI']['utility']}; gates {x['gates']}. {x.get('pairs','')}"]
        lines += [f"Retention {s['retention']}, combined promotion {s['promotion']}; independent exact family bootstrap: {s['independent']}.","Common-prefix and actual selection accounting: "+str(read(T/(panel+'_MECHANISM.json'))['arms'])]
    lines+=['','## Rapfi separate diagnostic',f"16 compact families, both colors; Q104/Q112 each vs identical Rapfi B128. Actual {locks['RAPFI']['counts']}; {rp['points']}; Q104-Q112 score {rp['paired_delta_score']}, CI {rp['score_CI']}; saving {rp['paired_delta_saving']}, CI {rp['saving_CI']}. Exploratory fixed-cap comparison, no direct NN transfer.",'','## Verification']
    audit={}
    for mode in ('DEV','MAIN','ARCHIVE'):
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:audit[mode]=json.load(f)['proof']
    lines += [f"Total main empirical games {total_games}, played moves {total_moves}, physical evaluator calls {total_nn}; NN/Rapfi separate units. 12 separate smoke games and parity excluded. Rust full suite 668 pass/0 fail/89 ignored, existing warnings. Genuine red-green free-choice test, actual 24 old/new matches, 6 per STOP/FREE/TARGET parity, 18 malformed rejected, three archive strict inference; inherited ten distinct tensor states with byte hashes rechecked. Exact rational 427 cases and Wolfram three True. NN independent coordinate/proof/cache/decision audits {audit}; independent natural exact solver record "+str({k:v for k,v in read(T/'EXACT_DEV_MAIN_ARCHIVE.json').items() if k!='rows'})+". Rapfi proof audit "+str(rp['proof'])+". All NN outputs not independently re-inferred. CPU-only 4 development/6 arena workers; native subagents unavailable; main not merged.",
    '', '## Drift watchdog and next',
    'Original 15f3a198 README/QUARTZ_THEORY and 7241ad0c frozen THESIS reread. Original P1 posterior expected simple regret reduction per compute is not established by cell fitting. Original A1_legacy_base P2 still untested; P3 emergent signatures not tuned or tested. K63 separates already available information from paid search, with a fixed empirical budget and price. K64 should inspect whether FREE accounts for the previously observed target advantage, then isolate marginal information acquisition with paired state-level intervention and independent families. Preserve negative gates; do not retune held-out outcomes. Consider risk-calibrated selective candidate sampling, different candidate coverage and separate Rapfi allocation.']
    (T/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',total_games,total_moves,total_nn,flush=True)
if __name__=='__main__':main()
