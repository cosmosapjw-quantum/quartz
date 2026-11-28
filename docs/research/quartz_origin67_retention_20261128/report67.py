from plan67 import D,T,read,sha,ARMS
import gzip,json
def main():
    p=read(D/'PLAN.json');fit=read(T/'FIT.json');ind=read(T/'INDEPENDENT.json');sg=read(T/'SIGN_DIAGNOSTIC.json');deg=read(T/'DEGENERACY.json')
    modes=('MAIN','ARCHIVE','NOVEL','TRANSFER');panels={m:read(T/(m+'_SUMMARY.json')) for m in modes};rp=read(T/'RAPFI_SUMMARY.json')
    locks={m:read(T/(m+'_LOCK.json')) for m in ('DEV',*modes,'RAPFI')}
    games=sum(x['counts']['games'] for x in locks.values());moves=sum(x['counts']['moves'] for x in locks.values())
    nn=sum(locks[m]['counts']['physical'] for m in ('DEV',*modes));rapfi_nn=locks['RAPFI']['counts']['nn']
    assert games==8028 and locks['DEV']['counts']['games']==2996
    promo={m:s['promotion'] for m,s in panels.items()}
    lines=['# Origin67: calibrated incumbent retention','','## Scientific verdict',
    f"Benefit gates {promo}. Combined { {k:all(x[k] for x in promo.values()) for k in ('CALV-VOTE','CALF-FOCUS','CALX-EXPAND')} }. No post-holdout selection. Parent Origin66 eb9b16a671e1e364bea1304564b9937669105e3a preserved.",
    '', '## Conditional signed prediction',
    'Five possible terminal score differences on actual fixed FREE-v-B32 continuation; retain MA4 anchor unless predicted signed mean positive. VOTE no extra search, FOCUS and EXPAND pay one existing native visit when available, then decide after sunk cost. New Python readout only; native binary unchanged. No prospective purchase policy. Proper scoring and decision gain assessed separately.',
    f"Historical MA4 occupancy census {p['population']}; 749 roots and {locks['DEV']['counts']}; previous 171 complete action atlas reused, not counted as new games. TRAIN/CAL opening family groups, disjoint D4 board and opening family identities. {fit['train_families']} TRAIN families, {fit['cal_families']} CAL families; exclusions {len(fit['excluded'])}; labels TRAIN {fit['train_labels']}, CAL {fit['cal_labels']}; actual pairs {sum(fit['train_labels'])}/{sum(fit['cal_labels'])}. Equal source-domain weights and atlas design weights; conditional on discordance. 60 fresh four-ply families disjoint 30 prior locks; 16 distinct tensor hashes including 3 additional archived legacy-AZ checkpoints (not independent training seed replication).",
    f"Fixed ridge .02, 15 features, final gradient {fit['final_gradient']}, exact independent scalar reproduction {ind['train_gradient_norm']}; temperature grid {fit['temperature_nll']}, T={fit['temperature']}. Model {fit['model_sha256']}.",
    '', '|split/domain/operation|pairs|NLL|constant NLL|Brier|constant Brier|signed gain|always gain|switches|help/harm|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for k,z in fit['metrics'].items():
        lines.append(f"|{k}|{z['pairs']}|{z['nll']:.8f}|{z['nll_constant']:.8f}|{z['brier']:.8f}|{z['brier_constant']:.8f}|{z['signed_switch_gain']:.8f}|{z['always_switch_gain']:.8f}|{z['switch_count']}|{z['help']}/{z['harm']}|")
    lines += ['', 'All-root CAL descriptive local and whole-game costs: '+str(ind['cal_all_roots']), 'Sign diagnostic (no retuning): '+str(sg['groups']), 'Observed feature-box interval and actual trajectory comparison: '+str(deg)]
    for panel,s in panels.items():
        lines+=['',f'## {panel} heldout arena',f"Actual {locks[panel]['counts']}. {'24 fresh families, seven fixed models' if panel=='MAIN' else '12 fresh families, three fixed models'}; both colors, each arm vs same B32.",'','|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            q=s['arms'][arm];z=q['point'];c=q['counts']
            lines += [f"|{arm}|{c['games']}|{c['wins']}/{c['draws']}/{c['losses']}|{z['score']:.9f}|{z['q_rate']:.9f}|{z['b_rate']:.9f}|{z['saving']:.9f}|{z['cpu_saving']:.9f}|{z['utility']:.9f}|"]
        for k,x in s['contrasts'].items():
            lines += [f"{k}: point score {x['point']['score']:.9f}, saving {x['point']['saving']:.9f}, signed utility {x['point']['utility']:.9f}; lower95 score {x['lower95']['score']:.9f}, saving {x['lower95']['saving']:.9f}, utility {x['lower95']['utility']:.9f}; utility CI {x['CI']['utility']}; gates {x['gates']}."]
        lines += [f"Primary benefit gates {s['promotion']}; independent exact family bootstrap: {s['independent']}.","Mechanism and common-prefix accounting: "+str(read(T/(panel+'_MECHANISM.json')))]
    lines+=['','## Rapfi separate comparison',f"Actual {locks['RAPFI']['counts']}; Q96/Q104 each vs identical Rapfi B128. {rp['points']}; score difference {rp['paired_delta_score']}, CI {rp['score_CI']}; saving difference {rp['paired_delta_saving']}, CI {rp['saving_CI']}. Prior Origin Rapfi family locks now included. Fixed-cap scope only.",'','## Verification']
    audit={}
    for mode in ('DEV',*modes):
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:audit[mode]=json.load(f)['proof']
    lines += [f"Actual new games {games}, moves {moves}, NN physical {nn}, Rapfi calls {rapfi_nn}; separate units. DEV2996, heldout NN4968, Rapfi64; four-game dev smoke excluded. Two Python red-green tests, independent full replay. Native unchanged (previous Origin66 Rust671/0/89; not rebuilt or rerun), exact byte hash {p['old_binary_sha256']}. Exact finite 4900 cases and Wolfram checks. Independent proof/cache/selection audit {audit}; independent coordinate exact solver "+str({k:v for k,v in read(T/'EXACT_ALL.json').items() if k!='rows'})+". Rapfi audit "+str(rp['proof'])+". No independent inference for all NN queries; no GPU/native subagents, no main merge. SciPy unavailable; NumPy stable gradient descent with proven conservative step, directional finite difference and independent scalar verification.",
    '', '## Drift and next',
    'Original Jan28 PRD and Jan29 paper, initial README/THEORY and frozen THESIS: information acquisition should improve decisions per resource, not merely probability score. Origin67 tested H1-style calibration on signed value, but training/cal sign predictions were operation-level fixed decisions, not state-selective retention. No new adaptive controller promoted. Retain FREE/MA4 and proof evidence. Next Origin68 should target conditional non-tie decision evidence and residual value of a cheap independent observation (e.g. opponent immediate threat/reply contrast and paired two-ply evidence), then develop prospective purchase of FOCUS/EXPAND from shared prefix. Need label enrichment with predecision features (or fresh root sampling) and distinct frozen suffix regimes, not post-hoc selection of losing cases; proper scoring of rare sign outcomes and prospective signed utility. Do not add repeated tables to force gain. Original generic asynchronous MERGE, real average-time budget, expert signatures remain open.']
    (T/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',games,moves,nn,rapfi_nn,flush=True)
if __name__=='__main__':main()
