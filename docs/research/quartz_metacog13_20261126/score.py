"""Actual response dynamics and independent terminal utilities; no retune."""
from experiment import *
import monitor_ref as mr
def summary(aa):
    t=[(a,v) for a in aa for v in a['meta']['steps']]
    n=len(t);ys=[v['response']['y'] for a,v in t]
    errs=[v['prediction_error'] for a,v in t]
    phase=[a['meta']['prefix']['monitor'] for a in aa]
    final=[a['meta']['steps'][-1]['monitor'] for a in aa]
    fc=[a['meta']['steps'][-1]['counters'] for a in aa]
    return dict(n=len(aa),blocks=n,channel_counts=dict(C.Counter(v['channel'] for a,v in t)),mean_response=S.mean(ys),negative_response=sum(y<0 for y in ys),zero_response=sum(y==0 for y in ys),
        mean_prediction_mse=S.mean(e*e for e in errs),zero_predictor_mse=S.mean(y*y for y in ys),mean_meta_kl=S.mean(mr.kl(v['proposal'],v['omega']) for a,v in t),
        zero_memory_action_difference=sum(mr.select(v['proposal'],v['uniform'])!=mr.select(v['omega'],v['uniform']) for a,v in t),
        mean_phi_start=S.mean(x['phi'] for x in phase),mean_phi_end=S.mean(x['phi'] for x in final),
        negative_phi_blocks=sum(v['response']['delta_phi']<0 for a,v in t),positive_A_blocks=sum(v['response']['delta_a']>0 for a,v in t),
        mean_A_start=S.mean(x['uncertainty'] for x in phase),mean_A_end=S.mean(x['uncertainty'] for x in final),
        sum_depletion=sum(v['response']['depletion'] for a,v in t),sum_transport=sum(v['response']['transport'] for a,v in t),
        sum_work=sum(v['response']['work'] for a,v in t),sum_reorganization=sum(v['response']['reorganization'] for a,v in t),
        mean_wall_ms=S.mean(a['transport_ns']/1e6 for a in aa),mean_CPU_ms=S.mean((a['measure']['rust_cpu_ns']+a['measure']['rapfi_cpu_ns'])/1e6 for a in aa),
        mean_rust_CPU_ms=S.mean(a['measure']['rust_cpu_ns']/1e6 for a in aa),mean_rapfi_CPU_ms=S.mean(a['measure']['rapfi_cpu_ns']/1e6 for a in aa),
        mean_native_ms=S.mean(a['whole_us']/1000 for a in aa),mean_NN=S.mean(a['NN_calls'] for a in aa),
        mean_block_ms=S.mean(v['block_us']/1000 for a,v in t),mean_monitor_read_ms=S.mean(v['read_us']/1000 for a,v in t),
        mean_plan_ms=S.mean(v['plan_us']/1000 for a,v in t),
        counters={k:sum(c[k] for c in fc) for k in ('select_nodes','select_changed','width_nodes','width_mass_added','extra_evals','proofs','proof_win','proof_scans','merge_blends','backup_changed','ledger_shared_children','ledger_observations','terminal_leaves')})
def main():
    raw=list(all_searches());cont=list(rows('C08_CONT_RAW.jsonl.gz'));co=read('C06_COHORT.json')['roots']
    assert sha(D/'C08_CONT_RAW.jsonl.gz')==read('C08_CONT_LOCK.json')['raw_sha256']
    uu={(r['root']['id'],r['action'],r['cap']):r['utility'] for r in cont};assert len(uu)==len(cont)
    ref={(x['root']['id'],x['response']['arm']):x['response'] for x in raw};assert len(ref)==len(raw)==len(co)*4
    metrics=[];used=set()
    for x in raw:
        r=x['root'];a=x['response'];i=r['id'];arm=a['arm'];pre=a['meta']['prefix']['result']['best'];final=a['best']
        b=ref[i,'MBASE'];assert b['meta']['prefix']['result']['best']==pre
        moves=(pre,final,b['best'])
        v={str(cap):[uu[i,j,cap] for j in moves] for cap in (128,512)}
        used.update((i,j,cap) for j in moves for cap in (128,512))
        metrics.append(dict(id=i,group=r['group'],arm=arm,prefix=pre,final=final,baseline=b['best'],changed=final!=b['best'],
          utility=v,gain={k:x[1]-x[2] for k,x in v.items()},prefix_gain={k:x[1]-x[0] for k,x in v.items()},
          channel_schedule=[t['channel'] for t in a['meta']['steps']],NN=a['NN_calls']))
    assert used==set(uu)
    groups={}
    for group in ('CAL','TEST','ALL'):
        v=[x for x in raw if group=='ALL' or x['root']['group']==group]
        s={}
        for arm in ARMS:
            aa=[x['response'] for x in v if x['response']['arm']==arm]
            z=summary(aa)
            m=[x for x in metrics if x['arm']==arm and (group=='ALL' or group==x['group'])]
            z.update(changed_final=sum(x['changed'] for x in m),changed_from_prefix=sum(x['prefix']!=x['final'] for x in m),
                mean_final_utility={str(cap):S.mean(x['utility'][str(cap)][1] for x in m) for cap in (128,512)},
                paired_gain={str(cap):S.mean(x['gain'][str(cap)] for x in m) for cap in (128,512)},
                prefix_gain={str(cap):S.mean(x['prefix_gain'][str(cap)] for x in m) for cap in (128,512)},
                improved={str(cap):sum(x['gain'][str(cap)]>0 for x in m) for cap in (128,512)},
                harmed={str(cap):sum(x['gain'][str(cap)]<0 for x in m) for cap in (128,512)})
            s[arm]=z
        assert s['META']['channel_counts']==s['YOKE']['channel_counts']
        ids={x['root']['id'] for x in v}
        s['feedback']={
            'roots_different_META_OPEN_schedule':sum([t['channel'] for t in ref[i,'META']['meta']['steps']]!=[t['channel'] for t in ref[i,'OPEN']['meta']['steps']] for i in ids),
            'roots_different_META_OPEN_final':sum(ref[i,'META']['best']!=ref[i,'OPEN']['best'] for i in ids),
            'roots_different_META_YOKE_final':sum(ref[i,'META']['best']!=ref[i,'YOKE']['best'] for i in ids),
            'roots_all_final_same':sum(len({ref[i,a]['best'] for a in ARMS})==1 for i in ids)}
        groups[group]=s
    choices={(i,m) for i,m,c in uu}
    disagree=[dict(id=i,action=m,U128=uu[i,m,128],U512=uu[i,m,512]) for i,m in sorted(choices) if uu[i,m,128]!=uu[i,m,512]]
    src=list(rows('C06_SOURCE_GAMES.jsonl.gz'))
    out=dict(identity=identity(),roots=len(co),requests=len(raw),blocks=15*len(raw),visits=sum(x['response']['visits'] for x in raw),NN=sum(x['response']['NN_calls'] for x in raw),
       source_games=len(src),source_plies=sum(len(g['rows']) for g in src),source_NN=sum(x['response']['NN_calls'] for g in src for x in g['rows']),source_lengths=[len(g['final_history']) for g in src],
       unique_actions=len(choices),continuations=len(cont),continuation_plies=sum(len(g['rows']) for g in cont),continuation_NN=sum(x['response']['NN_calls'] for g in cont for x in g['rows']),
       label_disagreement=disagree,groups=groups,hosts=[read(k)['host'] for k in ('C06_COHORT.json','C07_PRIMARY_LOCK.json','C07_YOKE_LOCK.json','C08_CONT_LOCK.json')],
       claim='feedback mechanism and proxy response; no posterior regret, calibrated VOC, brain model, matched-time or strength admission')
    write('C08_METRICS.json',metrics);write('C08_SUMMARY.json',out)
    print('COUNTS',{k:v for k,v in out.items() if k not in ('groups','hosts','identity')})
    for g in ('CAL','TEST','ALL'):
        for a,v in groups[g].items():print(g,a,v)
if __name__=='__main__':main()
