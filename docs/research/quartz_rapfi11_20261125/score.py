"""Signed policy-specific outcomes and paired diagnostic costs."""
from experiment import *
def main():
    raw=list(rows('C07_RAW.jsonl.gz'));cont=list(rows('C08_CONTINUATIONS.jsonl.gz'));co=read('C06_COHORT.json')['roots']
    assert sha(D/'C07_RAW.jsonl.gz')==read('C07_RAW_LOCK.json')['raw_sha256']
    assert sha(D/'C08_CONTINUATIONS.jsonl.gz')==read('C08_CONT_LOCK.json')['raw_sha256']
    u={(r['root']['id'],r['action'],r['cap']):r['utility'] for r in cont};assert len(u)==len(cont)
    ref={(v['root']['id'],v['prefix']):v['response'] for v in raw if v['response']['arm']=='DS0'}
    metrics=[];used=set()
    for v in raw:
        r=v['root'];p=v['prefix'];a=v['response'];i=r['id'];arm=a['arm'];z=ref[i,p]
        m=recommend(a,p);m0=recommend(z,p);vs={str(k):[u[i,x,k] for x in m] for k in (64,256)}
        for k in (64,256):
            used.update((i,x,k) for x in m)
        q=dict(id=i,group=r['group'],stratum=r['stratum'],prefix=p,arm=arm,moves=m,ref_moves=m0,utilities=vs,
            mid_gain={str(k):u[i,m[1],k]-u[i,m0[1],k] for k in (64,256)},
            final_gain={str(k):u[i,m[2],k]-u[i,m0[2],k] for k in (64,256)},changed_mid=m[1]!=m0[1],changed_final=m[2]!=m0[2],
            visits=a['visits'],nn=a['NN_calls'],wall_ms=a['transport_ns']/1e6,stopped=a['decision']['stopped'],target=a['decision']['target'],counter=a['decision']['phases'][-1]['counters'])
        metrics.append(q)
    assert used==set(u) and len(metrics)==len(co)*14
    groups={}
    for group in ('TRAIN','CAL','TEST','ALL'):
        for st in ('ALL','UNFILTERED','SENSITIVE'):
            for p in (8,16):
                for arm in ARMS:
                    rr=[r for r in metrics if (group=='ALL' or group==r['group']) and (st=='ALL' or st==r['stratum']) and r['prefix']==p and r['arm']==arm]
                    if not rr:continue
                    avg=lambda k:S.mean(r[k] for r in rr)
                    groups[f'{group}/{st}/{p}/{arm}']=dict(n=len(rr),mid_diff=sum(r['changed_mid'] for r in rr),final_diff=sum(r['changed_final'] for r in rr),
                        target_count=sum(r['target'] is not None for r in rr),stop_count=sum(r['stopped'] for r in rr),mean_visits=avg('visits'),mean_nn=avg('nn'),mean_wall_ms=avg('wall_ms'),
                        mean_mid_gain={str(k):S.mean(r['mid_gain'][str(k)] for r in rr) for k in (64,256)},
                        mean_final_gain={str(k):S.mean(r['final_gain'][str(k)] for r in rr) for k in (64,256)},
                        mid_improved={str(k):sum(r['mid_gain'][str(k)]>0 for r in rr) for k in (64,256)},
                        mid_harmed={str(k):sum(r['mid_gain'][str(k)]<0 for r in rr) for k in (64,256)},
                        final_improved={str(k):sum(r['final_gain'][str(k)]>0 for r in rr) for k in (64,256)},
                        final_harmed={str(k):sum(r['final_gain'][str(k)]<0 for r in rr) for k in (64,256)},
                        mechanisms={k:sum(r['counter'][k] for r in rr) for k in ('proofs','extra_evals','merge_blends','backup_changed','ledger_shared_children','ledger_observations')})
    q=list(rows('C08_COST_RAW.jsonl.gz'));assert len(q)==len(co)*20 and sha(D/'C08_COST_RAW.jsonl.gz')==read('C08_COST_LOCK.json')['raw_sha256']
    qq={}
    for r in q:
        k=r['id'],r['prefix'],r['rep'],r['arm'];assert k not in qq;qq[k]=r
    cost={}
    for group in ('ALL','TRAIN','CAL','TEST'):
        for p in (8,16):
            pairs=[(r,qq[r['id'],p,r['rep'],'DSSTOP']) for r in q if r['arm']=='DS0' and r['prefix']==p and (group=='ALL' or r['group']==group)]
            assert all(a['best']==b['best'] and a['order']==b['order'] and a['tick_hz']==b['tick_hz'] for a,b in pairs)
            cost[f'{group}/{p}']=dict(pairs=len(pairs),families=len({a['id'] for a,b in pairs}),mean_wall_ms=[S.mean(v['wall_ns']/1e6 for v in x) for x in zip(*pairs)],mean_native_ms=[S.mean(v['native_us']/1000 for v in x) for x in zip(*pairs)],mean_visits=[S.mean(v['visits'] for v in x) for x in zip(*pairs)],mean_NN=[S.mean(v['nn'] for v in x) for x in zip(*pairs)],total_rust_ticks=[sum(v['rust_ticks'] for v in x) for x in zip(*pairs)],total_rapfi_ticks=[sum(v['rapfi_ticks'] for v in x) for x in zip(*pairs)],tick_hz=pairs[0][0]['tick_hz'],mean_paired_wall_delta_ms=S.mean((b['wall_ns']-a['wall_ns'])/1e6 for a,b in pairs))
    actions={(r['root']['id'],r['action']) for r in cont}
    disagreement=[dict(id=i,move=m,U64=u[i,m,64],U256=u[i,m,256]) for i,m in sorted(actions) if u[i,m,64]!=u[i,m,256]]
    source=list(rows('C06_SOURCE_GAMES.jsonl.gz'))
    out=dict(identity=identity(),source_games=len(source),source_lengths=[len(g['final_history']) for g in source],source_plies=sum(len(g['rows']) for g in source),source_NN=sum(x['response']['NN_calls'] for g in source for x in g['rows']),probes=sum(len(g['probes']) for g in source),roots=len(co),requests=len(raw),snapshots=sum(len(v['response']['decision']['phases']) for v in raw),visits=sum(v['response']['visits'] for v in raw),nn=sum(v['response']['NN_calls'] for v in raw),unique_actions=len(actions),continuations=len(cont),continuation_plies=sum(len(g['rows']) for g in cont),continuation_NN=sum(x['response']['NN_calls'] for g in cont for x in g['rows']),label_disagreement=disagreement,groups=groups,cost=cost,scientific_promotion=False)
    write('C08_METRICS.json',metrics);write('C08_SUMMARY.json',out)
    print('SUMMARY', {k:v for k,v in out.items() if k not in ('groups','cost','identity')})
    for k,v in groups.items():
        if k.startswith(('TEST/ALL/','ALL/ALL/')):print(k,v)
    print('COST',cost)
if __name__=='__main__':main()
