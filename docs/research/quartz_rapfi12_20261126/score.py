"""Resource matching and paired family outcomes, no posthoc refitting."""
from experiment import *
def agg(aa):
    assert aa and all(a['measure']['tick_hz']==aa[0]['measure']['tick_hz'] for a in aa)
    hz=aa[0]['measure']['tick_hz'];n=len(aa)
    rt=sum(a['measure']['rust_ticks'] for a in aa);et=sum(a['measure']['rapfi_ticks'] for a in aa)
    wall=[a['transport_ns']/1e6 for a in aa]
    active=[a for a in aa if a['visits']>0]
    r=[a['resource'] for a in active]
    return dict(moves=n,visits=sum(a['visits'] for a in aa),NN=sum(a['NN_calls'] for a in aa),
        mean_visits=S.mean(a['visits'] for a in aa),mean_NN=S.mean(a['NN_calls'] for a in aa),
        mean_wall_ms=S.mean(wall),p95_wall_ms=sorted(wall)[math.ceil(.95*n)-1],total_wall_ns=sum(a['transport_ns'] for a in aa),
        mean_native_ms=S.mean(a['whole_us']/1000 for a in aa),mean_active_wall_ms=S.mean(a['transport_ns']/1e6 for a in active) if active else None,
        cpu_ms=1000*(rt+et)/hz,mean_cpu_ms=1000*(rt+et)/hz/n,rust_ticks=rt,rapfi_ticks=et,tick_hz=hz,
        python_cpu_ms=sum(a['measure']['python_cpu_ns'] for a in aa)/1e6,guarded=n-len(active),
        locks=sum(x['locked'] for x in r),saved_by_lock=sum(x['selected_cap']-a['visits'] for x,a in zip(r,active) if x['locked']),
        planned_cap_total=sum(x['selected_cap'] for x in r),causes=dict(C.Counter(a['cause'].split(' {')[0] for a in aa)),
        nominal_caps={str(k):v for k,v in C.Counter(a['cap'] for a in aa).items()},
        chosen_caps={str(k):v for k,v in C.Counter(x['selected_cap'] for x in r).items()},
        hard=sum(x['hard'] is True for x in r),easy=sum(x['hard'] is False for x in r))
def ratios(a,b):
    w=b['mean_wall_ms']/a['mean_wall_ms'];c=b['mean_cpu_ms']/a['mean_cpu_ms'] if a['mean_cpu_ms'] else None
    return dict(wall_ratio=w,cpu_ratio=c,wall_match=abs(w-1)<=.10,cpu_match=c is not None and abs(c-1)<=.20)
def quantiles(a):
    a=sorted(a);return [a[math.floor(.025*(len(a)-1))],a[math.ceil(.975*(len(a)-1))]]
def boot(games):
    ids=sorted({g['job']['family'] for g in games});assert len(ids)==8
    fam={i:[g for g in games if g['job']['family']==i] for i in ids}
    per=[]
    for i,gg in fam.items():
        assert len(gg)==2 and {g['candidate_color'] for g in gg}=={1,-1}
        z={k:[] for k in ('base','candidate')}
        for g in gg:
            for row in g['rows']:
                a=row['response'];z['candidate' if a['arm']==g['candidate'] else 'base'].append(a)
        per.append(dict(family=i,id=gg[0]['job']['id'],scores=[g['score'] for g in gg],
            mean_score=S.mean(g['score'] for g in gg),base=agg(z['base']),candidate=agg(z['candidate'])))
    rng=random.Random(2026112612);samples=[];wr=[];cr=[]
    for _ in range(10000):
        z=rng.choices(per,k=len(per));samples.append(S.mean(x['mean_score'] for x in z))
        a=sum(x['base']['total_wall_ns'] for x in z)/sum(x['base']['moves'] for x in z)
        b=sum(x['candidate']['total_wall_ns'] for x in z)/sum(x['candidate']['moves'] for x in z)
        wr.append(b/a)
        a=sum(x['base']['cpu_ms'] for x in z)/sum(x['base']['moves'] for x in z)
        b=sum(x['candidate']['cpu_ms'] for x in z)/sum(x['candidate']['moves'] for x in z)
        if a:cr.append(b/a)
    return dict(families=per,score_CI95=quantiles(samples),wall_ratio_CI95=quantiles(wr),cpu_ratio_CI95=quantiles(cr) if cr else None,bootstrap_reps=10000)
def main():
    plan=read('C06_SOURCE_PLAN.json');fit=read('C07_FROZEN_MIX.json');assert plan['identity']==fit['identity']==identity()
    sources=list(rows('C06_SOURCE_GAMES.jsonl.gz'));grid=list(rows('C07_GRID_RAW.jsonl.gz'));cal=list(rows('C07_CAL_RAW.jsonl.gz'));games=list(rows('C08_ARENA_RAW.jsonl.gz'))
    locks={k:read(k) for k in ('C06_SOURCE_LOCK.json','C07_GRID_LOCK.json','C07_CAL_LOCK.json','C08_ARENA_LOCK.json')}
    for name,lock in [('C06_SOURCE_GAMES.jsonl.gz','C06_SOURCE_LOCK.json'),('C07_GRID_RAW.jsonl.gz','C07_GRID_LOCK.json'),('C07_CAL_RAW.jsonl.gz','C07_CAL_LOCK.json'),('C08_ARENA_RAW.jsonl.gz','C08_ARENA_LOCK.json')]:
        assert sha(D/name)==locks[lock]['raw_sha256']
    cals={arm:agg([x['response'] for x in cal if x['response']['arm']==arm]) for arm in ARMS}
    calrat={arm:ratios(cals['P5PW'],cals[arm]) for arm in ARMS[1:]}
    # Actual same-root allocation contrast on calibration distribution.
    byroot=C.defaultdict(dict)
    for row in cal:byroot[row['root']['family'],row['root']['ply']][row['response']['arm']]=row['response']
    strat={}
    for hard in (False,True):
        z=[v for v in byroot.values() if v['P5ALLOC']['resource'] and v['P5ALLOC']['resource']['hard']==hard]
        if z:
            strat[str(hard)]=dict(roots=len(z),base=agg([v['P5PW'] for v in z]),allocation=agg([v['P5ALLOC'] for v in z]))
    comparisons={}
    hosts={k:v['host'] for k,v in locks.items()}
    host_ok=all(h['timing_gate'] for h in hosts.values())
    for arm in ARMS[1:]:
        gg=[g for g in games if g['candidate']==arm];assert len(gg)==16
        z={k:[] for k in ('base','candidate')}
        for g in gg:
            for row in g['rows']:
                a=row['response'];z['candidate' if a['arm']==arm else 'base'].append(a)
        a,b=agg(z['base']),agg(z['candidate']);ra=ratios(a,b)
        assert sum(g['score']==v for g in gg for v in (0,.5,1))==16
        admission=host_ok and fit['fit'][arm]['bracket'] and calrat[arm]['wall_match'] and calrat[arm]['cpu_match'] and ra['wall_match'] and ra['cpu_match']
        ci=boot(gg)
        score=S.mean(g['score'] for g in gg);assert score==S.mean(v['mean_score'] for v in ci['families'])
        comparisons[arm]=dict(games=len(gg),WDL=[sum(g['score']==v for g in gg) for v in (1,.5,0)],
            score=score,black_score=S.mean(g['score'] for g in gg if g['candidate_color']==1),white_score=S.mean(g['score'] for g in gg if g['candidate_color']==-1),
            base=a,candidate=b,arena_ratio=ra,cal_ratio=calrat[arm],resource_match=admission,
            scope='small paired exploratory arena, no superiority or Elo admission',lengths=[len(g['final_history']) for g in gg],**ci)
    out=dict(identity=identity(),source_games=len(sources),source_plies=sum(len(g['rows']) for g in sources),grid_requests=len(grid),train_roots=fit['n'],cal_roots=len(cal)//4,
        cal=cals,cal_ratios=calrat,cal_allocation_strata=strat,comparisons=comparisons,arena_games=len(games),arena_plies=sum(len(g['rows']) for g in games),
        arena_NN=sum(row['response']['NN_calls'] for g in games for row in g['rows']),arena_visits=sum(row['response']['visits'] for g in games for row in g['rows']),
        hosts=hosts,host_gate=host_ok,fit=fit['fit'],target_wall_ms=fit['target_wall_ns']/1e6,scientific_promotion=False)
    write('C08_SUMMARY.json',out)
    print('COUNTS',{k:v for k,v in out.items() if k not in ('comparisons','cal','cal_ratios','hosts','identity','fit','cal_allocation_strata')})
    print('CAL',[(k,v['mean_wall_ms'],v['mean_cpu_ms'],v['mean_visits'],v['mean_NN']) for k,v in cals.items()])
    print('CAL RATIO',calrat)
    for k,v in strat.items():print('ALLOC CAL STRATUM',k,'roots',v['roots'],'visits',v['base']['mean_visits'],v['allocation']['mean_visits'],'wall',v['base']['mean_wall_ms'],v['allocation']['mean_wall_ms'])
    for arm,v in comparisons.items():
        print(arm,'WDL',v['WDL'],'score',v['score'],'CI',v['score_CI95'],'ratio',v['arena_ratio'],'CAL',v['cal_ratio'],'MATCH',v['resource_match'])
        for nm in ('base','candidate'):print(nm,{k:v[nm][k] for k in ('moves','mean_wall_ms','mean_cpu_ms','mean_visits','mean_NN','guarded','locks','saved_by_lock','hard','easy','nominal_caps','chosen_caps','causes')})
        print('PAIRS',[(x['id'],x['scores'],x['mean_score']) for x in v['families']])
    print('HOST',[(k,h['seconds'],h['swap_pages_s'],h['max_load'],h['timing_gate']) for k,h in hosts.items()])
if __name__=='__main__':main()
