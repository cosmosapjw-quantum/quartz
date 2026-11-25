"""Native decoupled cost calibration and prospective WDL experiment; no gain refitting."""
import collections as C,json,math,os,random,statistics as S,sys,time
from pathlib import Path
import native as n
from native import D,ROOT,write,read,sha
from costcal import point_cost,envelope,tables,costs,score,available
B=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_nn_costcal_backups')
n.BIN=B/'mcts_demo'
SEEDS=(17,43,71,101,137,173)
ST=('TACTICAL','SELFPLAY')
ARMS=('PREFIX','FIXED8','RANDOM16','ENTROPY16','RATIO','DP','ROBUST_DP')
def snap():
    s=n.host();s['mono']=time.monotonic()
    p=Path('/proc/stat').read_text().splitlines()[0].split();v=list(map(int,p[1:9]))
    s['cpu_total']=sum(v);s['cpu_idle']=v[3]+v[4]
    return s
def interval(a,b):
    t=max(b['mono']-a['mono'],1e-9)
    swap=sum(b['swap_pages'][k]-a['swap_pages'][k] for k in a['swap_pages'])/t
    ticks=b['cpu_total']-a['cpu_total'];busy=1-(b['cpu_idle']-a['cpu_idle'])/max(1,ticks)
    return dict(swap_pages_per_s=swap,busy=busy,seconds=t,load1_max=max(a['load'][0],b['load'][0]),ok=swap<=512 and max(a['load'][0],b['load'][0])<=12)
def preflight():
    a=snap();time.sleep(1);b=snap();v=interval(a,b);assert a['cpus']==b['cpus']==24
    return dict(start=a,end=b,interval=v)
def groups(co,split,st):
    r=[r for r in co['rows'] if r['group']==split and r['stratum']==st]
    return [p for p in n.chunks(r) if len(p)==8]
def check_lock():
    n.lock()
    expected=next(line.split()[0] for line in (D/'C01_IDENTITY.txt').read_text().splitlines() if line.endswith('/GAIN_LOCK.json'))
    assert sha(D/'GAIN_LOCK.json')==expected
def fit():
    check_lock();co=read('C07_COHORT.json');g=read('GAIN_LOCK.json')['gain'];pre=preflight()
    write('C07_PREFLIGHT.json',pre)
    if not pre['interval']['ok']:
        print('HOST_BLOCKED',pre,flush=True);return
    rng=random.Random(202611250405);raw={k:[] for k in ('FIT','CAL','CHECK','PROFILE')};start=snap()
    with n.Server() as srv:
        warm=groups(co,'FIT','TACTICAL')[0];srv.call('SETUP',warm)
        for split in ('FIT','CAL','CHECK'):
            for st in ST:
                for k,pool in enumerate(groups(co,split,st)):
                    for seed in ((17,43) if split=='FIT' else (rng.choice(SEEDS),)):
                        a=srv.call('TRACE',pool,seed);a.update(stratum=st,pool=k,split=split,host=snap());raw[split].append(a)
            print('TIMING',split,len(raw[split]),flush=True)
        for rep in range(5):
            for st in ST:
                pool=groups(co,'CHECK',st)[0]
                for mode in (('SETUP','PREFIX') if rep%2==0 else ('PREFIX','SETUP')):
                    a=srv.call(mode,pool,SEEDS[rep]);a.update(stratum=st,rep=rep,host=snap());raw['PROFILE'].append(a)
    end=snap();env=interval(start,end)
    write('C07_TIMING_RAW.json',dict(start=start,end=end,interval=env,raw=raw))
    result={}
    for st in ST:
        fit=[costs(r) for a in raw['FIT'] if a['stratum']==st for r in a['rows']]
        d=point_cost(fit);cal=[a for a in raw['CAL'] if a['stratum']==st];ck=[a for a in raw['CHECK'] if a['stratum']==st]
        sc=[score(a,d) for a in cal];k,mult=envelope(sc);budget=min(100000,max(1000,4*d[1],2*d[2]))
        point=tables(g,d,1);rob=tables(g,d,mult) if mult is not None else None
        sr=[score(a,d) for a in ck];cov=None if mult is None else sum(s<=mult for s in sr)/len(sr)
        coord=[(c<=mult*x) for a in ck for r in a['rows'] for c,x in zip(costs(r),d)] if mult is not None else []
        prof={}
        for mode in ('SETUP','PREFIX'):
            rr=[a for a in raw['PROFILE'] if a['stratum']==st and a['mode']==mode]
            vals=[a['whole_us'] for a in rr];med=S.median(vals);mad=S.median(abs(v-med) for v in vals)/med
            prof[mode]=dict(whole_us=vals,median=med,relative_mad=mad,setup_median=S.median(a['setup_total_us'] for a in rr),drop_median=S.median(a['drop_us'] for a in rr),prefix_median=S.median(a['prefix_step_us'] for a in rr))
        pa=[available(a,point,budget) for a in ck];ra=[available(a,rob,budget) for a in ck] if rob else []
        result[st]=dict(point_us=d,fit_rows=len(fit),cal_scores=sc,cal_pools=len(cal),rank=k,multiplier=mult,budget_us=budget,point=point,robust=rob,check_scores=sr,check_pool_coverage=cov,check_coordinate_coverage=sum(coord)/len(coord) if coord else None,point_availability=pa,robust_availability=ra,point_availability_rate=sum(len(v)>=2 for v in pa)/len(pa),robust_availability_rate=sum(len(v)>=2 for v in ra)/len(ra) if ra else None,profiles=prof)
    maxload=max(a['host']['load'][0] for v in raw.values() for a in v)
    model=dict(strata=result,host_ok=env['ok'] and maxload<=12,host_max_load1=maxload,profile_ok=all(p['relative_mad']<=.20 for v in result.values() for p in v['profiles'].values()),availability_ok=all(v['point_availability_rate']>=.75 for v in result.values()),gain_sha=sha(D/'GAIN_LOCK.json'),timing_sha=sha(D/'C07_TIMING_RAW.json'),cohort_sha=sha(D/'C07_COHORT.json'),binary_sha=sha(n.BIN))
    write('C07_COST_MODEL.json',model)
    print('COST',json.dumps({st:{k:v[k] for k in ('point_us','cal_pools','rank','multiplier','budget_us','check_pool_coverage','check_coordinate_coverage','point_availability_rate','robust_availability_rate','profiles')} for st,v in result.items()}),flush=True)
    print('GATES',model['host_ok'],model['profile_ok'],model['availability_ok'],flush=True)
def live():
    check_lock();co=read('C07_COHORT.json');m=read('C07_COST_MODEL.json')
    assert m['gain_sha']==sha(D/'GAIN_LOCK.json') and m['cohort_sha']==sha(D/'C07_COHORT.json') and m['timing_sha']==sha(D/'C07_TIMING_RAW.json')
    pre=preflight();write('C08_PREFLIGHT.json',pre)
    if not pre['interval']['ok']:
        print('LIVE_HOST_BLOCKED',pre,flush=True);return
    start=snap();raw=[]
    with n.Server() as srv:
        srv.call('SETUP',groups(co,'FIT','TACTICAL')[0])
        for rep,seed in enumerate(SEEDS):
            for st in ST[::(-1 if rep%2 else 1)]:
                a0=m['strata'][st];arms=ARMS if a0['robust'] is not None else ARMS[:-1]
                for k,pool in enumerate(groups(co,'TEST',st)):
                    off=(rep+k)%len(arms)
                    for arm in arms[off:]+arms[:off]:
                        mode='DP' if arm=='ROBUST_DP' else arm
                        table=a0['robust'] if arm=='ROBUST_DP' else a0['point'] if arm in ('DP','RATIO') else None
                        a=srv.call(mode,pool,seed,a0['budget_us'],a0['point_us'] if arm!='PREFIX' else None,table)
                        a.update(arm=arm,stratum=st,pool=k,rep=rep,host=snap());raw.append(a)
            print('LIVE',rep,seed,len(raw),flush=True)
    end=snap();write('C08_LIVE_RAW.json',dict(start=start,end=end,interval=interval(start,end),runs=raw));print('BLIND',sha(D/'C08_LIVE_RAW.json'),flush=True)
def summarize():
    check_lock();co=read('C07_COHORT.json');m=read('C07_COST_MODEL.json');rr=read('C08_LIVE_RAW.json');runs=rr['runs']
    roots=[r for st in ST for p in groups(co,'TEST',st) for r in p]
    lab=n.oracle(roots);write('C08_LABELS.json',lab);z={r['id']:r['scores'] for r in lab if r['status']=='EXACT'}
    rng=random.Random(202611250406);by={};summary={}
    for a in runs:
        dest=by.setdefault((a['stratum'],a['arm'],a['seed']),{})
        for r in a['rows']:
            i=r['id']
            if i not in z:continue
            ss=r['steps'];u=[n.utility(z[i][s['best']]) for s in ss];opt=max(n.utility(z[i][c]) for c in r['legal'])
            assert i not in dest;dest[i]=(opt-u[-1],u[-1]-u[0],ss[-1]['visits'])
    for st in ST:
        arms={};ids=sorted(r['id'] for r in roots if r['stratum']==st and r['id'] in z);cfg=m['strata'][st]
        for arm in ARMS:
            a=[a for a in runs if a['stratum']==st and a['arm']==arm]
            if not a:continue
            x=[by[st,arm,s][i] for s in SEEDS for i in ids]
            d=dict(regret=S.mean(v[0] for v in x),gain=S.mean(v[1] for v in x),visits=S.mean(v[2] for v in x))
            for k in ('whole_us','transport_us','setup_total_us','prefix_step_us','plan_us','optional_us','continuation_step_us','drop_us','overrun_us'):
                d[k]=S.mean(sum(r[k] for r in a if r['seed']==s) for s in SEEDS)
            selected=C.Counter(v for r in a for _,v in r['selected']);planned=C.Counter(v for r in a for _,v in r['planned'])
            d.update(selected=dict(selected),planned=dict(planned),evals=S.mean(sum(r['steps'][-1]['eval_calls'] for a1 in a if a1['seed']==s for r in a1['rows']) for s in SEEDS),requests=len(a),overrun_pools=sum(r['overrun_us']>0 for r in a),max_overrun_us=max(r['overrun_us'] for r in a))
            # Actual selected coordinate upper-bound violations; no claim about unselected options.
            if arm=='ROBUST_DP':
                costs_live=[]
                for a1 in a:
                    for r in a1['rows']:
                        for j,s in enumerate(r['steps'][1:]):
                            c=sum(x['wall_us'] for x in r['steps'][1:j+2])
                            costs_live.append(c<=cfg['point_us'][j]*cfg['multiplier'])
                d['selected_coordinate_coverage']=sum(costs_live)/len(costs_live) if costs_live else None
                d['selected_coordinates']=len(costs_live)
            arms[arm]=d
        paired={}
        for candidate in ('DP','ROBUST_DP'):
            if candidate not in arms:continue
            for arm in arms:
                if arm==candidate:continue
                ds=[S.mean(by[st,arm,s][i][0]-by[st,candidate,s][i][0] for s in SEEDS) for i in ids]
                boot=sorted(S.mean(rng.choices(ds,k=len(ds))) for _ in range(3000))
                paired[candidate+'/'+arm]=dict(improvement=S.mean(ds),ci95=[boot[75],boot[2924]],n=len(ds),time_mismatch=abs(arms[candidate]['whole_us']/arms[arm]['whole_us']-1),transport_mismatch=abs(arms[candidate]['transport_us']/arms[arm]['transport_us']-1))
        eligible=[arm for arm in ('FIXED8','RANDOM16','ENTROPY16') if paired['DP/'+arm]['time_mismatch']<=.1]
        best=min(eligible,key=lambda a:arms[a]['regret']) if eligible else None
        accept=bool(best and paired['DP/'+best]['ci95'][0]>0 and paired['DP/RATIO']['time_mismatch']<=.1 and paired['DP/RATIO']['ci95'][0]>0)
        summary[st]=dict(n=len(ids),arms=arms,paired=paired,eligible=eligible,best_eligible=best,decision_time_pass=accept)
    maxload=max(a['host']['load'][0] for a in runs);hostok=rr['interval']['ok'] and maxload<=12
    out=dict(groups=summary,host_ok=hostok,host_max_load1=maxload,fit_host_ok=m['host_ok'],profile_ok=m['profile_ok'],availability_ok=m['availability_ok'],labels=dict(C.Counter(r['status'] for r in lab)),requests=len(runs),root_traces=sum(len(a['rows']) for a in runs),raw_sha=sha(D/'C08_LIVE_RAW.json'),promotion=bool(hostok and m['host_ok'] and m['profile_ok'] and m['availability_ok'] and all(v['decision_time_pass'] for v in summary.values()) and len(z)==len(roots)))
    write('C08_SUMMARY.json',out)
    print('SUMMARY',json.dumps(out),flush=True)
if __name__=='__main__':
    {'smoke':n.smoke,'fit':fit,'live':live,'score':summarize}[sys.argv[1]]()
