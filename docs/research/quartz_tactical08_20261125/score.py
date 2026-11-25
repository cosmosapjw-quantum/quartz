"""Frozen post-processing of actual ladder snapshots; no minimax oracle."""
import collections as C,json,math,random,statistics as S
from experiment import D,read,rows,write,ST,sha
def point(a,x):
    p=a['proof'];iv={m:(l,u) for m,l,u,rep in p['actions']};best=x['best'];l,u=iv[best]
    lo=max(0,max((v[0] for m,v in iv.items() if m!=best),default=-1)-u)/2
    hi=max(0,max((v[1] for m,v in iv.items() if m!=best),default=-1)-l)/2
    t=set(a['targets']);r=x['result'];es=r['edges'] if r else None
    E=None if not r or not t else any(e['move'] in t for e in es)
    V=None if not r or not t else any(e['move'] in t and e['n']>0 for e in es)
    G=None if not t else best in t
    return dict(model=a['model_identity']['model'],arm=a['arm'],id=a['id'],family=a['family'],stratum=a['stratum'],player=a['player'],budget=x['budget'],visits=x['visits'],target_rank=a['target_rank'],targeted=bool(t),exposed=E,visited=V,selected=G,best=best,materialized=None if not r else len(es),proof_optimal=hi==0,regret_lo=lo,regret_hi=hi,strictly_dominated=u<p['interval'][0],known_loss=l==u==-1,cert_stop=r is None,nn=0 if not r else r['evaluator_calls'],cumulative_us=x['cumulative_us'],step_us=x['step_us'],whole_us=a['whole_us'],transport_ns=a['transport_ns'])
def summarize(x):
    t=[v for v in x if v['targeted']];e=[v for v in t if v['exposed'] is not None]
    E=sum(v['exposed'] for v in e);V=sum(v['visited'] for v in e);G=sum(v['selected'] for v in e)
    if e:assert G<=V<=E<=len(e)
    return dict(n=len(x),target_n=len(t),search_target_n=len(e),exposed=E,visited=V,selected=sum(v['selected'] for v in t),failure_partition=[len(e)-E,E-V,V-G] if e else None,p_E=E/len(e) if e else None,p_V_given_E=V/E if E else None,p_S_given_V=G/V if V else None,median_rank=S.median(v['target_rank'] for v in e) if e else None,stop=sum(v['cert_stop'] for v in x),proof_optimal=sum(v['proof_optimal'] for v in x),regret_lower_mean=S.mean(v['regret_lo'] for v in x),regret_upper_mean=S.mean(v['regret_hi'] for v in x),strictly_dominated=sum(v['strictly_dominated'] for v in x),known_loss=sum(v['known_loss'] for v in x),mean_NN=S.mean(v['nn'] for v in x),mean_visits=S.mean(v['visits'] for v in x),median_cumulative_ms=S.median(v['cumulative_us'] for v in x)/1000,mean_cumulative_ms=S.mean(v['cumulative_us'] for v in x)/1000,median_materialized=S.median(v['materialized'] for v in x if v['materialized'] is not None) if any(v['materialized'] is not None for v in x) else None)
def main():
    allrows=[];raw=[];host=[]
    for model in ('F1','F2'):
        name=f'C08_TEST_{model}.jsonl.gz';r=list(rows(name));h=read(f'C08_TEST_{model}_HOST.json')
        assert h['raw_sha256']==sha(D/name) and len(r)==h['requests']
        raw+=r;host.append(h)
        for a in r:allrows.extend(point(a,x) for x in a['points'])
    assert len(raw)==80 and len(allrows)==304,(len(raw),len(allrows))
    by=C.defaultdict(list)
    for x in allrows:
        for st in (x['stratum'],'ALL')+ (('TACTICAL',) if x['targeted'] else ()):
            by[x['model'],x['arm'],x['budget'],st].append(x)
        if x['targeted']:by[x['model'],x['arm'],x['budget'],'BLACK' if x['player']==1 else 'WHITE'].append(x)
    summary={'/'.join(map(str,k)):summarize(v) for k,v in sorted(by.items())}
    p={(x['id'],x['model'],x['arm'],x['budget']):x for x in allrows}
    assert len(p)==len(allrows)
    diffs={}
    for name,model,arm,ref_model,ref_arm,budgets in [
        ('F2_minus_F1','F2','BASE','F1','BASE',[1,8,32,64,128,256]),
        ('EXPOSE_minus_BASE','F1','EXPOSE','F1','BASE',[1,8,32,64,128]),
        ('FOCUS_minus_EXPOSE','F1','FOCUS','F1','EXPOSE',[1,8,32,64,128]),
        ('GUARD_minus_BASE','F1','GUARD','F1','BASE',[8]),
        ('CERT_minus_BASE','F1','CERT','F1','BASE',[8])]:
        dd={}
        for b in budgets:
            v=[(x,p[x['id'],ref_model,ref_arm,b]) for x in allrows if x['model']==model and x['arm']==arm and x['budget']==b and x['targeted']]
            z=[int(x['selected'])-int(y['selected']) for x,y in v]
            dd[str(b)]=dict(families=len(v),improved=sum(q>0 for q in z),worsened=sum(q<0 for q in z),unchanged=sum(q==0 for q in z),selection_gain=S.mean(z),mean_NN_delta=S.mean(x['nn']-y['nn'] for x,y in v),mean_cumulative_ms_delta=S.mean(x['cumulative_us']-y['cumulative_us'] for x,y in v)/1000)
        diffs[name]=dd
    # Root-level time is measured once per actual request, not a sum over cumulative snapshots.
    resource={}
    for model,arm in sorted({(x['model_identity']['model'],x['arm']) for x in raw}):
        v=[x for x in raw if (x['model_identity']['model'],x['arm'])==(model,arm)]
        resource[f'{model}/{arm}']=dict(requests=len(v),visits=sum(x['visits'] for x in v),NN_rows=sum(x['nn_rows'] for x in v),mean_whole_ms=S.mean(x['whole_us'] for x in v)/1000,mean_transport_ms=S.mean(x['transport_ns'] for x in v)/1e6,mean_python_cpu_ms=S.mean(x['python_cpu_ns'] for x in v)/1e6,mean_rust_cpu_ms_tick_estimate=S.mean(x['rust_cpu_ticks']/x['tick_hz']*1000 for x in v),stop=sum(x['cause']!='VISITS' for x in v))
    # Stable rank changes across the two actual frozen models.
    base={a['id']:a for a in raw if a['arm']=='BASE' and a['model_identity']['model']=='F1'}
    alt={a['id']:a for a in raw if a['arm']=='BASE' and a['model_identity']['model']=='F2'}
    ranks=[dict(id=i,stratum=a['stratum'],player=a['player'],F1=a['target_rank'],F2=alt[i]['target_rank']) for i,a in base.items() if a['targets']]
    out=dict(scope='16 fresh constructed families, not natural self-play, eight target-bearing roots; descriptive no strength promotion',requests=len(raw),snapshots=len(allrows),visits=sum(x['visits'] for x in raw),NN_rows=sum(x['nn_rows'] for x in raw),groups=summary,paired=diffs,rank_pairs=ranks,resources=resource,host=host,timing_admission=False,scientific_promotion=False)
    write('C08_SUMMARY.json',out);write('C08_METRICS.json',allrows)
    for model,arm,bs in [('F1','BASE',[1,8,32,64,128,256]),('F2','BASE',[1,8,32,64,128,256]),('F1','EXPOSE',[1,8,32,64,128]),('F1','FOCUS',[1,8,32,64,128]),('F1','GUARD',[8]),('F1','CERT',[8])]:
        print(model,arm,[(b,{k:summary[f'{model}/{arm}/{b}/TACTICAL'][k] for k in ('exposed','visited','selected','stop','mean_NN','mean_cumulative_ms')}) for b in bs])
    print('RANK',ranks)
    print('RESOURCE',resource)
    print('HOST',[(v['model'],v['seconds'],v['swap_pages_s'],v['timing_gate'],v['max_load']) for v in host])
if __name__=='__main__':main()
