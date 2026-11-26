"""Descriptive post-outcome same-K38-checkpoint sensitivity, not an admission gate."""
from collections import defaultdict
from k39_plan import D,load,save,sha,SEED
import gzip,json,numpy as np
def run():
    out={};x=load('SUMMARY.json')
    rows=defaultdict(list)
    for fi in load('ARENA_LOCK.json')['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256']
        with gzip.open(p,'rt') as f:
            for l in f:
                g=json.loads(l)
                if g['match']=='L20_B32' and g['job']['seed'] in (101,102,103):rows[(g['job']['seed'],g['job']['id'])].append(g)
    a=defaultdict(list)
    for (s,i),gg in sorted(rows.items()):
        assert len(gg)==2
        v=[sum(g['score2'] for g in gg),0,0,0,0]
        for g in gg:
            for r in g['rows']:
                k=1 if r['player']==g['player'] else 3
                v[k]+=1;v[k+1]+=r['nn']
        a[s].append(v)
    rng=np.random.default_rng(SEED+197);z=np.zeros((4000,5))
    total=np.zeros(5)
    for s in (101,102,103):
        v=np.asarray(a[s]);assert v.shape==(48,5);total+=v.sum(axis=0)
        z+=v[rng.integers(0,len(v),size=(4000,len(v)))].sum(axis=1)
    score=np.sort(z[:,0]/(2*288));reduction=np.sort(1-(z[:,2]/z[:,1])/(z[:,4]/z[:,3]))
    out['same_K38_panel']=dict(games=288,score=float(total[0]/(2*288)),NI_lower=float(score[199]),NN_reduction=float(1-(total[2]/total[1])/(total[4]/total[3])),NN_CI=[float(reduction[99]),float(reduction[3899])])
    r=x['atlas']['TEST'];c=r['cost'];total=c['B32']-c['L20']
    out['TEST_fixed_root_cost']=dict(B32=c['B32'],B20=c['B20'],L32=c['L32'],L20=c['L20'],total_saved=total,blind_cap_cut=c['B32']-c['B20'],lock_at20=c['B20']-c['L20'],lock_at32=c['B32']-c['L32'],cap_reduction_after_lock=c['L32']-c['L20'],lock20_share=(c['B20']-c['L20'])/total)
    out['break_even_lambda_diagnostic']={s:v['uniform32']['all']['delta4']/(4*v['uniform32']['all']['cost_delta']) for s,v in x['atlas'].items()}
    out['scope']='Post-outcome sensitivity only: no revised price or mask, no additional performance admission, fixed checkpoint panel. Root cost decomposition valid only on matched root set.'
    save('SENSITIVITY.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':run()
