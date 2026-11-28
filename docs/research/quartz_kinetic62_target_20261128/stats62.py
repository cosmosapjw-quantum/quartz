from plan62 import D,T,ARMS,sha,read
import gzip,json,collections as C,random,math
from fractions import Fraction as F
import numpy as np
KEYS=('score2','games','Qnn','Qmoves','Bnn','Bmoves','Qcpu','Bcpu','wins','draws','losses','targets','switches','Qlogical','Blogical')
def calc(a):
    i={k:a[...,j] for j,k in enumerate(KEYS)}
    return dict(score=i['score2']/(2*i['games']),saving=1-(i['Qnn']/i['Qmoves'])/(i['Bnn']/i['Bmoves']),cpu_saving=1-(i['Qcpu']/i['Qmoves'])/(i['Bcpu']/i['Bmoves']),q_rate=i['Qnn']/i['Qmoves'],b_rate=i['Bnn']/i['Bmoves'],utility=(i['score2']/2-i['Qnn']/500)/i['games'],q_cost_per_game=i['Qnn']/i['games'])
def low(x):return float(np.sort(x)[399])
def interval(x):return [float(np.sort(x)[199]),float(np.sort(x)[7799])]
def arena(mode):
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:aud=json.load(f)
    rows=aud['rows'];N=64 if mode=='MAIN' else 16;M=7 if mode=='MAIN' else 3;assert len(rows)==2*N*M
    ids=sorted({r['id'] for r in rows});ii={j:i for i,j in enumerate(ids)};fdata=np.zeros((N,3,len(KEYS)),dtype=np.int64);cells=C.defaultdict(list);byid={}
    for r in rows:
        i=ii[r['id']];byid[i]=r['geometry']
        for k,arm in enumerate(ARMS):fdata[i,k]+=np.array([r['arms'][arm].get(x,0) for x in KEYS],dtype=np.int64)
    for i,c in sorted(byid.items()):cells[c].append(i)
    assert len(cells)==2 and all(len(v)==N//2 for v in cells.values()) and (fdata[:,:,1]==2*M).all()
    rng=np.random.default_rng(202611286200+(mode=='ARCHIVE'));ix=np.concatenate([rng.choice(v,(8000,len(v)),replace=True) for v in cells.values()],axis=1)
    agg=fdata.sum(axis=0);boot=fdata[ix].sum(axis=1);point=calc(agg);bs=calc(boot)
    out={'mode':mode,'arms':{},'contrasts':{},'audit_sha256':sha(T/(mode+'_AUDIT.json.gz'))}
    for k,arm in enumerate(ARMS):
        p={n:float(v[k]) for n,v in point.items()};b={n:low(v[:,k]) for n,v in bs.items()}
        out['arms'][arm]=dict(point=p,lower95=b,counts={x:int(v) for x,v in zip(KEYS,agg[k])})
    for t,b in [(1,0),(2,0),(2,1)]:
        x={n:float(v[t]-v[b]) for n,v in point.items()};y={n:low(v[:,t]-v[:,b]) for n,v in bs.items()}
        gates=dict(score_NI=y['score']>=-.02,absolute_saving=out['arms'][ARMS[t]]['lower95']['saving']>=.85,relative_saving=y['saving']>=-.05,cpu_positive=point['cpu_saving'][t]>0,signed_utility=y['utility']>0)
        out['contrasts'][ARMS[t]+'-'+ARMS[b]]=dict(point=x,lower95=y,CI={n:interval(v[:,t]-v[:,b]) for n,v in bs.items()},gates=gates,performance=all(gates[k] for k in ('score_NI','absolute_saving','relative_saving','cpu_positive')),benefit=all(gates.values()))
        if b==0:
            p=[r['pairs'][ARMS[t]] for r in rows]
            out['contrasts'][ARMS[t]+'-H4']['pairs']=dict(diverged=sum(z['diverged'] for z in p),improved=sum(z['delta2']>0 for z in p),worsened=sum(z['delta2']<0 for z in p))
    data=fdata.tolist();rng=random.Random(202611286210+(mode=='ARCHIVE'));vv=C.defaultdict(list)
    def single(s):return F(s[0],2*s[1]),1-F(s[2]*s[5],s[3]*s[4]),F(s[0]*250-s[2],500*s[1])
    for _ in range(8000):
        sel=[rng.choice(v) for v in cells.values() for j in v];a=[single([sum(data[i][k][j] for i in sel) for j in range(6)]) for k in range(3)]
        for t,b in [(2,0),(2,1)]:
            for i,key in enumerate(('score','saving','utility')):vv[ARMS[t]+'-'+ARMS[b],key].append(float(a[t][i]-a[b][i]))
    out['independent']={f'{a}:{k}':sorted(v)[399] for (a,k),v in vv.items()}
    (T/(mode+'_SUMMARY.json')).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(mode,json.dumps(out),flush=True)
def dev():
    with gzip.open(T/'DEV_AUDIT.json.gz','rt') as f:aud=json.load(f)
    pl=read(D/'PLAN.json');rows=aud['rows'];assert len(rows)==400;cs=C.defaultdict(list)
    for r in rows:
        j=pl['roots'][r['id']];p=r['pair'];a,b=p['a'],p['b'];y=p['score2'];m=p['moves'];base=y['H4'];v=y['FORCE_B']-base;g=y['TARGET']-base;h=y['H8']-base;oracle=p['regret2']
        assert g==oracle-p['target_residual2'] and g==int(m['TARGET']==b)*v
        q=r['arms'];cs[r['cell']].append(dict(regret=F(oracle,2),residual=F(p['target_residual2'],2),target_gain=F(g,2),h8_gain=F(h,2),h8_residual=F(p['h8_residual2'],2),oracle_positive=int(v>0),rival_negative=int(v<0),switch=int(m['TARGET']==b),helpful_switch=int(g>0),harmful_switch=int(g<0),missed_positive=int(v>0 and m['TARGET']==a),h8_outside=int(m['H8'] not in (a,b)),h8_better_oracle=int(p['h8_residual2']<0),target_extra=p['target_extra'],h8_extra=p['h8_extra'],target_utility=F(g,2)-F(q['TARGET']['Qnn']-q['H4']['Qnn'],500),h8_utility=F(h,2)-F(q['H8']['Qnn']-q['H4']['Qnn'],500)))
    N=sum(pl['population']);out={'cells':[],'weighted':{},'raw_counts':{},'audit_sha256':sha(T/'DEV_AUDIT.json.gz')}
    for key in cs[0][0]:
        mu=F();var=F();nraw=F()
        for c,Nc in enumerate(pl['population']):
            a=[F(r[key]) for r in cs[c]];n=len(a);assert n==80;m=sum(a,F())/n
            var+=F(Nc,N)**2*(1-F(n,Nc))*sum((x-m)**2 for x in a)/(n*(n-1));mu+=F(Nc,N)*m;nraw+=sum(a,F())
        out['weighted'][key]={'mean':float(mu),'exact':str(mu),'design_se':math.sqrt(float(var))};out['raw_counts'][key]=float(nraw)
    for c,Nc in enumerate(pl['population']):
        out['cells'].append(dict(cell=c,N=Nc,n=80,means={k:float(sum(F(r[k]) for r in cs[c])/80) for k in cs[c][0]}))
    assert F(out['weighted']['target_gain']['exact'])==F(out['weighted']['regret']['exact'])-F(out['weighted']['residual']['exact'])
    # Independent integer-lattice check for selected paired identities using raw audited score values.
    gain=F();reg=F();res=F()
    for r in rows:
        p=r['pair'];w=F(pl['population'][r['cell']],N*80)
        gain+=w*F(p['score2']['TARGET']-p['score2']['H4'],2);reg+=w*F(p['oracle2']-p['score2']['H4'],2);res+=w*F(p['oracle2']-p['score2']['TARGET'],2)
    assert gain==reg-res==F(out['weighted']['target_gain']['exact'])
    (T/'DEV_SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print('DEV',json.dumps(out),flush=True)
if __name__=='__main__':
    import sys
    dev() if sys.argv[1]=='DEV' else arena(sys.argv[1])
