"""Independent stdlib integer totals and family bootstrap."""
import collections as C,gzip,json,random,hashlib
from fractions import Fraction as F
from k56_plan import D,T,load,sha,SEED
ARMS=('control','cold','reuse','voc')
def raw(mode):
    for f in load(mode+'_LOCK.json')['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as fd:yield from map(json.loads,fd)
def add(a,b):
    for i,x in enumerate(b):a[i]+=x
def sums(a):return [sum(x[i] for x in a) for i in range(len(a[0]))]
def game(g):
    a=[g['score2'],1,0,0,0,0,0,0,0,0,0,0]
    for r in g['rows']:
        k=2 if r['arm']=='Q' else 7;a[k]+=1;a[k+1]+=r.get('actual_nn',r['nn']);a[k+2]+=r['nn'];a[k+3]+=r['cpu_ns'];a[k+4]+=r['wall_ns']
    return a
def gm(a):
    return dict(score=float(F(a[0],2*a[1])),NN_reduction=float(1-F(a[3]*a[7],a[8]*a[2])),logical_NN_reduction=float(1-F(a[4]*a[7],a[9]*a[2])),CPU_reduction=float(1-F(a[5]*a[7],a[10]*a[2])))
def pm(a):
    return dict(tau=float(F(a[0],2*a[1])),local_nn=float(F(a[2],a[1])),delta_Q=float(F(a[3],a[1])),delta_B=float(F(a[4],a[1])),delta_total=float(F(a[5],a[1])))
def ci(a):
    a=sorted(a);return dict(CI=[a[199],a[7799]],lower95=a[399])
def main():
    s=load('SUMMARY.json');rng=random.Random(SEED+989);nn=C.defaultdict(lambda:C.defaultdict(lambda:[0]*66))
    for x in raw('NN'):
        gs={a:game(x[a]) for a in ARMS};j=x['control']['job']['id'];q=[]
        for b in ARMS[1:]:
            l,r=gs['control'],gs[b];z=x['pairs'][b];q.extend([r[0]-l[0],1,z['local_nn'],r[3]-l[3],r[8]-l[8],r[3]+r[8]-l[3]-l[8]])
        for pan in ('all','old' if x['control']['seed'] in (101,102,103,105) else 'new'):add(nn[pan][j],q+sum((gs[a] for a in ARMS),[]))
    out=dict(nn={},rapfi={},replicates=8000,seed=SEED+989)
    for pan,x in nn.items():
        ids=sorted(x);assert len(ids)==48;q=sums(list(x.values()))
        for i,b in enumerate(ARMS[1:]):
            for k,v in pm(q[6*i:6*i+6]).items():assert abs(v-s['nn'][pan]['pairs'][b]['metrics'][k])<1e-14
        for i,a in enumerate(ARMS):
            for k,v in gm(q[18+12*i:30+12*i]).items():
                r=s['nn'][pan]['arms'][a];z=r['logical']['metrics']['NN_reduction'] if k=='logical_NN_reduction' else r['physical']['metrics'][k]
                assert abs(v-z)<1e-14
        bs={a:C.defaultdict(list) for a in ARMS+tuple(b+'_effect' for b in ARMS[1:])}
        for _ in range(8000):
            q=sums([x[j] for j in rng.choices(ids,k=len(ids))]);vs={a:gm(q[18+12*i:30+12*i]) for i,a in enumerate(ARMS)}
            vs.update({b+'_effect':pm(q[6*i:6*i+6]) for i,b in enumerate(ARMS[1:])})
            for a,z in vs.items():
                for k,v in z.items():bs[a][k].append(v)
        out['nn'][pan]={a:{k:ci(v) for k,v in z.items()} for a,z in bs.items()}
    aa={(g['job']['id'],g['color']):g for g in raw('R_96')};bb={(g['job']['id'],g['color']):g for g in raw('R_T')};assert set(aa)==set(bb)
    for pan in ('all','unresolved_tactical','supplement'):
        x=C.defaultdict(lambda:[0]*30)
        for k,c in aa.items():
            if pan!='all' and c['job']['stratum']!=pan:continue
            t=bb[k];a,b=game(c),game(t);local=sum(r['target']['extra_nn'] for r in t['rows'] if 'target' in r)
            add(x[k[0]],[b[0]-a[0],1,local,b[3]-a[3],b[8]-a[8],b[3]+b[8]-a[3]-a[8]]+a+b)
        ids=sorted(x);assert len(ids)==(72 if pan=='all' else 8 if pan=='unresolved_tactical' else 64);q=sums(list(x.values()))
        for name,z in [('96',gm(q[6:18])),('T',gm(q[18:30]))]:
            for k,v in z.items():
                if k!='logical_NN_reduction':assert abs(v-s['rapfi'][pan]['arms'][name]['metrics'][k])<1e-14
        for k,v in pm(q[:6]).items():assert abs(v-s['rapfi'][pan]['paired']['metrics'][k])<1e-14
        bs={a:C.defaultdict(list) for a in ('96','T','effect')}
        for _ in range(8000):
            q=sums([x[j] for j in rng.choices(ids,k=len(ids))]);vs={'96':gm(q[6:18]),'T':gm(q[18:30]),'effect':pm(q[:6])}
            for a,z in vs.items():
                for k,v in z.items():bs[a][k].append(v)
        out['rapfi'][pan]={a:{k:ci(v) for k,v in z.items()} for a,z in bs.items()}
    (T/'STATS_INDEPENDENT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT NN', {a:z['tau'] for a,z in out['nn']['all'].items() if a.endswith('_effect')},flush=True)
    print('INDEPENDENT RAPFI',out['rapfi'],flush=True)
if __name__=='__main__':main()
