"""Independent numerical checks of native count/moment evidence and choice."""
import math
from fractions import Fraction as F
import kinetic_ref as kr
import monitor_ref as mr
ARMS=('K16N','K16BAL','K16REV')
CAPS=(32,64)
def moments(r):
    ca=[x[0] for x in r['candidates']]
    idx={m:i for i,m in enumerate(ca)}
    n=[0]*len(ca);w=[0.]*len(ca)
    assert [e['move'] for e in r['edges']]==ca[:len(r['edges'])]
    assert len(ca)==len(idx)
    for e in r['edges']:
        i=idx[e['move']];n[i]=e['n'];w[i]=e['w'];assert type(n[i]) is int and n[i]>=0 and abs(w[i])<=n[i]+1e-6
        assert e['vl']==0 and abs(e['vvalue'])<1e-6 and -1e-6<=e['m2']<=(n[i]-w[i]*w[i]/n[i] if n[i] else 0)+1e-5
    assert sum(n)==r['root_n'];mr.eq(sum(w),r['root_w'])
    assert ca[max(range(len(n)),key=lambda i:(n[i],i))]==r['best']
    return ca,n,w
def pair(r):
    ca,n,w=moments(r);a=ca.index(r['best'])
    b=max((i for i,v in enumerate(n) if i!=a and v>0),key=lambda i:(n[i],i),default=None)
    if b is None:b=next(i for i in range(len(n)) if i!=a)
    return a,b
def verify(a):
    arm=a['arm'];cap=a['cap'];assert arm in ARMS and cap in CAPS
    if a.get('cause')=='GUARDED':assert a['evidence'] is None;return
    e=a['evidence'];pre=e['prefix'];ca,n,w=moments(pre);i,j=pair(pre)
    assert pre['root_n']==8 and e['pair']==[ca[i],ca[j]]
    assert set(ca)==set(a['legal']) and len(e['phases'])==3
    x=[w[k]/(n[k]+4) for k in (i,j)];mr.vec(x,e['prefix_score'])
    q=(cap-16)//2;order=[i,j,None] if arm=='K16BAL' else [j,i,None] if arm=='K16REV' else [None]*3
    old=n;oldw=w;last=pre;cur=8;prior=[x[1] for x in pre['candidates']];cnt=0
    for k,(t,b,d) in enumerate(zip(e['phases'],order,(q,q,8),strict=True)):
        r=t['result'];ca1,nn,ww=moments(r);cur+=d
        assert ca1==ca and r['candidates']==pre['candidates'] and r['root_hash']==pre['root_hash'] and t['n']==r['root_n']==cur
        assert r['evaluator_calls']>=last['evaluator_calls']
        delta=[v-u for u,v in zip(old,nn,strict=True)]
        assert delta==t['delta'] and min(delta)>=0 and sum(delta)==d
        assert all(abs(v-u)<=dn+1e-6 for u,v,dn in zip(oldw,ww,delta,strict=True))
        assert t['target']==(ca[b] if b is not None else None)
        if b is not None:assert delta[b]==d;cnt+=d
        mr.vec(t['flux'],kr.flow(old,nn,prior))
        if k==1:
            p=[(ww[l]-w[l]+4*z)/(nn[l]-n[l]+4) for l,z in zip((i,j),x,strict=True)]
            mr.vec(p,e['pulse_score'])
        if k==2:
            f=[ww[l]/(nn[l]+4) for l in (i,j)];mr.vec(f,e['final_score'])
        old,oldw,last=nn,ww,r
    assert last==a['result'] and cur==a['visits']==cap and a['NN_calls']==last['evaluator_calls']
    for name,vals in [('prefix',x),('pulse',p),('final',f)]:
        assert e[name+'_best']==ca[j if vals[1]>vals[0] else i] and all(abs(y)<=1+1e-8 for y in vals)
    assert e['native_best']==a['best']==last['best']
    return cnt
