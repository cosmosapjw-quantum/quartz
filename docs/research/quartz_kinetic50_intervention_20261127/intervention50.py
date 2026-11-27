"""One eligible decision; paired outcomes and exact NN cost decomposition."""
import collections as C
def request_mask(eligible,k,arm):
    assert k in (1,2) and 0<=eligible<=k and arm in ('CONTROL','EXTEND')
    if eligible==k:return None
    return 255 if arm=='EXTEND' and eligible==k-1 else 0
def semantic(x):
    if isinstance(x,dict):return {k:semantic(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns')}
    if isinstance(x,list):return [semantic(v) for v in x]
    return x
def cost(g,tag=None,after=-1):
    return sum(r['nn'] for i,r in enumerate(g['rows']) if i>after and (tag is None or r['arm']==tag))
def pair(c,t):
    assert c['job']==t['job'] and c['seed']==t['seed'] and c['color']==t['color'] and c['policy']=='CONTROL' and t['policy']=='EXTEND'
    k=c['job']['ordinal'];e=c['event'];assert e==t['event']
    assert sum(r['nn'] for r in c['rows'])==c['nn'] and sum(r['nn'] for r in t['rows'])==t['nn']
    local=0;changed=False;bin=None;act=None;ply=None;postQ=postB=0
    if e is None:
        assert c['history']==t['history'] and c['score2']==t['score2'] and c['nn']==t['nn']
        assert semantic(c['rows'])==semantic(t['rows'])
    else:
        assert semantic(c['rows'][:e])==semantic(t['rows'][:e])
        r,s=c['rows'][e],t['rows'][e]
        assert r['arm']==s['arm']=='Q' and r['history']==s['history'] and r['bin']==s['bin'] and r['low']==s['low']
        assert r['eligible_index']==s['eligible_index']==k
        assert r['mask']==0 and s['mask']==255 and not r['extended'] and s['extended']
        assert semantic(r['partial'])==semantic(s['partial'])
        assert s['nn']==s['physical_nn'] and r['nn']==r['physical_nn']
        assert r['nn']==r['low']['nn'] and r['move']==r['low']['move']
        assert s['visits']>=r['visits'] and s['nn']>=r['nn']
        assert all(m==mm and n<=nn for (m,n),(mm,nn) in zip(r['votes'],s['votes'],strict=True))
        local=s['nn']-r['nn'];changed=s['move']!=r['move'];bin=r['bin'];act=[r['move'],s['move']];ply=r['ply']
        postQ=cost(t,'Q',e)-cost(c,'Q',e);postB=cost(t,'B',e)-cost(c,'B',e)
        if not changed:
            assert c['history']==t['history'] and c['score2']==t['score2'] and postQ==postB==0
            assert semantic(c['rows'][e+1:])==semantic(t['rows'][e+1:])
    dq=cost(t,'Q')-cost(c,'Q');db=cost(t,'B')-cost(c,'B')
    assert dq==local+postQ and db==postB and dq+db==t['nn']-c['nn']
    return dict(job=c['job']['id'],split=c['job']['split'],seed=c['seed'],color=c['color'],ordinal=k,reached=e is not None,event=e,ply=ply,bin=bin,actions=act,changed=changed,score2=[c['score2'],t['score2']],delta2=t['score2']-c['score2'],local_nn=local,downstream_Q=postQ,downstream_B=postB,delta_Q=dq,delta_B=db,delta_total=dq+db,same_history=c['history']==t['history'])
def select_mask(rs):
    cost=[sum(r['local_nn'] for r in rs if r['bin']==z) for z in range(8)]
    gain=[sum(r['delta2'] for r in rs if r['bin']==z) for z in range(8)]
    cap=sum(cost)//4
    v=[(sum(gain[z] for z in range(8) if m>>z&1),-m.bit_count(),-m) for m in range(256) if sum(cost[z] for z in range(8) if m>>z&1)<=cap]
    g,_,neg=max(v);m=-neg
    return dict(mask=m,budget=cap,total_cost=sum(cost),selected_cost=sum(cost[z] for z in range(8) if m>>z&1),selected_delta2=g,feasible=len(v))
