from plan61 import D,T,P,P58,SEEDS,ARMS,sha,read,raw
import sys,collections as C,json,gzip,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P));import audit60 as base
a=base.a
def independent_cell(s,bd):
    if s['locked20'] or s['rival'] is None:return None
    a=s['move'];values={m:(n,w) for m,n,w,p in s['edges']};n,w=values[a]
    if not n:return 0
    superior=False
    for m,(v,z) in values.items():
        if m!=a and v and bd[m][1]>-1 and bd[m][1]>bd[a][0] and z*n>w*v:superior=True
    return (1 if w<0 else 3)+int(superior)
def adaptive(h,r,bds,table):
    e=r['early'];assert e['op']==6 and e['horizon']==8
    ss=e['snapshots'];sn=[];target=None;extend=False;counts=C.Counter()
    for x in ss:
        n=x['visits']
        if n==4:
            idx=independent_cell(x,bds);extend=idx is not None and table[idx]
            assert x['cell61']==idx and x['extend61']==extend
            if idx is not None:counts[f'{idx}:{"extend" if extend else "stop"}']+=1
            stop=idx is not None and not extend
        else:
            assert 'cell61' not in x and 'extend61' not in x
            stop=n==8 and extend and not x['locked20'] and x['rival'] is not None
        sn.append({k:v for k,v in x.items() if k not in ('cell61','extend61')})
        if stop:
            assert target is None and x is ss[-1];target=x
    assert e['reached']==(target is not None) and e['at']==target
    rr=dict(r);rr['early']=dict(e,snapshots=sn)
    if target is not None:
        hh=target['visits'];rr['early'].update(horizon=hh,op=1,reason='eligible',at=sn[-1],reached=True)
        assert e['reason']=='select61'
        a.inspect_early(h,rr,bds,dict(horizon=hh),'STOP')
    else:
        x=next((x for x in sn if x['visits']==16),None)
        reached=x is not None and not x['locked20'] and x['rival'] is not None
        reason='eligible' if reached else 'lock_before_horizon' if x is None else 'locked_at_horizon' if x['locked20'] else 'no_rival'
        rr['early'].update(horizon=16,op=0,reason=reason,at=x,reached=reached)
        a.inspect_early(h,rr,bds,dict(horizon=16),'CONTROL')
        assert e['reason']==('continue61' if ss else 'lock_before_horizon')
    return counts
base.adaptive=adaptive
def shard(arg):
    mode,meta=arg;a.CT.clear();a.EX.clear();path=T/meta['file'];assert sha(path)==meta['sha256'];pl=read(D/'PLAN.json')
    tab=read(D/'TABLE.json');tables={'SELECT':tab['table'],'CONTRAST':tab['contrast']};ct=C.Counter();rows=[];cells=C.defaultdict(C.Counter)
    for z in raw(path):
        if mode=='TRAIN':
            j=z['job'];assert j==pl['roots'][j['id']];gs=z['games'];ref=gs['4']
            for h in (4,8,0):
                g=gs[str(h)]
                c,_=base.audit_game(g,j['history'],j['color'],lambda i:('FULL' if h==0 else 'STOP',h or 16) if i==0 else ('STOP',4),[])
                ct['games']+=1;ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical'];ct['moves']+=c['moves']
                if h==4:
                    r=g['rows'][0];assert r['early']['at']==j['snapshot'] and r['partial']['bounds']==j['bounds']
                    bd={m:(l,u) for m,l,u in j['bounds']};assert independent_cell(j['snapshot'],bd)==j['cell']
                else:
                    aa,bb=ref['rows'][0],g['rows'][0];e=bb['early'];pa=z['pairs'][str(h)]
                    assert bb['early']['snapshots'][:len(aa['early']['snapshots'])]==aa['early']['snapshots']
                    qcost=lambda g,arm:sum(x['actual_nn'] for x in g['rows'] if x['arm']==arm)
                    assert pa==dict(gain2=g['score2']-ref['score2'],root_cost=bb['actual_nn']-aa['actual_nn'],changed=aa['move']!=bb['move'],qcost=qcost(g,'Q')-qcost(ref,'Q'),bcost=qcost(g,'B')-qcost(ref,'B'),reached=e['reached'])
                    assert pa['root_cost']>=0
                    if not pa['changed']:assert g['history']==ref['history'] and base.sem(g['rows'][1:])==base.sem(ref['rows'][1:])
                    ct[f'changed{h}']+=pa['changed'];ct[f'harm{h}']+=pa['gain2']>0;ct[f'benefit_stop{h}']+=pa['gain2']<0
            ct['roots']+=1;rows.append(j['id'])
        else:
            gs=z;g=gs['H4'];j=pl['jobs'][g['job']['id']];seed=g['seed'];color=g['color'];ag={}
            for arm in ARMS:
                g=gs[arm];assert g['job']==j and g['seed']==seed and g['color']==color and g['policy']==arm
                rule=lambda i:('PLAIN',20) if arm=='CONTROL' else ('STOP',4) if arm=='H4' else ('ADAPT',8)
                c,cl=base.audit_game(g,j['opening'],color,rule,tables.get(arm,[]));ag[arm]=c;cells[arm].update(cl)
                ct['games']+=1;ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical'];ct['moves']+=c['moves']
                ct[arm+'_attempts']+=c['attempts'];ct[arm+'_events']+=c['events']
            ct['episodes']+=1
            pairs={arm:dict(delta2=gs[arm]['score2']-gs['H4']['score2'],diverged=gs[arm]['history']!=gs['H4']['history']) for arm in ARMS if arm!='H4'}
            rows.append(dict(id=j['id'],seed=seed,color=color,geometry=j['geometry'],arms=ag,pairs=pairs))
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return rows,dict(a.CT),{k:dict(v) for k,v in cells.items()},[(list(k),v) for k,v in a.EX.items()]
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json')
    assert lock['source']==sha(D/('continuation61.py' if mode=='TRAIN' else 'arena61.py'))
    pl=read(D/'PLAN.json');rr=[];ct=C.Counter();cells=C.defaultdict(C.Counter);ex=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for r,c,cl,xx in pool.map(shard,[(mode,m) for m in lock['files']]):
            rr+=r;ct.update(c);ex+=xx
            for k,v in cl.items():cells[k].update(v)
    if mode=='TRAIN':assert sorted(rr)==list(range(len(pl['roots'])))
    else:assert len(rr)==672 and {(r['id'],r['seed'],r['color']) for r in rr}=={(j['id'],s,c) for j in pl['jobs'] for s in SEEDS for c in (-1,1)}
    assert ct['cache_physical']==lock['counts']['physical'] and ct['cache_logical']==lock['counts']['logical']
    out=dict(rows=rr,proof=dict(ct),cells={k:dict(v) for k,v in cells.items()},exact=ex,lock_sha256=sha(T/(mode+'_LOCK.json')),counts=lock['counts'])
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print(mode,'AUDIT',len(rr),dict(ct),out['cells'],flush=True)
if __name__=='__main__':main(sys.argv[1])
