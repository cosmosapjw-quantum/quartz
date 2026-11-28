"""Actual H4/H8/Q20 branch games on sampled frozen H4 occurrences."""
from plan61 import D,T,P58,SEEDS,HS,sha,read,raw,feature
import json,gzip,sys,collections as C,random,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P58))
import native58 as n
n.T=T
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def play(s,ev,j,h):
    from k46_engine import from_history,won,FULL
    history=j['history'].copy();rr=[];e0=ev.rows
    while True:
        a,b=from_history(history);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(history)%2);q=side==j['color']
        if not rr:
            assert q;r=s.early(history,h or 16,int(bool(h)))
        elif q:r=s.early(history,4,1)
        else:r=s.search(history,False)
        assert r['history']==history and r['move'] not in history and 'target' not in r
        r.update(arm='Q' if q else 'B',player=side,ply=len(history));rr.append(r);history.append(r['move'])
    g=dict(h=h,history=history,rows=rr,winner=w,score2=1+j['color']*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr))
    assert ev.rows-e0==g['actual_nn'];return g
def pair(base,other,h):
    a,b=base['rows'][0],other['rows'][0]
    assert a['history']==b['history'] and 'early' in a and 'early' in b
    assert b['early']['snapshots'][:len(a['early']['snapshots'])]==a['early']['snapshots']
    assert b['actual_nn']>=a['actual_nn']
    if a['move']==b['move']:
        assert base['score2']==other['score2'] and base['history']==other['history'] and sem(base['rows'][1:])==sem(other['rows'][1:])
    cost=lambda g,arm:sum(r['actual_nn'] for r in g['rows'] if r['arm']==arm)
    return dict(gain2=other['score2']-base['score2'],root_cost=b['actual_nn']-a['actual_nn'],changed=a['move']!=b['move'],qcost=cost(other,'Q')-cost(base,'Q'),bcost=cost(other,'B')-cost(base,'B'),reached=b['early']['reached'])
def worker(arg):
    seed,i,js,smoke=arg
    from k46_engine import load_eval
    p=read(D/'PLAN.json');ev=load_eval(seed);ct=C.Counter();path=T/f'{"TRAIN_SMOKE" if smoke else "TRAIN"}_{seed}_{i}.jsonl.gz';assert not path.exists()
    with n.NNServer(ev,bin=p['old_binary']) as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            gs={h:play(s,ev,j,h) for h in j['arm_order']}
            a=gs[4]['rows'][0]
            assert a['early']['reached'] and a['early']['at']==j['snapshot'] and a['partial']['bounds']==j['bounds'] and a['actual_nn']==j['root_nn']
            assert feature(a['early']['at'],a['partial']['bounds'])==j['cell']
            pairs={h:pair(gs[4],gs[h],h) for h in (8,0)}
            for g in gs.values():
                ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn']
            ct['roots']+=1
            for h,z in pairs.items():
                ct[f'changed{h}']+=z['changed'];ct[f'harm{h}']+=z['gain2']>0;ct[f'benefit_stop{h}']+=z['gain2']<0
            f.write(json.dumps(dict(job=j,games=gs,pairs=pairs),separators=(',',':'),allow_nan=False)+'\n')
    assert ev.rows==ct['physical']
    print('TRAIN SHARD',seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(smoke=False):
    p=read(D/'PLAN.json');assert sha(p['old_binary'])==p['old_binary_sha256'];roots=p['roots']
    if smoke:
        j=roots[0];return worker((j['seed'],99,[j],True))
    tasks=[]
    for s in SEEDS:
        js=[j for j in roots if j['seed']==s]
        tasks.extend((s,i,js[i::3],False) for i in range(3) if js[i::3])
    random.Random(202611286101).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=p['train_workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['roots']==len(roots) and ct['games']==3*len(roots)
    (T/'TRAIN_LOCK.json').write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'continuation61.py'),binary=p['old_binary_sha256']),indent=2,sort_keys=True)+'\n')
    print('TRAIN',dict(ct),flush=True)
if __name__=='__main__':main('--smoke' in sys.argv)
