"""Actual five-policy continuation games from K59 repeat-occupancy roots."""
from plan60 import D,T,P58,SEEDS,HS,sha,read,raw
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
        elif q:r=s.early(history,j['h0'],1)
        else:r=s.search(history,False)
        assert r['history']==history and r['move'] not in history and 'target' not in r
        r.update(arm='Q' if q else 'B',player=side,ply=len(history));rr.append(r);history.append(r['move'])
    g=dict(h=h,history=history,rows=rr,winner=w,score2=1+j['color']*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr))
    assert ev.rows-e0==g['actual_nn']
    return g
def pair(full,short,h):
    a,b=full['rows'][0],short['rows'][0]
    assert a['history']==b['history'] and 'early' in a and 'early' in b
    e=b['early'];assert e['horizon']==h and e['op']==1
    assert a['early']['snapshots'][:len(e['snapshots'])]==e['snapshots']
    assert a['actual_nn']>=b['actual_nn']
    if not e['reached']:assert a['move']==b['move'] and a['actual_nn']==b['actual_nn']
    if a['move']==b['move']:
        assert full['score2']==short['score2'] and full['history']==short['history'] and sem(full['rows'][1:])==sem(short['rows'][1:])
    return dict(eligible=e['reached'],gain2=full['score2']-short['score2'],root_cost=a['actual_nn']-b['actual_nn'],changed=a['move']!=b['move'])
def worker(arg):
    seed,i,js=arg
    from k46_engine import load_eval
    p=read(D/'PLAN.json');ev=load_eval(seed);ct=C.Counter();path=T/f'TRAIN_{seed}_{i}.jsonl.gz';assert not path.exists()
    with n.NNServer(ev,bin=p['old_binary']) as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            gs={h:play(s,ev,j,h) for h in j['arm_order']}
            pairs={h:pair(gs[0],gs[h],h) for h in HS}
            for g in gs.values():
                ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn']
            ct['roots']+=1;ct['eligible']+=sum(x['eligible'] for x in pairs.values());ct['changed']+=sum(x['changed'] for x in pairs.values())
            f.write(json.dumps(dict(job=j,games=gs,pairs=pairs),separators=(',',':'),allow_nan=False)+'\n')
    assert ev.rows==ct['physical']
    print('TRAIN SHARD',seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(smoke=False):
    p=read(D/'PLAN.json');assert sha(p['old_binary'])==p['old_binary_sha256']
    roots=p['roots']
    if smoke:
        j=roots[0];return worker((j['seed'],99,[j]))
    tasks=[]
    for s in SEEDS:
        js=[j for j in roots if j['seed']==s]
        tasks.extend((s,i,js[i::2]) for i in range(2) if js[i::2])
    random.Random(202611286001).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=p['train_workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['roots']==len(roots) and ct['games']==5*len(roots)
    (T/'TRAIN_LOCK.json').write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'continuation60.py'),binary=p['old_binary_sha256']),indent=2,sort_keys=True)+'\n')
    print('TRAIN',dict(ct),flush=True)
if __name__=='__main__':main('--smoke' in sys.argv)
