"""Same-position resource mechanism diagnostic; no policy fit or refit."""
from arena import *
def run(arg):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    seed,rr=arg;c,h,ct,meta=client(seed);out=[]
    try:
        for x in rr:
            oo=['base32','proof32','blind20','proof20'];random.Random(SEED+x['id']*100+x['ply']).shuffle(oo)
            d={}
            for k in oo:d[k]=decide(c,ct,x['history'],k,x['seed'])
            assert (d['proof32']['proof']==d['proof20']['proof'])
            if d['proof32']['kind']=='CERTIFICATE':assert d['proof32']['move']==d['proof20']['move'] and d['proof32']['NN']==d['proof20']['NN']==0
            out.append(dict(x,order=oo,results=d))
    finally:h.remove();c.stop()
    return out
def main():
    lk=load('GAMES_LOCK.json');rr=[];miss=[]
    for fi in lk['files']:
        if '_proof32_' not in fi['file']:continue
        p=D/fi['file'];assert sha(p)==fi['sha256']
        for g in lines(p):
            if g['candidate_side']!=1:continue
            j=g['job'];hh=g['history']
            for ply in (4,10):
                if len(hh)<=ply:miss.append([j['seed'],j['id'],ply]);continue
                rr.append(dict(model=j['seed'],id=j['id'],ply=ply,history=hh[:ply],seed=(j['game_seed']+ply*0x9e3779b97f4a7c15)&((1<<64)-1)))
    assert len(rr)+len(miss)==72
    with ProcessPoolExecutor(max_workers=3) as ex:res=list(ex.map(run,[(s,[x for x in rr if x['model']==s]) for s in SEEDS]))
    rows=[x for z in res for x in z];assert len(rows)==len(rr)
    path=D/'SAME_POSITION.jsonl.gz';assert not path.exists()
    with gzip.open(path,'wt',compresslevel=3) as fd:
        for x in rows:fd.write(json.dumps(x,separators=(',',':'))+'\n')
    out={}
    for k in ('base32','proof32','blind20','proof20'):
        aa=[x['results'][k] for x in rows]
        out[k]=dict(n=len(aa),NN=sum(a['NN'] for a in aa),physical=sum(a['physical'] for a in aa),wall_ns=sum(a['wall_ns'] for a in aa),cert=sum(a['kind']=='CERTIFICATE' for a in aa))
    for k,v in out.items():
        v['NN_reduction']=1-v['NN']/out['base32']['NN'];v['physical_reduction']=1-v['physical']/out['base32']['physical']
    save('SAME_POSITION_LOCK.json',dict(games=sha(D/'GAMES_LOCK.json'),sha256=sha(path),n=len(rows),missing=miss,summary=out,scope='post-lock same-position comparison at predetermined plies4 and10 of proof32 candidate-black games; not representative or independent performance evidence'))
    print('SAME_POSITION',out,'missing',len(miss),flush=True)
if __name__=='__main__':main()
