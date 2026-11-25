"""Check actual original MixNet outputs; no MCTS and no teacher inference."""
import subprocess as SP,json,math,hashlib
from pathlib import Path
from publish import D,ROOT,sha
import sys
sys.path.insert(0,str(ROOT/'docs/research/quartz_tactical08_20261125'))
from rules import replay,fixtures
P=ROOT.parent
EXE=P/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'
LOCK=json.loads((D/'RAPFI_LOCK.json').read_text());AS={x['name']:x for x in LOCK['files']}
W=Path(AS['Rapfi_NNUE_freestyle']['path'])
def call(hs):
    q=['RESET'];hist=[]
    for i,h in enumerate(hs):
        b,p,w=replay(h,'freestyle');assert w is None
        q.append('EVAL '+str(i+1)+' '+''.join(map(str,b)));hist.append((b,p))
    q.append('QUIT')
    r=SP.run([str(EXE),'--quartz-eval',str(W)],cwd=W.parent,input='\n'.join(q)+'\n',capture_output=True,text=True,timeout=30)
    assert r.returncode==0,(r.stderr,r.stdout[-1000:])
    lines=r.stdout.splitlines();assert 'QZ_READY 250615 mix9svq freestyle15' in lines and 'QZ_RESET' in lines and any('load weight from' in x and W.name in x for x in lines)
    values=[]
    for i,line in enumerate(x for x in lines if x.startswith('QZ_RESULT ')):
        t=line.split()
        assert len(t)==232, len(t)
        b,p=hist[i];assert list(map(int,t[1:4]))==[i+1,sum(v!=0 for v in b),p]
        w,l,d=map(float,t[4:7]);pi=list(map(float,t[7:]))
        assert all(math.isfinite(x) and x>=0 for x in [w,l,d]+pi)
        assert abs(w+l+d-1)<1e-6 and abs(sum(pi)-1)<1e-5
        assert all(pi[i]==0 for i,v in enumerate(b) if v)
        values.append(dict(history=hs[i],player=p,wdl=[w,d,l],value=w-l,policy=pi))
    assert len(values)==len(hs)
    return values
def main():
    f=fixtures();a=f['trans_a'];b=f['trans_b'];odd=a+[next(i for i in range(225) if i not in a)]
    c=f['long'];hs=[a,odd,c,b,a,[],odd]
    v=call(hs)
    assert v[0]==v[4]
    assert v[0]['policy']==v[3]['policy'] and v[0]['wdl']==v[3]['wdl']
    assert v[1]==v[6]
    fresh=call([c,odd,a])
    for x,y in zip(fresh,[v[2],v[1],v[0]]):assert x==y
    out=dict(upstream_commit=LOCK['source_commit'],bridge_sha256=sha(EXE),bridge_bytes=EXE.stat().st_size,weight_sha256=sha(W),config_sha256=sha(W.parent/'config.toml'),queries=len(v)+len(fresh),exact_repeat_and_reset=True,same_board_different_history=True,rows=v,independent_fresh=fresh,scope='real Rapfi policy and WDL, not minimax')
    (D/'RAPFI_EVAL_SMOKE.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('Actual Rapfi evaluator',len(v)+len(fresh),'queries, identical repeated/fresh outputs, both colors.')
if __name__=='__main__':main()
