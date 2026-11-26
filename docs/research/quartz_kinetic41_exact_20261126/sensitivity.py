"""Post-outcome exact first-action comparison, and censoring diagnostics. No refit."""
import json,gzip,collections as C, pathlib,hashlib
from k41_plan import D,SEEDS,load,save,sha
def raw(p):
    with gzip.open(p,'rt') as fd:yield from map(json.loads,fd)
def rows(mode):
    for f in load(mode+'_LOCK.json')['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];yield from raw(p)
def orient(h):
    a=sum(1<<m for i,m in enumerate(h) if i%2==0);b=sum(1<<m for i,m in enumerate(h) if i%2)
    return (a,b) if len(h)%2==0 else (b,a)
def main():
    p=D/'INDEPENDENT.jsonl.gz';assert sha(p)==load('AUDIT.json')['independent']['artifact_sha256']
    ir={(r['a'],r['b']):r for r in raw(p)};corpus=C.defaultdict(C.Counter)
    for sp in ('TRAIN','CAL','TEST'):
        for r in rows('EXACT_'+sp):
            z=r['exact'];j=r['job'];ref=ir[orient(j['opening'])];vals=ref['values'];best=max(vals.values());c=corpus[j['empties']]
            c['roots']+=1;c['complete']+=z['complete'];c['censored']+=not z['complete']
            c['wins']+=best==1;c['draws']+=best==0;c['losses']+=best==-1;c['mixed']+=len(set(vals.values()))>1
            c['python_nodes']+=z['nodes'];c['python_cpu_ns']+=r['cpu_ns'];c['cpp_nodes']+=ref['nodes'];c['cpp_wall_ns']+=ref['wall_ns']
    pairs=C.defaultdict(dict)
    for g in rows('ARENA'):
        if g['job']['split']!='LATE':continue
        a=g['rows'][0];p=pairs[g['job']['id'],g['seed']]
        assert a['arm'] not in p;p[a['arm']]=(g,a)
    ct=C.Counter();by=C.defaultdict(C.Counter)
    for (i,s),p in pairs.items():
        assert set(p)=={'Q41','B32'}
        g,q=p['Q41'];h=g['job']['opening'];g2,b=p['B32'];assert g2['job']['opening']==h
        vals=ir[orient(h)]['values'];v=max(vals.values());vq,vb=vals[str(q['move'])],vals[str(b['move'])]
        for c in (ct,by[s]):
            c['roots']+=1;c['q_bad']+=vq!=v;c['b_bad']+=vb!=v;c['q_regret2']+=v-vq;c['b_regret2']+=v-vb
            c['improved']+=vq>vb;c['worse']+=vq<vb;c['same_value']+=vq==vb;c['same_move']+=q['move']==b['move']
            c['q_exact']+=q['exact_complete'];c['q_NN']+=q['nn'];c['b_NN']+=b['nn'];c['q_CPU_ns']+=q['cpu_ns'];c['b_CPU_ns']+=b['cpu_ns']
    assert len(pairs)==256
    out=dict(corpus_by_empties={str(k):dict(v) for k,v in sorted(corpus.items())},first_action=dict(ct),by_seed={str(k):dict(v) for k,v in by.items()},scope='post-outcome diagnostic, exact minimax independent labels; CPU and C++ wall not equivalent timings, no refit or reclassification')
    save('SENSITIVITY.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
