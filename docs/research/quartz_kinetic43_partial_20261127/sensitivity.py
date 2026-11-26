"""Descriptive decomposition; no post-outcome policy admission or refit."""
import json,collections as C,gzip
from k43_plan import D,TIERS,load,save,sha
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    for x in load(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def main():
    x=load('SUMMARY.json');src={};arena={}
    for mode,s in x['source'].items():
        a=s['counts'];c={}
        for d,r in s['tiers'].items():
            net_l=a['L_cpu_ns']-r['cpu_ns'];saved_l=a['L_cpu_ns']+r['proof_cpu_ns']-r['cpu_ns']
            c[d]=dict(additional_NN_saved_beyond_lock=a['L_nn']-r['nn'],additional_CPU_saved_beyond_lock_ns=net_l,avoided_L_cpu_ns=saved_l,proof_cpu_ns=r['proof_cpu_ns'],proof_cost_over_avoided_L_cpu=r['proof_cpu_ns']/saved_l if saved_l>0 else None,conditional_NN_saving_vs_L=1-r['nn']/a['L_nn'],conditional_CPU_saving_vs_L=1-r['cpu_ns']/a['L_cpu_ns'])
        src[mode]=c
    for mode in ('NN_ARENA','R_ARENA'):
        c=C.Counter();by=C.defaultdict(C.Counter);pairs=C.defaultdict(list)
        for g in rows(mode):
            pairs[g['job']['id'],g['seed']].append(g)
            for r in g['rows']:
                q=r.get('partial')
                if not q:continue
                z=by[str(g['seed'])]
                for ct in (c,z):
                    ct['attempts']+=1;ct['queries']+=q['queries'];ct['scans']+=q['scans'];ct['checks']+=q['checks'];ct['proof_cpu_ns']+=r['proof_cpu_ns'];ct['exhausted']+=q['exhausted']
                    if q['cert'] is not None:
                        l,u=next((l,u) for m,l,u in q['bounds'] if m==q['cert'])
                        ct['cert']+=1;ct['cert_win']+=l==1;ct['cert_loss']+=u==-1;ct['cert_unknown']+=l<u;ct['cert_draw']+=l==u==0
                    elif r['replacement']:ct['replaced']+=1
        c['complementary_pairs']=sum(v[0]['score2']+v[1]['score2']==2 for v in pairs.values())
        c['both_Q_win']=sum(v[0]['score2']==v[1]['score2']==2 for v in pairs.values())
        c['both_Q_lose']=sum(v[0]['score2']==v[1]['score2']==0 for v in pairs.values())
        arena[mode]=dict(counts=dict(c),strata={k:dict(v) for k,v in by.items()})
    out=dict(source=src,arena=arena,scope='post-outcome descriptive source cost decomposition vs same-horizon lock and actual certificate types; no retuning or new performance gate; NN source cost estimate excludes separate Python common-gate cost on bypass')
    save('SENSITIVITY.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
