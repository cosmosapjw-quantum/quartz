"""Runtime mirror; sequential f64 accumulation and explicit numerical tie tolerance."""
def predict(m,x):
    g=[0.];k=[0.]
    for a,b in zip(m['gain'],m['cost']):
        v=w=0.
        for i in range(8):v+=a[i]*x[i];w+=b[i]*x[i]
        g.append(max(-1.,min(1.,v)));k.append(max(0.,min(12.,w)))
    op=0
    if not m['stop']:
        for i in (1,2):
            if g[i]-m['lambda']*k[i]>g[op]-m['lambda']*k[op]+1e-12:op=i
    return op,g,k
def main():
    from fit56 import records,choose
    from k56_plan import load,save
    import collections as C
    f=load('FROZEN.json');m=f['model'];out={}
    for p,rows in [('train',[r for r in records(55) if r['job']<48]),('cal',[r for r in records(55) if r['job']>=48]),('K54',list(records(54)))]:
        ct=C.Counter()
        for r in rows:
            o,g,k=predict(m,r['x']);ct['n']+=1;ct['op'+str(o)]+=1;ct['changed_from_float_fit']+=o!=choose(m,r['x'])[0]
            ct['pred_gain']+=g[o];ct['pred_cost']+=k[o];ct['all_cold_cost']+=r['k'][1]
            if o<len(r['g']):ct['gain2']+=round(r['g'][o]*2);ct['cost']+=r['k'][o]
            else:ct['unobserved']+=1
        out[p]=dict(ct)
    assert out['train']['pred_cost']<=f['budget']+1e-12
    save('RUNTIME_FROZEN.json',dict(counts=out,tie=1e-12,model=m,cal_admitted=out['cal'].get('gain2',0)>0 and 2*out['cal'].get('cost',0)<=out['cal']['all_cold_cost']))
    print('RUNTIME',out,flush=True)
if __name__=='__main__':main()
