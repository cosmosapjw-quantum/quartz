"""Fixed multinomial ridge and heldout family temperature calibration; no arena data."""
from plan67 import D,T,P,read,sha,raw,family
from retention67 import features,predict,expected,DVAL
from audit67 import independent_features
from collections import Counter,defaultdict
import numpy as np,math,json,gzip
def main():
    plan=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');atlas=read(P/'DEV_LOCK.json')
    assert sha(P/'DEV_LOCK.json')==plan['parent_lock_sha256']['DEV']
    rows=[];excluded=[];calboards={family(j['history']) for j in plan['roots'] if j['split']=='CAL'}
    calfamilies={family(j['source_opening']) for j in plan['roots'] if j['split']=='CAL'}
    roots=Counter();nochange=Counter()
    def add(j,source,split,gs,rs,w):
        board=family(j['history']);fam=family(j['source_opening'])
        if split=='TRAIN' and (board in calboards or fam in calfamilies):
            excluded.append(dict(source=source,id=j['id'],domain=j['domain'],reason='CAL board or family collision'));return
        assert (fam in calfamilies)==(split=='CAL')
        roots[source,split,j['domain']]+=1
        for op,r in rs.items():
            a,b,x=features(r,op);aa,bb,xx=independent_features(r,op)
            assert (a,b)==(aa,bb) and max(abs(v-u) for v,u in zip(x,xx))<1e-12
            assert a==j['incumbent'];y=gs[b]-gs[a];assert -2<=y<=2
            if a==b:
                assert y==0;nochange[source,split,j['domain'],op]+=1;continue
            rows.append(dict(source=source,split=split,domain=j['domain'],root=j['id'],family=list(fam),board=list(board),seed=j['seed'],op=op,a=a,b=b,x=x,y=y,w=w/3))
    for m in atlas['files']:
        f=P/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];gs={int(m):g['score2'] for m,g in z['forced'].items()};ops=z['operations'];base=z['forced'][str(j['incumbent'])]['rows'][0]
            add(j,'ATLAS','TRAIN',gs,{'VOTE':base,**{op:g['rows'][0] for op,g in ops.items()}},j['population']/j['sample'])
    for m in lock['files']:
        f=T/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];games=z['games'];gs={}
            for op,g in games.items():
                r=g['rows'][0];m0=r['played'];assert m0 not in gs or gs[m0]==g['score2'];gs[m0]=g['score2']
            add(j,'CENSUS',j['split'],gs,{op:games[op]['rows'][0] for op in ('VOTE','FOCUS','EXPAND')},1.)
    tr=[r for r in rows if r['split']=='TRAIN'];ca=[r for r in rows if r['split']=='CAL']
    assert tr and ca and not ({tuple(r['board']) for r in tr}&{tuple(r['board']) for r in ca})
    assert not ({tuple(r['family']) for r in tr}&{tuple(r['family']) for r in ca})
    def data(rr):
        total=defaultdict(float)
        for r in rr:total[r['source'],r['domain']]+=r['w']
        wt=np.array([r['w']/total[r['source'],r['domain']]/len(total) for r in rr]);X=np.array([r['x'] for r in rr]);Y=np.array([r['y']+2 for r in rr])
        assert abs(wt.sum()-1)<1e-12
        return X,Y,wt
    X,Y,w=data(tr);C,Z,v=data(ca)
    prior=(np.bincount(Y,weights=w*len(Y),minlength=5)+.5)/(len(Y)+2.5);off=np.log(prior);eye=np.eye(5)[Y]
    lam=.02
    def soft(z):
        zz=z-z.max(axis=1,keepdims=True);e=np.exp(zz);return e/e.sum(axis=1,keepdims=True)
    def loss(W):
        pr=soft(X@W.T+off);value=-np.dot(w,np.log(pr[np.arange(len(Y)),Y]))+lam/2*np.sum(W*W)
        grad=((pr-eye)*w[:,None]).T@X+lam*W
        return float(value),grad
    W=np.zeros((5,15));f,g=loss(W);rng=np.random.default_rng(2026112867);u=rng.normal(size=W.shape);u/=np.linalg.norm(u)
    eps=1e-5;fd=(loss(W+eps*u)[0]-loss(W-eps*u)[0])/(2*eps);anal=float(np.sum(g*u));assert abs(fd-anal)<1e-8
    # Softmax Hessian spectral norm <=1/2; use global conservative fixed step.
    step=1/(.5*np.max(np.sum(X*X,axis=1))+lam);trace=[]
    for it in range(20000):
        f,g=loss(W);norm=float(np.linalg.norm(g))
        if it%100==0:trace.append([it,f,norm])
        if norm<1e-9:break
        W-=step*g
    assert norm<1e-9,(it,norm)
    grid=[.5,.75,1.,1.25,1.5,2.,3.,4.]
    def nll(t):
        pr=soft((C@W.T+off)/t);return float(-np.dot(v,np.log(pr[np.arange(len(Z)),Z])))
    ts=[(t,nll(t)) for t in grid];temp=min(ts,key=lambda q:(q[1],abs(q[0]-1),q[0]))[0]
    model=dict(schema='origin67_five_class_ridge',dvalues=list(DVAL),offset=off.tolist(),coef=W.tolist(),temperature=temp,lambda_ridge=lam,prior=prior.tolist(),train_pairs=len(tr),cal_pairs=len(ca),dev_lock=sha(T/'DEV_LOCK.json'),atlas_lock=sha(P/'DEV_LOCK.json'),plan_sha256=sha(D/'PLAN.json'),feature_sha256=sha(D/'retention67.py'))
    (D/'MODEL.json').write_text(json.dumps(model,indent=2,sort_keys=True)+'\n')
    metrics={}
    for split,rr in [('TRAIN',tr),('CAL',ca)]:
        for dom in ('ALL','MAIN','ARCHIVE','NOVEL'):
            r0=[r for r in rr if dom=='ALL' or r['domain']==dom]
            if not r0:continue
            for op in ('ALL','VOTE','FOCUS','EXPAND'):
                rr0=[r for r in r0 if op=='ALL' or r['op']==op]
                if not rr0:continue
                xx,yy,ww=data(rr0);pr=soft((xx@W.T+off)/temp);gain=pr@np.array(DVAL);real=(yy-2)/2;sw=gain>0
                for r,p0 in zip(rr0,pr):assert np.max(np.abs(p0-np.array(predict(model,r['x']))))<1e-12
                metrics[split,dom,op]=dict(pairs=len(rr0),families=len({tuple(r['family']) for r in rr0}),label_counts=np.bincount(yy,minlength=5).tolist(),
                  nll=float(-np.dot(ww,np.log(pr[np.arange(len(yy)),yy]))),nll_constant=float(-np.dot(ww,np.log(prior[yy]))),
                  brier=float(np.dot(ww,np.sum((pr-np.eye(5)[yy])**2,axis=1))),brier_constant=float(np.dot(ww,np.sum((prior-np.eye(5)[yy])**2,axis=1))),
                  predicted_mean=float(np.dot(ww,gain)),observed_mean=float(np.dot(ww,real)),switch_fraction=float(np.dot(ww,sw)),switch_count=int(sw.sum()),help=int(np.sum(sw&(real>0))),harm=int(np.sum(sw&(real<0))),tie=int(np.sum(sw&(real==0))),
                  retained_help=int(np.sum(~sw&(real>0))),retained_harm=int(np.sum(~sw&(real<0))),signed_switch_gain=float(np.dot(ww,sw*real)),always_switch_gain=float(np.dot(ww,real)),oracle_gain=float(np.dot(ww,np.maximum(real,0))))
    out=dict(roots={':'.join(k):v for k,v in roots.items()},identical_proposals={':'.join(k):v for k,v in nochange.items()},excluded=excluded,train_labels=np.bincount(Y,minlength=5).tolist(),cal_labels=np.bincount(Z,minlength=5).tolist(),train_families=len({tuple(r['family']) for r in tr}),cal_families=len({tuple(r['family']) for r in ca}),gradient_check=[fd,anal],iterations=it,final_gradient=norm,objective=f,trace=trace,temperature_nll=ts,temperature=temp,metrics={':'.join(k):v for k,v in metrics.items()},model_sha256=sha(D/'MODEL.json'),prior=prior.tolist())
    (T/'FIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    with gzip.open(T/'DATA.json.gz','wt',compresslevel=6) as fh:json.dump(rows,fh,allow_nan=False,separators=(',',':'))
    print('FIT',len(tr),len(ca),'excluded',len(excluded),'labels',out['train_labels'],out['cal_labels'],'T',temp,'grad',norm,flush=True)
    for k,z in out['metrics'].items():
        if k.startswith('CAL:'):print(k,z,flush=True)
if __name__=='__main__':main()
