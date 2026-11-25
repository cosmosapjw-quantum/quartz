"""Independent grid, full graph and frontier replay, signed decision gain and durable restore."""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import collections as C,gzip,json,math,statistics as S,hashlib,subprocess as SP,tempfile
from publish import ROOT,D,B,BASE,BR,sha,git
from population import exclusions
import experiment as e
import native as n
LINES=[tuple((y+j*dy)*7+x+j*dx for j in range(4)) for y in range(6) for x in range(7) for dx,dy in ((1,0),(0,1),(1,1),(-1,1)) if 0<=x+3*dx<7 and 0<=y+3*dy<6]
def read(f):return json.loads((D/f).read_text())
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-7),(a,b)
def win(b,p):return any(all(b[k]==p for k in line) for line in LINES)
def replay(h):
    b=[0]*42;z=[0]*7
    for i,x in enumerate(h):
        c=int(x)-1;p=1-2*(i%2)
        assert 0<=c<7 and z[c]<6 and not win(b,p) and not win(b,-p)
        b[7*z[c]+c]=p;z[c]+=1
    return b,z
def immediate(b,z,p):
    return any(win(b[:7*z[c]+c]+[p]+b[7*z[c]+c+1:],p) for c in range(7) if z[c]<6)
def canon(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def cb(b):return min(tuple(b),tuple(b[7*y+x] for y in range(6) for x in range(6,-1,-1)))
def u(x):return F(1+(x>0)-(x<0),2)
def encode(b,p):
    return '%022x'%((int(p==1)<<84)|sum((v+1)<<(2*i) for i,v in enumerate(b)))
def decode(k):
    n=int(k,16);assert n>>85==0
    b=[((n>>(2*i))&3)-1 for i in range(42)];p=1 if n>>84 else -1
    assert all(v in (-1,0,1) for v in b)
    z=[]
    for c in range(7):
        col=[b[y*7+c] for y in range(6)];h=sum(x!=0 for x in col)
        assert all(v!=0 for v in col[:h]) and all(v==0 for v in col[h:]);z.append(h)
    assert b.count(1)-b.count(-1)==(p==-1)
    return b,p,z
def outcome(b,p):
    w1,w2=win(b,1),win(b,-1);assert not(w1 and w2)
    if w1:return p
    if w2:return -p
    if all(v!=0 for v in b):return 0
    return None
def fnv(b,p):
    z=0xcbf29ce484222325
    for v in b+[p]:z=((z^(v+2))*0x100000001b3)&((1<<64)-1)
    return '%016x'%z
def rb(a,move):
    iv=dict(a)
    return F(max(0,max((u for j,(_,u) in iv.items() if j!=move),default=-1)-iv[move][0]),2)
def graph_check(run):
    g=run['proof'];nodes={v['key']:v for v in g['graph']};assert len(nodes)==g['nodes']
    b,z=replay(run['history']);p=1-2*(len(run['history'])%2);root=encode(b,p);assert root==g['key']
    ordered=sorted(nodes,key=lambda k:-sum(x!=0 for x in decode(k)[0]))
    ref={};arcs=0;terminal_count=0;shared=C.Counter()
    for k in ordered:
        v=nodes[k];b,p,z=decode(k);term=outcome(b,p)
        if v['terminal'] is not None:
            assert v['terminal']==term and v['actions'] is None;iv=(term,term);terminal_count+=1
        elif v['actions'] is None:
            assert term is None;iv=(-1,1)
        else:
            assert term is None and len(v['actions'])==len({m for m,_ in v['actions']})
            assert {m for m,_ in v['actions']}=={i for i in range(7) if z[i]<6}
            aa=[]
            for m,c in v['actions']:
                cb0=b.copy();cb0[z[m]*7+m]=p
                assert c==encode(cb0,-p) and c in ref
                l,u0=ref[c];aa.append((-u0,-l));arcs+=1;shared[c]+=1
            iv=(max(x[0] for x in aa),max(x[1] for x in aa))
        assert list(iv)==v['interval'];ref[k]=iv
    reachable={root};todo=[root]
    while todo:
        k=todo.pop()
        for _,c in nodes[k]['actions'] or []:
            if c not in reachable:reachable.add(c);todo.append(c)
    assert reachable==set(nodes)
    assert g['interval']==list(ref[root])
    aa=[[m,[-ref[c][1],-ref[c][0]]] for m,c in nodes[root]['actions']]
    assert aa==g['actions']
    assert g['solved']==sum(l==u0 for l,u0 in ref.values())
    assert g['registered']==sum(v['terminal'] is not None or v['actions'] is not None for v in nodes.values())
    return dict(nodes=len(nodes),arcs=arcs,terminal=terminal_count,shared_nodes=sum(v>1 for v in shared.values()),solved=g['solved'])
def rw(a):
    return (min(2*rb(a,m) for m,_ in a),sum(u0-l for _,(l,u0) in a))
def frontier_reference(g):
    """Full topological recomputation for each single revelation, no native overlay reuse."""
    nodes={n['key']:n for n in g['graph']};root=g['key']
    order=sorted(nodes,key=lambda k:-sum(x!=0 for x in decode(k)[0]))
    base={k:tuple(n['interval']) for k,n in nodes.items()}
    if g['witness'] is not None:return None,0,0
    aa=lambda k,iv:[(m,(-iv[c][1],-iv[c][0])) for m,c in nodes[k]['actions']]
    d,w=rw(aa(root,base));seen={root};todo=C.deque([(root,[])]);best=None;scored=0;hyp=0
    while todo:
        k,path=todo.popleft();lo,hi=base[k]
        if lo==hi:continue
        if nodes[k]['actions'] is None:
            scored+=1;gs=[]
            for x in range(lo,hi+1):
                # Recompute every registered node in reverse legal depth.
                iv=dict(base);iv[k]=(x,x)
                for j in order:
                    if nodes[j]['actions'] is not None:
                        a=aa(j,iv);iv[j]=(max(l for _,(l,_) in a),max(u0 for _,(_,u0) in a))
                dd,ww=rw(aa(root,iv));assert dd<=d and ww<=w;gs.append([x,int(d-dd),w-ww]);hyp+=1
            v=max((x[1],x[2]) for x in gs)
            if v>(0,0) and (best is None or v>tuple(best['score']) or v==tuple(best['score']) and len(path)<len(best['path'])):
                best=dict(key=k,path=path,gains=gs,score=list(v))
            if scored>=64:break
            continue
        a=aa(k,base);lo0=max(l for _,(l,_) in a)
        for m,c in nodes[k]['actions']:
            l,h=base[c]
            if -l>=lo0 and l<h and c not in seen:
                seen.add(c);todo.append((c,path+[[k,m,c]]))
    return best,scored,hyp
def margin_ref(pre,post):
    a=pre['best'];b=post['best'];g=post['proof']
    if a==b or g is None or g['witness'] is not None or a not in g['allowed']:return None
    d={x['move']:x['n'] for x in post['result']['edges']}
    if d.get(a,0)==0 or d.get(b,0)==0:return None
    return F(d[b]-d[a],post['result']['root_n'])
def check_plan(a,pre,post,plan):
    arm=a['arm'];h0=pre['hooks'];h=post['hooks'];d=lambda k:h[k]-h0[k]
    if arm not in ('FOCUS','EXPAND','FRONTIER'):
        assert plan is None and d('local_claims')==0;return
    assert plan['mode']==arm and d('local_claims')==2
    if arm=='FOCUS':
        g=pre['proof'];aa=dict(g['actions']);allowed=set(g['allowed'])
        es=[(i,x) for i,x in enumerate(pre['result']['edges']) if x['n']>0 and x['move'] in allowed];es.sort(key=lambda p:(-p[1]['n'],p[0]))
        leaders=[[v['move'],v['n'],v['w']/v['n']] for _,v in es[:2]]
        assert len(plan['leaders'])==len(leaders)
        for a0,b0 in zip(plan['leaders'],leaders):assert a0[:2]==b0[:2];close(a0[2],b0[2])
        if len(leaders)==2:
            gap=abs(leaders[0][2]-leaders[1][2]);close(plan['gap'],gap)
            j=leaders[1][0];wanted=j if gap<=.25 and aa[j][0]<aa[j][1] else None
        else:assert plan['gap'] is None;wanted=None
        assert plan['target']==wanted and 0<=d('local_focus_edges')<=2 and (wanted is not None or d('local_focus_edges')==0)
    elif arm=='EXPAND':assert 0<=d('local_expand_edges')<=2
    else:
        assert 0<=plan['scored']<=64 and d('local_scored')==plan['scored']
        t=plan['target'];g=pre['proof']
        if t is None:assert d('local_frontier_edges')==d('local_frontier_hits')==0
        else:
            assert t['path'][0][0]==g['key'] and t['path'][-1][2]==t['key'] and len(t['path'])>=1
            for i,(k,m,c) in enumerate(t['path']):
                if i:assert k==t['path'][i-1][2]
                b,p,z=decode(k);assert outcome(b,p) is None and 0<=m<7 and z[m]<6
                b[z[m]*7+m]=p;assert encode(b,-p)==c
            assert [x[0] for x in t['gains']]==[-1,0,1]
            rd,wd=rw(g['actions'])
            assert all(0<=x<=rd and 0<=y<=wd for _,x,y in t['gains'])
            assert tuple(t['score'])==max((x,y) for _,x,y in t['gains']) and tuple(t['score'])>(0,0)
            assert d('local_frontier_edges')<=2*len(t['path']) and d('local_frontier_hits')<=8
def audit():
    e.lock()
    co=read('C07_COHORT.json');sm=e.read_gz('C06_SMOKE.json.gz');tim=e.read_gz('C07_TIMING_RAW.json.gz');cost=read('C07_COST_LOCK.json');guard=read('C07_RETENTION_LOCK.json');su=read('C08_SUMMARY.json');host=read('C08_LIVE_HOST.json')
    assert sha(D/'C07_ATTEMPTS.jsonl.gz')==co['attempt_sha'] and sha(D/'C08_LIVE_RAW.jsonl.gz')==su['raw_sha']==host['raw_sha']
    assert cost['cohort_sha']==sha(D/'C07_COHORT.json') and cost['diag_sha']==sha(D/'C07_DIAG_RAW.jsonl.gz')==guard['diag_sha'] and cost['timing_sha']==sha(D/'C07_TIMING_RAW.json.gz') and cost['binary_sha']==sha(e.BIN)
    assert guard['labels_sha']==sha(D/'C07_DIAG_LABELS.json')
    oldp,oldb,meta=exclusions();assert len(oldp)==co['old_families'] and len(oldb)==co['old_boards'] and meta==co['sources']
    ids={r['id']:r for r in co['rows']};seen=set();families=set();boards=set();accepted=set();events=0;counts=C.Counter();attempts=C.Counter()
    with gzip.open(D/'C07_ATTEMPTS.jsonl.gz','rt') as fp:
        for line in fp:
            a=json.loads(line);i=a['id'];h=a['history'];op=a['opening'];f=canon(op)
            assert i not in seen and f not in oldp|families and h.startswith(op);seen.add(i);families.add(f);attempts[a['stratum']+'/'+a['group']]+=1
            if a['stratum']=='SELFPLAY':
                hist=op
                for ev in a['events']:
                    b,z=replay(hist);assert not win(b,1) and not win(b,-1)
                    assert ev['ply']==len(hist) and all(0<=c<7 and z[c]<6 for c in (ev['action'],ev['recommendation']))
                    assert ev['explore'] or ev['action']==ev['recommendation'];assert ev['eval_calls']>=1
                    hist+=str(ev['action']+1);events+=1
                assert hist==h
            else:
                b,z=replay(h);assert not win(b,1) and not win(b,-1)
            if a['candidate'] is None:continue
            r=ids[i];b,z=replay(h);p=1-2*(len(h)%2)
            assert a['status']=='ACCEPT' and a['candidate']==h and i not in accepted and not win(b,p) and not win(b,-p);accepted.add(i)
            assert r['history']==h and r['board']==b and r['legal']==[j for j in range(7) if z[j]<6] and len(r['legal'])>=2
            assert r['family']==f and r['player']==p and r['ply']==len(h) and r['stratum']==a['stratum'] and len(h)%2==a['parity'] and r['group']==a['group']
            assert r['immediate']==immediate(b,z,p)
            if r['stratum']=='TACTICAL':assert 18<=len(h)<=25 and not r['immediate']
            elif r['stratum']=='HARD':assert 20<=len(h)<=25 and len(r['legal'])>=4 and not r['immediate'] and not immediate(b,z,-p)
            else:assert 16<=len(h)<=26
            k=cb(b);assert k not in oldb|boards;boards.add(k);counts[r['stratum']+'/'+r['group']+'/'+str(p)]+=1
    assert accepted==set(ids) and len(seen)==co['attempts'] and len(ids)==132
    assert all(counts[st+'/'+g+'/'+str(p)]==({'TRAIN':6,'CAL':4,'TEST':12}[g]) for st in e.ST for g in ('TRAIN','CAL','TEST') for p in (-1,1))
    assert all(a['standard']['result']==a['research']['result'] for a in sm['parity'])
    for st in e.ST:
        rr=[a for a in tim['rows'] if a['stratum']==st];v=cost['cost'][st]
        assert len(rr)==24 and all(ids[a['id']]['group']=='TRAIN' and a['arm']=='BASE' and a['mode']=='WORK' for a in rr)
        med=S.median(a['whole_us'] for a in rr);drop=S.median(a['drop_us'] for a in rr)
        assert v['median_whole_us']==med and v['median_drop_us']==drop and v['target']==min(20000,max(3000,math.ceil(med))) and v['reserve']==min(v['target']-1,math.ceil(drop)+150)
    tr={i:r for i,r in ids.items() if r['group']!='TEST'};test={i:r for i,r in ids.items() if r['group']=='TEST'}
    dl={r['id']:r['scores'] for r in read('C07_DIAG_LABELS.json') if r['status']=='EXACT'}
    labs={r['id']:r['scores'] for r in read('C08_LABELS.json') if r['status']=='EXACT'};assert set(dl)==set(tr) and set(labs)==set(test)
    for z,rr in ((dl,tr),(labs,test)):
        for i,v in z.items():assert len(v)==7 and all((x!=-1000)==(j in rr[i]['legal']) for j,x in enumerate(v))
    mirrors=0
    for st in e.ST:
        for r in sorted((x for x in test.values() if x['stratum']==st),key=lambda x:x['id'])[:3]:
            h=r['history'].translate(str.maketrans('1234567','7654321'))
            z=SP.run([str(n.EXT/'c4solver'),'-a','-b',str(n.EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=n.EXT,timeout=15,check=True).stdout.split()
            assert z[0]==h and list(map(int,z[1:]))==labs[r['id']][::-1];mirrors+=1
    # Full unique diagnostic prefix graphs and every available first frontier plan.
    graph_tot=C.Counter();dfirst=0;hyp=0;fscored=0;diag_prefix={};diag_grid=set();diag_pref_n=0;fit=C.defaultdict(list);diag_metrics=C.defaultdict(list)
    for a in e.records('C07_DIAG_RAW.jsonl.gz'):
        i=a['id'];r=tr[i];arm=a['arm'];assert a['history']==r['history'] and a['group']==r['group'] and a['stratum']==r['stratum'] and a['mode']=='DIAG' and a['seed']==17
        assert (i,arm) not in diag_grid;diag_grid.add((i,arm))
        p=a['prefix'];k=(i,arm=='BASE')
        if k not in diag_prefix:
            diag_prefix[k]=(p['result'],p['proof'])
            if arm!='BASE':graph_tot.update(graph_check(dict(history=a['history'],proof=p['proof'])))
        else:assert diag_prefix[k]==(p['result'],p['proof']);diag_pref_n+=1
        if arm=='FRONTIER' and a['steps']:
            b,n0,nh=frontier_reference(p['proof']);plan=a['steps'][0]['plan'];assert plan['target']==b and plan['scored']==n0
            dfirst+=1;hyp+=nh;fscored+=n0
        post=a['steps'][-1]['post'] if a['steps'] else p
        m=margin_ref(p,post)
        if m is None:assert a['retention_margin'] is None
        else:close(m,a['retention_margin'])
        assert a['guard_best']==a['best']
        z=dl[i];opt=max(u(x) for x in z if x!=-1000);gain=u(z[a['best']])-u(z[p['best']])
        diag_metrics[(r['group'],arm)].append((opt-u(z[a['best']]),gain))
        if arm!='BASE':fit[(r['group'],i)].append((m,-gain))
    assert len(diag_grid)==540 and len(diag_prefix)==120
    # Recompute TRAIN/CAL retention choice with rational margins and family means.
    grid={}
    for t in (F(1,20),F(1,10),F(1,5)):
        gs={}
        for g in ('TRAIN','CAL'):
            fs=[v for (gg,_),v in fit.items() if gg==g];ex=[v for v in fs if any(m is not None and m<=t for m,_ in v)]
            d=[v for f in fs for m,v in f if m is not None and m<=t]
            mean=sum(sum(v if m is not None and m<=t else 0 for m,v in f)/len(f) for f in fs)/len(fs)
            a=guard['grid'][str(float(t))][g];assert len(fs)==a['families'] and len(ex)==a['exposed_families'] and len(d)==a['exposed_pairs'] and sum(x>0 for x in d)==a['better'] and sum(x<0 for x in d)==a['worse'];close(mean,a['mean'])
            gs[g]=(len(ex),mean)
        grid[t]=gs
    ok=[t for t,g in grid.items() if g['TRAIN'][0]>=6 and g['TRAIN'][1]>0]
    c=max(ok,key=lambda t:(grid[t]['TRAIN'][1],-t)) if ok else F(0)
    t=c if c and grid[c]['CAL'][0]>=4 and grid[c]['CAL'][1]>=0 else F(0)
    close(c,guard['train_candidate']);close(t,guard['threshold'])
    del diag_prefix
    # Independent reconstruction of every actual TEST block, exact WDL and plan contract.
    results={};prefix={};pref_n=0;cert=0;action_n=0;stops=0;blocks=0;snaps=0;host_ok_requests=0
    expose=C.Counter();worker=C.Counter();hist=C.Counter();plan_out=C.defaultdict(list);per_request_swap=C.Counter();plan_contrib=C.Counter()
    for a in e.records('C08_LIVE_RAW.jsonl.gz'):
        i=a['id'];r=test[i];s=a['seed'];mo=a['mode'];arm=a['arm'];st=r['stratum'];assert (i,s,mo,arm) not in results
        assert a['history']==r['history'] and a['stratum']==st and a['model_identity']==n.IDENT and a['retention']==float(t)
        p=a['prefix'];ss=[p]+[x['post'] for x in a['steps']];f=ss[-1];blocks+=len(a['steps']);snaps+=len(ss)
        k=(i,s,arm=='BASE')
        if k in prefix:assert prefix[k]==(p['result'],p['proof']);pref_n+=1
        else:prefix[k]=(p['result'],p['proof'])
        z=labs[i];opt=max(u(j) for j in z if j!=-1000);vals=[];risks=[];widths=[]
        for j,x in enumerate(ss):
            v=x['result'];h=x['hooks'];g=x['proof']
            assert v['root_n']==32+8*j and v['root_hash']==fnv(r['board'],r['player'])
            assert {m for m,_ in v['candidates']}==set(r['legal']) and x['best'] in r['legal'] and v['best'] in r['legal']
            assert [j0['move'] for j0 in v['edges']]==[j0[0] for j0 in v['candidates'][:len(v['edges'])]]
            assert sum(j0['n'] for j0 in v['edges'])==v['root_n'];close(sum(j0['w'] for j0 in v['edges']),v['root_w'])
            for y in v['edges']:assert y['vl']==0 and abs(y['vvalue'])<1e-6 and abs(y['w'])<=y['n']+1e-6
            assert h['base_evals']+h['terminal_leaves']+h['graph_solved_skips']==v['root_n'] and v['evaluator_calls']==1+h['base_evals']+h['extra_evals']
            assert h['graph_repeat_skips']<=h['graph_solved_skips']
            val=u(z[x['best']]);vals.append(val)
            if arm=='BASE':assert g is None and x['best']==v['best']
            else:
                assert g['key']==encode(r['board'],r['player']);iv=dict(g['actions']);assert set(iv)==set(r['legal']) and g['best']==x['best']
                assert g['interval']==[max(l for l,_ in iv.values()),max(h0 for _,h0 in iv.values())]
                assert g['allowed']==[m for m,(l,h0) in g['actions'] if h0>=g['interval'][0]] and x['best'] in g['allowed']
                for m,(l,h0) in g['actions']:assert -1<=l<=(z[m]>0)-(z[m]<0)<=h0<=1;action_n+=1
                assert opt-val<=rb(g['actions'],x['best'])
                rd,wd=rw(g['actions']);risks.append(rd/2);widths.append(wd);w=g['witness']
                assert (w is not None)==(rd==0)
                if w is not None:assert val==opt and x['best']==w;cert+=1
                if j:
                    prev=ss[j-1]['proof'];assert prev['witness'] is None and all(dict(prev['actions'])[m][0]<=l<=h0<=dict(prev['actions'])[m][1] for m,(l,h0) in g['actions'])
            if j:
                step=a['steps'][j-1];prev=ss[j-1];check_plan(a,prev,x,step['plan'])
                # Distinguish proof reduction, recommendation flip and actual decision gain.
                delta=vals[-1]-vals[-2]
                flip=prev['best']!=x['best']
                assert flip or delta==0
                if arm=='FRONTIER' and step['plan']['target'] is not None:
                    tt=step['plan']['target'];sc=tt['score'][0];kind='positive_risk' if sc>0 else 'width_only'
                    plan_out[(st,mo,kind)].append((delta,risks[-2]-risks[-1],widths[-2]-widths[-1],step['plan_us']))
                    plan_contrib[(mo,kind,'blocks')]+=1
                    plan_contrib[(mo,kind,'risk_reduced')]+=risks[-1]<risks[-2]
                    plan_contrib[(mo,kind,'width_reduced')]+=widths[-1]<widths[-2]
        gain=vals[-1]-vals[0];diffs=[b-a0 for a0,b in zip(vals,vals[1:])];assert sum(diffs)==gain
        if arm!='BASE':assert risks[-1]<=risks[0] and widths[-1]<=widths[0]
        if a['cause']=='CERTIFIED':assert arm!='BASE' and f['proof']['witness'] is not None and vals[-1]==opt;stops+=1
        elif mo=='WORK':assert f['result']['root_n']==128 and a['cause']=='VISITS'
        elif mo=='NN':assert a['cause']=='NN' and f['result']['evaluator_calls']>=64 or a['cause']=='CAP' and f['result']['root_n']==256
        else:
            cfg=cost['cost'][st];assert a['target_us']==cfg['target'] and a['reserve_us']==cfg['reserve']
            assert a['cause']=='TIME' and a['before_drop_us']>=cfg['target']-cfg['reserve'] or a['cause']=='CAP' and f['result']['root_n']==256
        m=margin_ref(p,f)
        if m is None:assert a['retention_margin'] is None
        else:close(a['retention_margin'],m)
        gb=p['best'] if t and m is not None and m<=t else a['best'];assert gb==a['guard_best'];h=f['hooks'];h0=p['hooks']
        hc={k:h[k]-h0[k] for k in h if isinstance(h[k],int) and k!='worker_mask'}
        for key0 in ('local_focus_edges','local_expand_edges','local_frontier_edges','local_frontier_hits','local_scored','graph_repeat_skips','graph_solved_skips','graph_b_skip','merge_blends','extra_evals'):expose[(mo,arm,key0)]+=hc[key0]
        if arm=='WRITER':assert h['ledger_observations']==h['merge_blends']==0
        elif arm=='MERGE':assert h['ledger_observations']==h['backup_edges'] and h['temporal_blends']==0
        else:assert h['backup_edges']==h['ledger_observations']==h['merge_blends']==0
        if a['steps']:worker[(mo,arm,'roots')]+=1;worker[(mo,arm,'two_total')]+=h['worker_mask'].bit_count()>=2
        hist[(st,mo,arm,a['cause'])]+=1
        hh=a['host_interval'];host_ok_requests+=hh['ok'];assert hh['ok']==(hh['swap_pages_s']<=512 and hh['max_endpoint_load']<=12)
        for k,x in hh['swap_delta'].items():per_request_swap[k]+=x
        results[i,s,mo,arm]=dict(regret=opt-vals[-1],prefix_regret=opt-vals[0],gain=gain,guard_regret=opt-u(z[gb]),guard_gain=u(z[gb])-vals[-1],flips=sum(x['best']!=y['best'] for x,y in zip(ss,ss[1:])),better=sum(x>0 for x in diffs),worse=sum(x<0 for x in diffs),neutral_flip=sum(x['best']!=y['best'] and d==0 for x,y,d in zip(ss,ss[1:],diffs)),final_flip=int(p['best']!=f['best']),final_better=int(gain>0),final_worse=int(gain<0),whole_us=a['whole_us'],transport_us=a['transport_us'],plan_us=sum(x['plan_us'] for x in a['steps']),evals=f['result']['evaluator_calls'],optional_evals=f['result']['evaluator_calls']-p['result']['evaluator_calls'],visits=f['result']['root_n'],blocks=len(a['steps']),risk_reduction=None if not risks else risks[0]-risks[-1],width_reduction=None if not widths else widths[0]-widths[-1],stop=int(a['cause']=='CERTIFIED'),cap=int(a['cause']=='CAP'),cert_unsolved=int(a['cause']=='CERTIFIED' and f['proof']['interval'][0]!=f['proof']['interval'][1]),hooks=hc)
    assert len(results)==3888 and action_n==su['checked_action_intervals'] and cert==su['certificate_snapshots']
    for st in (*e.ST,'ALL'):
        rr=[i for i,r in test.items() if st=='ALL' or r['stratum']==st]
        for mo in e.MODES:
            g=su['groups'][st][mo]
            for arm,x in g['arms'].items():
                a=[results[i,s,mo,arm] for i in rr for s in e.SEEDS]
                for k in ('flips','better','worse','neutral_flip','final_flip','final_better','final_worse','stop','cap','cert_unsolved'):assert sum(v[k] for v in a)==x[k]
                for k in ('regret','prefix_regret','gain','guard_gain','guard_regret','whole_us','transport_us','plan_us','evals','optional_evals','visits','blocks','risk_reduction','width_reduction'):
                    b=[v[k] for v in a if v[k] is not None]
                    if b:close(sum(b)/len(b),x[k])
                    else:assert x[k] is None
                for k in x['hooks']:assert sum(v['hooks'][k] for v in a)==x['hooks'][k]
            for k,v in g['paired'].items():
                a,b=k.split('/');diff=[sum(results[i,s,mo,b]['regret']-results[i,s,mo,a]['regret'] for s in e.SEEDS)/2 for i in rr];close(sum(diff)/len(diff),v['delta'])
    dt=host['end']['mono']-host['start']['mono'];swap={k:host['end']['swap_pages'][k]-host['start']['swap_pages'][k] for k in host['start']['swap_pages']};close(sum(swap.values())/dt,host['interval']['swap_pages_s'])
    def flatten(x):return {'/'.join(k):float(v) if isinstance(v,F) else v for k,v in x.items()}
    pd={}
    for key0,v in plan_out.items():
        pd['/'.join(key0)]=dict(blocks=len(v),mean_signed_gain=float(sum(x[0] for x in v)/len(v)),mean_risk_reduction=float(sum(x[1] for x in v)/len(v)),mean_width_reduction=sum(x[2] for x in v)/len(v),mean_plan_us=sum(x[3] for x in v)/len(v),better=sum(x[0]>0 for x in v),worse=sum(x[0]<0 for x in v))
    return dict(source_attempts=len(seen),attempt_strata=dict(attempts),roots=len(ids),cohort=dict(counts),selfplay_events=events,diagnostic_requests=len(diag_grid),diag_prefix_groups=120,diag_prefix_equalities=diag_pref_n,full_diagnostic_graphs=60,graph_totals=dict(graph_tot),first_frontier_plans=dfirst,frontier_candidates_recomputed=fscored,hypothetical_values_recomputed=hyp,retention_independently_checked=True,retention_threshold=float(t),test_requests=len(results),test_exact=len(labs),mirror_vectors=mirrors,blocks=blocks,snapshots=snaps,prefix_groups=len(prefix),prefix_equalities=pref_n,root_action_intervals=action_n,valid_certificate_snapshots=cert,actual_stops=stops,worker_counts=flatten(worker),exposure=flatten(expose),termination_counts=flatten(hist),frontier_plan_outcomes=pd,frontier_plan_contrib=flatten(plan_contrib),diag_mean={ '/'.join(k):dict(regret=float(sum(x[0] for x in v)/len(v)),gain=float(sum(x[1] for x in v)/len(v))) for k,v in diag_metrics.items()},host_ok_requests=host_ok_requests,per_request_swap=dict(per_request_swap),swap_pages_delta=swap,host_ok=su['host_ok'],scientific_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();allowed=('src/mcts/mod.rs','src/mcts/select.rs','src/mcts/four_axis.rs','src/mcts/four_axis_math.rs','src/mcts/proof_graph.rs','src/mcts_server/four_axis_local_cpu_server.rs','src/mcts_server/four_axis_pool_plan.rs','src/mcts_server/four_axis_pool_cpu_server.rs','src/mcts_server.rs','src/mcts_server/four_axis_cpu_server.rs')
    assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in allowed for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-fouraxis05-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head and gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for p in files:assert hashlib.sha256(gg(r,'show',f'{head}:{p}')).hexdigest()==sha(ROOT/p),p
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(e.BIN)==read('C06_BUILD_IDENTITY.json')['binary_sha256']
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(clean),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,actual_restore=True,connectivity_fsck=True,binary_sha256=sha(e.BIN))
if __name__=='__main__':
    import sys
    print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2),flush=True)
