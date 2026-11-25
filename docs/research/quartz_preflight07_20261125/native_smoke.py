"""Bounded fixture requests only; no full corpus generation."""
import json,math,struct
from pathlib import Path
from rules import *
from engine import Engine,D,identity
def check(a,h,rule,inspect=False,deep=False):
    b,p,w=replay(h,rule)
    assert a['history']==h and a['rule']==rule and a['board']==b and a['player']==p
    assert a['rule_key']==rule_key(b,p,rule) and a['recent']==h[-7:]
    assert a['terminal']==(w is not None) and a['outcome']==(None if w is None else w*p)
    legal=[] if w is not None else [i for i,v in enumerate(b) if not v]
    assert a['legal']==legal
    if inspect:assert a['features']==features(h)
    if w is not None:assert a['proof'] is None;return
    pr=proof_reference(b,p,rule);assert a['proof']==pr
    if deep:
        # Independent direct child and reply rule checks, rather than merely
        # trusting the opponent-threat removal identity used in the fast scan.
        for m,l,u,rep in pr['actions']:
            b0,p0,w0=replay(h+[m],rule)
            if w0 is not None:assert l==u==w0*p and rep is None
            elif rep is not None:
                b1,p1,w1=replay(h+[m,rep],rule);assert w1==-p and l==u==-1
            else:
                assert (l,u)==(-1,1)
                assert not any(win_at(b0,j,1 if p0==1 else 2,rule) for j,x in enumerate(b0) if not x)
    if inspect:return
    assert a['best'] in legal and a['visits']<=a['requested_visits']
    if a['cause']=='CERTIFIED':
        assert a['arm']=='CERT' and a['best']==pr['witness'] and a['result'] is None and a['visits']==a['evaluator_calls']==a['nn_rows']==0
    else:
        v=a['result'];assert a['cause']=='VISITS' and a['visits']==a['requested_visits']==v['root_n'] and v['root_hash']==a['tt_hash']
        assert v['best'] in legal and {m for m,_ in v['candidates']}==set(legal)
        assert [e['move'] for e in v['edges']]==[m for m,_ in v['candidates'][:len(v['edges'])]]
        assert sum(e['n'] for e in v['edges'])==v['root_n']
        assert abs(sum(e['w'] for e in v['edges'])-v['root_w'])<1e-5
        assert all(e['vl']==0 and abs(e['vvalue'])<1e-6 and abs(e['w'])<=e['n']+1e-6 for e in v['edges'])
        assert a['nn_rows']==v['evaluator_calls']==a['evaluator_calls']>=1 and 1<=a['nn_batches']<=a['nn_rows']
        assert a['root_nn_feature_sha256']==hashlib.sha256(struct.pack('<'+'f'*3825,*features(h))).hexdigest()
    assert a['whole_us']>=a['search_us']+a['drop_us']+a['setup_us']
def main():
    f=fixtures();rows=[];bad=[];search=[]
    with Engine(2) as e:
        for name in ('immediate','overline','double_loss','trans_a','trans_b','long','draw_pre','draw'):
            for rule in ('freestyle','standard'):
                h=f[name];a=e.inspect(h,rule);assert 'error' not in a,a
                check(a,h,rule,True,deep=name in ('immediate','overline','double_loss','draw_pre'))
                a['fixture']=name;rows.append(a)
        get=lambda nm,rule:next(a for a in rows if a['fixture']==nm and a['rule']==rule)
        x=get('trans_a','freestyle');y=get('trans_b','freestyle')
        assert x['rule_key']==y['rule_key'] and x['board_hash']==y['board_hash'] and x['tt_hash']!=y['tt_hash'] and x['features']!=y['features']
        assert 109 in get('overline','freestyle')['proof']['own'] and 109 not in get('overline','standard')['proof']['own']
        assert get('draw_pre','freestyle')['proof']['interval']==[0,0] and get('draw','freestyle')['outcome']==0
        for rule in ('omok','renju','caro'):
            a=e.inspect(f['forbidden'],rule);assert 'error' not in a,a
            assert a['proof'] is None and (112 in a['legal'])==(rule=='caro')
            a['fixture']='forbidden';rows.append(a)
            q=dict(cmd='research_gomoku15_long_search',history=f['forbidden'],rule=rule,arm='BASE',visits=8,threads=1,seed=17)
            a=e.request(q);assert 'error' in a;bad.append(dict(request=q,response=a))
        for q in [
            dict(cmd='research_gomoku15_long_inspect',rule='freestyle',history=[0,0]),
            dict(cmd='research_gomoku15_long_inspect',rule='freestyle',history=[225]),
            dict(cmd='research_gomoku15_long_inspect',rule='freestyle',history=f['immediate']+[109,1]),
            dict(cmd='research_gomoku15_long_inspect',rule='freestyle',history=[],oracle=[1]),
            dict(cmd='research_gomoku15_long_inspect',rule='unknown',history=[])]:
            a=e.request(q);assert 'error' in a;bad.append(dict(request=q,response=a))
        for nm,rule,arm,threads in [
            ('immediate','freestyle','BASE',1),('immediate','freestyle','CERT',1),
            ('trans_a','freestyle','BASE',2),('trans_b','standard','BASE',1),
            ('long','freestyle','BASE',2),('long','freestyle','CERT',2)]:
            a=e.search(f[nm],rule,arm,8,threads,17);check(a,f[nm],rule)
            a['fixture']=nm;search.append(a)
            print('NN_SMOKE',nm,rule,arm,'visits',a['visits'],'NN',a['nn_rows'],'time_ms',a['transport_ns']/1e6,flush=True)
        params=e.model_parameters;cfg={k:e.cfg[k] for k in ('filters','blocks','vh','ch')}
    assert len(search)==6
    o=dict(identity=identity(),model_parameters=params,model_config=cfg,inspect=rows,search=search,rejected=bad)
    (D/'C06_NATIVE_SMOKE.json').write_text(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print('Native smoke',len(rows),len(search),len(bad),'parameters',params,flush=True)
if __name__=='__main__':main()
