"""Post-primary bounded comparator repair only; frozen live policy unchanged."""
from acquire import *
EXT=ROOT.parent/'quartz_kinetic20_ext_target_20261126/mcts_demo'
def main():
    identity0=read('RUST_EXT_BUILD_IDENTITY.json');assert sha(EXT)==identity0['binary_sha256']
    f=read('FROZEN.json');m=f['controller'];lk=read('ATLAS_LOCK.json')
    assert lk['sha256']==sha(D/'ATLAS.jsonl.gz') and lk['frozen']==sha(D/'FROZEN.json')
    path=D/'EXT_MATCH.jsonl.gz';assert not path.exists();rows=[];nn=vis=0
    ll={x['root']['id']:x for x in read('TEST_LABELS.json')['rows'] if x['status']=='EXACT'}
    with Server(EXT) as s,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in raw(D/'ATLAS.jsonl.gz'):
            h=r['root']['history'];a=s.request(h,'K20LIVE',controller=m);b=s.request(h,'K20N64')
            for now,prev in [(a,r['live']),(b,r['normal'])]:
                assert 'error' not in now and now['prefix']==prev['prefix'] and now['result']==prev['result'] and now['best']==prev['best'] and now['NN_calls']==prev['NN_calls'] and now['decision']['channel']==prev['decision']['channel']
            x=s.request(h,'K20MATCHX',nn_budget=a['NN_calls'])
            assert 'error' not in x,x
            assert x['prefix']==a['prefix'] and x['best'] in r['root']['legal'] and x['visits']<=256
            assert x['NN_calls']<=a['NN_calls'] and (x['NN_calls']==a['NN_calls'] or x['visits']==256)
            if r['match']['NN_calls']==a['NN_calls']:assert r['match']['result']==x['result'] and r['match']['best']==x['best']
            emit(fd,dict(root=r['root'],live=a,normal=b,matchx=x))
            for z in (a,b,x):nn+=z['NN_calls'];vis+=z['visits']
            u={int(k):(v+1)/2 for k,v in ll[r['root']['id']]['independent'].items()}
            rows.append(dict(id=r['root']['id'],gain=u[a['best']]-u[x['best']],live_regret=max(u.values())-u[a['best']],match_regret=max(u.values())-u[x['best']],live_NN=a['NN_calls'],match_NN=x['NN_calls'],live_visits=a['visits'],match_visits=x['visits'],live_cpu=a['cpu_ns']/1e6,match_cpu=x['cpu_ns']/1e6,live_wall=a['wall_ns']/1e6,match_wall=x['wall_ns']/1e6))
    def summ(a):
        if not a:return dict(n=0)
        return dict(n=len(a),**{k:S.mean(x[k] for x in a) for k in rows[0] if k!='id'},positive=sum(x['gain']>0 for x in a),negative=sum(x['gain']<0 for x in a))
    exact=[x for x in rows if x['live_NN']==x['match_NN']]
    out=dict(identity=identity0['binary_sha256'],atlas=lk['sha256'],frozen=lk['frozen'],sha256=sha(path),requests=len(rows)*3,visits=vis,NN=nn,all=summ(rows),matched=summ(exact),unmatched=summ([x for x in rows if x not in exact]),rows=rows)
    save('EXT_SUMMARY.json',out)
    print('EXT',json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
    # Explicit malformed request and exact proof certificate on tactical root, all comparators.
    with Server(EXT) as s:
        bad=s.request('121212','K20LIVE')
        assert 'error' in bad
        bad2=s.request('121212','K20MATCHX',nn_budget=0)
        assert 'error' in bad2
        for arm,opts in [('K20LIVE',dict(controller=m)),('K20N64',{}),('K20MATCHX',dict(nn_budget=30)),('K20TIME',dict(tail_budget_us=300))]:
            x=s.request('121212',arm,**opts);assert 'error' not in x,x
            assert x['decision']['channel']=='CERTIFIED' and x['best']==0 and x['visits']==8 and [0,1.,1.] in x['decision']['proof']
    print('malformed controller/budget rejected; exact immediate-win proof all 4 arms PASS',flush=True)
if __name__=='__main__':main()
