"""Descriptive signed deterministic-policy utility, not minimax or game Elo."""
import collections as C,statistics as S,math
from experiment import *
def main():
    raw=list(rows('C07_RAW.jsonl.gz'));cont=list(rows('C08_CONTINUATIONS.jsonl.gz'))
    assert sha(D/'C07_RAW.jsonl.gz')==read('C07_RAW_LOCK.json')['raw_sha256']
    assert sha(D/'C08_CONTINUATIONS.jsonl.gz')==read('C08_CONT_LOCK.json')['raw_sha256']
    u={(r['root']['id'],r['action']):r['utility'] for r in cont};assert len(u)==len(cont)
    p={};rows0=[];choices=C.defaultdict(set)
    for r in raw:
        root=r['root'];a=r['response'];i=root['id'];arm=a['arm'];ph=a['axis']['phases']
        m=[x['result']['best'] for x in ph];v=[u[i,j] for j in m]
        choices[i].update(m)
        margins=[];shares=[]
        for x in ph:
            visits=sorted((e['n'] for e in x['result']['edges']),reverse=True)
            margins.append(visits[0]-(visits[1] if len(visits)>1 else 0));shares.append(visits[0]/x['visits'])
        q=dict(id=i,family=root['family'],group=root['group'],arm=arm,moves=m,utilities=v,signed_1=v[1]-v[0],signed_2=v[2]-v[1],signed_total=v[2]-v[0],margin=margins,leader_share=shares,visit_lock_after64=margins[1]>32,NN_calls=a['NN_calls'],whole_ms=a['whole_us']/1000,transport_ms=a['transport_ns']/1e6,phase_ms=[x['step_us']/1000 for x in ph],phase_nn=[ph[0]['result']['evaluator_calls'],ph[1]['result']['evaluator_calls']-ph[0]['result']['evaluator_calls'],ph[2]['result']['evaluator_calls']-ph[1]['result']['evaluator_calls']],counters=ph[-1]['counters'])
        assert q['signed_total']==q['signed_1']+q['signed_2']
        p[i,arm]=q;rows0.append(q)
    assert len(p)==len(raw)==96 and len(choices)==16
    for a in rows0:
        ref=p[a['id'],'AX0']
        assert a['moves'][0]==ref['moves'][0] and a['utilities'][0]==ref['utilities'][0]
        a['paired_64']=a['utilities'][1]-ref['utilities'][1];a['paired_96']=a['utilities'][2]-ref['utilities'][2]
        a['different_64']=a['moves'][1]!=ref['moves'][1];a['different_96']=a['moves'][2]!=ref['moves'][2]
        a['nn_delta']=a['NN_calls']-ref['NN_calls']
        a['time_delta_ms']=a['transport_ms']-ref['transport_ms']
        if a['visit_lock_after64']:assert a['moves'][1]==a['moves'][2]
    assert set(u)=={(i,m) for i,ms in choices.items() for m in ms}
    groups={}
    for group in ('TRAIN','CAL','TEST','ALL'):
        for arm in ARMS:
            rr=[r for r in rows0 if r['arm']==arm and (group=='ALL' or r['group']==group)]
            n=len(rr);avg=lambda k:S.mean(r[k] for r in rr)
            groups[group+'/'+arm]=dict(n=n,utilities=[S.mean(r['utilities'][i] for r in rr) for i in range(3)],signed_1=avg('signed_1'),signed_2=avg('signed_2'),signed_total=avg('signed_total'),paired_64=avg('paired_64'),paired_96=avg('paired_96'),different_64=sum(r['different_64'] for r in rr),different_96=sum(r['different_96'] for r in rr),improved_64=sum(r['paired_64']>0 for r in rr),worse_64=sum(r['paired_64']<0 for r in rr),improved_96=sum(r['paired_96']>0 for r in rr),worse_96=sum(r['paired_96']<0 for r in rr),switch_1=sum(r['moves'][1]!=r['moves'][0] for r in rr),switch_2=sum(r['moves'][2]!=r['moves'][1] for r in rr),locked_after64=sum(r['visit_lock_after64'] for r in rr),median_prefix_share=S.median(r['leader_share'][0] for r in rr),mean_prefix_share=S.mean(r['leader_share'][0] for r in rr),mean_NN=avg('NN_calls'),mean_NN_delta=avg('nn_delta'),mean_wall_ms=avg('transport_ms'),mean_native_ms=avg('whole_ms'),mean_time_delta_ms=avg('time_delta_ms'),mean_phase_ms=[S.mean(r['phase_ms'][i] for r in rr) for i in range(3)],mean_phase_nn=[S.mean(r['phase_nn'][i] for r in rr) for i in range(3)],exposure={k:sum(r['counters'][k] for r in rr) for k in ('select_nodes','select_changed','select_deep','width_nodes','width_mass_added','width_deep','base_evals','extra_evals','proofs','proof_win','proof_scans','proof_checks','merge_blends','backup_changed','ledger_shared_children','ledger_observations','terminal_leaves')})
    out=dict(identity=identity(),scope='deterministic common continuation policy conditional returns, not minimax or independent games for arm efficacy',source_games=len(list(rows('C06_SOURCE_GAMES.jsonl.gz'))),roots=len(choices),requests=len(raw),phase_snapshots=3*len(raw),visits=sum(r['response']['visits'] for r in raw),NN_calls=sum(r['NN_calls'] for r in rows0),unique_continuations=len(cont),same_all_recommendations=sum(len(v)==1 for v in choices.values()),distinct_choices={i:sorted(v) for i,v in choices.items()},continuation_searched_plies=sum(len(r['rows']) for r in cont),continuation_NN=sum(x['response']['NN_calls'] for r in cont for x in r['rows']),groups=groups,source_length=[len(r['final_history']) for r in rows('C06_SOURCE_GAMES.jsonl.gz')],continuation_length=[len(r['final_history']) for r in cont],host=[read('C06_COHORT.json')['host'],read('C07_RAW_LOCK.json')['host'],read('C08_CONT_LOCK.json')['host']],scientific_promotion=False)
    write('C08_METRICS.json',rows0);write('C08_SUMMARY.json',out)
    print('SUMMARY', {k:v for k,v in out.items() if k not in ('groups','identity','host')})
    for k,v in groups.items():
        if k.startswith(('TEST/','ALL/')):print(k,{q:v[q] for q in ('utilities','paired_64','paired_96','different_64','different_96','locked_after64','median_prefix_share','mean_NN','mean_wall_ms','mean_phase_ms','exposure')})
if __name__=='__main__':main()
