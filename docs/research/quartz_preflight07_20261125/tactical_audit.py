"""Post-pilot diagnostic only. No search, policy changes or additional games."""
import collections as C,json,math,statistics as S,subprocess as SP
from audit import ref
from collector import load_record
from publish import ROOT,D,B,git

def finite_tree(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite_tree(v)
    elif isinstance(x,list):
        for v in x:finite_tree(v)

def main():
    sides={1:C.Counter(),-1:C.Counter()};ranks={1:[],-1:[]}
    for f in sorted((D/'P2_CORPUS').glob('shard_*/*.json.gz')):
        r=load_record(f);finite_tree(r)
        h=list(r['job']['opening']);rule='freestyle'
        for row in r['moves']:
            b,p,w=ref.replay(h,rule)
            assert w is None and type(row['player']) is int and row['player']==p
            assert type(row['z']) is int and row['z']==r['winner_black']*p
            a=row['native'];pr=a['proof'];own=set(pr['own']);opp=set(pr['opponent']);res=a['result'];m=row['played_move'];c=sides[p];c['positions']+=1
            for move,lo,hi,reply in pr['actions']:
                if reply is not None:
                    bb=b.copy();assert not bb[move];bb[move]=1 if p==1 else 2
                    assert not bb[reply] and ref.win_at(bb,reply,2 if p==1 else 1,rule) and lo==hi==-1
                    c['direct_reply_checks']+=1
                if lo==hi==1:
                    assert ref.win_at(b,move,1 if p==1 else 2,rule)
                    c['direct_win_checks']+=1
                if move==m and hi<pr['interval'][0]:c['strictly_dominated_played']+=1
            if own:
                c['immediate_win_positions']+=1
                es=[e for e in res['edges'] if e['move'] in own]
                vs=[e for e in es if e['n']>0]
                c['winning_materialized_positions']+=bool(es)
                c['winning_visited_positions']+=bool(vs)
                c['winning_played_positions']+=m in own
                ranks[p].append(min(i+1 for i,(k,_) in enumerate(res['candidates']) if k in own))
                for e in vs:
                    assert abs(e['w']-e['n'])<1e-8 and abs(e['m2'])<1e-8
                    c['terminal_win_edge_snapshots']+=1;c['terminal_win_samples']+=e['n']
            if not own and len(opp)==1:
                c['single_opponent_threat_without_own_win']+=1
                c['single_opponent_threat_blocked']+=m in opp
            h.append(m)
        _,_,w=ref.replay(h,rule);assert w==r['winner_black']
    out={}
    for p in (1,-1):
        c=sides[p];out['black' if p==1 else 'white']={**dict(c),'best_winning_prior_rank_mean':S.mean(ranks[p]),'best_winning_prior_rank_median':S.median(ranks[p]),'best_winning_prior_rank_max':max(ranks[p])}
    assert sum(c['positions'] for c in sides.values())==1396
    assert sum(c['winning_played_positions'] for c in sides.values())==16
    return dict(scope='posthoc all pilot splits, descriptive only; not held-out performance',by_color=out,finite_records=16,exact_integer_player_and_z_rows=1396,no_new_search=True)
if __name__=='__main__':
    print(json.dumps(main(),indent=2,sort_keys=True))
