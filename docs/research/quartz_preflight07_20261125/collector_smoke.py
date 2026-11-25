"""Two bounded constructed-prefix smoke games. Never a production collection."""
import json,tempfile
from pathlib import Path
from rules import fixtures
from engine import Engine,D,identity
from collector import *
def main():
    f=fixtures()
    jobs=[
      dict(game_id='SMOKE_DRAW',family='fixture_draw',family_index=0,replicate=0,split='SMOKE',shard=0,opening=f['draw_pre'],seed=17,black_arm='BASE',white_arm='BASE',max_ply=225),
      dict(game_id='SMOKE_LONG_CENSOR',family='fixture_long',family_index=1,replicate=0,split='SMOKE',shard=1,opening=f['long'],seed=43,black_arm='BASE',white_arm='BASE',max_ply=153),
    ]
    p=dict(schema=SMOKE_SCHEMA,generation_enabled=False,mode='SMOKE_FIXTURE',rule='freestyle',visits=8,threads=1,temperature_plies=12,max_ply=225,jobs=jobs,identity=identity())
    root=D/'C07_SMOKE';assert not root.exists()
    init_store(root,p);out=[]
    with Engine(1) as e:
        for j in jobs:
            r=play_game(e,p,j);assert 1<=len(r['moves'])<=3
            file=save_record(root,p,j,r);assert existing(root,p,j)==file
            out.append(dict(id=j['game_id'],initial_plies=len(j['opening']),final_plies=len(r['final_history']),searched=len(r['moves']),status=r['status'],winner=r['winner_black'],NN_rows=sum(x['native']['nn_rows'] for x in r['moves']),transport_ms=sum(x['native']['transport_ns'] for x in r['moves'])/1e6,bytes=file.stat().st_size,path=file.name))
            print(out[-1],flush=True)
    assert out[0]['status']=='COMPLETED' and out[0]['winner']==0 and out[0]['final_plies']==225
    assert out[1]['status']=='CENSORED' and out[1]['winner'] is None
    audit=verify_store(root);assert audit['committed']==2 and audit['missing']==0
    # Resume must discover both immutable records without new search.
    assert all(existing(root,p,j) for j in jobs)
    (D/'C07_SMOKE_SUMMARY.json').write_text(json.dumps(dict(games=out,verification=audit,production_games=0),indent=2,sort_keys=True)+'\n')
    print('SMOKE_VERIFY',audit,flush=True)
if __name__=='__main__':main()
