"""Create only disabled immutable plan metadata, not a game population."""
from pathlib import Path
import json
from engine import D,identity
from collector import PLAN_SCHEMA,digest,collection_lock
p=dict(schema=PLAN_SCHEMA,generation_enabled=False,mode='SELFPLAY',rule='freestyle',games=10000,replicates_per_family=2,seed=2026112506,opening_plies=6,opening_radius=3,visits=128,threads=1,temperature_plies=12,max_ply=225,shards=4,arm='BASE',candidate_arm='CERT',min_free_gib=50,min_available_gib=4,estimated_compressed_bytes_per_ply=8192,identity=identity(),collection_lock=collection_lock(),candidate_visits=128)
for name,q in [('PLAN_DISABLED.json',p),('ARENA_PLAN_DISABLED.json',dict(p,mode='PAIRED_ARENA'))]:
    path=D/name
    if path.exists():assert json.loads(path.read_text())['generation_enabled'] is False
    path.write_text(json.dumps(q,indent=2,sort_keys=True)+'\n');print(name,digest(q))
