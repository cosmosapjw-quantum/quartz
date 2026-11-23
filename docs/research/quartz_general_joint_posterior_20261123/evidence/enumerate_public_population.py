from pathlib import Path
import sys
sys.dont_write_bytecode = True
import importlib.util, json, hashlib, subprocess
from collections import Counter, defaultdict
R = Path("/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123")
D = R / "docs/research/quartz_general_game_metareasoning_20261123"
P = D / "coding/c26_campaign.py"
spec=importlib.util.spec_from_file_location("c26_readonly", P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old_public=[json.loads(line) for line in (D/"coding/data/public_sources.jsonl").read_text().splitlines()]
old_boards={m.canonicalize(m.replay(s["history"]),s["history"])[0] for s in old_public}
assert len(old_public)==len(old_boards)==300
reachable=m.enumerate_reachable();can={}
for board,history in reachable.items():
    if m.terminal(board):continue
    cb,ch=m.canonicalize(board,history)
    if cb not in can or ch<can[cb]:can[cb]=ch
rows=[]
for b,h in can.items():
    if b in old_boards:continue
    sha=hashlib.sha256(("C27-available-v1|"+m.board_text(b)).encode()).hexdigest()
    moves=m.legal_moves(b)
    terminal_children=sum(m.terminal(m.apply(b,a)) for a in moves)
    rows.append({"board":list(b),"history":list(h),"ply":len(h),"available_hash":sha,
                 "legal_children":len(moves),"public_terminal_children":terminal_children,
                 "nonterminal_children":len(moves)-terminal_children})
rows.sort(key=lambda x:(x["ply"],x["available_hash"]))
# Distinct root identity does not mean disjoint descendant sets.
@__import__("functools").lru_cache(None)
def descendants(b):
    result=set()
    for a in m.legal_moves(b):
        child=m.apply(b,a);result.add(m.canonicalize(child,())[0])
        result.update(descendants(child))
    return frozenset(result)
new_frame=[x for x in rows if x["ply"] in (4,5,6)]
overlap_old=sum(bool(descendants(tuple(x["board"])) & old_boards) for x in new_frame)
paired=0
for i,x in enumerate(new_frame):
    dx=descendants(tuple(x["board"]))
    for y in new_frame[i+1:]:
        if dx & descendants(tuple(y["board"])):paired+=1
assert m.oracle.cache_info().currsize==0, "No minimax label oracle allowed in this investigation"
result={"schema":"c27-source-context-v1","ssot":"87c15dc828d6eea519c219fd8a412fa21dfc464d",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_harness_sha256":hashlib.sha256(P.read_bytes()).hexdigest(),
        "C26_public_sources_sha256":hashlib.sha256((D/"coding/data/public_sources.jsonl").read_bytes()).hexdigest(),
        "reachable_board_count":len(reachable),"canonical_nonterminal_count":len(can),
        "canonical_population_by_ply":dict(sorted(Counter(len(h) for h in can.values()).items())),
        "excluded_C26_by_ply":dict(sorted(Counter(s["ply"] for s in old_public).items())),
        "available_count":len(rows),"available_by_ply":dict(sorted(Counter(x["ply"] for x in rows).items())),
        "available_with_at_least_3_nonterminal_children_by_ply":dict(sorted(Counter(x["ply"] for x in rows if x["nonterminal_children"]>=3).items())),
        "new_same_ply_frame":len(new_frame),
        "new_same_ply_roots_with_a_C26_root_descendant":overlap_old,
        "new_same_ply_pairs_sharing_a_descendant":paired,
        "new_same_ply_pair_count":len(new_frame)*(len(new_frame)-1)//2,
        "oracle_calls":m.oracle.cache_info().currsize,
        "available_roots":rows,
        "scope":"Public legal-state enumeration and canonical identity only. No new minimax labels, fitted parameters, policy runs or outcome-selected eligibility."}
Path("/tmp/quartz_c27_context.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="available_roots"},sort_keys=True))
