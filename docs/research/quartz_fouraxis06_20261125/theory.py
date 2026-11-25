from publish import publish,D,ROOT
import subprocess
P=str(D.relative_to(ROOT))+'/'
(D/'ORIGIN_WATCHDOG.md').write_text("""# FOURAXIS-06 original motivation and boundary
Authority 87c15dc828d6eea519c219fd8a412fa21dfc464d, tree 021a61b574557378ed5917322e10c9884d3bdc24 verified GitHub/local. Earliest 15f3a1986a00d548d3da56dab8233b336e931aee: KL surprise and p_flip*e_delta-cost; paper pp5–6 Eq1 Gibbs prior and Eq2 VOC, FOCUS/EXPAND/MERGE/STOP; PRD §§0–2 CPU-only game-agnostic, same mean time higher Elo. FA05 9b8a4718f7f92d83db2fb00e5806c6c4bd8cbfc2: 337 positive-risk plans only34 observed risk decreases; ACTIVE candidate on HARD, not general; FRONTIER diagnostic, guard disabled. Return to actual Gomoku15 neural engine from NN01/02 rather than escalate surrogate. Prior SSOT already has COG05 policy-only and COG06 fork witnesses, not the 17-plane Torch value-policy model; reuse conceptual exact tactical proof, do not claim discovery. Actual frozen F1 smoke checkpoint SHA57b93ba867ff8f9b74f91adef6b38f02f94de0bd7dc177785d668d32e1dcc3ae. Exact rule identity separate from neural history identity, build strict full-history endpoint, bounded tactical proof adapter, real data schema and resumable disabled collector. No large corpus, no training, no production merge, no autonomous launch. Tiny explicitly tagged smoke only, no performance claim. FA05 TEST excluded from fitting, diagnostic TRAIN/CAL first-block reuse only. Native subagents NOT_RUN unavailable; independent arithmetic/rule/record validation.
""")
publish('R01','Construct bounded portable exact proof adapter and long-game data collection readiness; current user explicitly prohibits large generation. Source full-history neural replay and actual native search; quantify prior first-block information separately.',[P+'ORIGIN_WATCHDOG.md',P+'publish.py',P+'theory.py'])
publish('R02','Primary sources: RIF International Rules https://www.renju.net/rifrules §§4,5,9.1–9.3,12: black exact five vs white overline, recursive double-three exceptions, opening/pass rules; current engine Renju cap200 and simplified patterns not certified as official RIF. Allis/van den Herik/Huntjens Go-moku solved by new search techniques 1996 https://scispace.com/papers/go-moku-solved-by-new-search-techniques-499rbmit0l threat-space and proof-number search for two variants, not arbitrary positions/variants. Silver et al https://arxiv.org/html/1712.01815 Methods text: visit pi and terminal self-play z estimate policy-conditioned outcomes, not exact minimax; use D4 only under rule invariance. FA05 BAI and exact proof references retained. Existing source src/games/gomoku15.rs, game.rs, mcts_server/fork16_cpu_server.rs, prior15_cpu_server.rs and NN01 engine.py read. Exa/SciSpace/Parallel checked. New research is a bounded actual NN transfer and trace pipeline, not new game solution.')
publish('R03','Define legal placement-only 15x15 freestyle and standard exact-five, alternating black+1 white-1, no capture/pass/swap/resignation, terminal at winning placement or225. Other variants engine inspection only; proof/collection reject. Board[225] 0/1/2, move indices row*15+col, full legal ordered history, proof key board/side/rule, statistical TT opt-in includes last7 history; exact structural keys not probabilistic hashes. Fixed checkpoints; no online oracle.')
(D/'DERIVATION.md').write_text("""# Game rule transfer and local intervention inference
For a legal reachable unrestricted no-capture alternating state, K=(variant,board,side) is exact game-state identity for freestyle/standard; legal set and terminal value derived from it. Use 2 bits/cell (450 bits), side1 bit, variant3 bits, 454 bits fit8 u64 words; no collision by decoding. No D4 canonicalization in native proof (action orientation preserved); D4 orbit for grouping only. K does not encode NN features: AlphaZero 17 planes reconstruct current and7 previous boards with fixed current-player orientation. Histories with same board/side but different recent order can have different features. For statistical search use opt-in mixed 64-bit TT fingerprint including rule and ordered recent7, not collision-free; no proof ever sourced from it. Root reconstructed by validated full history, never board-only from_board. Separate input equality test compares actual first Torch row with independently encoded planes.
Let F_p(B) be set of empty placements producing a winning maximal contiguous run (>=5 freestyle, =5 standard), no preexisting winner. If current legal move a is in F_p(B), its WDL=+1. Otherwise if last empty cell, terminal draw0. Else if F_{-p}(B)\\{a} nonempty, opponent has legal immediate win and a has WDL=-1. Lemma: for these two variants adding an enemy stone at a does not change the contiguous run length at any other empty b for color -p (both empty/enemy stop the run), so F_{-p}(B+a)=F_{-p}(B)\\{a}; no pass/capture and both all empty cells legal. Not valid without rule-specific proof for Caro blocked ends or black forbidden variants. Remaining actions unknown[-1,1], including unexpanded ones. Full legal action intervals, regret bound r_a=max(0,max_{b!=a}U_b-L_a)/2, witness iff r_a=0; keep exact interval channel separate from N/Q. If all actions exact losses, any is minimax optimal (not necessarily best practical defense vs imperfect opponent); certificate STOP preserves WDL, not policy-conditioned win probability. Root certificate adapter only, no general internal Gomoku graph or all-four-axis theorem. Prior COG06 fork proof can be composed later under conditions; not re-prove here.
For root local computation c, response vector Y=(signed decision change, proof risk drop, actual cost, repeat count). g=E[delta U|do(c),H], h=E[delta R|do(c),H]=p(Rdrop>0|c,H)*E[drop|drop>0,c,H], VOC=w_g g+w_r h-lambda E[C]. Max hypothetical drop not expected drop. For actual randomized action assigned with known probability p_c(H)>0, Horvitz-Thompson E[1{A=c}Y/p_c(H)|H]=E[Y(c)|H] under consistency/randomization/no interference. Paired branch replay same full prefix is stronger for local comparison; follow common continuation and report resource differences. Existing FA05 deterministic first block TRAIN/CAL comparison is descriptive matched-prefix, no stochastic treatment propensity. Independently compute probability and gain for each arm conditional on unresolved prefix; no TEST fits and no new controller. Complementarity example three actions initially[-1,1]; revealing one exact loss still D=2; two exact losses gives remaining action witness (lower -1>=competitors -1), D=0; marginal risk gain not submodular in general.
Game corpus: X_t includes full ordered history, rule, model and policy config, seed, actual visits/evals/NN rows, prior and observed edge N/W/M2, exact intervals and certificate provenance, selected action and behavior probability, search/whole time. Preserve terminal winner w in black sign; for completed game z_t=w*p_t (p_t=+1/-1), draw0. Cap/interruption is CENSORED, z_t=null, never false draw; immediate certificate with zero NN/visits has policy target null, no artificial counts. z_t estimates outcome under behavior policy, not V*, and cannot label unplayed alternatives. Full games/paired opening clusters are independent analysis units; plies correlated. Equal per-game weighting E[(1/T_g)sum_t f] differs from position-weighted E[sum f]/E[T_g]; long survivors selected by game length, preserve all natural game lengths and censored endpoints, report conditional length strata/at-risk denominators rather than rejecting short wins or suppressing winning moves. For paired arena same opening with swapped policy colors, D_g in[-1,1] difference of candidate vs baseline score (average both games), use family bootstrap/paired variance; same actual mean wall time per move is ultimate objective. IID game Bernoulli normal approximation N~z^2*.25/eps^2=9604 for eps=.01, paired worst variance1 gives38416; descriptive sizing, no guarantee for dependent games. Hypothetical long game data budget G*mean moves, bytes per move from smoke only rough; hard preflight worst225 plies and disk reserve. User explicitly forbids bulk, dry-run default and generation_enabled false with explicit future hash authorization, no training.
""")
publish('R04','Candidate two-ply exact tactical adapter vs general graph; choose linear full legal root scan, safe witness and no proof from NN/TT. Design full history statistical identity, prospective game-level split, natural survival distribution and recorded resource/decision trajectories.',[P+'DERIVATION.md'])
publish('R05','Prove opponent threat removal identity under both admitted rule variants, exact interval and regret certificate, exact 454-bit packing and history distinction; local probability factorization, randomized estimator, paired cluster and censoring algebra.')
publish('R06','Counterexamples: same board different recent history -> different 17 planes, same board different rule overline -> different values; Caro end-block effects, Renju rule/cap divergence; simultaneous exact losses complementary risk; zero-visit certificate not NN policy; do not label capped games draw or outcomes minimax.')
(D/'theory_check.py').write_text("""from fractions import Fraction as F
from itertools import product
from collections import defaultdict
import gzip,json,statistics as S,random
from pathlib import Path
from publish import D,ROOT,sha
# Exact two-ply local line lemma for all ternary lines of length8 and all
# empty pairs; four directions follow separately, excluding occupied a.
n=0
def wins(row,b,p,exact):
    k=1
    for d in (-1,1):
        j=b+d
        while 0<=j<len(row) and row[j]==p:k+=1;j+=d
    return k==5 if exact else k>=5
for row in product(range(3),repeat=8):
    for p in (1,2):
        for ex in (False,True):
            old={i for i,x in enumerate(row) if x==0 and wins(row,i,p,ex)}
            for a,v in enumerate(row):
                if v:continue
                b=list(row);b[a]=3-p
                new={i for i,x in enumerate(b) if x==0 and wins(b,i,p,ex)}
                assert new==old-{a};n+=1
def rw(iv):
    return min(max(0,max((h for j,(_,h) in enumerate(iv) if i!=j),default=-1)-l) for i,(l,u) in enumerate(iv))
assert rw([(-1,1)]*3)==2 and rw([(-1,-1),(-1,1),(-1,1)])==2 and rw([(-1,-1),(-1,-1),(-1,1)])==0
for g in (-1,0,1):
    for p in (-1,1):
        assert -g*p==g*(-p);n+=1
# Earlier disjoint FA05 TRAIN/CAL diagnostic first-block observations. Verify
# exact Git blob identity, no FA05 TEST accessed. Matched prefix only.
p=ROOT.parent/'quartz_fouraxis05_20261125/docs/research/quartz_fouraxis05_20261125'
for name in ('C07_DIAG_RAW.jsonl.gz','C07_DIAG_LABELS.json'):
    import subprocess as sp
    assert sp.check_output(['git','hash-object',str(p/name)],cwd=ROOT,text=True).strip()==sp.check_output(['git','rev-parse','9b8a4718f7f92d83db2fb00e5806c6c4bd8cbfc2:docs/research/quartz_fouraxis05_20261125/'+name],cwd=ROOT,text=True).strip()
z={x['id']:x['scores'] for x in json.loads((p/'C07_DIAG_LABELS.json').read_text()) if x['status']=='EXACT'}
utility=lambda x:F(1+(x>0)-(x<0),2)
by={}
with gzip.open(p/'C07_DIAG_RAW.jsonl.gz','rt') as f:
    for line in f:
        a=json.loads(line)
        if a['arm']=='BASE' or not a['steps']:continue
        assert a['group'] in ('TRAIN','CAL') and a['mode']=='DIAG'
        pre=a['prefix'];post=a['steps'][0]['post'];iv=lambda q:[x[1] for x in q['proof']['actions']]
        d=F(rw(iv(pre))-rw(iv(post)),2);assert d>=0
        g=utility(z[a['id']][post['best']])-utility(z[a['id']][pre['best']])
        by[a['group'],a['id'],a['arm']]=(pre['result'],pre['proof'],d,g,a['steps'][0]['plan_us']+a['steps'][0]['search_us'])
out={}
for group in ('TRAIN','CAL'):
    out[group]={}
    for arm in ('SAFE','ACTIVE','FOCUS','EXPAND','FRONTIER','EVAL','WRITER','MERGE'):
        rows=[(v,by[g,i,'SAFE']) for (g,i,a),v in by.items() if g==group and a==arm]
        assert rows
        for v,r in rows:assert v[:2]==r[:2]
        d=[v[2] for v,r in rows];x=[v[3] for v,r in rows];delta=[v[3]-r[3] for v,r in rows]
        out[group][arm]=dict(families=len(rows),risk_drop_count=sum(y>0 for y in d),mean_risk_drop=float(S.mean(d)),p_risk=float(F(sum(y>0 for y in d),len(d))),conditional_risk_drop=float(S.mean([y for y in d if y>0])) if any(d) else None,signed_gain=float(S.mean(x)),paired_vs_safe=float(S.mean(delta)),positive=sum(y>0 for y in x),negative=sum(y<0 for y in x),mean_plan_search_us=S.mean(v[4] for v,r in rows))
(D/'R07_LOCAL_EVIDENCE.json').write_text(json.dumps(dict(assertions=n,sources={name:sha(p/name) for name in ('C07_DIAG_RAW.jsonl.gz','C07_DIAG_LABELS.json')},groups=out,scope='previous exact TRAIN/CAL first block, unresolved common prefix only; descriptive, no new label or generation'),indent=2,sort_keys=True)+'\n')
print('Exact line and terminal-label checks',n)
print(json.dumps(out,sort_keys=True))
""")
out=subprocess.check_output(['python3',str(D/'theory_check.py')],text=True)
(D/'R07.log').write_text(out+'Wolfram: eight words for 451 bits (454 also fits), normal IID proportion N9604; cost expression and clustered design-effect check.\\n')
publish('R07','Executed exhaustive ternary line lemma, label checks, exact complementarity and actual prior FA05 TRAIN/CAL first-block comparisons. Wolfram arithmetic. No new corpus.',[P+'theory_check.py',P+'R07.log',P+'R07_LOCAL_EVIDENCE.json'])
publish('R08','Admit exact portable two-ply root proof for freestyle/standard and collection preparation only. Existing prior first-block data sparse, no learned controller promotion; do not claim full Gomoku proof graph or RIF compliance.')
(D/'CONTRACT.md').write_text("""# Frozen readiness contract
Direct SSOT and exact10 FA05 native files transplanted. Actual prior F1 Gomoku15 17-plane checkpoint external lock, TorchCPU prior venv, no substitute NN. Strict new research endpoint accepts full validated history, explicit rule (freestyle/standard proof and NN; omok/renju/caro inspection only), inspect terminal or legal, reject invalid/duplicate/post-terminal and unknown fields, return board/side/rule key, history-aware TT fingerprint, full features on inspect, exact complete root intervals and witness for admitted variants. Native BASE and CERT use same real policy/value NN and fixed visits with1 or2 threads, fresh tree each move (resident within each search only); BASE logs proof diagnostic but does not use witness; CERT stops before NN on exact witness, otherwise same native MCTS and no dominated recommendation. Do not falsely imply all four axes ported. Actual policy/edge stats and NN counters, proof timings, full transport, no online labels. Rule proof keys exact8 words, statistical fingerprint64 and recent7, production default untouched. No arbitrary board parser or rule alias.
Bounded tests: independent full-grid legality/terminal/one/two-ply certificate, transposition same proof key different history and feature input, overline standard vs freestyle, draw and illegal histories, Renju/Omok forbidden fixture and unsupported proof, Caro blocked-end counterexample; no large synthetic games, deterministic constructed draw-safe long fixture only. Actual frozen Torch model forward hooks compare first root row vs independent feature array, calls and NN batches/rows accounting, 1/2 thread searches on <=6 fixed fixture roots; smoke collector max2 game records and <=8 searched plies/game; one near-terminal and one >=100-ply nonterminal prefix, preserve censoring, no large games or training.
Production preparation: versioned immutable JSON plan default generation_enabled=false, dry-run never launches engine. Future explicit execution requires enabled plan + exact plan digest authorization, disk/memory preflight and limits; deterministic opening families (D4 canonical, no outcome filtering), game IDs/seeds/splits assigned before play, per-game independent compressed atomic commit, checksums and replay on resume, shard disjointness, single writer per shard, failure stops, no fake terminal label. Actual move choice from visits (sample early then greedy) or cert, log behavior p and policy target (null if zero-visit cert), all native trace and root search telemetry; independent validator reconstructs board/history, legal/proof/labels/stats, checks file identity/manifest/shard, canonical split grouping and cross-split duplicates report (position leakage audit post-opening, not guaranteed absent). Future arena protocol color-swapped paired openings same model, compare actual per-move wall/CPU cost and game outcomes; not execute arena or bulk. Proposed corpus plan illustrative10000 games, natural up to225 plies, no minimum-length filtering. Current low disk reserve fails future bulk preflight; prepare only. Native subagent absent. Each stage push+verified incremental bundle, remove only same-run superseded; no old unique assets removed.
""")
publish('R09','Freeze executable adapter and disabled long-game collector contract, bounded smoke and independent audit. Game-level outcomes distinct from minimax. User forbids bulk.',[P+'CONTRACT.md'])
publish('R10','Handoff to native history-safe Gomoku root adapter, actual CPU Torch bridge, durable streaming collector and validated dry-run; no large generation.')
