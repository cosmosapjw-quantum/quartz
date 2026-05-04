"""
QUARTZ Evaluation System — Final Version
=========================================

Architecture: Three-layer separation of concerns.

  Layer 1 — PromotionGate (2-agent, per checkpoint)
  Layer 2 — SanityCheck (diagnostic, periodic)
  Layer 3 — RatingLadder (Glicko-2, analytical)
  Layer 4 — ScaleCalibrator (affine published Elo)
"""

import math
import time
import logging
import random
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# §1 Glicko-2 Core

@dataclass
class RatingParams:
    mu0_elo: float = 1500.0
    rd0_elo: float = 350.0
    sigma0: float = 0.06
    tau: float = 0.5
    scale: float = 173.7178
    anchor_target_elo: float = 0.0
    max_match_batch: int = 16

@dataclass
class RatingRecord:
    mu_elo: float; rd_elo: float; sigma: float
    games_played: int = 0; last_period: int = 0
    published_elo: Optional[float] = None
    history: List[float] = field(default_factory=list)
    def init_published(self):
        if self.published_elo is None: self.published_elo = self.mu_elo

class Glicko2:
    @staticmethod
    def to_g2(mu_elo, rd_elo, P): return (mu_elo - P.mu0_elo)/P.scale, rd_elo/P.scale
    @staticmethod
    def to_elo(mu, phi, P): return mu*P.scale + P.mu0_elo, phi*P.scale
    @staticmethod
    def g(phi): return 1.0/math.sqrt(1.0 + 3.0*phi*phi/(math.pi*math.pi))
    @staticmethod
    def E(mu, mu_j, phi_j):
        x = Glicko2.g(phi_j) * (mu - mu_j)
        if x >= 0.0:
            z = math.exp(-min(x, 700.0))
            return 1.0 / (1.0 + z)
        z = math.exp(max(x, -700.0))
        return z / (1.0 + z)
    @staticmethod
    def _update_from_accumulators(mu, phi, sigma, s_g2E, s_gsE, tau):
        if s_g2E <= 0 or not math.isfinite(s_g2E) or not math.isfinite(s_gsE):
            return mu, phi, sigma
        v = 1.0/s_g2E; delta = v*s_gsE
        a = math.log(sigma*sigma); eps = 1e-6
        def f(x):
            ex=math.exp(x)
            return (ex*(delta**2-phi**2-v-ex)/(2.0*(phi**2+v+ex)**2))-(x-a)/(tau**2)
        A=a; B=math.log(delta**2-phi**2-v) if delta**2>phi**2+v else a-tau
        if delta**2<=phi**2+v:
            k=1
            while f(B)<0 and k<100: k+=1; B=a-k*tau
        fA,fB = f(A),f(B)
        for _ in range(100):
            if abs(B-A)<=eps: break
            C=A+(A-B)*fA/(fB-fA); fC=f(C)
            if fC*fB<=0: A,fA=B,fB
            else: fA*=0.5
            B,fB=C,fC
        sig2=math.exp(A/2.0); phi_s=math.sqrt(phi**2+sig2**2)
        phi2=1.0/math.sqrt(1.0/phi_s**2+1.0/v)
        mu2=mu+phi2**2*s_gsE
        return mu2, phi2, sig2
    @staticmethod
    def update(mu, phi, sigma, terms, tau):
        s_g2E = 0.0
        s_gsE = 0.0
        for g, E, s in terms:
            s_g2E += g*g*E*(1-E)
            s_gsE += g*(s-E)
        return Glicko2._update_from_accumulators(mu, phi, sigma, s_g2E, s_gsE, tau)
    @staticmethod
    def update_weighted(mu, phi, sigma, terms, tau):
        s_g2E = 0.0
        s_gsE = 0.0
        for g, E, s, count in terms:
            if count <= 0:
                continue
            s_g2E += count * g*g*E*(1-E)
            s_gsE += count * g*(s-E)
        return Glicko2._update_from_accumulators(mu, phi, sigma, s_g2E, s_gsE, tau)


# §1.5 GameSpec + Anchor

@dataclass
class GameSpec:
    game_type: str; board_size: int = 0; rules_hash: str = ""
    search_budget_type: str = "simulations"; search_budget: int = 800
    opening_book_hash: str = ""; evaluator_version: str = ""; threads: int = 1
    def spec_hash(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()[:16]

@dataclass
class AnchorEntry:
    id: str; engine_hash: str; target_elo: float; weight: float = 1.0
    role: str = "anchor"; description: str = ""

@dataclass
class AnchorManifest:
    version: str = "v1"; game_spec: Optional[GameSpec] = None
    anchors: List[AnchorEntry] = field(default_factory=list); created: str = ""
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path,"w") as f:
            json.dump({"version":self.version,"game_spec":self.game_spec.__dict__ if self.game_spec else {},
                       "anchors":[a.__dict__ for a in self.anchors],"created":self.created or time.strftime("%Y-%m-%d")},f,indent=2)
    @classmethod
    def load(cls, path: Path) -> "AnchorManifest":
        with open(path) as f: d=json.load(f)
        gs=GameSpec(**d["game_spec"]) if d.get("game_spec") else None
        return cls(version=d.get("version","v1"),game_spec=gs,
                   anchors=[AnchorEntry(**a) for a in d.get("anchors",[])],created=d.get("created",""))
    @classmethod
    def default_gomoku7(cls):
        return cls(game_spec=GameSpec(game_type="gomoku",board_size=7,search_budget=800),
            anchors=[AnchorEntry(id="random",engine_hash="random_v1",target_elo=0.0,weight=1.0,
                                 description="Uniform random legal moves"),
                     AnchorEntry(id="search_only",engine_hash="uct_uniform_128_v1",target_elo=200.0,
                                 weight=1.0,description="UCT + uniform prior, 128 sims")])


# §1.6 Scale Calibrator

class ScaleCalibrator:
    def __init__(self, manifest: AnchorManifest):
        self.manifest=manifest; self.a=1.0; self.b=0.0; self._history=[]
    def calibrate(self, ladder, min_expected=0.05, max_expected=0.95):
        points=[]
        for anchor in self.manifest.anchors:
            if anchor.id not in ladder.players: continue
            rec=ladder.get(anchor.id)
            if rec.games_played==0: continue
            if rec.rd_elo>ladder.P.rd0_elo*0.95: continue
            points.append((rec.mu_elo, anchor.target_elo, anchor.weight))
        if len(points)<1: self.a,self.b=1.0,0.0
        elif len(points)==1:
            r,t,_=points[0]; self.a=1.0; self.b=t-r
        else:
            W=sum(w for _,_,w in points); Sr=sum(w*r for r,_,w in points)/W
            St=sum(w*t for _,t,w in points)/W; Srr=sum(w*r*r for r,_,w in points)/W
            Srt=sum(w*r*t for r,t,w in points)/W; denom=Srr-Sr*Sr
            if abs(denom)>1e-10:
                self.a=(Srt-Sr*St)/denom; self.b=St-self.a*Sr
                if self.a<=0 or self.a>3.0 or self.a<0.3: self.a=1.0; self.b=St-Sr
            else: self.a=1.0; self.b=St-Sr
        self._history.append((self.a,self.b)); return self.a,self.b
    def published_elo(self, internal_elo): return self.a*internal_elo+self.b
    def residuals(self, ladder):
        return {a.id: round(self.published_elo(ladder.get(a.id).mu_elo)-a.target_elo,1)
                for a in self.manifest.anchors if a.id in ladder.players}


# §2 Rating Ladder

class RatingLadder:
    def __init__(self, P=None):
        self.P=P or RatingParams(); self.players={}; self._period=0; self._bootstrapped=False
        self._ensure("random"); self._ensure("search_only"); self._recenter(); self._bootstrapped=True
    def _ensure(self, name):
        if name not in self.players:
            start=self.P.anchor_target_elo if self._bootstrapped else self.P.mu0_elo
            rec=RatingRecord(mu_elo=start,rd_elo=self.P.rd0_elo,sigma=self.P.sigma0)
            rec.init_published(); self.players[name]=rec
        return self.players[name]
    def _recenter(self):
        rand=self._ensure("random"); off=self.P.anchor_target_elo-rand.mu_elo
        if abs(off)<1e-12: return
        for r in self.players.values():
            r.mu_elo+=off
            if r.published_elo is not None: r.published_elo+=off
    def get(self, name): return self._ensure(name)
    def _split_match_counts(self, wins, draws, losses):
        total = wins + draws + losses
        if total <= 0:
            return []
        batch = max(1, int(getattr(self.P, "max_match_batch", 16)))
        remaining = {"wins": int(wins), "draws": int(draws), "losses": int(losses)}
        chunks = []
        while total > 0:
            chunk_n = min(batch, total)
            alloc = {"wins": 0, "draws": 0, "losses": 0}
            remainders = []
            used = 0
            for key in ("wins", "draws", "losses"):
                raw = remaining[key] * chunk_n / total
                take = min(remaining[key], int(math.floor(raw)))
                alloc[key] = take
                used += take
                remainders.append((raw - take, key))
            remainders.sort(reverse=True)
            while used < chunk_n:
                picked = False
                for _, key in remainders:
                    if remaining[key] > alloc[key]:
                        alloc[key] += 1
                        used += 1
                        picked = True
                        if used >= chunk_n:
                            break
                if not picked:
                    break
            remaining["wins"] -= alloc["wins"]
            remaining["draws"] -= alloc["draws"]
            remaining["losses"] -= alloc["losses"]
            total -= alloc["wins"] + alloc["draws"] + alloc["losses"]
            chunks.append((alloc["wins"], alloc["draws"], alloc["losses"]))
        return [chunk for chunk in chunks if sum(chunk) > 0]
    def record_match(self, player, opponent, wins, draws, losses):
        total=wins+draws+losses
        if total<=0: return
        P=self.P; A,B=self.get(player),self.get(opponent)
        for cw, cd, cl in self._split_match_counts(wins, draws, losses):
            chunk_total = cw + cd + cl
            muA,phiA=Glicko2.to_g2(A.mu_elo,A.rd_elo,P); muB,phiB=Glicko2.to_g2(B.mu_elo,B.rd_elo,P)
            gB,EB=Glicko2.g(phiB),Glicko2.E(muA,muB,phiB); gA,EA=Glicko2.g(phiA),Glicko2.E(muB,muA,phiA)
            termsA=((gB,EB,1.0,cw),(gB,EB,0.5,cd),(gB,EB,0.0,cl))
            termsB=((gA,EA,1.0,cl),(gA,EA,0.5,cd),(gA,EA,0.0,cw))
            mu2,phi2,sig2=Glicko2.update_weighted(muA,phiA,A.sigma,termsA,P.tau)
            nextA_mu,nextA_rd=Glicko2.to_elo(mu2,phi2,P)
            mu2,phi2,sig2_b=Glicko2.update_weighted(muB,phiB,B.sigma,termsB,P.tau)
            nextB_mu,nextB_rd=Glicko2.to_elo(mu2,phi2,P)
            A.mu_elo,A.rd_elo,A.sigma=nextA_mu,nextA_rd,sig2
            B.mu_elo,B.rd_elo,B.sigma=nextB_mu,nextB_rd,sig2_b
            A.games_played+=chunk_total; B.games_played+=chunk_total
            A.last_period=self._period; B.last_period=self._period
        A.history.append(A.mu_elo); B.history.append(B.mu_elo)
        self._recenter()
    def update_from_period(self, matches):
        P=self.P; pre={}
        for p,o,w,d,l in matches:
            for name in (p,o):
                if name not in pre:
                    rec=self.get(name); mu,phi=Glicko2.to_g2(rec.mu_elo,rec.rd_elo,P)
                    pre[name]=(mu,phi,rec.sigma)
        player_terms={}; player_games={}
        for p,o,w,d,l in matches:
            total=w+d+l
            if total<=0: continue
            mu_p,phi_p,_=pre[p]; mu_o,phi_o,_=pre[o]
            gO=Glicko2.g(phi_o); EO=Glicko2.E(mu_p,mu_o,phi_o)
            terms_p=player_terms.setdefault(p,[])
            if w: terms_p.append((gO,EO,1.0,w))
            if d: terms_p.append((gO,EO,0.5,d))
            if l: terms_p.append((gO,EO,0.0,l))
            gP=Glicko2.g(phi_p); EP=Glicko2.E(mu_o,mu_p,phi_p)
            terms_o=player_terms.setdefault(o,[])
            if l: terms_o.append((gP,EP,1.0,l))
            if d: terms_o.append((gP,EP,0.5,d))
            if w: terms_o.append((gP,EP,0.0,w))
            player_games[p]=player_games.get(p,0)+total; player_games[o]=player_games.get(o,0)+total
        for name,terms in player_terms.items():
            if not terms: continue
            mu,phi,sigma=pre[name]
            mu2,phi2,sig2=Glicko2.update_weighted(mu,phi,sigma,terms,P.tau)
            rec=self.get(name); rec.mu_elo,rec.rd_elo=Glicko2.to_elo(mu2,phi2,P)
            rec.sigma=sig2; rec.games_played+=player_games.get(name,0)
            rec.last_period=self._period; rec.history.append(rec.mu_elo)
        self._recenter()
    def advance_period(self):
        self._period+=1; P=self.P
        for rec in self.players.values():
            _,phi=Glicko2.to_g2(rec.mu_elo,rec.rd_elo,P)
            phi_new=min(math.sqrt(phi**2+rec.sigma**2),P.rd0_elo/P.scale)
            rec.rd_elo=phi_new*P.scale
    @property
    def period(self): return self._period
    def summary(self):
        return {name:{"elo":round(r.mu_elo,1),"rd":round(r.rd_elo,1),"sigma":round(r.sigma,5),"games":r.games_played}
                for name,r in sorted(self.players.items(),key=lambda x:-x[1].mu_elo)}
    def save(self, path):
        data={name:{"mu_elo":r.mu_elo,"rd_elo":r.rd_elo,"sigma":r.sigma,"games_played":r.games_played,
                     "last_period":r.last_period,"history":r.history} for name,r in self.players.items()}
        data["_meta"]={"period":self._period}
        path.parent.mkdir(parents=True,exist_ok=True)
        with open(path,"w") as f: json.dump(data,f,indent=2)
    def load(self, path):
        with open(path) as f: data=json.load(f)
        meta=data.pop("_meta",{}); self._period=meta.get("period",0); self._bootstrapped=True
        for name,d in data.items():
            rec=RatingRecord(mu_elo=d["mu_elo"],rd_elo=d["rd_elo"],sigma=d["sigma"],
                             games_played=d.get("games_played",0),last_period=d.get("last_period",0),
                             history=d.get("history",[])); rec.init_published(); self.players[name]=rec


# §3 Statistical Utilities

def wilson_ci(wins, total, z=1.96):
    if total<=0: return (0.0,0.0)
    p=wins/total; d=1.0+z*z/total; c=p+z*z/(2*total)
    w=z*math.sqrt((p*(1-p)+z*z/(4*total))/total)
    return (max(0.0,(c-w)/d), min(1.0,(c+w)/d))

def score_rate_ci(wins, draws, total, z=1.96):
    if total<=0: return 0.0,(0.0,0.0)
    sr=(wins+0.5*draws)/total; losses=total-wins-draws
    var=(wins*(1-sr)**2+draws*(0.5-sr)**2+losses*(0-sr)**2)/total
    se=math.sqrt(var/total) if var>0 else 0.0
    return sr,(max(0.0,sr-z*se),min(1.0,sr+z*se))

def binomial_p_value(wins, total, p0=0.5):
    if total<30: return 1.0
    var=total*p0*(1-p0)
    if var<=0: return 1.0 if wins==total*p0 else 0.0
    z=abs(wins-total*p0)/math.sqrt(var)
    return min(1.0, 2.0*(1.0-0.5*(1.0+math.erf(z/1.4142135))))

Z_TABLE={0.90:1.645, 0.95:1.960, 0.99:2.576}


def estimate_match_delta_elo(wins, draws, losses, prior_games=8.0, max_abs=800.0):
    total=wins+draws+losses
    if total<=0: return 0.0
    scored=wins+0.5*draws
    sr=(scored+0.5*prior_games)/(total+prior_games)
    sr=min(max(sr,1e-6),1.0-1e-6)
    delta=400.0*math.log10(sr/(1.0-sr))
    return max(-max_abs,min(max_abs,delta))


# §4 Engine & Game Protocols

@runtime_checkable
class Engine(Protocol):
    def select_move(self, state) -> Tuple[int, Dict[str,Any]]: ...
    def reset(self) -> None: ...
    def name(self) -> str: ...

@runtime_checkable
class GameAdapter(Protocol):
    def clone(self) -> "GameAdapter": ...
    def apply_move(self, action: int) -> None: ...
    def is_terminal(self) -> bool: ...
    def outcome_for_black(self) -> Optional[float]: ...
    def current_player(self) -> int: ...
    def legal_moves(self) -> List[int]: ...

class RandomEngine:
    def __init__(self, seed=None): self._rng=random.Random(seed)
    def select_move(self, state):
        return self._rng.choice(state.legal_moves()),{"time_used_ms":0,"simulations":0}
    def reset(self): pass
    def name(self): return "random"


# §5 Match Protocol

@dataclass
class MoveRecord:
    ply:int; player:int; action:int; time_ms:float; sims:int
    root_entropy: Optional[float]=None

@dataclass
class GameRecord:
    game_id:str; engine_black:str; engine_white:str; outcome:str
    score_black: Optional[float]; move_count:int; total_time_ms:float
    moves: List[MoveRecord]=field(default_factory=list)
    opening: List[int]=field(default_factory=list)
    seed: Optional[int]=None; error: Optional[str]=None; is_void:bool=False
    search_manifest_hash: Optional[str]=None
    search_manifest_hash_black: Optional[str]=None
    search_manifest_hash_white: Optional[str]=None
    search_manifest_mismatch_reason: Optional[str]=None
    search_summary: Optional[Dict[str,Any]]=None
    def to_jsonl(self):
        d={"game_id":self.game_id,"outcome":self.outcome,"score_black":self.score_black,
           "engines":[self.engine_black,self.engine_white],"moves":self.move_count,
           "time_ms":round(self.total_time_ms,1),"is_void":self.is_void}
        if self.error: d["error"]=self.error
        if self.search_manifest_hash is not None:
            d["search_manifest_hash"] = self.search_manifest_hash
        if self.search_manifest_hash_black is not None:
            d["search_manifest_hash_black"] = self.search_manifest_hash_black
        if self.search_manifest_hash_white is not None:
            d["search_manifest_hash_white"] = self.search_manifest_hash_white
        if self.search_manifest_mismatch_reason is not None:
            d["search_manifest_mismatch_reason"] = self.search_manifest_mismatch_reason
        return json.dumps(d,ensure_ascii=False)

class MatchRunner:
    def __init__(self, game_factory, opening_book=None, seed=None, max_moves=500):
        self.game_factory=game_factory; self.opening_book=opening_book or []
        self.rng=random.Random(seed); self.max_moves=max_moves
    def play_game(self, eng_black, eng_white, game_id, opening_idx=None, collect_moves=True):
        game=self.game_factory(); eng_black.reset(); eng_white.reset()
        engines={0:eng_black,1:eng_white}; moves=[]; opening_applied=[]
        game_seed=self.rng.randint(0,2**31)
        if opening_idx is not None and opening_idx<len(self.opening_book):
            for a in self.opening_book[opening_idx]:
                if game.is_terminal() or a not in game.legal_moves(): break
                game.apply_move(a); opening_applied.append(a)
        t0=time.time(); ply=len(opening_applied)
        try:
            while not game.is_terminal() and ply<self.max_moves:
                p=game.current_player(); action,meta=engines[p].select_move(game)
                if hasattr(game, "apply_engine_meta") and meta.get("terminal", False):
                    game.apply_engine_meta(action, meta)
                    break
                if collect_moves:
                    moves.append(MoveRecord(ply=ply,player=p,action=action,
                        time_ms=meta.get("time_used_ms",0),sims=meta.get("simulations",0),
                        root_entropy=meta.get("root_entropy")))
                applied=False
                if hasattr(game, "apply_engine_meta"):
                    applied=bool(game.apply_engine_meta(action, meta))
                if not applied:
                    game.apply_move(action)
                ply+=1
            elapsed=(time.time()-t0)*1000
            if game.is_terminal():
                if hasattr(game, "is_void_result") and game.is_void_result():
                    return GameRecord(
                        game_id=game_id,
                        engine_black=eng_black.name(),
                        engine_white=eng_white.name(),
                        outcome="void",
                        score_black=None,
                        move_count=ply,
                        total_time_ms=elapsed,
                        moves=moves,
                        opening=opening_applied,
                        seed=game_seed,
                        is_void=True,
                    )
                r=game.outcome_for_black()
                if r is None or r==0: outcome,sb="draw",0.5
                elif r>0: outcome,sb="black_win",1.0
                else: outcome,sb="white_win",0.0
            else: outcome,sb="draw",0.5
            return GameRecord(game_id=game_id,engine_black=eng_black.name(),engine_white=eng_white.name(),
                outcome=outcome,score_black=sb,move_count=ply,total_time_ms=elapsed,
                moves=moves,opening=opening_applied,seed=game_seed)
        except Exception as e:
            return GameRecord(game_id=game_id,engine_black=eng_black.name(),engine_white=eng_white.name(),
                outcome="void",score_black=None,move_count=ply,total_time_ms=(time.time()-t0)*1000,
                moves=moves,opening=opening_applied,seed=game_seed,error=str(e),is_void=True)
    def play_match(self, eng_a, eng_b, num_games, color_swap=True):
        records=[]; ob_n=len(self.opening_book) if self.opening_book else 0; idx=0
        pairs=num_games//2 if color_swap else 0
        for i in range(pairs):
            oi=i%ob_n if ob_n else None
            records.append(self.play_game(eng_a,eng_b,f"g{idx:04d}",oi)); idx+=1
            records.append(self.play_game(eng_b,eng_a,f"g{idx:04d}",oi)); idx+=1
        for _ in range(num_games-2*pairs):
            oi=idx%ob_n if ob_n else None
            records.append(self.play_game(eng_a,eng_b,f"g{idx:04d}",oi)); idx+=1
        return records
    def play_match_tally_range(self, eng_a, eng_b, pair_start, pair_count, single_start, single_count, logger=None):
        engine_name=eng_a.name()
        tally=MatchTally(engine_name=engine_name,opponent_name=eng_b.name())
        ob_n=len(self.opening_book) if self.opening_book else 0
        for i in range(pair_start, pair_start + pair_count):
            oi=i%ob_n if ob_n else None
            idx=2*i
            rec=self.play_game(eng_a,eng_b,f"g{idx:04d}",oi,collect_moves=False)
            _accumulate_tally_record(tally, rec, engine_name)
            if logger is not None: logger.log(rec)
            rec=self.play_game(eng_b,eng_a,f"g{idx+1:04d}",oi,collect_moves=False)
            _accumulate_tally_record(tally, rec, engine_name)
            if logger is not None: logger.log(rec)
        for idx in range(single_start, single_start + single_count):
            oi=idx%ob_n if ob_n else None
            rec=self.play_game(eng_a,eng_b,f"g{idx:04d}",oi,collect_moves=False)
            _accumulate_tally_record(tally, rec, engine_name)
            if logger is not None: logger.log(rec)
        return tally
    def play_match_tally(self, eng_a, eng_b, num_games, color_swap=True, logger=None):
        pairs=num_games//2 if color_swap else 0
        single_start=2*pairs
        single_count=num_games-single_start
        return self.play_match_tally_range(eng_a, eng_b, 0, pairs, single_start, single_count, logger=logger)

    def play_match_tally_batched(self, eng_a, eng_b, num_games, color_swap=True, logger=None):
        if hasattr(eng_a, "play_match_tally_against"):
            return eng_a.play_match_tally_against(
                eng_b,
                self.game_factory,
                self.opening_book,
                num_games,
                color_swap=color_swap,
                logger=logger,
                max_moves=self.max_moves,
                seed=self.rng.randint(0, 2**31),
            )
        if not hasattr(eng_a, "select_moves_batch") or not hasattr(eng_b, "select_moves_batch"):
            raise TypeError("batched tally requires engines with select_moves_batch(states)")

        eng_a.reset()
        eng_b.reset()
        engine_name = eng_a.name()
        tally = MatchTally(engine_name=engine_name, opponent_name=eng_b.name())
        ob_n = len(self.opening_book) if self.opening_book else 0
        sessions = []

        def _create_game_record(game_id, eng_black, eng_white, game, opening_applied, game_seed, total_time_ms, move_count, error=None):
            if game.is_terminal():
                if hasattr(game, "is_void_result") and game.is_void_result():
                    return GameRecord(
                        game_id=game_id,
                        engine_black=eng_black.name(),
                        engine_white=eng_white.name(),
                        outcome="void",
                        score_black=None,
                        move_count=move_count,
                        total_time_ms=total_time_ms,
                        moves=[],
                        opening=opening_applied,
                        seed=game_seed,
                        is_void=True,
                        error=error,
                    )
                r = game.outcome_for_black()
                if r is None or r == 0:
                    outcome, sb = "draw", 0.5
                elif r > 0:
                    outcome, sb = "black_win", 1.0
                else:
                    outcome, sb = "white_win", 0.0
            else:
                outcome, sb = "draw", 0.5
            return GameRecord(
                game_id=game_id,
                engine_black=eng_black.name(),
                engine_white=eng_white.name(),
                outcome=outcome,
                score_black=sb,
                move_count=move_count,
                total_time_ms=total_time_ms,
                moves=[],
                opening=opening_applied,
                seed=game_seed,
                error=error,
                is_void=bool(error),
            )

        def _append_session(eng_black, eng_white, game_id, opening_idx=None):
            game = self.game_factory()
            opening_applied = []
            game_seed = self.rng.randint(0, 2**31)
            if opening_idx is not None and opening_idx < len(self.opening_book):
                for action in self.opening_book[opening_idx]:
                    if game.is_terminal() or action not in game.legal_moves():
                        break
                    game.apply_move(action)
                    opening_applied.append(action)
            sessions.append({
                "game_id": game_id,
                "game": game,
                "eng_black": eng_black,
                "eng_white": eng_white,
                "opening": opening_applied,
                "seed": game_seed,
                "ply": len(opening_applied),
                "total_time_ms": 0.0,
                "done": bool(game.is_terminal()),
                "error": None,
            })

        pairs = num_games // 2 if color_swap else 0
        for i in range(pairs):
            opening_idx = i % ob_n if ob_n else None
            _append_session(eng_a, eng_b, f"g{2*i:04d}", opening_idx)
            _append_session(eng_b, eng_a, f"g{2*i+1:04d}", opening_idx)
        for idx in range(2 * pairs, num_games):
            opening_idx = idx % ob_n if ob_n else None
            _append_session(eng_a, eng_b, f"g{idx:04d}", opening_idx)

        while True:
            active = []
            for sess in sessions:
                game = sess["game"]
                if sess["done"]:
                    continue
                if game.is_terminal() or sess["ply"] >= self.max_moves:
                    sess["done"] = True
                    continue
                active.append(sess)
            if not active:
                break

            grouped = {id(eng_a): [], id(eng_b): []}
            engines = {id(eng_a): eng_a, id(eng_b): eng_b}
            for sess in active:
                game = sess["game"]
                mover = sess["eng_black"] if game.current_player() == 0 else sess["eng_white"]
                grouped[id(mover)].append(sess)

            progressed = False
            for engine_key, batch in grouped.items():
                if not batch:
                    continue
                progressed = True
                engine = engines[engine_key]
                states = [sess["game"] for sess in batch]
                t0 = time.time()
                try:
                    results = engine.select_moves_batch(states)
                except Exception as exc:
                    for sess in batch:
                        sess["error"] = str(exc)
                        sess["done"] = True
                    continue
                batch_elapsed_ms = max(0.0, (time.time() - t0) * 1000.0)
                share_ms = batch_elapsed_ms / max(1, len(batch))
                if len(results) != len(batch):
                    for sess in batch:
                        sess["error"] = f"batch result length mismatch: expected {len(batch)} got {len(results)}"
                        sess["done"] = True
                    continue
                for sess, (action, meta) in zip(batch, results):
                    game = sess["game"]
                    move_time_ms = float(meta.get("time_used_ms", 0.0) or 0.0)
                    sess["total_time_ms"] += move_time_ms if move_time_ms > 0.0 else share_ms
                    applied = False
                    if hasattr(game, "apply_engine_meta"):
                        applied = bool(game.apply_engine_meta(action, meta))
                    if not applied:
                        game.apply_move(action)
                    sess["ply"] += 1
                    if game.is_terminal() or sess["ply"] >= self.max_moves:
                        sess["done"] = True
            if not progressed:
                break

        for sess in sessions:
            game = sess["game"]
            played = sess["ply"]
            rec = _create_game_record(
                sess["game_id"],
                sess["eng_black"],
                sess["eng_white"],
                game,
                sess["opening"],
                sess["seed"],
                sess["total_time_ms"],
                played,
                error=sess["error"],
            )
            _accumulate_tally_record(tally, rec, engine_name)
            if logger is not None:
                logger.log(rec)
        return tally


# §6 Match Tallying

@dataclass
class MatchTally:
    engine_name:str; opponent_name:str; wins:int=0; draws:int=0; losses:int=0; errors:int=0; voids:int=0; total:int=0
    @property
    def scored(self): return self.wins+self.draws+self.losses
    @property
    def score_rate(self):
        n=self.scored; return (self.wins+0.5*self.draws)/n if n>0 else 0.0
    @property
    def win_rate(self):
        n=self.scored; return self.wins/n if n>0 else 0.0

def _accumulate_tally_record(tally, record, engine_name):
    tally.total += 1
    if record.is_void:
        if record.error:
            tally.errors += 1
        else:
            tally.voids += 1
        return
    is_black = record.engine_black == engine_name
    if not tally.opponent_name:
        tally.opponent_name = record.engine_white if is_black else record.engine_black
    if record.outcome == "draw":
        tally.draws += 1
    elif record.outcome == "black_win":
        if is_black: tally.wins += 1
        else: tally.losses += 1
    elif record.outcome == "white_win":
        if is_black: tally.losses += 1
        else: tally.wins += 1

def tally_match(records, engine_name):
    t=MatchTally(engine_name=engine_name,opponent_name="")
    for r in records:
        _accumulate_tally_record(t, r, engine_name)
    return t


# §7 Layer 1 — Promotion Gate

class PromotionVerdict(Enum):
    PROMOTE="promote"; REJECT="reject"; NEED_MORE="need_more"

@dataclass
class PromotionResult:
    verdict:PromotionVerdict; score_rate:float; score_rate_ci:Tuple[float,float]
    wilson_lower:float; p_value:float; is_significant:bool; games_scored:int; reason:str
    def to_dict(self):
        return {"verdict":self.verdict.value,"score_rate":round(self.score_rate,4),
                "score_rate_ci":tuple(round(x,4) for x in self.score_rate_ci),
                "wilson_lower":round(self.wilson_lower,4),"p_value":round(self.p_value,6),
                "is_significant":self.is_significant,"games_scored":self.games_scored,"reason":self.reason}

@dataclass
class PromotionConfig:
    threshold:float=0.55; min_games:int=200; confidence:float=0.95; require_significance:bool=True

class PromotionGate:
    def __init__(self, config=None): self.cfg=config or PromotionConfig()
    def evaluate(self, tally):
        n=tally.scored; z=Z_TABLE.get(self.cfg.confidence,1.96)
        sr,sr_ci=score_rate_ci(tally.wins,tally.draws,n,z)
        scored=tally.wins+0.5*tally.draws
        wl=wilson_ci(scored,n,z)[0]; pv=binomial_p_value(scored,n,0.5)
        sig=n>=self.cfg.min_games and pv<(1-self.cfg.confidence)
        if n<self.cfg.min_games: verdict=PromotionVerdict.NEED_MORE; reason=f"Only {n}/{self.cfg.min_games} scored games"
        elif sr>=self.cfg.threshold:
            if self.cfg.require_significance and not sig: verdict=PromotionVerdict.NEED_MORE; reason=f"sr={sr:.3f}≥{self.cfg.threshold} but p={pv:.4f} not sig"
            else: verdict=PromotionVerdict.PROMOTE; reason=f"sr={sr:.3f}≥{self.cfg.threshold}"
        else: verdict=PromotionVerdict.REJECT; reason=f"sr={sr:.3f}<{self.cfg.threshold}"
        return PromotionResult(verdict=verdict,score_rate=sr,score_rate_ci=sr_ci,wilson_lower=wl,
                               p_value=pv,is_significant=sig,games_scored=n,reason=reason)


# §8 Layer 2 — Sanity Check

@dataclass
class SanityResult:
    passed:bool; win_rate:float; games_played:int; reason:str
    def to_dict(self): return {"passed":self.passed,"win_rate":round(self.win_rate,3),"games":self.games_played,"reason":self.reason}

class SanityCheck:
    def __init__(self, min_score_rate=0.90, num_games=20):
        self.min_sr=min_score_rate; self.num_games=num_games
    def check(self, candidate, game_factory, seed=None):
        probe = game_factory()
        if getattr(probe, "supports_random_baseline", True) is False:
            return SanityResult(passed=True, win_rate=0.0, games_played=0,
                                reason="skipped: no local random baseline for this game")
        rand_eng=RandomEngine(seed=seed)
        runner=MatchRunner(game_factory,seed=seed,max_moves=300)
        tally=runner.play_match_tally(candidate,rand_eng,self.num_games,color_swap=True)
        sr=tally.score_rate
        ok=sr>=self.min_sr
        reason=f"sr vs random={sr:.3f} {'≥' if ok else '<'} {self.min_sr}"
        return SanityResult(passed=ok,win_rate=sr,games_played=tally.scored,reason=reason)


# §9 Champion Tracker

@dataclass
class ChampionState:
    model_id:str; generation:int=0; elo:Optional[float]=None
    promotion_history:List[Dict[str,Any]]=field(default_factory=list)

class ChampionTracker:
    def __init__(self, initial_id="gen_0", save_path=None, bridge_size=3):
        self.champion=ChampionState(model_id=initial_id,generation=0)
        self.save_path=save_path; self.bridge_size=bridge_size; self.bridge=[]
    @classmethod
    def load(cls, path, bridge_size=3):
        with open(path) as f: data=json.load(f)
        tracker=cls(initial_id=data.get("champion","gen_0"),save_path=path,bridge_size=bridge_size)
        tracker.champion=ChampionState(model_id=data.get("champion","gen_0"),
            generation=data.get("generation",0),elo=data.get("elo"),
            promotion_history=data.get("history",[]))
        tracker.bridge=list(data.get("bridge",[]))
        return tracker
    def try_promote(self, candidate_id, generation, result, published_elo=None):
        if result.verdict!=PromotionVerdict.PROMOTE: return False
        old_id=self.champion.model_id
        if old_id not in self.bridge: self.bridge.append(old_id)
        while len(self.bridge)>self.bridge_size: self.bridge.pop(0)
        entry={"from":old_id,"to":candidate_id,"generation":generation,
               "score_rate":result.score_rate,"games":result.games_scored,
               "timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
        self.champion=ChampionState(model_id=candidate_id,generation=generation,
            elo=published_elo,promotion_history=self.champion.promotion_history+[entry])
        if self.save_path: self._save()
        return True
    def get_bridge_ids(self): return list(self.bridge)
    def _save(self):
        self.save_path.parent.mkdir(parents=True,exist_ok=True)
        with open(self.save_path,"w") as f:
            json.dump({"champion":self.champion.model_id,"generation":self.champion.generation,
                       "elo":self.champion.elo,"history":self.champion.promotion_history,
                       "bridge":self.bridge},f,indent=2)


# §10 JSONL Logger

class MatchLogger:
    def __init__(self, path): self.path=path; self._f=None
    def __enter__(self): self.path.parent.mkdir(parents=True,exist_ok=True); self._f=open(self.path,"a"); return self
    def __exit__(self,*a):
        if self._f: self._f.close()
    def log(self, rec): self._f.write(rec.to_jsonl()+"\n"); self._f.flush()
    def log_lines(self, lines):
        if lines:
            self._f.write("".join(f"{line}\n" for line in lines))
            self._f.flush()


# §11 Training Evaluator

@dataclass
class EvalConfig:
    num_games:int=200; promotion_threshold:float=0.55; confidence:float=0.95
    sanity_check_interval:int=5; sanity_min_score:float=0.90; sanity_games:int=20
    anchor_match_games:int=20; bridge_match_interval:int=10; bridge_size:int=3
    color_swap:bool=True; max_moves:int=500; seed:Optional[int]=None
    log_path:Optional[str]=None; ladder_path:Optional[str]=None; champion_path:Optional[str]=None
    parallel_workers:int=1

@dataclass
class EvalResult:
    eval_id:str; timestamp:str; candidate:str; champion:str
    valid_eval:bool=True; invalid_reason:Optional[str]=None
    promotion:Dict[str,Any]=field(default_factory=dict)
    tally:Optional[Dict[str,Any]]=None; sanity:Optional[Dict[str,Any]]=None
    elo:Optional[Dict[str,Any]]=None; published:Optional[Dict[str,Any]]=None
    duration_s:float=0.0; conditions:Dict[str,Any]=field(default_factory=dict)
    def to_dict(self): return {k:v for k,v in self.__dict__.items() if v is not None}

class TrainingEvaluator:
    def __init__(self, config=None, manifest=None):
        self.cfg=config or EvalConfig()
        self.gate=PromotionGate(PromotionConfig(threshold=self.cfg.promotion_threshold,
            min_games=self.cfg.num_games,confidence=self.cfg.confidence))
        self.sanity=SanityCheck(min_score_rate=self.cfg.sanity_min_score,num_games=self.cfg.sanity_games)
        self.ladder=RatingLadder()
        if self.cfg.ladder_path and Path(self.cfg.ladder_path).exists():
            self.ladder.load(Path(self.cfg.ladder_path))
        self.manifest=manifest or AnchorManifest.default_gomoku7()
        self.calibrator=ScaleCalibrator(self.manifest)
        if self.cfg.champion_path and Path(self.cfg.champion_path).exists():
            self.champion=ChampionTracker.load(Path(self.cfg.champion_path),bridge_size=self.cfg.bridge_size)
        else:
            self.champion=ChampionTracker(save_path=Path(self.cfg.champion_path) if self.cfg.champion_path else None,
                                           bridge_size=self.cfg.bridge_size)
        self._eval_count=0
    def _build_match_chunks(self, num_games, color_swap, workers):
        workers=max(1, workers)
        if color_swap:
            pairs=num_games//2
            singles=num_games-(2*pairs)
            if pairs <= 0:
                return [(0,0,0,num_games)]
            worker_count=min(workers, pairs)
            base, extra = divmod(pairs, worker_count)
            chunks=[]; pair_start=0
            for wi in range(worker_count):
                pair_count=base + (1 if wi < extra else 0)
                chunks.append((pair_start, pair_count, 0, 0))
                pair_start += pair_count
            if singles:
                ps, pc, _, _ = chunks[-1]
                chunks[-1] = (ps, pc, 2*pairs, singles)
            return chunks
        worker_count=min(workers, num_games) if num_games > 0 else 1
        base, extra = divmod(num_games, worker_count)
        chunks=[]; single_start=0
        for wi in range(worker_count):
            single_count=base + (1 if wi < extra else 0)
            chunks.append((0, 0, single_start, single_count))
            single_start += single_count
        return chunks
    def _merge_tallies(self, tallies, engine_name, opponent_name):
        merged=MatchTally(engine_name=engine_name, opponent_name=opponent_name)
        for tally in tallies:
            merged.wins += tally.wins
            merged.draws += tally.draws
            merged.losses += tally.losses
            merged.errors += tally.errors
            merged.voids += tally.voids
            merged.total += tally.total
        return merged
    def _run_parallel_chunk(self, candidate_factory, champion_factory, game_factory, opening_book, chunk, collect_logs):
        pair_start, pair_count, single_start, single_count = chunk
        candidate = candidate_factory()
        champion = champion_factory()
        runner = MatchRunner(game_factory, opening_book, self.cfg.seed, self.cfg.max_moves)
        lines = []
        logger = None
        if collect_logs:
            class _ListLogger:
                def __init__(self): self.lines=[]
                def log(self, rec): self.lines.append(rec.to_jsonl())
            logger = _ListLogger()
        try:
            tally = runner.play_match_tally_range(
                candidate, champion, pair_start, pair_count, single_start, single_count, logger=logger)
            if logger is not None:
                lines = logger.lines
            return tally, lines
        finally:
            try: candidate.reset()
            except Exception: pass
            try: champion.reset()
            except Exception: pass

    def evaluate_checkpoint(self, candidate, champion, game_factory,
                            candidate_id="", generation=0, opening_book=None,
                            candidate_factory=None, champion_factory=None):
        t0=time.time(); cfg=self.cfg; self._eval_count+=1
        candidate_key=candidate_id or candidate.name()
        champion_key=self.champion.champion.model_id
        use_batched = (
            hasattr(candidate, "select_moves_batch") and hasattr(champion, "select_moves_batch")
        )
        use_parallel = (
            cfg.parallel_workers > 1 and candidate_factory is not None and champion_factory is not None
        )
        if use_batched:
            runner = MatchRunner(game_factory, opening_book, cfg.seed, cfg.max_moves)
            if cfg.log_path:
                with MatchLogger(Path(cfg.log_path)) as ml:
                    tally = runner.play_match_tally_batched(
                        candidate, champion, cfg.num_games, cfg.color_swap, logger=ml)
            else:
                tally = runner.play_match_tally_batched(candidate, champion, cfg.num_games, cfg.color_swap)
        elif use_parallel:
            chunks = self._build_match_chunks(cfg.num_games, cfg.color_swap, cfg.parallel_workers)
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                parts = [
                    pool.submit(
                        self._run_parallel_chunk,
                        candidate_factory,
                        champion_factory,
                        game_factory,
                        opening_book,
                        chunk,
                        bool(cfg.log_path),
                    )
                    for chunk in chunks
                ]
                chunk_results = [future.result() for future in parts]
            tally = self._merge_tallies(
                [chunk_tally for chunk_tally, _ in chunk_results],
                candidate.name(),
                champion.name(),
            )
            if cfg.log_path:
                with MatchLogger(Path(cfg.log_path)) as ml:
                    for _, lines in chunk_results:
                        ml.log_lines(lines)
        else:
            runner=MatchRunner(game_factory,opening_book,cfg.seed,cfg.max_moves)
            if cfg.log_path:
                with MatchLogger(Path(cfg.log_path)) as ml:
                    tally=runner.play_match_tally(candidate,champion,cfg.num_games,cfg.color_swap,logger=ml)
            else:
                tally=runner.play_match_tally(candidate,champion,cfg.num_games,cfg.color_swap)
        promo=self.gate.evaluate(tally)
        result=EvalResult(eval_id=str(uuid.uuid4()),timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            candidate=candidate_key,champion=champion_key,
            conditions={"num_games":cfg.num_games,"color_swap":cfg.color_swap,"seed":cfg.seed,"generation":generation})
        result.tally={"wins":tally.wins,"draws":tally.draws,"losses":tally.losses,
                      "errors":tally.errors,"voids":tally.voids,"total":tally.total,
                      "scored":tally.scored,"score_rate":round(tally.score_rate,4)}
        result.promotion=promo.to_dict()
        if tally.scored <= 0:
            result.valid_eval = False
            result.invalid_reason = (
                f"zero scored games (errors={tally.errors}, voids={tally.voids}, total={tally.total})"
            )
            result.duration_s=time.time()-t0
            return result
        # Sanity check
        if self._eval_count%cfg.sanity_check_interval==0:
            sr=self.sanity.check(candidate,game_factory,cfg.seed)
            result.sanity=sr.to_dict()
        # Ladder update
        self.ladder.advance_period()
        self.ladder.record_match(candidate_key,champion_key,tally.wins,tally.draws,tally.losses)
        cand_rec=self.ladder.get(candidate_key); champ_rec=self.ladder.get(champion_key)
        a,b=self.calibrator.calibrate(self.ladder)
        cand_pub_raw=round(self.calibrator.published_elo(cand_rec.mu_elo),1)
        champ_pub=round(self.calibrator.published_elo(champ_rec.mu_elo),1)
        cand_pub=max(cand_pub_raw,champ_pub) if promo.verdict==PromotionVerdict.PROMOTE else cand_pub_raw
        match_delta=estimate_match_delta_elo(tally.wins,tally.draws,tally.losses)
        ladder_delta=cand_rec.mu_elo-champ_rec.mu_elo
        result.elo={"candidate":round(cand_rec.mu_elo,1),"champion":round(champ_rec.mu_elo,1),
                     "delta":round(match_delta,1),"ladder_delta":round(ladder_delta,1)}
        result.published={"candidate_abs":cand_pub,"candidate_abs_raw":cand_pub_raw,
                          "champion_abs":champ_pub,"delta":round(cand_pub-champ_pub,1),
                          "scale_a":round(a,4),"offset_b":round(b,1)}
        if promo.verdict==PromotionVerdict.PROMOTE:
            self.champion.try_promote(candidate_key,generation,promo,published_elo=cand_pub)
        else:
            self.champion.champion.elo=champ_pub
            if self.champion.save_path: self.champion._save()
        if cfg.ladder_path: self.ladder.save(Path(cfg.ladder_path))
        result.duration_s=time.time()-t0
        return result


# §12 Self-Tests

def _run_all():
    # Glicko-2
    terms=[]
    for mu_j,phi_j,s_j in [(-0.5756,0.1727,1.0),(0.2878,0.5756,0.0),(1.1513,1.7269,0.0)]:
        terms.append((Glicko2.g(phi_j),Glicko2.E(0.0,mu_j,phi_j),s_j))
    mu2,phi2,sig2=Glicko2.update(0.0,1.1513,0.06,terms,0.5)
    assert abs(Glicko2.g(1.7269)-0.7242)<0.002; assert abs(mu2-(-0.2069))<0.02
    assert abs(phi2-0.8722)<0.02; assert abs(sig2-0.05999)<0.001
    print("[PASS] Glicko-2 math")
    # Promotion
    gate=PromotionGate(PromotionConfig(threshold=0.55,min_games=10,require_significance=False))
    t=MatchTally("c","b",wins=7,draws=2,losses=1); r=gate.evaluate(t)
    assert r.verdict==PromotionVerdict.PROMOTE
    t2=MatchTally("c","b",wins=4,draws=2,losses=4); r2=gate.evaluate(t2)
    assert r2.verdict==PromotionVerdict.REJECT
    print("[PASS] Promotion gate")
    # Champion tracker
    ct=ChampionTracker(initial_id="gen_0",bridge_size=2)
    promote=PromotionResult(PromotionVerdict.PROMOTE,0.65,(0.55,0.75),0.55,0.01,True,200,"ok")
    ct.try_promote("gen_5",5,promote); assert ct.champion.model_id=="gen_5"
    ct.try_promote("gen_10",10,promote); assert ct.bridge==["gen_0","gen_5"]
    ct.try_promote("gen_15",15,promote); assert ct.bridge==["gen_5","gen_10"]
    print("[PASS] Champion tracker + bridge")
    # Wilson CI
    sr,ci=score_rate_ci(40,20,100); assert abs(sr-0.50)<0.001
    print("[PASS] Score rate CI")
    # RD inflation
    ladder=RatingLadder(); ladder.record_match("m","random",10,0,0)
    rd0=ladder.get("m").rd_elo; ladder.advance_period()
    assert ladder.get("m").rd_elo>rd0
    print("[PASS] RD inflation")
    # Batch vs sequential
    ls=RatingLadder(); ls.record_match("A","B",8,1,1); ls.record_match("A","C",1,1,8)
    lb=RatingLadder(); lb.update_from_period([("A","B",8,1,1),("A","C",1,1,8)])
    assert abs(ls.get("A").mu_elo-lb.get("A").mu_elo)>0.5
    print("[PASS] Batch vs sequential")
    # Scale calibrator
    mf=AnchorManifest(anchors=[AnchorEntry(id="A0",engine_hash="r",target_elo=0.0),
                                AnchorEntry(id="A1",engine_hash="u",target_elo=400.0)])
    ld=RatingLadder(); ld.record_match("A0","A1",2,1,7)
    cal=ScaleCalibrator(mf); a,b=cal.calibrate(ld)
    assert abs(cal.published_elo(ld.get("A0").mu_elo))<50
    assert abs(cal.published_elo(ld.get("A1").mu_elo)-400)<50
    print(f"[PASS] Scale calibrator (a={a:.3f}, b={b:.1f})")
    print(f"\n[ALL PASS] 7 tests passed.")

if __name__=="__main__":
    import sys,argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: _run_all(); sys.exit(0)
    print("Use TrainingEvaluator programmatically. --self-test for verification.")
