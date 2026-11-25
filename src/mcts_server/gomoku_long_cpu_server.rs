//! Long-game collection research seam; strict ordered history, real NN callback.
//! Exact proof only for unrestricted freestyle and exact-five Standard.
use super::*;
use crate::games::gomoku15::{Gomoku15,GomokuVariant};
use crate::mcts::four_axis::Hooks;
use serde::Deserialize;
use serde_json::{json,Value};
#[path="gomoku_long_proof.rs"]
mod proof;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{
    cmd:String,rule:String,history:Vec<u16>,
    #[serde(default)]arm:Option<String>,#[serde(default)]visits:Option<u32>,
    #[serde(default)]threads:Option<usize>,#[serde(default)]seed:Option<u64>,
}
fn micros(t:Instant)->u64{t.elapsed().as_micros() as u64}
fn rule(s:&str)->Option<(GomokuVariant,u8)>{
    match s{
        "freestyle"=>Some((GomokuVariant::Freestyle,0)),
        "standard"=>Some((GomokuVariant::Standard,1)),
        "omok"=>Some((GomokuVariant::Omok,2)),
        "renju"=>Some((GomokuVariant::Renju,3)),
        "caro"=>Some((GomokuVariant::Caro,4)),
        _=>None
    }
}
fn proof_json(p:&proof::Proof)->Value{
    json!({"kind":"rule_two_ply_v1","own":p.own,"opponent":p.opponent,
      "actions":p.bounds.iter().map(|b|json!([b.action,b.lo,b.hi,b.reply])).collect::<Vec<_>>(),
      "interval":p.interval,"witness":p.witness,"risk2":p.risk2})
}
fn inner(line:&str)->Result<Value,String>{
    let started=Instant::now();
    let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    let (variant,rid)=rule(&q.rule).ok_or("explicit rule required")?;
    if q.history.len()>225{return Err("history too long".into());}
    let inspect=q.cmd=="research_gomoku15_long_inspect";
    if !inspect&&q.cmd!="research_gomoku15_long_search"{return Err("bad command".into());}
    if inspect&&(q.arm.is_some()||q.visits.is_some()||q.threads.is_some()||q.seed.is_some()){return Err("inspect has no search options".into());}
    let mut state=Gomoku15::new(variant).with_research_history_tt();
    for &a in &q.history{
        if a>=225||state.is_terminal()||!state.legal_moves().contains(&a){return Err("illegal, duplicate or post-terminal history".into());}
        state=state.apply_move(a);
    }
    let replay_us=micros(started);
    let legal=state.legal_moves();let terminal=state.is_terminal();let side=state.current_player();
    let k=proof::key(state.board_as_12(),side,rid);
    let t=Instant::now();
    let p=if rid<=1&&!terminal{
        let p=proof::scan(state.board_as_12(),if side==1{1}else{2},rid==1);
        assert_eq!(p.bounds.iter().map(|b|b.action).collect::<Vec<_>>(),legal);
        // Crosscheck current-side winning candidates against actual game rules.
        assert_eq!(p.own,legal.iter().copied().filter(|&a|state.is_winning_move(a)).collect::<Vec<_>>());
        Some(p)
    }else{None};
    let proof_us=micros(t);
    let mut out=json!({"schema":"quartz_gomoku_long_v1","rule":q.rule,"history":q.history,
        "board":state.board_as_12().as_slice(),"player":side,
        "rule_key":k.iter().rev().map(|x|format!("{x:016x}")).collect::<String>(),
        "board_hash":format!("{:016x}",state.hash()),"tt_hash":format!("{:016x}",state.tt_hash()),
        "recent":state.research_recent_moves(),"legal":legal,"terminal":terminal,
        "outcome":if terminal{Some(state.outcome())}else{None},
        "proof":p.as_ref().map(proof_json),"replay_us":replay_us,"proof_us":proof_us});
    if inspect{
        out["features"]=json!(state.encode_planes());out["whole_us"]=json!(micros(started));return Ok(out);
    }
    if rid>1{return Err("rule is inspection-only; proof and collection not admitted".into());}
    if terminal||legal.is_empty(){return Err("search requires nonterminal state".into());}
    let arm=q.arm.as_deref().ok_or("arm required")?;
    let visits=q.visits.ok_or("visits required")?;
    let threads=q.threads.ok_or("threads required")?;
    let seed=q.seed.ok_or("seed required")?;
    if !matches!(arm,"BASE"|"CERT")||!matches!(threads,1|2)||visits<threads as u32||visits>4096||visits%threads as u32!=0{return Err("bad search budget or arm".into());}
    let proof=p.as_ref().unwrap();
    if arm=="CERT"{
        if let Some(w)=proof.witness{
            out["arm"]=json!(arm);out["requested_visits"]=json!(visits);out["threads"]=json!(threads);out["seed"]=json!(seed);
            out["best"]=json!(w);out["cause"]=json!("CERTIFIED");out["result"]=Value::Null;
            out["visits"]=json!(0);out["evaluator_calls"]=json!(0);out["setup_us"]=json!(0);
            out["search_us"]=json!(0);out["drop_us"]=json!(0);out["whole_us"]=json!(micros(started));return Ok(out);
        }
    }
    let t=Instant::now();
    let eval=make_eval::<Gomoku15>(SearchProfile::BaselineStrict,threads,4,1500,225,false);
    let mut cfg=apply_search_profile(gomoku15_quartz(variant),SearchProfile::BaselineStrict).with_seed(seed);
    cfg.dirichlet=None;cfg.temperature=0.;cfg.root_forced_win=false;
    // Preserve full candidate list; the original configured progressive widening
    // may materialize only a prefix. Both arms use exactly this same configuration.
    let engine=MctsEngine::new(state,eval,cfg);
    let setup_us=micros(t);
    let h=Hooks::new_modes(0,0,0,false);
    let t=Instant::now();engine.four_axis_block(visits,threads,&h);let search_us=micros(t);
    let result=engine.four_axis_snapshot();
    let normal=result["best"].as_u64().ok_or("no native best move")? as u16;
    assert_eq!(result["root_n"].as_u64(),Some(visits as u64));
    let best=if arm=="CERT"{
        let lo=proof.interval.0;
        if proof.bounds.iter().any(|b|b.action==normal&&b.hi>=lo){normal}
        else{
            let edges=result["edges"].as_array().unwrap();
            proof.bounds.iter().filter(|b|b.hi>=lo).max_by_key(|b|{
                (edges.iter().find(|e|e["move"].as_u64()==Some(b.action as u64)).and_then(|e|e["n"].as_u64()).unwrap_or(0),std::cmp::Reverse(b.action))
            }).unwrap().action
        }
    }else{normal};
    let evals=engine.evaluator_call_count();
    let t=Instant::now();drop(engine);drop(h);let drop_us=micros(t);
    out["arm"]=json!(arm);out["requested_visits"]=json!(visits);out["threads"]=json!(threads);out["seed"]=json!(seed);
    out["best"]=json!(best);out["cause"]=json!("VISITS");out["result"]=result;
    out["visits"]=json!(visits);out["evaluator_calls"]=json!(evals);
    out["setup_us"]=json!(setup_us);out["search_us"]=json!(search_us);out["drop_us"]=json!(drop_us);
    out["whole_us"]=json!(micros(started));Ok(out)
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}

// P3: same resident root, cumulative native search checkpoints. No changes to
// old long_search or ordinary production command. Oracle-free exact rule guard.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LadderReq{cmd:String,rule:String,history:Vec<u16>,arm:String,budgets:Vec<u32>,seed:u64}
fn ladder(line:&str)->Result<Value,String>{
    let start=Instant::now();
    let q:LadderReq=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_gomoku15_tactical_ladder"||!matches!(q.arm.as_str(),"BASE"|"EXPOSE"|"FOCUS"|"GUARD"|"CERT"){return Err("bad ladder command or arm".into());}
    let (variant,rid)=rule(&q.rule).ok_or("explicit rule required")?;
    if rid>1||q.history.len()>225{return Err("unsupported ladder rule/history".into());}
    if q.budgets.is_empty()||q.budgets.len()>6||q.budgets.iter().any(|n|![1,8,32,64,128,256].contains(n))||q.budgets.windows(2).any(|p|p[0]>=p[1]){return Err("bad ladder budgets".into());}
    if matches!(q.arm.as_str(),"GUARD"|"CERT")&&q.budgets!=[8]{return Err("proof output uses one 8-visit cap".into());}
    let mut state=Gomoku15::new(variant).with_research_history_tt();
    for &a in &q.history{
        if a>=225||state.is_terminal()||!state.legal_moves().contains(&a){return Err("illegal, duplicate or post-terminal history".into());}
        state=state.apply_move(a);
    }
    if state.is_terminal(){return Err("terminal ladder root".into());}
    let replay_us=micros(start);let side=state.current_player();let legal=state.legal_moves();
    let key=proof::key(state.board_as_12(),side,rid);
    let t=Instant::now();let p=proof::scan(state.board_as_12(),if side==1{1}else{2},rid==1);
    assert_eq!(p.bounds.iter().map(|b|b.action).collect::<Vec<_>>(),legal);
    assert_eq!(p.own,legal.iter().copied().filter(|&a|state.is_winning_move(a)).collect::<Vec<_>>());
    let targets=proof::target_moves(&p);let proof_us=micros(t);
    if matches!(q.arm.as_str(),"EXPOSE"|"FOCUS")&&targets.is_empty(){return Err("targeted arm requires tactical target".into());}
    let stop=match q.arm.as_str(){"GUARD"=>proof::guard(&p),"CERT"=>p.witness,_=>None};
    let mut out=json!({"schema":"quartz_gomoku_tactical_v1","rule":q.rule,"history":q.history,"arm":q.arm,"budgets":q.budgets,"seed":q.seed,"threads":1,
        "board":state.board_as_12().as_slice(),"player":side,"rule_key":key.iter().rev().map(|x|format!("{x:016x}")).collect::<String>(),
        "tt_hash":format!("{:016x}",state.tt_hash()),"board_hash":format!("{:016x}",state.hash()),"recent":state.research_recent_moves(),
        "legal":legal,"proof":proof_json(&p),"targets":targets,"replay_us":replay_us,"proof_us":proof_us});
    if let Some(w)=stop{
        out["cause"]=json!(if q.arm=="GUARD"{"GUARDED"}else{"CERTIFIED"});
        out["best"]=json!(w);out["initial"]=Value::Null;out["target_rank"]=Value::Null;out["target"]=Value::Null;
        out["root_floor"]=json!(0);out["forced_first"]=Value::Null;out["points"]=json!([{"budget":8,"visits":0,"best":w,"result":null,"step_us":0,"cumulative_us":micros(start)}]);
        out["visits"]=json!(0);out["evaluator_calls"]=json!(0);out["setup_us"]=json!(0);out["plan_us"]=json!(0);out["drop_us"]=json!(0);out["whole_us"]=json!(micros(start));return Ok(out);
    }
    let t=Instant::now();
    let eval=make_eval::<Gomoku15>(SearchProfile::BaselineStrict,1,4,1500,225,false);
    let mut cfg=apply_search_profile(gomoku15_quartz(variant),SearchProfile::BaselineStrict).with_seed(q.seed);
    cfg.dirichlet=None;cfg.temperature=0.;cfg.root_forced_win=false;
    let engine=MctsEngine::new(state,eval,cfg);let setup_us=micros(t);
    let t=Instant::now();
    let initial=engine.four_axis_snapshot();
    let target=initial["candidates"].as_array().unwrap().iter().enumerate().find_map(|(i,c)|{
        let m=c[0].as_u64().unwrap() as u16;
        if targets.contains(&m){Some((i+1,m))}else{None}
    });
    assert_eq!(target.is_some(),!targets.is_empty());
    let floor=if matches!(q.arm.as_str(),"EXPOSE"|"FOCUS"){target.unwrap().0}else{0};
    let first=if q.arm=="FOCUS"{Some(target.unwrap().1 as usize)}else{None};
    let h=Hooks::new_modes(0,0,0,false).with_tactical_root(floor,first);let plan_us=micros(t);
    let mut points=Vec::new();let mut prev=0;
    for &budget in &q.budgets{
        let t=Instant::now();engine.four_axis_block(budget-prev,1,&h);let step_us=micros(t);prev=budget;
        let result=engine.four_axis_snapshot();
        let best=result["best"].as_u64().ok_or("no native recommendation")? as u16;
        assert_eq!(result["root_n"].as_u64(),Some(budget as u64));
        assert_eq!(result["candidates"],initial["candidates"]);
        points.push(json!({"budget":budget,"visits":budget,"best":best,"result":result,"step_us":step_us,"cumulative_us":micros(start)}));
    }
    let best=points.last().unwrap()["best"].as_u64().unwrap();let evals=engine.evaluator_call_count();
    let t=Instant::now();drop(engine);drop(h);let drop_us=micros(t);
    out["cause"]=json!("VISITS");out["best"]=json!(best);out["initial"]=initial;out["target_rank"]=json!(target.map(|x|x.0));out["target"]=json!(target.map(|x|x.1));
    out["root_floor"]=json!(floor);out["forced_first"]=json!(first);out["points"]=json!(points);out["visits"]=json!(prev);out["evaluator_calls"]=json!(evals);
    out["setup_us"]=json!(setup_us);out["plan_us"]=json!(plan_us);out["drop_us"]=json!(drop_us);out["whole_us"]=json!(micros(start));Ok(out)
}
pub(super) fn handle_ladder(line:&str)->Value{match ladder(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
