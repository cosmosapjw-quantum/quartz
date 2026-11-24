//! Explicit research-only native pool using existing neural evaluator and parallel resident MCTS.
use super::*;
use serde::{Deserialize,Serialize};
use std::collections::HashSet;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job { id:String, history:String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    cmd:String, mode:String, seed:u64, model_path:String, model_identity:String,
    jobs:Vec<Job>, #[serde(default)] budget_us:u64, #[serde(default)] floor_us:u64,
    #[serde(default)] cells:Vec<[f64;2]>,
}
#[derive(Serialize,Clone)]
struct Step { visits:u32, best:usize, policy:Vec<(usize,f32)>, eval_calls:u64, wall_us:u64, root_hash:String }
#[derive(Serialize)]
struct Row { id:String, history:String, legal:Vec<usize>, root_hash:String, cell:usize, entropy:f64, top:f64, threat:bool, setup_us:u64, steps:Vec<Step> }
fn us(t:Instant)->u64 {t.elapsed().as_micros().max(1) as u64}
fn win_at(board:&[i8;42],col:usize,height:usize,p:i8)->bool {
    for (dx,dy) in [(1i32,0i32),(0,1),(1,1),(1,-1)] {
        let mut n=1;
        for sign in [-1,1] {
            let mut x=col as i32+dx*sign; let mut y=height as i32+dy*sign;
            while (0..7).contains(&x) && (0..6).contains(&y) && board[y as usize*7+x as usize]==p {
                n+=1; x+=dx*sign;y+=dy*sign;
            }
        }
        if n>=4{return true;}
    }
    false
}
fn threat(game:&Connect4)->bool {
    let b=game.board_flat();let h=game.heights();let p=-game.current_player();
    game.legal_moves().into_iter().any(|c|win_at(&b,c,h[c] as usize,p))
}
fn cell(e:f64,m:f64,t:bool,l:usize)->usize {
    ((e>=0.55) as usize) | (((m>=0.75) as usize)<<1) | ((t as usize)<<2) | (((l<=4) as usize)<<3)
}
fn advance(engine:&MctsEngine<Connect4>,n:u32)->Result<Step,String> {
    let t=Instant::now();
    let stat=engine.run_par(&FixedIterations::new(n),2);
    let wall_us=us(t);
    if stat.root_visits!=n {return Err(format!("visit mismatch {} != {}",stat.root_visits,n));}
    let best=engine.best_move().ok_or("no root decision")?;
    let policy=engine.pi_target(1.);
    if !engine.root_state().legal_moves().contains(&best) {return Err("illegal decision".into());}
    Ok(Step{visits:n,best,policy,eval_calls:engine.evaluator_call_count(),wall_us,root_hash:format!("{:016x}",engine.root_state().tt_hash())})
}
fn key(s:&str,seed:u64)->u64 {
    let mut z=seed^0xcbf29ce484222325;
    for b in s.bytes(){z=(z^b as u64).wrapping_mul(0x100000001b3);}
    z^=z>>30;z=z.wrapping_mul(0xbf58476d1ce4e5b9);z^=z>>27;z=z.wrapping_mul(0x94d049bb133111eb);z^(z>>31)
}
fn inner(line:&str)->Result<serde_json::Value,String> {
    let start=Instant::now();
    let q:Request=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_intervention_cpu" || !matches!(q.mode.as_str(),"GEN"|"TRACE"|"PREFIX"|"FIXED8"|"RANDOM16"|"ENTROPY16"|"SIGNED16")
        || q.jobs.is_empty() || q.jobs.len()>256 || q.budget_us>10_000_000 {return Err("bad mode, pool size or budget".into());}
    if q.mode=="SIGNED16" && (q.cells.len()!=16 || q.cells.iter().any(|r|!r[0].is_finite()||r[0].abs()>1.||!r[1].is_finite()||r[1]<=0.)) {return Err("invalid signed cell model".into());}
    if q.mode!="SIGNED16" && !q.cells.is_empty(){return Err("unexpected model cells".into());}
    let mut seen=HashSet::new();
    let (eval,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut engines=Vec::with_capacity(q.jobs.len());
    let mut rows=Vec::with_capacity(q.jobs.len());
    for j in q.jobs {
        if !seen.insert(j.id.clone()) {return Err("duplicate root ID".into());}
        let state=Connect4::from_history(&j.history)?;
        if state.is_terminal(){return Err("terminal root".into());}
        let legal=state.legal_moves();
        let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);
        cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
        let st=Instant::now();let engine=MctsEngine::new(state.clone(),eval.clone(),cfg);let setup_us=us(st);
        let prefix=advance(&engine,4)?;
        // Normalize by the full action alphabet, consistent with COG02.
        let z: f64=prefix.policy.iter().map(|a|a.1 as f64).sum();
        if !z.is_finite() || z<=0. {return Err("invalid visit distribution".into());}
        let entropy=prefix.policy.iter().map(|a|a.1 as f64/z).filter(|p|*p>0.).map(|p|-p*p.ln()).sum::<f64>()/7f64.ln();
        let top=prefix.policy.iter().map(|a|a.1 as f64/z).fold(0.,f64::max);
        let t=threat(&state);
        let c=cell(entropy,top,t,legal.len());
        let root_hash=prefix.root_hash.clone();
        rows.push(Row{id:j.id,history:j.history,legal,root_hash,cell:c,entropy,top,threat:t,setup_us,steps:vec![prefix]});
        engines.push(engine);
    }
    let prefix_us=us(start);
    let optional=Instant::now();
    let mut order:Vec<usize>=(0..rows.len()).collect();
    match q.mode.as_str(){
        "FIXED8"=>order.sort_by(|&a,&b|rows[a].id.cmp(&rows[b].id)),
        "RANDOM16"=>order.sort_by_key(|&i|(key(&rows[i].id,q.seed),rows[i].id.clone())),
        "ENTROPY16"=>order.sort_by(|&a,&b|rows[b].entropy.total_cmp(&rows[a].entropy).then_with(||rows[a].id.cmp(&rows[b].id))),
        "SIGNED16"=>order.sort_by(|&a,&b|{let x=q.cells[rows[a].cell];let y=q.cells[rows[b].cell];(y[0]/y[1]).total_cmp(&(x[0]/x[1])).then_with(||rows[a].id.cmp(&rows[b].id))}),
        _=>{}
    }
    let mut selected=Vec::new();
    if !matches!(q.mode.as_str(),"PREFIX"|"GEN") {
        for i in order {
            if q.mode!="TRACE" && (optional.elapsed().as_micros() as u64).saturating_add(q.floor_us)>q.budget_us {break;}
            selected.push(rows[i].id.clone());
            let targets:&[u32]=if q.mode=="TRACE"{&[8,16,32]}else if q.mode=="FIXED8"{&[8]}else{&[8,16]};
            for &n in targets {
                let s=advance(&engines[i],n)?;
                if s.root_hash!=rows[i].root_hash || s.eval_calls<rows[i].steps.last().unwrap().eval_calls{return Err("resident invariant".into());}
                rows[i].steps.push(s);
            }
        }
    }
    let optional_us=us(optional);
    drop(engines);
    let whole_us=us(start);
    Ok(serde_json::json!({"schema":"quartz_nn_intervention_v1","mode":q.mode,"seed":q.seed,"threads":2,"model_identity":q.model_identity,
        "cache_hit":hit,"prefix_us":prefix_us,"optional_us":optional_us,"whole_us":whole_us,"budget_us":q.budget_us,"floor_us":q.floor_us,
        "overrun_us":optional_us.saturating_sub(q.budget_us),"selected":selected,"rows":rows}))
}
pub(super) fn handle(line:&str)->serde_json::Value {
    match inner(line){Ok(v)=>v,Err(e)=>serde_json::json!({"error":e})}
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn cell_and_threat(){
        assert_eq!(cell(.55,.75,true,4),15);assert_eq!(cell(.549,.749,false,5),0);
        let g=Connect4::from_history("324252").unwrap();
        assert!(threat(&g));
    }
}
