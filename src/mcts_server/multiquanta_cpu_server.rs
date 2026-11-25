//! Explicit research-only native pool using existing neural evaluator and parallel resident MCTS.
use super::*;
use serde::{Deserialize,Serialize};
use std::collections::HashSet;
#[path="multiquanta_plan.rs"]
mod plan;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job { id:String, history:String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    cmd:String, mode:String, seed:u64, model_path:String, model_identity:String,
    jobs:Vec<Job>, #[serde(default)] budget_us:u64, #[serde(default)] floor_us:u64,
    #[serde(default)] cells:Vec<[[f64;2];3]>,
    #[serde(default)] floors:Vec<u64>,
}
#[derive(Serialize,Clone)]
struct Step { visits:u32, best:usize, policy:Vec<(usize,f32)>, eval_calls:u64, search_us:u64, wall_us:u64, root_hash:String }
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
    let search_us=us(t);
    if stat.root_visits!=n {return Err(format!("visit mismatch {} != {}",stat.root_visits,n));}
    let best=engine.best_move().ok_or("no root decision")?;
    let policy=engine.pi_target(1.);
    if !engine.root_state().legal_moves().contains(&best) {return Err("illegal decision".into());}
    Ok(Step{visits:n,best,policy,eval_calls:engine.evaluator_call_count(),search_us,wall_us:us(t),root_hash:format!("{:016x}",engine.root_state().tt_hash())})
}
fn key(s:&str,seed:u64)->u64 {
    let mut z=seed^0xcbf29ce484222325;
    for b in s.bytes(){z=(z^b as u64).wrapping_mul(0x100000001b3);}
    z^=z>>30;z=z.wrapping_mul(0xbf58476d1ce4e5b9);z^=z>>27;z=z.wrapping_mul(0x94d049bb133111eb);z^(z>>31)
}
fn inner(line:&str)->Result<serde_json::Value,String> {
    let start=Instant::now();
    let q:Request=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_multiquanta_cpu" || !matches!(q.mode.as_str(),"TRACE"|"SETUP"|"PREFIX"|"FIXED8"|"RANDOM16"|"ENTROPY16"|"RATIO"|"DP")
        || q.jobs.is_empty() || q.jobs.len()>8 || q.budget_us>100_000 {return Err("bad mode, pool size or budget".into());}
    let learned=matches!(q.mode.as_str(),"DP"|"RATIO");
    let adaptive=!matches!(q.mode.as_str(),"TRACE"|"PREFIX"|"SETUP");
    if adaptive && (q.floors.len()!=3 || q.floors.iter().any(|x|*x==0)){return Err("invalid calibrated floors".into());}
    if learned && (q.cells.len()!=16 || q.cells.iter().flatten().any(|r|!r[0].is_finite()||r[0].abs()>1.||!r[1].is_finite()||r[1]<=0.||r[1]>1e9)) {return Err("invalid signed cell model".into());}
    if !learned && !q.cells.is_empty(){return Err("unexpected model cells".into());}
    let mut seen=HashSet::new();
    let mt=Instant::now();let (eval,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let model_us=us(mt);
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
        let t=threat(&state);
        let root_hash=format!("{:016x}",engine.root_state().tt_hash());
        if q.mode=="SETUP" {
            rows.push(Row{id:j.id,history:j.history,legal,root_hash,cell:0,entropy:0.,top:0.,threat:t,setup_us,steps:vec![]});
        } else {
            let prefix=advance(&engine,4)?;
            let z:f64=prefix.policy.iter().map(|a|a.1 as f64).sum();
            if !z.is_finite()||z<=0.{return Err("invalid visit distribution".into());}
            let entropy=prefix.policy.iter().map(|a|a.1 as f64/z).filter(|p|*p>0.).map(|p|-p*p.ln()).sum::<f64>()/7f64.ln();
            let top=prefix.policy.iter().map(|a|a.1 as f64/z).fold(0.,f64::max);
            let c=cell(entropy,top,t,legal.len());
            rows.push(Row{id:j.id,history:j.history,legal,root_hash,cell:c,entropy,top,threat:t,setup_us,steps:vec![prefix]});
        }
        engines.push(engine);
    }
    let prefix_us=us(start);
    let setup_total_us=rows.iter().map(|r|r.setup_us).sum::<u64>();
    let prefix_step_us=rows.iter().flat_map(|r|r.steps.iter()).map(|s|s.wall_us).sum::<u64>();
    let optional=Instant::now();
    let mut order:Vec<usize>=(0..rows.len()).collect();
    let mut plans:Vec<(usize,usize)>=vec![];
    let mut predicted_ticks=0usize;let mut predicted_gain=0.;
    match q.mode.as_str(){
        "FIXED8"=>{order.sort_by(|&a,&b|rows[a].id.cmp(&rows[b].id));plans=order.into_iter().map(|i|(i,1)).collect();}
        "RANDOM16"=>{order.sort_by_key(|&i|(key(&rows[i].id,q.seed),rows[i].id.clone()));plans=order.into_iter().map(|i|(i,2)).collect();}
        "ENTROPY16"=>{order.sort_by(|&a,&b|rows[b].entropy.total_cmp(&rows[a].entropy).then_with(||rows[a].id.cmp(&rows[b].id)));plans=order.into_iter().map(|i|(i,2)).collect();}
        "TRACE"=>{plans=order.into_iter().map(|i|(i,3)).collect();}
        "RATIO"|"DP"=>{
            order.sort_by(|&a,&b|rows[a].id.cmp(&rows[b].id));
            let opts:Vec<plan::Options>=order.iter().map(|&i|{
                let mut a=[(0usize,0.);4];
                for j in 1..4{let [g,c]=q.cells[rows[i].cell][j-1];a[j]=((c/50.).ceil() as usize,g);}
                a
            }).collect();
            let k=(q.budget_us/50) as usize;
            let choices=if q.mode=="DP"{plan::dp(&opts,k)}else{plan::ratio(&opts,k)};
            for (i,&j) in choices.iter().enumerate(){if j!=0{
                predicted_ticks+=opts[i][j].0;predicted_gain+=opts[i][j].1;plans.push((order[i],j));
            }}
            if predicted_ticks>k{return Err("plan violates estimated capacity".into());}
        }
        _=>{}
    }
    let plan_us=us(optional);
    let plan_record:Vec<_>=plans.iter().map(|&(i,j)|(rows[i].id.clone(),[4,8,16,32][j])).collect();
    let mut selected=Vec::new();let mut actual_step_us=0u64;
    for (i,j) in plans{
        if adaptive && (optional.elapsed().as_micros() as u64).saturating_add(q.floors[j-1])>q.budget_us{break;}
        selected.push((rows[i].id.clone(),[4,8,16,32][j]));
        for &n in &[8,16,32][..j]{
            let s=advance(&engines[i],n)?;
            if s.root_hash!=rows[i].root_hash || s.eval_calls<rows[i].steps.last().unwrap().eval_calls{return Err("resident invariant".into());}
            actual_step_us+=s.wall_us;rows[i].steps.push(s);
        }
    }
    let optional_us=us(optional);let dt=Instant::now();
    drop(engines);let drop_us=us(dt);
    let whole_us=us(start);
    Ok(serde_json::json!({"schema":"quartz_nn_multiquanta_v1","mode":q.mode,"seed":q.seed,"threads":2,"model_identity":q.model_identity,
        "cache_hit":hit,"model_us":model_us,"prefix_us":prefix_us,"setup_total_us":setup_total_us,"prefix_step_us":prefix_step_us,
        "plan_us":plan_us,"continuation_step_us":actual_step_us,"optional_us":optional_us,"drop_us":drop_us,"whole_us":whole_us,"budget_us":q.budget_us,"floors":q.floors,
        "overrun_us":if adaptive{optional_us.saturating_sub(q.budget_us)}else{0},"planned":plan_record,"selected":selected,"predicted_ticks":predicted_ticks,"predicted_gain":predicted_gain,"rows":rows}))

}
pub(super) fn handle(line:&str)->serde_json::Value {
    match inner(line){Ok(v)=>v,Err(e)=>serde_json::json!({"error":e})}
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn cell_and_threat(){
        assert_eq!(cell(0.55,0.75,true,4),15);assert_eq!(cell(0.549,0.749,false,5),0);
        let g=Connect4::from_history("324252").unwrap();
        assert!(threat(&g));
    }
}
