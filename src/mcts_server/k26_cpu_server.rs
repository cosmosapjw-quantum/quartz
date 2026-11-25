//! K26 two resident resource-bounded feedback stages, CPU research only.
use super::*;
use crate::mcts::four_axis::{Hooks,LiteRoot};
use serde::Deserialize;
use serde_json::{json,Value};
const MASKS:[u8;6]=[0,1,2,4,8,15];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table{first:[u8;4],second:[[u8;8];16]}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,method:String,seed:u64,model_path:String,model_identity:String,
    #[serde(default)]path:Option<[u8;2]>,#[serde(default)]table:Option<Table>,#[serde(default)]trace:bool}
fn us(t:Instant)->u64{t.elapsed().as_micros() as u64}
fn aggregate(lanes:&[LiteRoot])->Value{
    let a=&lanes[0];let mut n=vec![0u32;a.len];let mut w=vec![0.;a.len];let mut nn=0;let mut visits=0;
    for r in lanes{assert_eq!(a.len,r.len);assert_eq!(a.moves,r.moves);assert_eq!(a.prior,r.prior);
        nn+=r.nn;visits+=r.visits;for i in 0..a.len{n[i]+=r.n[i];w[i]+=r.w[i];}}
    let i=n.iter().enumerate().max_by_key(|&(_,v)|*v).unwrap().0;
    json!({"moves":a.moves[..a.len],"n":n,"w":w,"visits":visits,"nn":nn,"best":a.moves[i]})
}
fn bin(a:&Value)->usize{
    let mut n:Vec<_>=a["n"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()).collect();n.sort_unstable();
    let top=n[n.len()-1];let gap=top-n[n.len()-2];2*usize::from(4*top>=3*a["visits"].as_u64().unwrap())+usize::from(gap>=4)
}
fn interval(root:&Connect4)->Vec<(usize,f64,f64)>{
    root.legal_moves().iter().map(|&m|{let s=root.apply_move(m);
        let(l,u)=if s.is_terminal(){let v=(1.-s.outcome() as f64)/2.;(v,v)}
        else if s.legal_moves().iter().any(|&a|s.is_winning_move(a)){(0.,0.)}else{(0.,1.)};
        (m,l,u)}).collect()
}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    let (count,threads,wave)=match q.method.as_str(){
        "SERIAL"=>(1,1,false),"TREE2"=>(1,2,false),"TREE4"=>(1,4,false),"TREE8"=>(1,8,false),
        "WAVE4"=>(1,4,true),"ROOT2"|"ROOT2D"=>(2,1,false),_=>return Err("bad K26 method".into())};
    if q.cmd!="research_connect4_k26_cpu"||q.path.is_some()==q.table.is_some()||
       q.path.as_ref().is_some_and(|p|p.iter().any(|v|!MASKS.contains(v)))||
       q.table.as_ref().is_some_and(|t|t.first.iter().chain(t.second.iter().flatten()).any(|v|!MASKS.contains(v))){
        return Err("bad K26 profile contract".into());}
    if rayon::current_num_threads()<threads.max(count){return Err("insufficient workers".into());}
    let state=Connect4::from_history(&q.history)?;let legal=state.legal_moves();
    if state.is_terminal()||legal.len()<3{return Err("bad root".into());}
    let(ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let t=Instant::now();let mut engines=Vec::new();let mut hs=Vec::new();
    for i in 0..count{
        let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed.wrapping_add(i as u64));
        cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
        cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));
        let mut e=MctsEngine::new(state.clone(),ev.clone(),cfg);
        if q.method=="ROOT2D"{e.k26_root_diversity(q.seed.wrapping_add(i as u64));}
        engines.push(e);hs.push(Hooks::new_modes(0,0,0,false).with_k25(q.trace));
    }
    let setup_us=us(t);let t=Instant::now();
    if count==2{rayon::join(||engines[0].four_axis_k25_block(4,1,&hs[0]),||engines[1].four_axis_k25_block(4,1,&hs[1]));}
    else{engines[0].four_axis_k25_block(8,1,&hs[0]);}
    let prefix_us=us(t);let pre:Vec<_>=engines.iter().map(|e|e.four_axis_lite()).collect();
    let prefix=aggregate(&pre);let prefix_bin=bin(&prefix);
    let ph:Vec<_>=hs.iter().map(|h|h.k25_report(false)).collect();
    let mut stages=Vec::new();let mut chosen=[0u8;2];let mut mid_bin=0;let mut end=prefix.clone();
    for j in 0..2{
        let profile=if let Some(p)=q.path{p[j]}else{let table=q.table.as_ref().unwrap();
            if j==0{table.first[prefix_bin]}else{table.second[chosen[0] as usize][mid_bin]}};
        chosen[j]=profile;
        let t=Instant::now();for h in &mut hs{h.set_k25_mask(profile);}
        let control_us=us(t);
        let t=Instant::now();let budgets=if count==2{
            let(a,b)=rayon::join(||{let a=us(t);let z=engines[0].four_axis_k26_budget(4,16,1,false,&hs[0]);(z,(a,us(t)))},
                ||{let a=us(t);let z=engines[1].four_axis_k26_budget(4,16,1,false,&hs[1]);(z,(a,us(t)))});vec![a,b]
        }else{vec![(engines[0].four_axis_k26_budget(8,32,threads,wave,&hs[0]),(0,us(t)))]};
        let wall_us=us(t);
        let ls:Vec<_>=engines.iter().map(|e|e.four_axis_lite()).collect();end=aggregate(&ls);
        let hooks:Vec<_>=hs.iter().map(|h|h.k25_report(q.trace&&j==1)).collect();
        let shared=hooks.iter().any(|h|h["ledger"]["shared"].as_u64().unwrap()>0);
        let b=2*bin(&end)+usize::from(shared);
        if j==0{mid_bin=b;}
        stages.push(json!({"profile":profile,"state":end,"lanes":ls,"hooks":hooks,"budgets":budgets,"bin":b,"control_us":control_us,"wall_us":wall_us}));
    }
    let proof=interval(&state);let ub=proof.iter().map(|p|p.2).fold(0.,f64::max);
    let cert=proof.iter().find(|p|p.1>=ub).map(|p|p.0);
    let best=cert.unwrap_or(end["best"].as_u64().unwrap() as usize);
    let snapshots:Vec<_>=engines.iter().map(|e|e.four_axis_snapshot()).collect();
    let nn=end["nn"].as_u64().unwrap();let visits=end["visits"].as_u64().unwrap();drop(engines);drop(hs);
    Ok(json!({"schema":"quartz_k26_c4_v1","history":q.history,"method":q.method,"seed":q.seed,"model_identity":q.model_identity,
       "legal":legal,"prefix":prefix,"prefix_lanes":pre,"prefix_hooks":ph,"prefix_bin":prefix_bin,"mid_bin":mid_bin,
       "chosen":chosen,"stages":stages,"result":end,"snapshots":snapshots,"best":best,"proof":proof,"NN_calls":nn,"visits":visits,
       "cache_hit":hit,"setup_us":setup_us,"prefix_us":prefix_us,"whole_us":us(start)}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
