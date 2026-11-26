//! K28 shared resident tree, protected NN pools and schedule-conditioned control.
use super::*;
use crate::mcts::four_axis::{Hooks,LiteRoot};
use serde::Deserialize;
use serde_json::{json,Value};
// mask, workers, mode (0 common,1 split,2 floor), reserve.
const ACTIONS:[(u8,usize,u8,u64);20]=[
 (0,1,0,0),(0,2,0,0),(0,4,0,0),(0,8,0,0),
 (15,1,0,0),(15,2,0,0),(15,4,0,0),(15,8,0,0),
 (15,1,1,4),(15,2,1,4),(15,4,1,4),(15,8,1,4),
 (15,1,2,4),(15,2,2,4),(15,4,2,4),(15,8,2,4),
 (4,8,2,4),(11,4,0,0),(15,8,2,2),(15,4,2,2)];
const STOP:usize=20;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table{first:[usize;4],second:[[usize;8];20]}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,seed:u64,model_path:String,model_identity:String,budget:u64,
    #[serde(default)]path:Option<[usize;2]>,#[serde(default)]table:Option<Table>,#[serde(default)]trace:bool}
fn us(t:Instant)->u64{t.elapsed().as_micros() as u64}
fn state(r:&LiteRoot)->Value{
    json!({"moves":r.moves[..r.len],"n":r.n[..r.len],"w":r.w[..r.len],"visits":r.visits,"nn":r.nn,"best":r.best})
}
fn bin(r:&LiteRoot)->usize{
    let mut n=r.n[..r.len].to_vec();n.sort_unstable();
    let top=n[n.len()-1];let gap=top-n[n.len()-2];2*usize::from(4*top>=3*r.visits)+usize::from(gap>=4)
}
fn interval(root:&Connect4)->Vec<(usize,f64,f64)>{
    root.legal_moves().iter().map(|&m|{let s=root.apply_move(m);
        let(l,u)=if s.is_terminal(){let v=(1.-s.outcome() as f64)/2.;(v,v)}
        else if s.legal_moves().iter().any(|&a|s.is_winning_move(a)){(0.,0.)}else{(0.,1.)};
        (m,l,u)}).collect()
}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_k28_cpu"||!matches!(q.budget,8|16|32|64)||q.path.is_some()==q.table.is_some()||
       q.path.as_ref().is_some_and(|p|p[0]>=ACTIONS.len()||p[1]>STOP)||
       q.table.as_ref().is_some_and(|t|t.first.iter().any(|&a|a>=ACTIONS.len())||t.second.iter().flatten().any(|&a|a>STOP)){
        return Err("bad K28 profile contract".into());}
    if rayon::current_num_threads()<8{return Err("K28 requires pool of eight workers".into());}
    let root=Connect4::from_history(&q.history)?;let legal=root.legal_moves();
    if root.is_terminal()||legal.len()<3{return Err("bad root".into());}
    let(ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let t=Instant::now();let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);
    cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
    cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));
    let e=MctsEngine::new(root.clone(),ev,cfg);let mut h=Hooks::new_modes(0,0,0,false).with_k25(q.trace);
    let setup_us=us(t);let t=Instant::now();e.four_axis_k25_block(8,1,&h);
    let prefix_us=us(t);let pre=e.four_axis_lite();let pb=bin(&pre);let ph=h.k25_report(false);
    let mut stages=Vec::new();let mut chosen=[0usize;2];let mut mb=0;let mut end=pre.clone();
    for j in 0..2{
        let a=if let Some(p)=q.path{p[j]}else{let table=q.table.as_ref().unwrap();if j==0{table.first[pb]}else{table.second[chosen[0]][mb]}};
        chosen[j]=a;if a==STOP{assert_eq!(j,1);break;}
        let(mask,workers,mode,reserve)=ACTIONS[a];let t=Instant::now();h.set_k25_mask(mask);
        let controller_us=us(t);let t=Instant::now();
        let mut b=if mode==2{e.four_axis_k28_floor(q.budget,(q.budget*4) as u32,workers,reserve,&h)}
        else{e.four_axis_k27_budget(q.budget,(q.budget*4) as u32,workers,false,reserve,&h)};
        if mode!=2{b["borrowed"]=json!(0);b["pre_extra"]=b["protected_extra"].clone();}
        let wall_us=us(t);end=e.four_axis_lite();let hh=h.k25_report(q.trace&&j==1);
        let z=2*bin(&end)+usize::from(hh["ledger"]["shared"].as_u64().unwrap()>0);if j==0{mb=z;}
        stages.push(json!({"action":a,"mask":mask,"workers":workers,"mode":mode,"reserve":reserve,"state":state(&end),"lite":end,
            "hooks":hh,"budget":b,"bin":z,"control_us":controller_us,"wall_us":wall_us}));
    }
    let proof=interval(&root);let ub=proof.iter().map(|p|p.2).fold(0.,f64::max);let cert=proof.iter().find(|p|p.1>=ub).map(|p|p.0);
    let mid_best=cert.unwrap_or(stages[0]["state"]["best"].as_u64().unwrap() as usize);let best=cert.unwrap_or(end.best);let snap=e.four_axis_snapshot();let nn=end.nn;let visits=end.visits;
    drop(e);drop(h);
    Ok(json!({"schema":"quartz_k28_c4_v1","history":q.history,"seed":q.seed,"budget":q.budget,"model_identity":q.model_identity,
        "legal":legal,"prefix":state(&pre),"prefix_lite":pre,"prefix_hooks":ph,"prefix_bin":pb,"mid_bin":mb,"chosen":chosen,"mid_best":mid_best,"stopped":chosen[1]==STOP,
        "stages":stages,"result":state(&end),"snapshot":snap,"best":best,"proof":proof,"NN_calls":nn,"visits":visits,
        "cache_hit":hit,"setup_us":setup_us,"prefix_us":prefix_us,"whole_us":us(start)}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
