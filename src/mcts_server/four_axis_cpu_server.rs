//! Research-only CPU Connect4 factorial endpoint.
use super::*;
use crate::mcts::four_axis::{Hooks,X};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,mask:u8,seed:u64,visits:u32,threads:usize,model_path:String,model_identity:String,#[serde(default)] standard:bool}
fn inner(line:&str)->Result<serde_json::Value,String>{
    let t=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_fouraxis_cpu"||q.mask>15||!matches!(q.visits,64|512)||!matches!(q.threads,1|2)||q.standard&&q.mask!=0{return Err("bad fouraxis contract".into());}
    let state=Connect4::from_history(&q.history)?;
    if state.is_terminal(){return Err("terminal root".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);
    cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;
    if q.mask&X!=0{cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));}
    let st=Instant::now();let engine=MctsEngine::new(state,ev,cfg);let setup_us=st.elapsed().as_micros() as u64;
    let initial=engine.four_axis_snapshot();let h=Hooks::new(q.mask);
    let st=Instant::now();
    if q.standard{let a=engine.run_par(&FixedIterations::new(q.visits),q.threads);if a.root_visits!=q.visits{return Err("standard visits".into());}}
    else{engine.run_four_axis(q.visits,q.threads,&h);}
    let search_us=st.elapsed().as_micros() as u64;
    let result=engine.four_axis_snapshot();let hooks=h.snapshot();let st=Instant::now();
    drop(engine);let drop_us=st.elapsed().as_micros() as u64;
    Ok(serde_json::json!({"schema":"quartz_fouraxis_v1","history":q.history,"mask":q.mask,"seed":q.seed,"threads":q.threads,"standard":q.standard,"visits":q.visits,"model_identity":q.model_identity,"cache_hit":hit,"initial":initial,"result":result,"hooks":hooks,"setup_us":setup_us,"search_us":search_us,"drop_us":drop_us,"whole_us":t.elapsed().as_micros() as u64}))
}
pub(super) fn handle(line:&str)->serde_json::Value{match inner(line){Ok(v)=>v,Err(e)=>serde_json::json!({"error":e})}}
