//! Research-only certificate-aware Connect4 endpoint.
use super::*;
use crate::mcts::four_axis::{Hooks,X};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,arm:String,seed:u64,mode:String,model_path:String,model_identity:String,#[serde(default)]target_us:u64,#[serde(default)]reserve_us:u64}
fn inner(line:&str)->Result<serde_json::Value,String>{
    let t=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_fouraxis_cpu"{return Err("bad cmd".into());}
    let (sx,vm,bm,gm,stop)=match q.arm.as_str(){
        "BASE"=>(0,0,0,0,false),"SX"=>(3,0,0,0,false),"SAMPLE"=>(3,2,0,0,false),
        "ROOT"=>(3,2,0,1,false),"ACTIVE"=>(3,2,0,2,false),"ACTIVE_STOP"=>(3,2,0,2,true),
        "FULL"=>(3,4,0,0,false),"ACTIVE_FULL"=>(3,4,0,2,false),
        "TEMPORAL"=>(3,0,2,0,false),"MERGE"=>(3,0,3,0,false),_=>return Err("bad arm".into())
    };
    if !matches!(q.mode.as_str(),"VISITS"|"TIME"|"GRAPH"|"BTRACE"|"PREFIX"|"STANDARD"){return Err("bad mode".into());}
    if matches!(q.mode.as_str(),"PREFIX"|"STANDARD")&&q.arm!="BASE"{return Err("bad parity arm".into());}
    if (q.mode=="BTRACE")!=matches!(q.arm.as_str(),"TEMPORAL"|"MERGE"){return Err("bad BTRACE contract".into());}
    if q.mode=="GRAPH"&&gm==0{return Err("graph mode needs graph arm".into());}
    if q.mode=="TIME"&&!(2000..=10000).contains(&q.target_us)||q.mode=="TIME"&&q.reserve_us>=q.target_us{return Err("bad budget".into());}
    let state=Connect4::from_history(&q.history)?;if state.is_terminal(){return Err("terminal root".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;
    if sx&X!=0{cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));}
    let st=Instant::now();let engine=MctsEngine::new(state.clone(),ev,cfg);let setup_us=st.elapsed().as_micros() as u64;
    let ph=Hooks::new_modes(sx,0,0,false);
    let st=Instant::now();
    if q.mode=="STANDARD"{engine.run_par(&FixedIterations::new(8),1);}else{engine.four_axis_block(8,1,&ph);}
    let prefix_us=st.elapsed().as_micros() as u64;
    let prefix=engine.four_axis_snapshot();let mut h=Hooks::new_modes(sx,vm,bm,false).with_graph(gm);
    if q.mode=="BTRACE"{h=h.with_b_trace();}
    h.proof_register(&state);
    let mut blocks=0;let mut cause="PREFIX";let st=Instant::now();
    let max_visits=if matches!(q.mode.as_str(),"BTRACE"|"TIME"){512}else{128};
    if !matches!(q.mode.as_str(),"PREFIX"|"STANDARD"){
        loop{
            let visits=8+8*blocks;
            if stop{
                let b=engine.best_move().unwrap();
                let p=h.root_proof(&state,b,&[],false);
                if !p["witness"].is_null(){cause="CERTIFIED";break;}
            }
            if q.mode=="TIME"&&t.elapsed().as_micros() as u64>=q.target_us-q.reserve_us{cause="TIME";break;}
            if visits>=max_visits{cause=if q.mode=="TIME"{"CAP"}else{"VISITS"};break;}
            engine.four_axis_block(8,2,&h);blocks+=1;
        }
    }
    let continuation_us=st.elapsed().as_micros() as u64;
    let result=engine.four_axis_snapshot();let normal=result["best"].as_u64().unwrap() as usize;
    let vs:Vec<_>=result["edges"].as_array().unwrap().iter().map(|e|(e["move"].as_u64().unwrap() as usize,e["n"].as_u64().unwrap() as u32)).collect();
    let proof=h.root_proof(&state,normal,&vs,q.mode=="GRAPH");
    let best=if proof.is_null(){normal}else{proof["best"].as_u64().unwrap() as usize};
    let hooks=h.snapshot();let before_drop_us=t.elapsed().as_micros() as u64;let st=Instant::now();drop(engine);let drop_us=st.elapsed().as_micros() as u64;
    Ok(serde_json::json!({"schema":"quartz_fouraxis03_v1","history":q.history,"arm":q.arm,"seed":q.seed,"mode":q.mode,"model_identity":q.model_identity,"cache_hit":hit,"prefix":prefix,"prefix_hooks":ph.snapshot(),"result":result,"best":best,"proof":proof,"hooks":hooks,"blocks":blocks,"cause":cause,"target_us":q.target_us,"reserve_us":q.reserve_us,"setup_us":setup_us,"prefix_us":prefix_us,"continuation_us":continuation_us,"before_drop_us":before_drop_us,"drop_us":drop_us,"whole_us":t.elapsed().as_micros() as u64}))
}
pub(super) fn handle(line:&str)->serde_json::Value{match inner(line){Ok(v)=>v,Err(e)=>serde_json::json!({"error":e})}}
