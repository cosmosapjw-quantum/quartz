//! Research-only mechanism-resolved Connect4 endpoint.
use super::*;
use crate::mcts::four_axis::{Hooks,X};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,arm:String,seed:u64,mode:String,model_path:String,model_identity:String}
fn inner(line:&str)->Result<serde_json::Value,String>{
    let t=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_fouraxis_cpu"{return Err("bad cmd".into());}
    let (sx,vm,bm)=match q.arm.as_str(){
        "BASE"=>(0,0,0),"SX"=>(3,0,0),"WRITER"=>(3,0,1),"TEMPORAL"=>(3,0,2),
        "MERGE"=>(3,0,3),"COMBINED"=>(3,0,4),"LEGACY_V"=>(3,1,0),"PROOF_V"=>(3,2,0),
        "TOP2_V"=>(3,3,0),"FULL_V"=>(3,4,0),"JOINT"=>(3,3,3),_=>return Err("bad arm".into())
    };
    if !matches!(q.mode.as_str(),"VISITS"|"NN"|"TIME"|"TRACE_PREFIX"|"DIAG"|"STANDARD"|"DEEP"){return Err("bad mode".into());}
    if matches!(q.mode.as_str(),"TRACE_PREFIX"|"DIAG"|"STANDARD")&&q.arm!="BASE"{return Err("bad diagnostic arm".into());}
    let state=Connect4::from_history(&q.history)?;if state.is_terminal(){return Err("terminal root".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    if q.mode=="DIAG"{
        let base=ev.evaluate(&state);let mut p=base.policy.clone();
        p.sort_by(|a,b|b.1.total_cmp(&a.1).then_with(||state.move_to_idx(a.0).cmp(&state.move_to_idx(b.0))));
        let mut rows=vec![];let mut nn=1;
        for (mv,prior) in p{
            let child=state.apply_move(mv);let term=child.is_terminal();
            let val=if term{-child.outcome()}else{nn+=1;-ev.evaluate(&child).value};
            rows.push(serde_json::json!({"move":state.move_to_idx(mv),"prior":prior,"terminal":term,"h":val}));
        }
        return Ok(serde_json::json!({"history":q.history,"base":base.value,"rows":rows,"nn_calls":nn,"model_identity":q.model_identity}));
    }
    let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;
    if sx&X!=0{cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));}
    let st=Instant::now();let engine=MctsEngine::new(state,ev,cfg);let setup_us=st.elapsed().as_micros() as u64;
    let ph=Hooks::new_modes(sx,0,0,q.mode=="TRACE_PREFIX");
    let st=Instant::now();
    if q.mode=="STANDARD"{engine.run_par(&FixedIterations::new(8),1);}
    else{engine.four_axis_block(8,1,&ph);}
    let prefix_us=st.elapsed().as_micros() as u64;
    let prefix=engine.four_axis_snapshot();let prefix_hooks=ph.snapshot();let h=Hooks::new_modes(sx,vm,bm,false);
    let mut blocks=0;let mut cause="PREFIX";
    let st=Instant::now();
    if !matches!(q.mode.as_str(),"TRACE_PREFIX"|"STANDARD"){
        loop{
            let visits=8+8*blocks;
            if (q.mode=="VISITS"&&visits>=64)||(q.mode=="DEEP"&&visits>=512){cause="VISITS";break;}
            if q.mode=="NN"&&engine.evaluator_call_count()>=48{cause="NN";break;}
            if q.mode=="TIME"&&t.elapsed().as_micros()>=2500{cause="TIME";break;}
            if visits>=512{cause="CAP";break;}
            engine.four_axis_block(8,2,&h);blocks+=1;
        }
    }
    let continuation_us=st.elapsed().as_micros() as u64;let before_drop_us=t.elapsed().as_micros() as u64;
    let result=engine.four_axis_snapshot();let hooks=h.snapshot();let st=Instant::now();drop(engine);let drop_us=st.elapsed().as_micros() as u64;
    Ok(serde_json::json!({"schema":"quartz_fouraxis02_v1","history":q.history,"arm":q.arm,"seed":q.seed,"mode":q.mode,"model_identity":q.model_identity,"cache_hit":hit,"prefix":prefix,"prefix_hooks":prefix_hooks,"result":result,"hooks":hooks,"blocks":blocks,"cause":cause,"setup_us":setup_us,"prefix_us":prefix_us,"continuation_us":continuation_us,"before_drop_us":before_drop_us,"drop_us":drop_us,"whole_us":t.elapsed().as_micros() as u64}))
}
pub(super) fn handle(line:&str)->serde_json::Value{match inner(line){Ok(v)=>v,Err(e)=>serde_json::json!({"error":e})}}
