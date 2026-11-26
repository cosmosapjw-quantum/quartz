use super::*;
use crate::mcts::four_axis::Hooks;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request35{cmd:String,history:Vec<u16>,mask:Option<u8>,table:Option<[u8;4]>,cap:u32,threads:usize,lock:bool,seed:u64,#[serde(default)]trace:bool}
fn inner(line:&str)->Result<Value,String>{
    let started=Instant::now();let q:Request35=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_rapfi_k35"||q.history.len()>224{return Err("bad K34 request".into());}
    if !matches!(q.cap,80|128)||!matches!(q.threads,1|4)||q.mask.is_some()==q.table.is_some()||q.mask.is_some_and(|m|m>15)||q.table.is_some_and(|m|m.iter().any(|x|*x>15)){return Err("bad K35 options".into());}
    let(cap,locked,threads)=(q.cap,q.lock,q.threads);
    let eval=session().lock().unwrap().as_ref().cloned().ok_or("session not open")?;
    let mut s=Gomoku15::new(GomokuVariant::Freestyle);
    for &m in &q.history{if m>=225||s.is_terminal()||!s.legal_moves().contains(&m){return Err("illegal history".into());}s=s.apply_move(m);}
    if s.is_terminal(){return Err("terminal root".into());}
    let t=Instant::now();let proof=super::k34::scan(s.board_as_12(),if s.current_player()==1{1}else{2});
    let proof_us=t.elapsed().as_micros();
    let mut out=json!({"schema":"quartz_rapfi_k35_v1","history":q.history,"request_mask":q.mask,"table":q.table,"locked":locked,"seed":q.seed,"cap":cap,"threads":threads,
        "player":s.current_player(),"proof":proof,"proof_us":proof_us});
    if let Some(m)=proof["gate"].as_u64(){
        out["best"]=json!(m);out["cause"]=proof["kind"].clone();out["visits"]=json!(0);out["NN_calls"]=json!(0);
        out["root_eval"]=Value::Null;out["initial"]=Value::Null;out["prefix"]=Value::Null;out["waves"]=json!([]);out["result"]=Value::Null;
        out["hooks"]=Value::Null;out["bin"]=Value::Null;out["selected_mask"]=Value::Null;out["bridge_wait_ns"]=json!(0);out["eval_ipc_ns"]=json!(0);out["whole_us"]=json!(started.elapsed().as_micros());return Ok(out)
    }
    eval.0.lock().unwrap().reset()?;
    let mut cfg=gomoku15_quartz(GomokuVariant::Freestyle).with_seed(q.seed);
    cfg.root_forced_win=false;cfg.exact_terminal_value=false;cfg.fpu_reduction=0.;cfg.dirichlet=None;cfg.temperature=0.;
    cfg.vl_mode=crate::mcts::parallel::VlMode::Disabled;cfg.gvoc=None;cfg.quartz=None;
    let engine=MctsEngine::new(s,eval.clone(),cfg);let init=engine.four_axis_snapshot();
    assert_eq!(engine.evaluator_call_count(),1);
    let mut hook=Hooks::new_modes(0,0,0,false).with_k34_rapfi(q.trace);
    let a=engine.four_axis_k26_budget(15,16,1,false,&hook);let mid=engine.four_axis_snapshot();
    let prefix=json!({"budget":a,"result":mid,"hooks":hook.k25_report(false)});
    let v0={let b=eval.0.lock().unwrap();let f=b.first.as_ref().unwrap();f.w-f.l};
    let bin=usize::from(v0.abs()>=0.5)+2*usize::from(engine.vote_min_distance().is_none_or(|d|d>=4));
    let mask=q.mask.unwrap_or_else(||q.table.unwrap()[bin]);
    out["bin"]=json!(bin);out["selected_mask"]=json!(mask);
    hook.set_k25_mask(mask);
    let mut waves=Vec::new();let mut cause="VISIT_CAP";
    loop{
        let (n,nn,_)=engine.k34_progress();
        // Stop only when the entire finite remaining visit allowance cannot change winner.
        let d=engine.vote_min_distance();
        if nn>=cap as u64{cause="NN_TARGET";break;}
        if n>=cap{break;}
        if locked&&d.is_none_or(|x|x>cap-n){cause="VOTE_LOCK";break;}
        let v=(cap-n).min(8);let t=Instant::now();
        let budget=engine.four_axis_k26_budget(cap as u64-nn,v,threads,false,&hook);
        let (pn,pc,pb)=engine.k34_progress();
        waves.push(json!({"before_n":n,"before_nn":nn,"before_d":d,"budget":budget,
            "after_n":pn,"after_nn":pc,"after_best":pb,"after_d":engine.vote_min_distance(),"step_us":t.elapsed().as_micros()}));
    }
    let result=engine.four_axis_snapshot();let best=result["best"].as_u64().ok_or("missing best")?;
    let (calls,ipc,wait,first)={let b=eval.0.lock().unwrap();(b.calls,b.elapsed_ns,b.wait_ns,b.first.as_ref().ok_or("missing root NN")?.json())};
    assert_eq!(calls,engine.evaluator_call_count());assert!(calls<=cap as u64);
    let visits=result["root_n"].as_u64().unwrap();assert!(visits<=cap as u64);
    out["best"]=json!(best);out["cause"]=json!(cause);out["NN_calls"]=json!(calls);out["visits"]=json!(visits);
    out["root_eval"]=first;out["initial"]=init;out["prefix"]=prefix;out["waves"]=json!(waves);out["result"]=result;
    out["hooks"]=hook.k25_report(q.trace);out["eval_ipc_ns"]=json!(ipc);out["bridge_wait_ns"]=json!(wait);out["whole_us"]=json!(started.elapsed().as_micros());
    Ok(out)
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
