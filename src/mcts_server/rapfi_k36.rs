use super::*;
use crate::mcts::four_axis::Hooks;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request36{cmd:String,history:Vec<u16>,cap:Option<u32>,horizons:Option<[u32;8]>,lock:bool,seed:u64,#[serde(default)]trace:bool}
fn valid_cap(h:u32)->bool{matches!(h,80|96|112|128|256)}
fn inner(line:&str)->Result<Value,String>{
    let started=Instant::now();let q:Request36=serde_json::from_str(line).map_err(|e|e.to_string())?;
    let k52=q.cmd=="research_rapfi_k52";let hybrid=matches!(q.cmd.as_str(),"research_rapfi_k47"|"research_rapfi_k48")||k52;
    if hybrid&&!k52&&((q.cmd=="research_rapfi_k47"&&q.cap!=Some(128))||(q.cmd=="research_rapfi_k48"&&!q.cap.is_some_and(|v|matches!(v,80|96|112|128)))||!q.lock||q.horizons.is_some()){return Err("K47 Rapfi requires locked128".into());}
    if (!hybrid&&q.cmd!="research_rapfi_k36")||q.history.len()>224{return Err("bad K36 request".into());}
    if k52&&(!q.lock||!(q.cap.is_some_and(|v|matches!(v,104|112))&&q.horizons.is_none()||q.cap.is_none()&&q.horizons==Some([128,128,128,128,96,96,96,96]))){return Err("invalid K52 Rapfi cap".into());}
    if q.cap.is_some()==q.horizons.is_some()||q.cap.is_some_and(|h|!valid_cap(h)&&!(k52&&h==104))||q.horizons.is_some_and(|hs|hs.iter().any(|h|!valid_cap(*h)))||q.horizons.is_some()&&!q.lock{return Err("bad K36 options".into());}
    let(mut cap,locked,threads)=(q.cap.unwrap_or(256),q.lock,1usize);
    let eval=session().lock().unwrap().as_ref().cloned().ok_or("session not open")?;
    let mut s=Gomoku15::new(GomokuVariant::Freestyle);
    for &m in &q.history{if m>=225||s.is_terminal()||!s.legal_moves().contains(&m){return Err("illegal history".into());}s=s.apply_move(m);}
    if s.is_terminal(){return Err("terminal root".into());}
    let t=Instant::now();let proof=super::k34::scan(s.board_as_12(),if s.current_player()==1{1}else{2});
    let proof_us=t.elapsed().as_micros();
    let mut out=json!({"schema":"quartz_rapfi_k36_v1","history":q.history,"request_mask":0,"table":Value::Null,"requested_cap":q.cap,"horizons":q.horizons,"locked":locked,"seed":q.seed,"cap":cap,"threads":threads,
        "player":s.current_player(),"proof":proof,"proof_us":proof_us});
    if hybrid{out["partial"]=Value::Null;out["original_move"]=Value::Null;out["replacement"]=json!(false);}
    if let Some(m)=proof["gate"].as_u64(){
        out["best"]=json!(m);out["cause"]=proof["kind"].clone();out["visits"]=json!(0);out["NN_calls"]=json!(0);
        out["root_eval"]=Value::Null;out["initial"]=Value::Null;out["prefix"]=Value::Null;out["waves"]=json!([]);out["result"]=Value::Null;
        out["hooks"]=Value::Null;out["cap"]=json!(q.cap.unwrap_or(128));out["probe"]=Value::Null;out["probe_wave"]=Value::Null;out["bin"]=Value::Null;out["selected_mask"]=Value::Null;out["bridge_wait_ns"]=json!(0);out["eval_ipc_ns"]=json!(0);out["whole_us"]=json!(started.elapsed().as_micros());return Ok(out)
    }
    if hybrid{
        let p=if s.current_player()==1{1}else{2};
        let b:Vec<u8>=s.board_as_12().iter().map(|&x|if x==0{0}else if x==p{1}else{2}).collect();
        let part=crate::mcts_server::k47_proof::combined(&b,15)?;
        if !part["cold"]["receipt"]["gate"].is_null(){return Err("K34/K47 common gate mismatch".into());}
        out["partial"]=part;
        if let Some(m)=out["partial"]["cert"].as_u64(){
            out["best"]=json!(m);out["cause"]=json!("PARTIAL47");out["visits"]=json!(0);out["NN_calls"]=json!(0);
            out["root_eval"]=Value::Null;out["initial"]=Value::Null;out["prefix"]=Value::Null;out["probe"]=Value::Null;out["result"]=Value::Null;
            out["waves"]=json!([]);out["hooks"]=Value::Null;out["bin"]=Value::Null;out["selected_mask"]=Value::Null;
            out["bridge_wait_ns"]=json!(0);out["eval_ipc_ns"]=json!(0);out["certificate_checks"]=json!(0);
            out["whole_us"]=json!(started.elapsed().as_micros());return Ok(out)
        }
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
    hook.set_k25_mask(0);
    let mut probe=None;
    out["selected_mask"]=json!(0);
    let mut waves=Vec::new();let mut cause="VISIT_CAP";let mut certificate_checks=0u64;
    loop{
        let (n,nn,_)=engine.k34_progress();
        if probe.is_none()&&(n>=32||nn>=32){
            assert!(n<=39&&n+1<80-n);
            let r=engine.four_axis_snapshot();let best=r["best"].as_u64().unwrap();
            let prior=r["candidates"][0][0].as_u64().unwrap();
            let anchor=prefix["result"]["best"].as_u64().unwrap();
            let d=engine.vote_min_distance();
            let bin=usize::from(prior!=best)+2*usize::from(anchor!=best)+4*usize::from(d.is_none_or(|d|d>=8));
            if let Some(hs)=q.horizons{cap=hs[bin];}
            probe=Some(r);out["bin"]=json!(bin);out["probe_wave"]=json!(waves.len());out["cap"]=json!(cap);
        }
        // Stop only when the entire finite remaining visit allowance cannot change winner.
        let d=engine.vote_min_distance();
        if nn>=cap as u64{cause="NN_TARGET";break;}
        if n>=cap{break;}
        let certified=engine.vote_certificate_quiescent(cap);
        certificate_checks+=1;
        assert_eq!(certified,d.is_none_or(|x|x>cap-n));
        if locked&&certified{cause="VOTE_LOCK";break;}
        let v=(cap-n).min(8);let t=Instant::now();
        let budget=engine.four_axis_k26_budget(cap as u64-nn,v,threads,false,&hook);
        let (pn,pc,pb)=engine.k34_progress();
        waves.push(json!({"before_n":n,"before_nn":nn,"before_d":d,"budget":budget,
            "after_n":pn,"after_nn":pc,"after_best":pb,"after_d":engine.vote_min_distance(),"step_us":t.elapsed().as_micros()}));
    }
    out["certificate_checks"]=json!(certificate_checks);
    out["probe"]=probe.expect("common probe completed");
    let result=engine.four_axis_snapshot();let best=result["best"].as_u64().ok_or("missing best")?;
    let (calls,ipc,wait,first)={let b=eval.0.lock().unwrap();(b.calls,b.elapsed_ns,b.wait_ns,b.first.as_ref().ok_or("missing root NN")?.json())};
    assert_eq!(calls,engine.evaluator_call_count());assert!(calls<=cap as u64);
    let visits=result["root_n"].as_u64().unwrap();assert!(visits<=cap as u64);
    let mut chosen=best;
    if hybrid{
        let candidates=result["candidates"].as_array().ok_or("missing candidates")?;
        let edges=result["edges"].as_array().ok_or("missing edges")?;
        let vs:Vec<(u16,u32)>=candidates.iter().enumerate().map(|(i,c)|(c[0].as_u64().unwrap() as u16,edges.get(i).map(|e|{assert_eq!(e["move"],c[0]);e["n"].as_u64().unwrap() as u32}).unwrap_or(0))).collect();
        chosen=crate::mcts_server::k47_proof::replace(best as u16,&vs,&out["partial"]["bounds"],15) as u64;
        out["original_move"]=json!(best);out["replacement"]=json!(chosen!=best);
    }
    out["best"]=json!(chosen);out["cause"]=json!(cause);out["NN_calls"]=json!(calls);out["visits"]=json!(visits);
    out["root_eval"]=first;out["initial"]=init;out["prefix"]=prefix;out["waves"]=json!(waves);out["result"]=result;
    out["hooks"]=hook.k25_report(q.trace);out["eval_ipc_ns"]=json!(ipc);out["bridge_wait_ns"]=json!(wait);out["whole_us"]=json!(started.elapsed().as_micros());
    Ok(out)
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
