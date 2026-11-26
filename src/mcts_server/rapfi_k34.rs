//! Shared exact root certificate and strict two-resource four-axis search.
//! The existing Rapfi bridge is board/side-only and serialized by its mutex.
use super::*;
use crate::mcts::four_axis::Hooks;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request34{cmd:String,history:Vec<u16>,arm:String,seed:u64,#[serde(default)]trace:bool}
fn windows()->&'static Vec<[usize;5]>{
    static W:OnceLock<Vec<[usize;5]>>=OnceLock::new();
    W.get_or_init(||{let mut out=Vec::new();for r in 0..15i32{for c in 0..15i32{
        for (dr,dc) in [(0,1),(1,0),(1,1),(1,-1)]{
            if (0..15).contains(&(r+4*dr))&&(0..15).contains(&(c+4*dc)){
                out.push(std::array::from_fn(|t|((r+t as i32*dr)*15+c+t as i32*dc) as usize));
            }
        }
    }}assert_eq!(out.len(),572);out})
}
fn scan(b:&[u8;225],p:u8)->Value{
    let mut own=[false;225];let mut opp=[false;225];let mut adj:Vec<Vec<usize>>=vec![Vec::new();225];
    let mut pairs=0;
    for l in windows(){
        let mut n=0;let mut m=0;let mut e=[0;5];let mut k=0;
        for &i in l{match b[i]{0=>{e[k]=i;k+=1},x if x==p=>n+=1,_=>m+=1}}
        if k==1&&n==4{own[e[0]]=true;}
        if k==1&&m==4{opp[e[0]]=true;}
        if k==2&&n==3{adj[e[0]].push(e[1]);adj[e[1]].push(e[0]);pairs+=1;}
    }
    let legal:Vec<_>=(0..225).filter(|&i|b[i]==0).collect();assert!(!legal.is_empty());
    let ownv:Vec<_>=legal.iter().copied().filter(|&i|own[i]).collect();let oppv:Vec<_>=legal.iter().copied().filter(|&i|opp[i]).collect();
    let mut rows=Vec::new();let mut win3=Vec::new();
    for &a in &legal{
        let mut threats=Vec::new();let mut reply=None;
        let(l,u,kind)=if own[a]{(1,1,"WIN1")}
        else if legal.len()==1{(0,0,"DRAW")}
        else if let Some(&r)=oppv.iter().find(|&&r|r!=a){reply=Some(r);(-1,-1,"LOSE2")}
        else{
            threats.extend(ownv.iter().copied().filter(|&i|i!=a));
            threats.extend(adj[a].iter().copied());threats.sort_unstable();threats.dedup();
            if threats.len()>=2{win3.push(a);(1,1,"WIN3")}else{(-1,1,"UNKNOWN")}
        };
        rows.push(json!([a,l,u,kind,reply,threats]));
    }
    let old=ownv.first().copied().or_else(||if oppv.len()==1{Some(oppv[0])}else if legal.len()==1{Some(legal[0])}else{None});
    let (gate,kind)=if let Some(&a)=ownv.first(){(Some(a),"WIN1")}
    else if let Some(&a)=win3.first(){(Some(a),"WIN3")}
    else if let Some(a)=old{(Some(a),if legal.len()==1{"SOLE"}else{"BLOCK"})}
    else{(None,"UNRESOLVED")};
    json!({"own":ownv,"opponent":oppv,"win3":win3,"actions":rows,"gate":gate,"kind":kind,"old_guard":old,"windows":windows().len(),"pair_windows":pairs})
}
fn inner(line:&str)->Result<Value,String>{
    let started=Instant::now();let q:Request34=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_rapfi_k34"||q.history.len()>224{return Err("bad K34 request".into());}
    let (cap,mask,locked,threads)=match q.arm.as_str(){
        "B128"=>(128u32,0u8,false,1usize),"B80"=>(80,0,false,1),
        "F128"=>(128,15,false,1),"F80"=>(80,15,false,1),"F80L"=>(80,15,true,1),"F80L4"=>(80,15,true,4),
        _=>return Err("bad arm".into())};
    let eval=session().lock().unwrap().as_ref().cloned().ok_or("session not open")?;
    let mut s=Gomoku15::new(GomokuVariant::Freestyle);
    for &m in &q.history{if m>=225||s.is_terminal()||!s.legal_moves().contains(&m){return Err("illegal history".into());}s=s.apply_move(m);}
    if s.is_terminal(){return Err("terminal root".into());}
    let t=Instant::now();let proof=scan(s.board_as_12(),if s.current_player()==1{1}else{2});
    let proof_us=t.elapsed().as_micros();
    let mut out=json!({"schema":"quartz_rapfi_k34_v1","history":q.history,"arm":q.arm,"seed":q.seed,"cap":cap,"threads":threads,
        "player":s.current_player(),"proof":proof,"proof_us":proof_us});
    if let Some(m)=proof["gate"].as_u64(){
        out["best"]=json!(m);out["cause"]=proof["kind"].clone();out["visits"]=json!(0);out["NN_calls"]=json!(0);
        out["root_eval"]=Value::Null;out["initial"]=Value::Null;out["prefix"]=Value::Null;out["waves"]=json!([]);out["result"]=Value::Null;
        out["hooks"]=Value::Null;out["eval_ipc_ns"]=json!(0);out["whole_us"]=json!(started.elapsed().as_micros());return Ok(out)
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
    let (calls,ipc,first)={let b=eval.0.lock().unwrap();(b.calls,b.elapsed_ns,b.first.as_ref().ok_or("missing root NN")?.json())};
    assert_eq!(calls,engine.evaluator_call_count());assert!(calls<=cap as u64);
    let visits=result["root_n"].as_u64().unwrap();assert!(visits<=cap as u64);
    out["best"]=json!(best);out["cause"]=json!(cause);out["NN_calls"]=json!(calls);out["visits"]=json!(visits);
    out["root_eval"]=first;out["initial"]=init;out["prefix"]=prefix;out["waves"]=json!(waves);out["result"]=result;
    out["hooks"]=hook.k25_report(q.trace);out["eval_ipc_ns"]=json!(ipc);out["whole_us"]=json!(started.elapsed().as_micros());
    Ok(out)
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
