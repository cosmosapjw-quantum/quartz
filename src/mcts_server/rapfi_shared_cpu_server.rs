//! Research-only common evaluator experiment. Both arms use an actual,
//! separately loaded Rapfi mix9svq NN; no alpha-beta teacher or fallback.
use super::*;
use super::gomoku_long_cpu_server::{proof,proof_json};
use serde::Deserialize;
use serde_json::{json,Value};
use std::io::BufReader;
use std::process::{Child,ChildStdin,ChildStdout,Command,Stdio};
#[derive(Clone)]
struct Output{board:String,side:i8,w:f32,l:f32,d:f32,p:Vec<f32>}
impl Output{
    fn json(&self)->Value{json!({"board":self.board,"player":self.side,"win":self.w,"loss":self.l,"draw":self.d,"value":self.w-self.l,"policy":self.p})}
}
fn parse(line:&str,id:u64,board:&str,side:i8)->Result<Output,String>{
    let t:Vec<_>=line.split_whitespace().collect();
    if t.len()!=232||t[0]!="QZ_RESULT"||t[1].parse::<u64>().ok()!=Some(id)||t[2].parse::<usize>().ok()!=Some(board.bytes().filter(|&c|c!=b'0').count())||t[3].parse::<i8>().ok()!=Some(side){return Err("bridge response identity/length mismatch".into());}
    let x:Vec<f32>=t[4..].iter().map(|v|v.parse::<f32>().map_err(|_|"invalid float".to_string())).collect::<Result<_,_>>()?;
    if x.iter().any(|v|!v.is_finite()||*v<0.)||(x[..3].iter().sum::<f32>()-1.).abs()>1e-5||(x[3..].iter().sum::<f32>()-1.).abs()>1e-5{return Err("invalid WDL/policy distribution".into());}
    let p=x[3..].to_vec();
    if board.bytes().zip(&p).any(|(c,v)|c!=b'0'&&*v!=0.){return Err("occupied action prior".into());}
    Ok(Output{board:board.into(),side,w:x[0],l:x[1],d:x[2],p})
}
struct Bridge{
    child:Child,stdin:ChildStdin,stdout:BufReader<ChildStdout>,
    seq:u64,calls:u64,elapsed_ns:u128,first:Option<Output>,
}
impl Bridge{
    fn read(&mut self)->Result<String,String>{
        let mut line=String::new();
        for _ in 0..128{
            line.clear();
            let n=self.stdout.by_ref().take(65536).read_line(&mut line).map_err(|e|e.to_string())?;
            if n==0||!line.ends_with('\n'){return Err("bridge EOF or oversized line".into());}
            if line.starts_with("MESSAGE "){continue;}
            return Ok(line.trim_end().to_owned());
        }
        Err("too many bridge messages".into())
    }
    fn send(&mut self,line:&str)->Result<(),String>{
        writeln!(self.stdin,"{line}").and_then(|_|self.stdin.flush()).map_err(|e|e.to_string())
    }
    fn open()->Result<Self,String>{
        let exe=std::env::var_os("QUARTZ_RAPFI_EVAL_EXE").ok_or("missing QUARTZ_RAPFI_EVAL_EXE")?;
        let weight=std::env::var_os("QUARTZ_RAPFI_EVAL_WEIGHT").ok_or("missing QUARTZ_RAPFI_EVAL_WEIGHT")?;
        let cwd=std::path::Path::new(&weight).parent().ok_or("missing model directory")?;
        let mut child=Command::new(exe).arg("--quartz-eval").arg(&weight).current_dir(cwd).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().map_err(|e|e.to_string())?;
        let stdin=child.stdin.take().ok_or("missing stdin")?;
        let stdout=BufReader::new(child.stdout.take().ok_or("missing stdout")?);
        let mut b=Self{child,stdin,stdout,seq:0,calls:0,elapsed_ns:0,first:None};
        if b.read()? != "QZ_READY 250615 mix9svq freestyle15"{return Err("wrong Rapfi evaluator handshake".into());}
        Ok(b)
    }
    fn reset(&mut self)->Result<(),String>{
        self.send("RESET")?;
        if self.read()?!="QZ_RESET"{return Err("reset not acknowledged".into());}
        self.calls=0;self.elapsed_ns=0;self.first=None;Ok(())
    }
    fn eval(&mut self,s:&Gomoku15)->Result<Output,String>{
        let board:String=s.board_as_12().iter().map(|v|(b'0'+*v) as char).collect();
        self.seq+=1;let id=self.seq;
        let t=Instant::now();
        self.send(&format!("EVAL {id} {board}"))?;
        let line=self.read()?;
        let a=parse(&line,id,&board,s.current_player())?;
        self.elapsed_ns+=t.elapsed().as_nanos();
        self.calls+=1;if self.first.is_none(){self.first=Some(a.clone());}
        Ok(a)
    }
}
impl Drop for Bridge{
    fn drop(&mut self){let _=self.send("QUIT");let _=self.child.wait();}
}
struct SharedEval(Mutex<Bridge>);
impl Evaluator<Gomoku15> for SharedEval{
    fn evaluate(&self,s:&Gomoku15)->EvalResult<u16>{
        assert!(!s.is_terminal());
        let a=self.0.lock().unwrap().eval(s).expect("Rapfi evaluator failed; no fallback");
        let legal=s.legal_moves();
        assert!(!legal.is_empty());
        let policy=legal.into_iter().map(|m|(m,a.p[m as usize])).collect();
        EvalResult{policy,value:a.w-a.l}
    }
}
static SESSION:OnceLock<Mutex<Option<Arc<SharedEval>>>>=OnceLock::new();
fn session()->&'static Mutex<Option<Arc<SharedEval>>>{SESSION.get_or_init(||Mutex::new(None))}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request{
    cmd:String,
    #[serde(default)]rule:Option<String>,
    #[serde(default)]history:Option<Vec<u16>>,
    #[serde(default)]arm:Option<String>,
    #[serde(default)]visits:Option<u32>,
    #[serde(default)]seed:Option<u64>,
}
fn inner(line:&str)->Result<Value,String>{
    let q:Request=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd=="research_rapfi_shared_open"||q.cmd=="research_rapfi_shared_close"{
        if q.rule.is_some()||q.history.is_some()||q.arm.is_some()||q.visits.is_some()||q.seed.is_some(){return Err("open/close do not accept search options".into());}
        let mut s=session().lock().unwrap();
        if q.cmd=="research_rapfi_shared_open"{
            if s.is_some(){return Err("session already open".into());}
            let t=Instant::now();*s=Some(Arc::new(SharedEval(Mutex::new(Bridge::open()?))));
            return Ok(json!({"schema":"quartz_rapfi_shared_v1","status":"OPEN","ready":"250615 mix9svq freestyle15","load_us":t.elapsed().as_micros()}));
        }
        let child=s.take().ok_or("session not open")?;
        drop(child);return Ok(json!({"schema":"quartz_rapfi_shared_v1","status":"CLOSED"}));
    }
    if q.cmd!="research_rapfi_shared_search"||q.rule.as_deref()!=Some("freestyle"){return Err("freestyle search only".into());}
    let h=q.history.ok_or("history required")?;let arm=q.arm.ok_or("arm required")?;
    let cap=q.visits.ok_or("visits required")?;let seed=q.seed.ok_or("seed required")?;
    let axis=match arm.as_str(){"AX0"=>Some((0,0,0)),"AXS"=>Some((1,0,0)),"AXX"=>Some((2,0,0)),"AXV"=>Some((0,3,0)),"AXB"=>Some((0,0,3)),"AXALL"=>Some((3,3,3)),_=>None};
    if (!matches!(arm.as_str(),"STANDARD_PW"|"STANDARD_FULL"|"QUARTZ")&&axis.is_none())||!(1..=1024).contains(&cap)||h.len()>224||axis.is_some()&&cap!=96{return Err("bad arm/budget/history".into());}
    let start=Instant::now();
    let eval=session().lock().unwrap().as_ref().cloned().ok_or("session not open")?;
    let variant=GomokuVariant::Freestyle;
    // Rapfi evaluator is board/side-only. Both arms use the ordinary same-rule
    // board TT fingerprint, not a neural move-history feature identity.
    let mut s=Gomoku15::new(variant);
    for &m in &h{
        if m>=225||s.is_terminal()||!s.legal_moves().contains(&m){return Err("invalid history".into());}
        s=s.apply_move(m);
    }
    if s.is_terminal(){return Err("terminal root".into());}
    let legal=s.legal_moves();let side=s.current_player();
    let p=proof::scan(s.board_as_12(),if side==1{1}else{2},false);
    assert_eq!(p.bounds.iter().map(|b|b.action).collect::<Vec<_>>(),legal);
    assert_eq!(p.own,legal.iter().copied().filter(|&m|s.is_winning_move(m)).collect::<Vec<_>>());
    let board=s.board_as_12().to_vec();
    let mut out=json!({"schema":"quartz_rapfi_shared_v1","status":"OK","arm":arm,"rule":"freestyle","history":h,"player":side,"legal":legal,"board":board,"seed":seed,"cap":cap,"proof":proof_json(&p),"evaluator":"rapfi_250615_mix9svq","threads":1});
    // Shared, game-rule-only guard; no advantage attributed to Quartz.
    if let Some(m)=proof::guard(&p){
        out["axis"]=Value::Null;out["cause"]=json!("GUARDED");out["best"]=json!(m);out["visits"]=json!(0);out["NN_calls"]=json!(0);out["root_eval"]=Value::Null;out["initial"]=Value::Null;out["result"]=Value::Null;out["quartz"]=Value::Null;out["eval_ipc_ns"]=json!(0);out["whole_us"]=json!(start.elapsed().as_micros());return Ok(out);
    }
    eval.0.lock().unwrap().reset()?;
    let mut cfg=gomoku15_quartz(variant).with_seed(seed);
    cfg.root_forced_win=false;cfg.exact_terminal_value=false;cfg.fpu_reduction=0.;cfg.dirichlet=None;cfg.temperature=0.;
    cfg.vl_mode=crate::mcts::parallel::VlMode::Disabled;
    // The original run_quartz does not use the separate legacy GVOC state.
    cfg.gvoc=None;
    if arm!="QUARTZ"{cfg.quartz=None;}
    if arm=="STANDARD_FULL"{cfg.pw=None;}
    let qcfg=cfg.quartz.clone();
    let engine=MctsEngine::new(s,eval.clone(),cfg);
    let init=engine.four_axis_snapshot();
    let t=Instant::now();
    let mut axis_output=Value::Null;
    let (visits,cause,telemetry)=if let Some((mask,v,b))=axis{
        use crate::mcts::four_axis::Hooks;
        let mut hook=if b==3{Hooks::new_modes(0,0,5,false).with_gomoku_exact_merge()}else{Hooks::new_modes(0,0,0,false)};
        let mut phases=Vec::new();
        for (i,(m,vm,bm)) in [(0,0,if b==3{5}else{0}),(mask,v,b),(0,0,0)].into_iter().enumerate(){
            hook.mask=m;hook.configure_local(0,vm,bm,true);
            let t0=Instant::now();engine.four_axis_block(32,1,&hook);
            let step=t0.elapsed().as_micros();
            phases.push(json!({"visits":32*(i+1),"step_us":step,"result":engine.four_axis_snapshot(),"counters":hook.snapshot()}));
        }
        axis_output=json!({"phases":phases,"exact_merge":b==3,"b_trace":hook.gomoku_backup_trace(),"mask":mask,"vmode":v,"bmode":b});
        (96u32,"AXIS_FIXED".to_string(),Value::Null)
    }else if let Some(qcfg)=qcfg{
        let mut ctrl=QuartzController::new(cap,qcfg);
        let stats=engine.run_quartz(&mut ctrl);
        let z=ctrl.last_stats();
        let checks=ctrl.halt_telemetry().into_iter().map(|c|json!({"visits":c.root_visits,"elapsed_ms":c.elapsed_ms,"p_flip":c.p_flip,"voc_total":c.voc_total,"voc_focus":c.voc_focus,"voc_expand":c.voc_expand,"voc_merge":c.voc_merge,"channel":c.voc_argmax_channel,"decision":c.decision,"triggered":c.triggered})).collect::<Vec<_>>();
        let sel=engine.selection_telemetry.snapshot();
        (stats.root_visits,format!("{:?}",stats.stop_reason),json!({"p_flip":z.p_flip,"hbar_eff":z.hbar_eff,"voc_total":z.unified.voc_total,"voc_focus":z.unified.voc_focus,"voc_expand":z.unified.voc_expand,"voc_merge":z.unified.voc_merge,"checks":checks,"root_selects":sel.root_selects,"penalty_abs_sum":sel.selected_penalty_abs_sum,"penalty_mode_invokes":sel.penalty_mode_invoke_count}))
    }else{
        let mut ctrl=FixedIterations::new(cap);let s=engine.run(&mut ctrl);(s.root_visits,format!("{:?}",s.stop_reason),Value::Null)
    };
    let search_us=t.elapsed().as_micros();let result=engine.four_axis_snapshot();let best=result["best"].as_u64().ok_or("no best move")?;
    assert!(visits>0&&visits<=cap);
    assert_eq!(result["root_n"].as_u64(),Some(visits as u64));
    if arm!="QUARTZ"{assert_eq!(visits,cap);}
    let native_calls=engine.evaluator_call_count();
    let (calls,ipc,first)={let b=eval.0.lock().unwrap();(b.calls,b.elapsed_ns,b.first.as_ref().ok_or("missing root NN eval")?.json())};
    assert_eq!(calls,native_calls as u64);
    drop(engine);drop(eval);
    out["cause"]=json!(cause);out["best"]=json!(best);out["visits"]=json!(visits);out["axis"]=axis_output;out["NN_calls"]=json!(calls);out["root_eval"]=first;out["initial"]=init;out["result"]=result;out["quartz"]=telemetry;out["search_us"]=json!(search_us);out["eval_ipc_ns"]=json!(ipc);out["whole_us"]=json!(start.elapsed().as_micros());Ok(out)
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn strict_bridge_parser(){
        let b="0".repeat(225);let row=format!("QZ_RESULT 1 0 1 0.3 0.5 0.2 {}",(0..225).map(|_|"0.00444444444").collect::<Vec<_>>().join(" "));
        assert!(parse(&row,1,&b,1).is_ok());
        assert!(parse(&row,2,&b,1).is_err());
        assert!(parse(&row,1,&b,-1).is_err());
        assert!(parse(&row.replace("0.3","NaN"),1,&b,1).is_err());
        assert!(parse(&row.replace("0.3","0.8"),1,&b,1).is_err());
    }
}
