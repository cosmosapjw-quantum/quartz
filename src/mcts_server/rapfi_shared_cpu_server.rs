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
struct KModel{kind:String,first_gain:Vec<Vec<f64>>,first_cost:Vec<Vec<f64>>,second_gain:Vec<Vec<f64>>,second_cost:Vec<Vec<f64>>}
impl KModel{
    fn valid(&self,arm:&str)->bool{
        let kind=match arm{"KRAW"=>"RAW","KGIBBS"=>"GIBBS","KFLUX"=>"FLUX",_=>return false};
        if self.kind!=kind{return false;}
        [&self.first_gain,&self.first_cost,&self.second_gain,&self.second_cost].iter().all(|x|
            x.len()==5&&x[0].iter().all(|v|*v==0.)&&x.iter().all(|r|r.len()==crate::mcts::kinetic_response::DIM&&r.iter().all(|v|v.is_finite()&&v.abs()<=1e6)))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request{
    cmd:String,
    #[serde(default)]rule:Option<String>,
    #[serde(default)]history:Option<Vec<u16>>,
    #[serde(default)]arm:Option<String>,
    #[serde(default)]visits:Option<u32>,
    #[serde(default)]seed:Option<u64>,
    #[serde(default)]prefix:Option<u32>,
    #[serde(default)]schedule:Option<Vec<u8>>,
    #[serde(default)]model:Option<KModel>,
}
fn inner(line:&str)->Result<Value,String>{
    let q:Request=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd=="research_rapfi_shared_open"||q.cmd=="research_rapfi_shared_close"{
        if q.rule.is_some()||q.history.is_some()||q.arm.is_some()||q.visits.is_some()||q.seed.is_some()||q.prefix.is_some()||q.schedule.is_some()||q.model.is_some(){return Err("open/close do not accept search options".into());}
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
    let exposure=matches!(arm.as_str(),"K15N"|"K15S"|"K15X"|"K15V"|"K15B"|"K15F4"|"K15FH"|"K15XH"|"K15STOP"|"K15LOCK");
    if exposure&&(![32,96].contains(&cap)||q.prefix.is_some()||q.schedule.is_some()||q.model.is_some()){return Err("bad K15 budget/options".into());}
    let kinetic=matches!(arm.as_str(),"KINETIC"|"KRAW"|"KGIBBS"|"KFLUX");
    if kinetic&&(cap!=96||q.prefix.is_some())||arm=="KINETIC"&&(!q.schedule.as_ref().is_some_and(|v|v.len()==2&&v.iter().all(|&x|x<5))||q.model.is_some())||matches!(arm.as_str(),"KRAW"|"KGIBBS"|"KFLUX")&&(q.schedule.is_some()||!q.model.as_ref().is_some_and(|v|v.valid(&arm)))||!kinetic&&q.model.is_some(){return Err("bad kinetic cap/schedule/model".into());}
    let meta=matches!(arm.as_str(),"META"|"OPEN"|"YOKE"|"MBASE");
    if meta&&(cap!=256||q.prefix.is_some())||arm=="YOKE"&&!q.schedule.as_ref().is_some_and(|v|v.len()==15&&v.iter().all(|&x|x<5))||arm!="YOKE"&&arm!="KINETIC"&&q.schedule.is_some(){return Err("bad meta cap/prefix/schedule".into());}
    let p5=matches!(arm.as_str(),"P5PW"|"P5FULL"|"P5Q"|"P5ALLOC");
    if p5&&(![32,48,64,96,128,160,192,256].contains(&cap)||q.prefix.is_some()){return Err("bad P5 budget or prefix".into());}
    let axis=match arm.as_str(){"AX0"=>Some((0,0,0)),"AXS"=>Some((1,0,0)),"AXX"=>Some((2,0,0)),"AXV"=>Some((0,3,0)),"AXB"=>Some((0,0,3)),"AXALL"=>Some((3,3,3)),_=>None};
    if (!matches!(arm.as_str(),"STANDARD_PW"|"STANDARD_FULL"|"QUARTZ")&&axis.is_none()&&!p5&&!meta&&!kinetic&&!exposure&&!matches!(arm.as_str(),"DS0"|"DSF"|"DSX"|"DSL"|"DSV"|"DSB"|"DSSTOP"))||!(1..=1024).contains(&cap)||h.len()>224||axis.is_some()&&cap!=96{return Err("bad arm/budget/history".into());}
    let ds=matches!(arm.as_str(),"DS0"|"DSF"|"DSX"|"DSL"|"DSV"|"DSB"|"DSSTOP");
    if ds&&(cap!=32||!matches!(q.prefix,Some(8|16)))||!ds&&q.prefix.is_some(){return Err("decision arms require cap32 and prefix8/16 only".into());}
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
        out["exposure"]=Value::Null;out["kinetic"]=Value::Null;out["meta"]=Value::Null;out["resource"]=Value::Null;out["decision"]=Value::Null;out["axis"]=Value::Null;out["cause"]=json!("GUARDED");out["best"]=json!(m);out["visits"]=json!(0);out["NN_calls"]=json!(0);out["root_eval"]=Value::Null;out["initial"]=Value::Null;out["result"]=Value::Null;out["quartz"]=Value::Null;out["eval_ipc_ns"]=json!(0);out["whole_us"]=json!(start.elapsed().as_micros());return Ok(out);
    }
    eval.0.lock().unwrap().reset()?;
    let mut cfg=gomoku15_quartz(variant).with_seed(seed);
    cfg.root_forced_win=false;cfg.exact_terminal_value=false;cfg.fpu_reduction=0.;cfg.dirichlet=None;cfg.temperature=0.;
    cfg.vl_mode=crate::mcts::parallel::VlMode::Disabled;
    // The original run_quartz does not use the separate legacy GVOC state.
    cfg.gvoc=None;
    if arm!="QUARTZ"&&arm!="P5Q"{cfg.quartz=None;}
    if arm=="STANDARD_FULL"||arm=="P5FULL"{cfg.pw=None;}
    let qcfg=cfg.quartz.clone();
    let engine=MctsEngine::new(s,eval.clone(),cfg);
    let init=engine.four_axis_snapshot();
    let t=Instant::now();
    let mut axis_output=Value::Null;let mut decision_output=Value::Null;let mut resource_output=Value::Null;let mut meta_output=Value::Null;let mut kinetic_output=Value::Null;let mut exposure_output=Value::Null;
    let (visits,cause,telemetry)=if exposure{
        use crate::mcts::{four_axis::Hooks,decision_response as dr,kinetic_response as kr,meta_feedback as mf};
        let v0={let b=eval.0.lock().unwrap();let a=b.first.as_ref().ok_or("missing initial evaluator output")?;(a.w-a.l) as f64};
        let mut hook=Hooks::new_modes(0,0,5,false).with_gomoku_exact_merge();
        let pre=8u32;let high=if cap==32{16}else{56};
        let t0=Instant::now();engine.four_axis_block(pre,1,&hook);
        let pre_us=t0.elapsed().as_micros();let snap=engine.four_axis_snapshot();let cc=hook.snapshot();
        let mon=meta_monitor(&snap,&cc,v0);
        let prior:Vec<_>=snap["candidates"].as_array().unwrap().iter().map(|x|x[1].as_f64().unwrap()).collect();
        let ca=snap["candidates"].as_array().unwrap();let visible=snap["edges"].as_array().unwrap().len();
        let (l,rival,hidden)=dr::candidates(&mon.n,visible);
        let mv=|i:usize|ca[i][0].as_u64().unwrap() as usize;
        let target=match arm.as_str(){"K15F4"|"K15FH"=>rival,"K15XH"=>hidden,_=>None};
        let dose=if arm=="K15F4"{4}else{high};
        let force=target.map(|i|(i,mv(i)));
        let gg=[mon.phi,4.*mon.uncertainty,mon.js/std::f64::consts::LN_2,mon.hidden,mon.kl_prior,mon.entropy,0.,0.];
        let feats=std::array::from_fn::<_,3,_>(|k|kr::features(&prior,&mon.n,visible,mon.shared,v0,None,&gg,&[0.;6],k));
        let prefix=json!({"result":snap,"counters":cc,"monitor":meta_mon_json(&mon),"features":feats,"step_us":pre_us});
        let mut phases=Vec::new();let mut current=pre;let mut old_mon=mon;
        let mut locked=false;
        if arm=="K15LOCK"{
            while current<cap{
                if engine.vote_min_distance().is_none_or(|d|d>cap-current){locked=true;break;}
                let b=(cap-current).min(8);let t0=Instant::now();engine.four_axis_block(b,1,&hook);current+=b;
                let r=engine.four_axis_snapshot();let c=hook.snapshot();let m=meta_monitor(&r,&c,v0);
                let f=kr::flux(&old_mon.n,&m.n,&prior);
                phases.push(json!({"result":r,"counters":c,"flux":f,"monitor":meta_mon_json(&m),"step_us":t0.elapsed().as_micros(),"n":current}));
                old_mon=m;
            }
        }else if arm!="K15STOP"{
            hook.mask=match arm.as_str(){"K15S"=>1,"K15X"=>2,_=>0};
            hook.configure_local(0,if arm=="K15V"{3}else{0},if arm=="K15B"{3}else{5},true);
            if force.is_some(){hook.set_decision_force(force,pre,pre+dose);}
            let t0=Instant::now();engine.four_axis_block(dose,1,&hook);current+=dose;
            let r=engine.four_axis_snapshot();let c=hook.snapshot();let m=meta_monitor(&r,&c,v0);
            let f=kr::flux(&old_mon.n,&m.n,&prior);
            phases.push(json!({"result":r,"counters":c,"flux":f,"monitor":meta_mon_json(&m),"step_us":t0.elapsed().as_micros(),"n":current}));
            old_mon=m;
            if force.is_some(){hook.set_decision_force(None,0,0);}
            hook.mask=0;hook.configure_local(0,0,5,true);
            let t0=Instant::now();engine.four_axis_block(cap-current,1,&hook);current=cap;
            let r=engine.four_axis_snapshot();let c=hook.snapshot();let m=meta_monitor(&r,&c,v0);
            let f=kr::flux(&old_mon.n,&m.n,&prior);
            phases.push(json!({"result":r,"counters":c,"flux":f,"monitor":meta_mon_json(&m),"step_us":t0.elapsed().as_micros(),"n":current}));
            old_mon=m;
        }
        let trace=hook.gomoku_backup_trace();let rows=trace.as_array().unwrap();assert_eq!(rows.len(),current as usize);
        let mut forced=vec![0u32;ca.len()];
        if let Some((rank,m))=force{
            for tr in &rows[pre as usize..(pre+dose) as usize]{assert_eq!(tr["path"][0].as_u64(),Some(m as u64));forced[rank]+=1;}
        }
        let (raw,sub,w,tv)=dr::injection(&old_mon.n,&forced);
        assert_eq!(engine.best_move().unwrap() as usize,mv(raw));
        let min_d=engine.vote_min_distance();if locked{assert!(min_d.is_none_or(|d|d>cap-current));}
        exposure_output=json!({"prefix":prefix,"phases":phases,"dose":if force.is_some(){dose}else{0},"target":target.map(mv),"target_rank":target,"rival":rival.map(mv),"hidden":hidden.map(mv),"leader":mv(l),"forced":forced,"raw_best":mv(raw),"sub_best":mv(sub),"forced_fraction":w,"injection_tv":tv,"locked":locked,"min_d":min_d,"b_trace":trace});
        (current,(if locked{"K15_LOCKED"}else if arm=="K15STOP"{"K15_STOPPED"}else{"K15_FIXED"}).to_string(),Value::Null)
    }else if kinetic{
        use crate::mcts::{four_axis::Hooks,kinetic_response as kr,meta_feedback as mf};
        let v0={let b=eval.0.lock().unwrap();let a=b.first.as_ref().ok_or("missing initial evaluator output")?;(a.w-a.l) as f64};
        let mut hook=Hooks::new_modes(0,0,5,false).with_gomoku_exact_merge();
        let t0=Instant::now();engine.four_axis_block(16,1,&hook);
        let pre_us=t0.elapsed().as_micros();
        let mut prev=engine.four_axis_snapshot();let cc=hook.snapshot();
        let mut mon=meta_monitor(&prev,&cc,v0);
        let prior:Vec<_>=prev["candidates"].as_array().unwrap().iter().map(|x|x[1].as_f64().unwrap()).collect();
        let make_features=|r:&Value,m:&mf::Monitor,first:Option<usize>,f:&[f64;6],dp:f64,da:f64|{
            let g=[m.phi,4.*m.uncertainty,m.js/std::f64::consts::LN_2,m.hidden,m.kl_prior,m.entropy,dp,4.*da];
            std::array::from_fn::<_,3,_>(|k|kr::features(&prior,&m.n,r["edges"].as_array().unwrap().len(),m.shared,v0,first,&g,f,k))
        };
        let mut feats=make_features(&prev,&mon,None,&[0.;6],0.,0.);
        let prefix=json!({"result":prev,"counters":cc,"monitor":meta_mon_json(&mon),"features":feats,"step_us":pre_us});
        let mut steps=Vec::new();let mut first=None;
        for (i,quantum) in [32u32,32,16].into_iter().enumerate(){
            let t0=Instant::now();
            let (ch,decision)=if i==2{(0,Value::Null)}
                else if arm=="KINETIC"{(q.schedule.as_ref().unwrap()[i] as usize,Value::Null)}
                else{
                    let model=q.model.as_ref().unwrap();let kind=match arm.as_str(){"KRAW"=>0,"KGIBBS"=>1,_=>2};
                    let (g,c)=if i==0{(&model.first_gain,&model.first_cost)}else{(&model.second_gain,&model.second_cost)};
                    let (ch,gg,cc,ss)=kr::choose(&feats[kind],g,c);
                    (ch,json!({"gain":gg,"cost_ms":cc,"net":ss,"kind":kind}))
                };
            if i==0{first=Some(ch);}
            hook.mask=match ch{1=>1,2=>2,_=>0};
            hook.configure_local(0,if ch==3{3}else{0},if ch==4{3}else{5},true);
            let plan_us=t0.elapsed().as_micros();let t0=Instant::now();
            engine.four_axis_block(quantum,1,&hook);
            let block_us=t0.elapsed().as_micros();let t0=Instant::now();
            let next=engine.four_axis_snapshot();let next_c=hook.snapshot();
            let nxt=meta_monitor(&next,&next_c,v0);
            let f=kr::flux(&mon.n,&nxt.n,&prior);
            let nn=next["evaluator_calls"].as_u64().unwrap()-prev["evaluator_calls"].as_u64().unwrap();
            let res=mf::response(&mon,&nxt,nn as u32);
            feats=make_features(&next,&nxt,first,&f,res.delta_phi,res.delta_a);
            let read_us=t0.elapsed().as_micros();
            steps.push(json!({"channel":ch,"quantum":quantum,"decision":decision,"result":next,"counters":next_c,"monitor":meta_mon_json(&nxt),"features":feats,"flux":f,"nn":nn,"delta_phi":res.delta_phi,"delta_a":res.delta_a,"plan_us":plan_us,"block_us":block_us,"read_us":read_us}));
            prev=next;mon=nxt;
        }
        kinetic_output=json!({"prefix":prefix,"steps":steps,"b_trace":hook.gomoku_backup_trace(),"price":kr::PRICE});
        (96u32,"KINETIC_FIXED".to_string(),Value::Null)
    }else if meta{
        use crate::mcts::four_axis::Hooks;
        use crate::mcts::meta_feedback as mf;
        let v0={let b=eval.0.lock().unwrap();let a=b.first.as_ref().ok_or("missing initial evaluator output")?;(a.w-a.l) as f64};
        let mut hook=Hooks::new_modes(0,0,5,false).with_gomoku_exact_merge();
        let t0=Instant::now();engine.four_axis_block(16,1,&hook);let pre_us=t0.elapsed().as_micros();
        let mut prev=engine.four_axis_snapshot();let mut counters=hook.snapshot();
        let mut mon=meta_monitor(&prev,&counters,v0);
        let prefix=json!({"result":prev,"counters":counters,"monitor":meta_mon_json(&mon),"step_us":pre_us});
        let mut memory=[0f64;5];let mut steps=Vec::new();
        for i in 0..15{
            let t0=Instant::now();
            let (omega,z)=mf::policy(&mon,if arm=="META"{&memory}else{&[0.;5]});
            let u=mf::uniform(seed,i);
            let channel=match arm.as_str(){"MBASE"=>0,"YOKE"=>q.schedule.as_ref().unwrap()[i] as usize,_=>mf::select(&z,u)};
            let before=memory;
            hook.mask=match channel{1=>1,2=>2,_=>0};
            hook.configure_local(0,if channel==3{3}else{0},if channel==4{3}else{5},true);
            let plan_us=t0.elapsed().as_micros();
            let t0=Instant::now();engine.four_axis_block(16,1,&hook);let block_us=t0.elapsed().as_micros();
            let t0=Instant::now();let next=engine.four_axis_snapshot();let next_c=hook.snapshot();
            let nxt=meta_monitor(&next,&next_c,v0);
            let nn=next["evaluator_calls"].as_u64().unwrap()-prev["evaluator_calls"].as_u64().unwrap();
            let res=mf::response(&mon,&nxt,nn as u32);
            memory[channel]=mf::update_memory(memory[channel],res.y);
            let read_us=t0.elapsed().as_micros();
            steps.push(json!({"channel":channel,"uniform":u,"omega":omega,"proposal":z,
                "memory_before":before,"memory_after":memory,"prediction_error":res.y-before[channel],
                "response":{"work":res.work,"reorganization":res.reorganization,"delta_phi":res.delta_phi,"depletion":res.depletion,"transport":res.transport,"delta_a":res.delta_a,"y":res.y,"nn":nn},
                "monitor":meta_mon_json(&nxt),"result":next,"counters":next_c,"plan_us":plan_us,"block_us":block_us,"read_us":read_us}));
            prev=next;counters=next_c;mon=nxt;
        }
        meta_output=json!({"prefix":prefix,"steps":steps,"b_trace":hook.gomoku_backup_trace(),"mode":arm,"beta":mf::BETA,"kappa":mf::KAPPA,"eta":mf::ETA});
        (256u32,"META_FIXED".to_string(),Value::Null)
    }else if p5&&arm!="P5Q"{
        use crate::mcts::four_axis::Hooks;
        let hook=Hooks::new_modes(0,0,0,false);
        engine.four_axis_block(16,1,&hook);
        let prefix_d=engine.vote_min_distance();
        let prefix_best=engine.best_move().unwrap();
        let selected=if arm=="P5ALLOC"{crate::mcts::four_axis_math::allocation_cap(cap,prefix_d)}else{cap};
        let mut n=16;let mut locked=false;let mut checks=0;
        while n<selected{
            checks+=1;
            if engine.vote_min_distance().is_none_or(|d|d>selected-n){locked=true;break;}
            let block=8.min(selected-n);engine.four_axis_block(block,1,&hook);n+=block;
        }
        resource_output=json!({"selected_cap":selected,"prefix_d":prefix_d,"prefix_best":prefix_best,"hard":if arm=="P5ALLOC"{Some(prefix_d.is_some_and(|d|d<=4))}else{None},"locked":locked,"checks":checks,"final_d":engine.vote_min_distance()});
        (n,(if locked{"P5_LOCKED"}else{"P5_FIXED"}).to_string(),Value::Null)
    }else if ds{
        use crate::mcts::four_axis::Hooks;
        use crate::mcts::four_axis_math::vote_reach;
        let pre=q.prefix.unwrap();
        let merge=arm=="DSB";
        let mut hook=if merge{Hooks::new_modes(0,0,5,false).with_gomoku_exact_merge()}else{Hooks::new_modes(0,0,0,false)};
        let mut phases=Vec::new();let mut current=0u32;let mut target=None;let mut target_info=Value::Null;
        let mut stopped=false;
        let mut ends=vec![pre,pre+4,pre+8,pre+16,32];ends.sort_unstable();ends.dedup();
        for end in ends{
            // STOP uses exactly the standard phase boundaries after prefix
            // and after the 8-visit intervention slot, never mid-iteration.
            let t0=Instant::now();engine.four_axis_block(end-current,1,&hook);
            let step=t0.elapsed().as_micros();current=end;
            let r=engine.four_axis_snapshot();let ca=r["candidates"].as_array().unwrap();let es=r["edges"].as_array().unwrap();
            let mut counts=vec![0u32;ca.len()];
            for (i,e) in es.iter().enumerate(){assert_eq!(e["move"],ca[i][0]);counts[i]=e["n"].as_u64().unwrap() as u32;}
            assert_eq!(counts.iter().sum::<u32>(),current);
            let (leader,dist)=vote_reach(&counts);
            assert_eq!(r["best"],ca[leader][0]);
            let min_d=dist.iter().flatten().min().copied();
            let locked=min_d.is_none_or(|d|d>32-current);
            phases.push(json!({"visits":current,"step_us":step,"result":r,"counters":hook.snapshot(),"min_d":min_d,"locked":locked}));
            if arm=="DSSTOP"&&current!=pre+4&&current<32&&locked{stopped=true;break;}
            if current==pre{
                let reachable=|i:usize|dist[i].is_some_and(|d|d<=32-pre);
                let idx=match arm.as_str(){
                    "DSF"=>(0..es.len()).filter(|&i|counts[i]>0&&reachable(i)).max_by_key(|&i|(counts[i],i)),
                    "DSX"=>(es.len()..ca.len()).find(|&i|reachable(i)),
                    "DSL"=>Some(leader),
                    _=>None
                };
                target=idx.map(|i|(i,ca[i][0].as_u64().unwrap() as usize));
                if let Some((i,m))=target{target_info=json!({"rank":i,"move":m,"d":dist[i],"n":counts[i],"materialized":i<es.len(),"leader":ca[leader][0]});}
                hook.set_decision_force(target,pre,pre+4);
                hook.configure_local(0,if arm=="DSV"{3}else{0},if merge{3}else{0},true);
            }else if current==pre+4{
                hook.set_decision_force(None,0,0);
            }else if current==pre+8{
                hook.configure_local(0,0,0,true);
            }
        }
        decision_output=json!({"prefix":pre,"phases":phases,"target":target_info,"stopped":stopped,"b_trace":hook.gomoku_backup_trace(),"exact_merge":merge});
        (current,(if stopped{"DS_LOCKED"}else{"DS_FIXED"}).to_string(),Value::Null)
    }else if let Some((mask,v,b))=axis{
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
        let (stats,locked)=if arm=="P5Q"{engine.run_quartz_vote_locked(&mut ctrl)}else{(engine.run_quartz(&mut ctrl),false)};
        if arm=="P5Q"{resource_output=json!({"selected_cap":cap,"prefix_d":null,"prefix_best":null,"hard":null,"locked":locked,"checks":null,"final_d":engine.vote_min_distance()});}
        let z=ctrl.last_stats();
        let checks=ctrl.halt_telemetry().into_iter().map(|c|json!({"visits":c.root_visits,"elapsed_ms":c.elapsed_ms,"p_flip":c.p_flip,"voc_total":c.voc_total,"voc_focus":c.voc_focus,"voc_expand":c.voc_expand,"voc_merge":c.voc_merge,"channel":c.voc_argmax_channel,"decision":c.decision,"triggered":c.triggered})).collect::<Vec<_>>();
        let sel=engine.selection_telemetry.snapshot();
        (stats.root_visits,if locked{"P5_LOCKED".to_string()}else{format!("{:?}",stats.stop_reason)},json!({"p_flip":z.p_flip,"hbar_eff":z.hbar_eff,"voc_total":z.unified.voc_total,"voc_focus":z.unified.voc_focus,"voc_expand":z.unified.voc_expand,"voc_merge":z.unified.voc_merge,"checks":checks,"root_selects":sel.root_selects,"penalty_abs_sum":sel.selected_penalty_abs_sum,"penalty_mode_invokes":sel.penalty_mode_invoke_count}))
    }else{
        let mut ctrl=FixedIterations::new(cap);let s=engine.run(&mut ctrl);(s.root_visits,format!("{:?}",s.stop_reason),Value::Null)
    };
    let search_us=t.elapsed().as_micros();let result=engine.four_axis_snapshot();let best=result["best"].as_u64().ok_or("no best move")?;
    assert!(visits>0&&visits<=if arm=="P5ALLOC"{2*cap}else{cap});
    assert_eq!(result["root_n"].as_u64(),Some(visits as u64));
    if arm!="QUARTZ"&&arm!="DSSTOP"&&arm!="K15STOP"&&arm!="K15LOCK"&&!p5{assert_eq!(visits,cap);}
    let native_calls=engine.evaluator_call_count();
    let (calls,ipc,first)={let b=eval.0.lock().unwrap();(b.calls,b.elapsed_ns,b.first.as_ref().ok_or("missing root NN eval")?.json())};
    assert_eq!(calls,native_calls as u64);
    drop(engine);drop(eval);
    out["cause"]=json!(cause);out["best"]=json!(best);out["visits"]=json!(visits);out["axis"]=axis_output;out["decision"]=decision_output;out["resource"]=resource_output;out["meta"]=meta_output;out["kinetic"]=kinetic_output;out["exposure"]=exposure_output;out["NN_calls"]=json!(calls);out["root_eval"]=first;out["initial"]=init;out["result"]=result;out["quartz"]=telemetry;out["search_us"]=json!(search_us);out["eval_ipc_ns"]=json!(ipc);out["whole_us"]=json!(start.elapsed().as_micros());Ok(out)
}
fn meta_monitor(r:&Value,c:&Value,v0:f64)->crate::mcts::meta_feedback::Monitor{
    let ca=r["candidates"].as_array().unwrap();let es=r["edges"].as_array().unwrap();
    let p:Vec<_>=ca.iter().map(|x|x[1].as_f64().unwrap()).collect();
    let mut n=vec![0u32;ca.len()];let mut w=vec![0.;ca.len()];
    for (i,e) in es.iter().enumerate(){assert_eq!(e["move"],ca[i][0]);n[i]=e["n"].as_u64().unwrap() as u32;w[i]=e["w"].as_f64().unwrap();}
    assert_eq!(n.iter().sum::<u32>(),r["root_n"].as_u64().unwrap() as u32);
    crate::mcts::meta_feedback::monitor(&p,&n,&w,es.len(),c["ledger_shared_children"].as_u64().unwrap(),v0)
}
fn meta_mon_json(m:&crate::mcts::meta_feedback::Monitor)->Value{
    json!({"phi":m.phi,"uncertainty":m.uncertainty,"js":m.js,"hidden":m.hidden,"shared":m.shared,"kl_prior":m.kl_prior,"entropy":m.entropy})
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
