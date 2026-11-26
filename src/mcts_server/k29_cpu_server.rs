//! Research endpoint for real two-block decision laws and CPU leaf batching.
use super::*;
use crate::mcts::four_axis::{Hooks,LiteRoot};
use serde::Deserialize;
use serde_json::{json,Value};
#[path="k29_queue.rs"] mod queue;
#[path="k29_belief.rs"] mod belief;
const ARMS:[(u8,usize,usize,bool,u64);8]=[
    (0,1,0,false,0),(0,4,0,false,0),(0,8,0,false,0),
    (0,4,4,false,0),(0,8,8,false,0),(0,8,8,true,0),
    (15,8,0,false,4),(15,8,8,true,4)];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,seed:u64,model_path:String,model_identity:String,
    #[serde(default)]arm:Option<usize>,#[serde(default)]policy:Option<String>,
    #[serde(default)]belief_path:Option<String>,#[serde(default)]belief_identity:Option<String>,
    #[serde(default)]trace:bool}
fn us(t:Instant)->u64{t.elapsed().as_micros() as u64}
fn model(q:&Req)->Result<Arc<ConvConnect4Evaluator>,String>{
    if q.model_identity.len()!=64||!q.model_identity.bytes().all(|x|x.is_ascii_hexdigit()){return Err("bad model identity".into());}
    let path=std::fs::canonicalize(&q.model_path).map_err(|e|e.to_string())?;
    let k=format!("{}|{}",path.display(),q.model_identity);
    static CACHE:OnceLock<Mutex<HashMap<String,Arc<ConvConnect4Evaluator>>>>=OnceLock::new();
    let cache=CACHE.get_or_init(||Mutex::new(HashMap::new()));
    if let Some(m)=cache.lock().unwrap().get(&k){return Ok(m.clone());}
    let m=Arc::new(ConvConnect4Evaluator::from_file(path.to_str().ok_or("bad model path")?)?);
    cache.lock().unwrap().insert(k,m.clone());Ok(m)
}
fn interval(root:&Connect4)->Vec<(usize,f64,f64)>{
    root.legal_moves().iter().map(|&m|{let s=root.apply_move(m);
        let(l,u)=if s.is_terminal(){let v=(1.-s.outcome() as f64)/2.;(v,v)}
        else if s.legal_moves().iter().any(|&a|s.is_winning_move(a)){(0.,0.)}else{(0.,1.)};
        (m,l,u)}).collect()
}
fn features(root:&Connect4,r:&LiteRoot,anchor:usize,proof:&[(usize,f64,f64)])->(Vec<usize>,Vec<f64>,Vec<(u8,u8)>){
    let mut ix:Vec<_>=(0..r.len).collect();
    ix.sort_by(|&a,&b|{let aa=r.moves[a]==anchor;let bb=r.moves[b]==anchor;
        bb.cmp(&aa).then(r.n[b].cmp(&r.n[a])).then(a.cmp(&b))});
    let mut f=vec![root.ply_count() as f64/42.];let mut rank=Vec::new();let mut allowed=Vec::new();
    let ht=root.heights();
    for i in ix{let m=r.moves[i];let (_,lo,hi)=proof.iter().find(|p|p.0==m).unwrap();
        rank.push(m);allowed.push(((2.*lo) as u8,(2.*hi) as u8));
        f.extend([(r.n[i] as f64/r.visits as f64).sqrt(),r.prior[i].max(0.).sqrt(),
            0.5+0.5*r.w[i]/(r.n[i] as f64+4.),(r.m2[i].max(0.)/(r.n[i] as f64+1.)).sqrt()/2.,
            ht[m] as f64/6.,(m as f64-3.).abs()/3.,*lo,*hi]);
    }
    (rank,f,allowed)
}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_k29_cpu"||q.arm.is_some()==q.policy.is_some()||q.arm.is_some_and(|a|a>8)||
       q.policy.as_ref().is_some_and(|p|!matches!(p.as_str(),"prior"|"continue"|"priced"))||
       (q.policy.is_some()!=(q.belief_path.is_some()&&q.belief_identity.is_some()))||
       (q.arm.is_some()&&(q.belief_path.is_some()||q.belief_identity.is_some())){return Err("bad K29 contract".into());}
    if rayon::current_num_threads()<8{return Err("K29 needs eight-worker pool".into());}
    let root=Connect4::from_history(&q.history)?;let legal=root.legal_moves();
    if root.is_terminal()||legal.len()<3{return Err("bad root".into());}
    let model=model(&q)?;
    let belief_model=if let Some(p)=&q.belief_path{Some(belief::load(p,q.belief_identity.as_ref().unwrap())?)}else{None};
    let ev=Arc::new(queue::QueueEval::new(model));
    let t=Instant::now();let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);
    cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));
    let e=MctsEngine::new(root.clone(),ev.clone(),cfg);let mut h=Hooks::new_modes(0,0,0,false).with_k25(q.trace);
    let setup_us=us(t);let t=Instant::now();e.four_axis_k25_block(8,1,&h);let prefix_us=us(t);
    let pre=e.four_axis_lite();let ph=h.k25_report(false);h.set_k25_mask(0);
    let t=Instant::now();let first=e.four_axis_k27_budget(32,128,1,false,0,&h);let first_us=us(t);
    let mid=e.four_axis_lite();let mh=h.k25_report(false);
    let proof=interval(&root);let ub=proof.iter().map(|p|p.2).fold(0.,f64::max);let cert=proof.iter().find(|p|p.1>=ub).map(|p|p.0);
    let anchor=cert.unwrap_or(mid.best);let(rank,f,allowed)=features(&root,&mid,anchor,&proof);
    let t=Instant::now();let prediction=belief_model.as_ref().map(|m|m.predict(&f,&allowed));
    let arm=if let Some(a)=q.arm{a}else{
        let p=prediction.as_ref().unwrap();match q.policy.as_deref().unwrap(){"prior"=>8,"continue"=>p.arm,"priced"=>p.priced,_=>unreachable!()}
    };
    let control_us=us(t);let mut batch=Value::Null;let mut tail=Value::Null;let mut tail_us=0;let mut queue_setup_us=0;let mut queue_join_us=0;
    if arm<8{
        let(mask,workers,width,dedup,reserve)=ARMS[arm];h.set_k25_mask(mask);
        let t=Instant::now();let j=if width>0{Some(ev.start(width,dedup,q.trace))}else{None};queue_setup_us=us(t);
        let t=Instant::now();tail=if reserve>0{e.four_axis_k28_floor(32,128,workers,reserve,&h)}
            else{e.four_axis_k27_budget(32,128,workers,false,0,&h)};tail_us=us(t);
        let t=Instant::now();if let Some(j)=j{batch=ev.finish(j);}queue_join_us=us(t);
    }
    let end=e.four_axis_lite();let eh=h.k25_report(q.trace);let snap=e.four_axis_snapshot();
    let native=cert.unwrap_or(end.best);let z=rank.iter().position(|&a|a==native).unwrap();
    let best=if let Some(p)=&prediction{cert.unwrap_or(rank[if arm==8{p.prior}else{p.post[arm][z]}])}else{native};
    let logical=end.nn;let physical=logical-batch["coalesced"].as_u64().unwrap_or(0);
    if !batch.is_null(){assert_eq!(batch["logical"],logical-mid.nn);assert_eq!(batch["physical"].as_u64().unwrap()+batch["coalesced"].as_u64().unwrap(),logical-mid.nn);}
    if arm==8{assert_eq!(mid.nn,end.nn);assert_eq!(mid.visits,end.visits);}
    drop(e);drop(h);drop(ev);
    Ok(json!({"schema":"quartz_k29_c4_v1","history":q.history,"model_identity":q.model_identity,"belief_identity":q.belief_identity,
        "policy":q.policy,"arm":arm,"legal":legal,"proof":proof,"prefix":pre,"prefix_hooks":ph,
        "first":first,"mid":mid,"mid_hooks":mh,"mid_best":anchor,"rank":rank,"features":f,"allowed":allowed,
        "tail":tail,"end":end,"end_hooks":eh,"snapshot":snap,"native_best":native,"z":z,"best":best,
        "NN_calls":logical,"physical_calls":physical,"visits":end.visits,"queue":batch,
        "prediction":prediction.map(|p|p.report),"setup_us":setup_us,"prefix_us":prefix_us,"first_us":first_us,
        "control_us":control_us,"tail_us":tail_us,"queue_setup_us":queue_setup_us,"queue_join_us":queue_join_us,"whole_us":us(start)}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
