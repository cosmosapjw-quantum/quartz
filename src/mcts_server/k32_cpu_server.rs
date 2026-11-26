//! Research endpoint for real two-block decision laws and CPU leaf batching.
use super::*;
use crate::mcts::four_axis::{Hooks,LiteRoot};
use serde::Deserialize;
use serde_json::{json,Value};
#[path="k30_queue.rs"] mod queue;
#[path="k31_proof.rs"] mod replay31;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,seed:u64,model_path:String,model_identity:String,
    arm:usize,#[serde(default)]trace:bool}
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
    if q.cmd!="research_connect4_k32_cpu"||q.arm>10{return Err("bad K32 contract".into());}
    if rayon::current_num_threads()<8{return Err("K32 needs eight-worker pool".into());}
    let root=Connect4::from_history(&q.history)?;let legal=root.legal_moves();
    if root.is_terminal()||legal.len()<3{return Err("bad root".into());}
    let model=model(&q)?;
    let ev=Arc::new(queue::QueueEval::new(model,&q.model_path,&q.model_identity));
    let t=Instant::now();let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);
    cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));
    let e=MctsEngine::new(root.clone(),ev.clone(),cfg);let arm=q.arm;
    let graph=(2..=8).contains(&arm);let mut h=Hooks::new_modes(0,0,0,false).with_k25(q.trace||arm==10);
    if graph{h=h.with_k32();h.k32_init(&root);}
    let setup_us=us(t);let t=Instant::now();e.four_axis_k25_block(8,1,&h);let prefix_us=us(t);
    let pre=e.four_axis_lite();let ph=h.k25_report(false);h.set_k25_mask(0);
    let t=Instant::now();let first=e.four_axis_k27_budget(32,128,1,false,0,&h);let first_us=us(t);
    let mid=e.four_axis_lite();let mh=h.k25_report(false);
    let proof=interval(&root);let ub=proof.iter().map(|p|p.2).fold(0.,f64::max);let cert=proof.iter().find(|p|p.1>=ub).map(|p|p.0);
    let anchor=cert.unwrap_or(mid.best);let(rank,f,allowed)=features(&root,&mid,anchor,&proof);
    let mg=h.k32_report(&root,&mid,anchor,false,false);
    let mut batch=Value::Null;let mut tail=Value::Null;let mut replay=Value::Null;let mut rule_steps=Value::Null;let mut tail_us=0;let mut queue_setup_us=0;let mut queue_join_us=0;
    if arm==1{
        h.set_k25_mask(0);let t=Instant::now();tail=e.four_axis_k27_budget(32,128,1,false,0,&h);tail_us=us(t);
    }else if arm==3{
        let t=Instant::now();rule_steps=e.k32_rule_expand(&h,anchor);tail_us=us(t);
    }else if (4..=9).contains(&arm){
        let full=arm>=7;let worker=if full{8}else{1};let guide=match arm{5=>2,6|7|8=>3,_=>0};
        h.set_k25_mask(if full{15}else{0});if graph{h.k32_activate();}
        let t=Instant::now();let j=if arm>=8{Some(ev.start(8,true,true,q.trace))}else{None};queue_setup_us=us(t);
        let t=Instant::now();tail=e.four_axis_k32_budget(32,128,worker,if full{4}else{0},guide,graph,&mut h);tail_us=us(t);
        let t=Instant::now();if let Some(j)=j{batch=ev.finish(j);}queue_join_us=us(t);
    }else if arm==10{
        let t=Instant::now();let r=h.k25_report(true);replay=replay31::execute(&root,&mid,anchor,&r["trace"],8,q.trace);tail_us=us(t);
    }
    let end=e.four_axis_lite();let eh=h.k25_report(q.trace);let snap=e.four_axis_snapshot();
    let native=cert.unwrap_or(end.best);
    let eg=h.k32_report(&root,&end,native,q.trace,true);
    let best=if graph{eg["best"].as_u64().unwrap() as usize}else if arm==10{replay["end"]["best"].as_u64().unwrap() as usize}else{native};
    let rules=if graph{eg["rule_work"].as_u64().unwrap()}else if arm==10{replay["end"]["rule_work"].as_u64().unwrap()}else{0};
    let logical=end.nn;let physical=logical-batch["coalesced"].as_u64().unwrap_or(0);
    if !batch.is_null(){assert_eq!(batch["logical"],logical-mid.nn);}
    drop(e);drop(h);drop(ev);
    Ok(json!({"schema":"quartz_k32_c4_v1","history":q.history,"model_identity":q.model_identity,"arm":arm,"legal":legal,"proof":proof,
        "prefix":pre,"prefix_hooks":ph,"first":first,"mid":mid,"mid_hooks":mh,"mid_best":anchor,"rank":rank,"features":f,"allowed":allowed,
        "tail":tail,"end":end,"end_hooks":eh,"snapshot":snap,"native_best":native,"best":best,"mid_graph":mg,"end_graph":eg,"rule_steps":rule_steps,"replay_graph":replay,"rule_work":rules,
        "NN_calls":logical,"physical_calls":physical,"visits":end.visits,"queue":batch,
        "setup_us":setup_us,"prefix_us":prefix_us,"first_us":first_us,"tail_us":tail_us,"queue_setup_us":queue_setup_us,"queue_join_us":queue_join_us,"whole_us":us(start)}))

}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
