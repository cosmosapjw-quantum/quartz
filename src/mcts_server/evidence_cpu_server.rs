//! Research-only matched resident root queries; no oracle in online path.
use super::*;
use crate::mcts::{decision_response as dr,evidence_pair as ep,four_axis::Hooks,kinetic_response as kr};
use serde::Deserialize;
use serde_json::{json,Value};
fn moments(r:&Value)->(Vec<u32>,Vec<f64>){
    let ca=r["candidates"].as_array().unwrap();let es=r["edges"].as_array().unwrap();
    let mut n=vec![0u32;ca.len()];let mut w=vec![0.;ca.len()];
    for (i,e) in es.iter().enumerate(){assert_eq!(e["move"],ca[i][0]);n[i]=e["n"].as_u64().unwrap() as u32;w[i]=e["w"].as_f64().unwrap();}
    assert_eq!(n.iter().sum::<u32>() as u64,r["root_n"].as_u64().unwrap());(n,w)
}
pub(super) fn run<G:GameState>(engine:&MctsEngine<G>,arm:&str,cap:u32)->Result<Value,String>{
    if !matches!(arm,"K16N"|"K16BAL"|"K16REV"|"K17EXP")||![32,64].contains(&cap){return Err("bad pair arm/cap".into());}
    let mut h=Hooks::new_modes(0,0,0,false);
    let t=Instant::now();engine.four_axis_block(8,1,&h);let pre_us=t.elapsed().as_micros();
    let prefix=engine.four_axis_snapshot();let (n,w)=moments(&prefix);
    let ca=prefix["candidates"].as_array().unwrap();let (a,b,_)=dr::candidates(&n,prefix["edges"].as_array().unwrap().len());
    let b=b.ok_or("pair requires at least two legal moves")?;
    let third=if arm=="K17EXP" {
        Some((0..ca.len()).find(|&i|i!=a&&i!=b).ok_or("expansion requires three legal moves")?)
    }else{None};
    let mv=|i:usize|ca[i][0].as_u64().unwrap() as usize;
    let x=[ep::score(n[a],w[a]),ep::score(n[b],w[b])];
    let q=(cap-16)/2;let order=match arm{"K16BAL"=>[Some(a),Some(b),None],"K16REV"=>[Some(b),Some(a),None],"K17EXP"=>[Some(a),third,None],_=>[None,None,None]};
    let mut prev=prefix.clone();let mut phases=Vec::new();let mut cur=8;let prior:Vec<_>=ca.iter().map(|z|z[1].as_f64().unwrap()).collect();
    let mut pulse=[0.;2];let mut final_score=[0.;2];
    for (i,k) in [q,q,8].into_iter().enumerate(){
        let target=order[i];h.set_decision_force(target.map(|j|(j,mv(j))),cur,cur+k);
        let t=Instant::now();engine.four_axis_block(k,1,&h);let step_us=t.elapsed().as_micros();cur+=k;
        let next=engine.four_axis_snapshot();assert_eq!(next["candidates"],prefix["candidates"]);
        let (old_n,_)=moments(&prev);let (nn,ww)=moments(&next);
        let delta:Vec<_>=nn.iter().zip(&old_n).map(|(&v,&u)|v.checked_sub(u).unwrap()).collect();
        assert_eq!(delta.iter().sum::<u32>(),k);
        if let Some(j)=target{assert_eq!(delta[j],k);}
        if i==1{pulse=[ep::pulse_score(x[0],nn[a]-n[a],ww[a]-w[a]),ep::pulse_score(x[1],nn[b]-n[b],ww[b]-w[b])];}
        if i==2{final_score=[ep::score(nn[a],ww[a]),ep::score(nn[b],ww[b])];}
        let flux=kr::flux(&old_n,&nn,&prior);
        phases.push(json!({"n":cur,"target":target.map(mv),"delta":delta,"flux":flux,"result":next,"step_us":step_us}));
        prev=next;
    }
    let mut out=json!({"pair":[mv(a),mv(b)],"prefix":prefix,"prefix_us":pre_us,"phases":phases,
        "prefix_score":x,"pulse_score":pulse,"final_score":final_score,"prefix_best":ep::choose(mv(a),mv(b),x[0],x[1]),
        "pulse_best":ep::choose(mv(a),mv(b),pulse[0],pulse[1]),"final_best":ep::choose(mv(a),mv(b),final_score[0],final_score[1]),
        "native_best":prev["best"]});
    if let Some(c)=third {out["reservoir"]=json!([mv(a),mv(b),mv(c)]);}
    Ok(out)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,arm:String,cap:u32,seed:u64,model_path:String,model_identity:String}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_evidence_cpu"||!matches!(q.arm.as_str(),"K16N"|"K16BAL"|"K16REV"|"K17EXP")||![32,64].contains(&q.cap){return Err("bad contract".into());}
    let state=Connect4::from_history(&q.history)?;let legal=state.legal_moves();
    if state.is_terminal()||legal.len()<2{return Err("terminal or singleton root".into());}
    if q.arm=="K17EXP"&&legal.len()<3{return Err("expansion requires three legal moves".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
    let engine=MctsEngine::new(state.clone(),ev,cfg);
    let evidence=run(&engine,&q.arm,q.cap)?;let result=engine.four_axis_snapshot();let calls=engine.evaluator_call_count();
    drop(engine);
    Ok(json!({"schema":"quartz_k16_c4_v1","history":q.history,"arm":q.arm,"cap":q.cap,"seed":q.seed,"legal":legal,"model_identity":q.model_identity,"cache_hit":hit,"evidence":evidence,"result":result,"best":result["best"],"visits":q.cap,"NN_calls":calls,"whole_us":start.elapsed().as_micros()}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
