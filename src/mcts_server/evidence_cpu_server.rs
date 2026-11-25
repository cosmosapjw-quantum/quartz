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
// K20 fixed-channel signed-gain table. Online code has no label/oracle access.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Controller {
    gains:[[f64;6];4],costs:[[f64;6];4],switch:[[[bool;2];3];6],
}
impl Controller{
    fn valid(&self)->bool{
        self.gains.iter().flatten().all(|x|x.is_finite()&&x.abs()<=1.)&&
        self.costs.iter().flatten().all(|x|x.is_finite()&&*x>=0.&&*x<=64.)&&
        self.switch.iter().all(|r|!r[0][0]&&!r[0][1])
    }
}
const CHANNELS:[&str;6]=["K16N:32","K16BAL:32","K17EXP:32","K16N:64","K16BAL:64","K17EXP:64"];
fn proof_intervals(state:&Connect4)->Vec<(usize,f64,f64)>{
    state.legal_moves().iter().map(|&m|{
        let next=state.apply_move(m);
        let (l,u)=if next.is_terminal(){
            let v=(1.-next.outcome() as f64)/2.;(v,v)
        }else if next.legal_moves().iter().any(|&a|next.is_winning_move(a)){(0.,0.)}
        else{(0.,1.)};(m,l,u)
    }).collect()
}
fn certified(proof:&[(usize,f64,f64)])->Option<usize>{
    let ub=proof.iter().map(|p|p.2).fold(0.,f64::max);
    proof.iter().find(|p|p.1>=ub).map(|p|p.0)
}
fn k20_bin(n:&[u32],w:&[f64],p:[usize;3])->usize{
    let score=|i:usize|ep::score(n[i],w[i]);
    2*usize::from(n[p[0]]>=6)+usize::from(score(p[0])-score(p[1]).max(score(p[2]))>=0.15)
}
fn k20_choose(m:&Controller,b:usize)->Option<usize>{
    let mut ch=None;let mut best=0.;
    for i in 0..6{let v=m.gains[b][i]-0.002*m.costs[b][i];if v>best+1e-15{best=v;ch=Some(i);}}
    ch
}
fn k20_run(engine:&MctsEngine<Connect4>,q:&Req,proof:&[(usize,f64,f64)])->Result<Value,String>{
    let mut h=Hooks::new_modes(0,0,0,false);
    let t=Instant::now();
    engine.four_axis_block(8,1,&h);
    let prefix=engine.four_axis_snapshot();let prefix_us=t.elapsed().as_micros();
    let tail=Instant::now();let (n,w)=moments(&prefix);
    let ca=prefix["candidates"].as_array().unwrap();
    let (a,b,_)=dr::candidates(&n,ca.len());let b=b.ok_or("pair")?;
    let c=(0..ca.len()).find(|&i|i!=a&&i!=b).ok_or("triple")?;
    let panel=[a,b,c];let mv=|i:usize|ca[i][0].as_u64().unwrap() as usize;
    let bin=k20_bin(&n,&w,panel);let proof_best=certified(proof);
    let mut cur=8u32;let mut selected=None;let mut phases=Vec::new();
    let mut channel="STOP";let mut pred_gain=0.;let mut pred_cost=0.;
    if let Some(i)=proof_best{
        channel="CERTIFIED";selected=Some(i);
    }else if matches!(q.arm.as_str(),"K20MATCH"|"K20MATCHX"|"K20TIME"){
        channel=match q.arm.as_str(){"K20MATCH"=>"MATCH","K20MATCHX"=>"MATCHX",_=>"TIME"};
        let cap=match q.arm.as_str(){"K20MATCH"=>64,"K20MATCHX"=>256,_=>128};
        while cur<cap && if matches!(q.arm.as_str(),"K20MATCH"|"K20MATCHX"){engine.evaluator_call_count()<q.nn_budget.unwrap()}else{tail.elapsed().as_micros()<q.tail_budget_us.unwrap() as u128}{
            engine.four_axis_block(1,1,&h);cur+=1;
        }
    }else{
        let chosen=if q.arm=="K20N64"{Some(3)}else{k20_choose(q.controller.as_ref().unwrap(),bin)};
        if let Some(i)=chosen{
            channel=CHANNELS[i];
            if let Some(m)=q.controller.as_ref(){pred_gain=m.gains[bin][i];pred_cost=m.costs[bin][i];}
            let cap=if i<3{32}else{64};let k=(cap-16)/2;
            let order=match i%3{1=>[Some(a),Some(b),None],2=>[Some(a),Some(c),None],_=>[None,None,None]};
            for (target,block) in order.into_iter().zip([k,k,8]){
                h.set_decision_force(target.map(|j|(j,mv(j))),cur,cur+block);
                engine.four_axis_block(block,1,&h);cur+=block;
                let r=engine.four_axis_snapshot();
                phases.push(json!({"n":cur,"target":target.map(mv),"result":r}));
            }
            if i%3!=0{
                let r=engine.four_axis_snapshot();let(nn,ww)=moments(&r);
                let score=|j:usize|ep::score(nn[panel[j]],ww[panel[j]]);
                let mut j=0;
                for jj in 1..3{if score(jj)>score(j){j=jj;}}
                let high=usize::from(score(j)-score(0)>=0.15);
                let m=q.controller.as_ref().unwrap();
                let candidate=mv(panel[j]);let anchor=mv(a);
                let upper=proof.iter().find(|p|p.0==candidate).unwrap().2;
                let lower=proof.iter().find(|p|p.0==anchor).unwrap().1;
                selected=Some(if j!=0&&m.switch[i][j][high]&&upper>lower{candidate}else{anchor});
            }
        }else{selected=Some(mv(a));}
    }
    let result=engine.four_axis_snapshot();
    let best=selected.unwrap_or(result["best"].as_u64().unwrap() as usize);
    let tail_us=tail.elapsed().as_micros();
    Ok(json!({"prefix":prefix,"result":result,"best":best,"visits":cur,
        "decision":{"channel":channel,"bin":bin,"panel":panel.map(mv),"pred_gain":pred_gain,"pred_cost":pred_cost,
        "prefix_us":prefix_us,"tail_us":tail_us,"phases":phases,"proof":proof}}))
}
// K21 geometry of a score proxy; never interpreted as a posterior.
#[derive(serde::Serialize)]
struct SeqState{
    #[serde(flatten)]root:crate::mcts::four_axis::LiteRoot,
    bin:usize,kl:f64,drift:f64,bregman:f64,phi:f64,net:Option<f64>,
}
fn k21_phi(r:&crate::mcts::four_axis::LiteRoot)->(f64,[f64;7],[f64;7]){
    let mut x=[0.;7];let mut q=[0.;7];let mut mx=f64::NEG_INFINITY;
    for i in 0..r.len{x[i]=(ep::score(r.n[i],r.w[i])+1.)/2.;mx=mx.max(x[i]);}
    let mut z=0.;for i in 0..r.len{q[i]=((x[i]-mx)/0.25).exp();z+=q[i];}
    for i in 0..r.len{q[i]/=z;}
    (mx+0.25*(z/r.len as f64).ln(),q,x)
}
fn k21_state(r:crate::mcts::four_axis::LiteRoot,previous:Option<&crate::mcts::four_axis::LiteRoot>)->SeqState{
    let (phi,q,x)=k21_phi(&r);
    let (f0,p0,x0)=if let Some(z)=previous{assert_eq!(r.moves,z.moves);k21_phi(z)}else{
        (0.5,std::array::from_fn(|i|if i<r.len{1./r.len as f64}else{0.}),[0.5;7])
    };
    let mut kl=0.;let mut drift=0.;
    for i in 0..r.len{kl+=p0[i]*(p0[i]/q[i]).ln();drift+=p0[i]*(x[i]-x0[i]);}
    let bregman=phi-f0-drift;assert!((bregman-0.25*kl).abs()<1e-12);
    let mut a=0;let mut b=0;for &n in &r.n[..r.len]{if n>=a{b=a;a=n;}else if n>b{b=n;}}
    let bin=4*usize::from(a as f64/r.visits as f64>=0.75)+2*usize::from(a-b>=2)+usize::from(kl>=0.02);
    SeqState{root:r,bin,kl,drift,bregman,phi,net:None}
}
fn k21_run(engine:&MctsEngine<Connect4>,q:&Req,proof:&[(usize,f64,f64)])->Result<Value,String>{
    let h=Hooks::new_modes(0,0,0,false);
    let core=Instant::now();engine.four_axis_block(8,1,&h);
    let mut states=Vec::with_capacity(8);
    states.push(k21_state(engine.four_axis_lite(),None));
    let mut visits=8;let mut cause="BUDGET";let mut meta_ns=0u128;
    let cert=certified(proof);
    if cert.is_some(){cause="CERTIFIED";}
    else if matches!(q.arm.as_str(),"K21MATCH"|"K21TIME"){
        cause=if q.arm=="K21MATCH"{"MATCH"}else{"TIME"};
        while visits<256 && if q.arm=="K21MATCH"{engine.evaluator_call_count()<q.nn_budget.unwrap()}else{core.elapsed().as_micros()<q.tail_budget_us.unwrap() as u128}{
            engine.four_axis_block(1,1,&h);visits+=1;
        }
        if visits>8{states.push(k21_state(engine.four_axis_lite(),Some(&states[0].root)));}
    }else{
        while visits<64{
            if q.arm=="K21LIVE"{
                let m=Instant::now();let t=(visits/8-1) as usize;let z=states.last_mut().unwrap();
                let v=q.seq_table.as_ref().unwrap()[t][z.bin];z.net=Some(v);meta_ns+=m.elapsed().as_nanos();
                if v<=0.{cause="STOP";break;}
            }
            engine.four_axis_block(8,1,&h);visits+=8;
            let m=Instant::now();let z=k21_state(engine.four_axis_lite(),Some(&states.last().unwrap().root));meta_ns+=m.elapsed().as_nanos();states.push(z);
        }
    }
    let core_us=core.elapsed().as_micros();let best=cert.unwrap_or(states.last().unwrap().root.best);
    // Serialization only after decision/continuation and core measurement.
    Ok(json!({"states":states,"best":best,"visits":visits,"cause":cause,"proof":proof,"core_us":core_us,
       "meta_ns":meta_ns,"result":engine.four_axis_snapshot()}))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,arm:String,cap:u32,seed:u64,model_path:String,model_identity:String,#[serde(default)]controller:Option<Controller>,#[serde(default)]nn_budget:Option<u64>,#[serde(default)]tail_budget_us:Option<u64>,#[serde(default)]seq_table:Option<[[f64;8];7]>}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    let k20=matches!(q.arm.as_str(),"K20LIVE"|"K20N64"|"K20MATCH"|"K20MATCHX"|"K20TIME");
    let k21=matches!(q.arm.as_str(),"K21TRACE"|"K21LIVE"|"K21MATCH"|"K21TIME");
    if q.cmd!="research_connect4_evidence_cpu"||(!k20&&!k21&&!matches!(q.arm.as_str(),"K16N"|"K16BAL"|"K16REV"|"K17EXP"))||![32,64].contains(&q.cap){return Err("bad contract".into());}
    if (k20||k21)&&q.cap!=64 || (q.arm=="K20LIVE")!=q.controller.is_some() || q.controller.as_ref().is_some_and(|m|!m.valid()) ||
        matches!(q.arm.as_str(),"K20MATCH"|"K20MATCHX"|"K21MATCH")!=q.nn_budget.is_some() || q.nn_budget.is_some_and(|n|n==0||n>65) ||
        matches!(q.arm.as_str(),"K20TIME"|"K21TIME")!=q.tail_budget_us.is_some() || q.tail_budget_us.is_some_and(|n|n>1000000){return Err("bad K20/K21 controller/budget".into());}
    if (q.arm=="K21LIVE")!=q.seq_table.is_some() || q.seq_table.as_ref().is_some_and(|t|t.iter().flatten().any(|x|!x.is_finite()||x.abs()>2.)){return Err("bad K21 table".into());}
    let state=Connect4::from_history(&q.history)?;let legal=state.legal_moves();
    if state.is_terminal()||legal.len()<2{return Err("terminal or singleton root".into());}
    if (q.arm=="K17EXP"||k20||k21)&&legal.len()<3{return Err("expansion requires three legal moves".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
    let engine=MctsEngine::new(state.clone(),ev,cfg);
    if k21 {
        let proof=proof_intervals(&state);
        let mut out=k21_run(&engine,&q,&proof)?;
        let calls=engine.evaluator_call_count();drop(engine);
        out.as_object_mut().unwrap().extend(json!({"schema":"quartz_k21_c4_v1","history":q.history,"arm":q.arm,"cap":q.cap,"seed":q.seed,"legal":legal,"model_identity":q.model_identity,"cache_hit":hit,"NN_calls":calls,"whole_us":start.elapsed().as_micros()}).as_object().unwrap().clone());
        return Ok(out);
    }
    if k20 {
        let proof=proof_intervals(&state);
        let mut out=k20_run(&engine,&q,&proof)?;
        let calls=engine.evaluator_call_count();
        drop(engine);
        let obj=out.as_object_mut().unwrap();
        obj.extend(json!({"schema":"quartz_k20_c4_v1","history":q.history,"arm":q.arm,"cap":q.cap,"seed":q.seed,"legal":legal,"model_identity":q.model_identity,"cache_hit":hit,"NN_calls":calls,"whole_us":start.elapsed().as_micros()}).as_object().unwrap().clone());
        return Ok(out);
    }
    let evidence=run(&engine,&q.arm,q.cap)?;let result=engine.four_axis_snapshot();let calls=engine.evaluator_call_count();
    drop(engine);
    Ok(json!({"schema":"quartz_k16_c4_v1","history":q.history,"arm":q.arm,"cap":q.cap,"seed":q.seed,"legal":legal,"model_identity":q.model_identity,"cache_hit":hit,"evidence":evidence,"result":result,"best":result["best"],"visits":q.cap,"NN_calls":calls,"whole_us":start.elapsed().as_micros()}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
