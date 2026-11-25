"""Apply bounded K20 research endpoint patch to K19 source."""
from pathlib import Path
p=Path('src/mcts_server/evidence_cpu_server.rs')
s=p.read_text()
a='#[derive(Deserialize)]\n#[serde(deny_unknown_fields)]\nstruct Req'
assert s.count(a)==1
s=s.replace(a,'''// K20 fixed-channel signed-gain table. Online code has no label/oracle access.
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
    }else if q.arm=="K20MATCH"||q.arm=="K20TIME"{
        channel=if q.arm=="K20MATCH"{"MATCH"}else{"TIME"};
        let cap=if q.arm=="K20MATCH"{64}else{128};
        while cur<cap && if q.arm=="K20MATCH"{engine.evaluator_call_count()<q.nn_budget.unwrap()}else{tail.elapsed().as_micros()<q.tail_budget_us.unwrap() as u128}{
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
'''+a)
s=s.replace('model_identity:String}', 'model_identity:String,#[serde(default)]controller:Option<Controller>,#[serde(default)]nn_budget:Option<u64>,#[serde(default)]tail_budget_us:Option<u64>}')
a='if q.cmd!="research_connect4_evidence_cpu"||!matches!(q.arm.as_str(),"K16N"|"K16BAL"|"K16REV"|"K17EXP")||![32,64].contains(&q.cap){return Err("bad contract".into());}'
b='''let k20=matches!(q.arm.as_str(),"K20LIVE"|"K20N64"|"K20MATCH"|"K20TIME");
    if q.cmd!="research_connect4_evidence_cpu"||(!k20&&!matches!(q.arm.as_str(),"K16N"|"K16BAL"|"K16REV"|"K17EXP"))||![32,64].contains(&q.cap){return Err("bad contract".into());}
    if k20&&q.cap!=64 || (q.arm=="K20LIVE")!=q.controller.is_some() || q.controller.as_ref().is_some_and(|m|!m.valid()) ||
        (q.arm=="K20MATCH")!=q.nn_budget.is_some() || q.nn_budget.is_some_and(|n|n==0||n>65) ||
        (q.arm=="K20TIME")!=q.tail_budget_us.is_some() || q.tail_budget_us.is_some_and(|n|n>1000000){return Err("bad K20 controller/budget".into());}'''
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('if q.arm=="K17EXP"&&legal.len()<3','if (q.arm=="K17EXP"||k20)&&legal.len()<3')
a='let evidence=run(&engine,&q.arm,q.cap)?;'
b='''if k20 {
        let proof=proof_intervals(&state);
        let mut out=k20_run(&engine,&q,&proof)?;
        let calls=engine.evaluator_call_count();
        drop(engine);
        let obj=out.as_object_mut().unwrap();
        obj.extend(json!({"schema":"quartz_k20_c4_v1","history":q.history,"arm":q.arm,"cap":q.cap,"seed":q.seed,"legal":legal,"model_identity":q.model_identity,"cache_hit":hit,"NN_calls":calls,"whole_us":start.elapsed().as_micros()}).as_object().unwrap().clone());
        return Ok(out);
    }
    let evidence=run(&engine,&q.arm,q.cap)?;'''
assert s.count(a)==1;s=s.replace(a,b)
p.write_text(s)
