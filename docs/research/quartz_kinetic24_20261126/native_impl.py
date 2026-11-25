from pathlib import Path
# Research-only serial exact board ledger, all existing modes unchanged.
p=Path('src/mcts/four_axis.rs');s=p.read_text()
a='pub struct ProofPlan{pub stop:bool,pub allowed:Vec<usize>}'
assert s.count(a)==1
s=s.replace(a,a+'''
// K24 atomic evidence, not visit statistics. Fixed board-only deterministic
// Connect4 evaluator, one root/epoch, serial. Never used in standard search.
#[derive(Default)]
struct Context24{sum:f64,n:u64,leaf:std::collections::BTreeMap<(Key,bool),f32>}
#[derive(Default)]
struct Merge24{
    entries:std::collections::BTreeMap<Key,std::collections::BTreeMap<(Key,usize),Context24>>,
    ptrs:std::collections::BTreeMap<Key,usize>,
    reverse_ptrs:std::collections::BTreeMap<usize,Key>,
    trace:Vec<MergeTrace24>,
}
#[derive(serde::Serialize)]
struct MergeStep24{raw_n:u64,unique_n:usize,novel_n:usize,raw_mean:Option<f64>,novel_mean:Option<f64>,incoming:f32,backup:f32}
#[derive(serde::Serialize)]
struct MergeTrace24{path:Vec<usize>,leaf:f32,terminal:bool,mode:u8,steps:Vec<MergeStep24>}
impl Merge24{
    fn identity(&mut self,k:Key,p:usize){
        if let Some(old)=self.ptrs.insert(k,p){assert_eq!(old,p,"same board must share actual node");}
        if let Some(old)=self.reverse_ptrs.insert(p,k){assert_eq!(old,k,"actual node aliases two boards");}
    }
    fn observe(&mut self,child:Key,ctx:(Key,usize),leaf:(Key,bool),raw:f32)->(u64,usize,usize,Option<f64>,Option<f64>){
        let contexts=self.entries.entry(child).or_default();
        let mut all=std::collections::BTreeMap::new();
        let mut raw_sum=0.;let mut raw_n=0;
        for (k,v) in contexts.iter(){
            if let Some(x)=v.leaf.get(&leaf){assert_eq!(*x,raw,"deterministic leaf value mismatch");}
            if *k==ctx{continue;}
            raw_sum+=v.sum;raw_n+=v.n;
            for (&l,&x) in &v.leaf{
                if let Some(old)=all.insert(l,x){assert_eq!(old,x,"cross-context identity mismatch");}
            }
        }
        let unique_n=all.len();
        all.remove(&leaf);
        if let Some(own)=contexts.get(&ctx){
            for (l,x) in &own.leaf{
                if let Some(old)=all.remove(l){assert_eq!(old,*x,"own/cross identity mismatch");}
            }
        }
        let novel_n=all.len();
        let raw_mean=if raw_n>0{Some(raw_sum/raw_n as f64)}else{None};
        let novel_mean=if novel_n>0{Some(all.values().map(|&v|v as f64).sum::<f64>()/novel_n as f64)}else{None};
        let own=contexts.entry(ctx).or_default();own.n+=1;own.sum+=raw as f64;
        if let Some(old)=own.leaf.insert(leaf,raw){assert_eq!(old,raw);}
        (raw_n,unique_n,novel_n,raw_mean,novel_mean)
    }
    fn counts(&self)->Value{
        let contexts:usize=self.entries.values().map(|x|x.len()).sum();
        let shared=self.entries.values().filter(|x|x.len()>1).count();
        let observations:u64=self.entries.values().flat_map(|x|x.values()).map(|x|x.n).sum();
        let unique_context:usize=self.entries.values().flat_map(|x|x.values()).map(|x|x.leaf.len()).sum();
        json!({"children":self.entries.len(),"contexts":contexts,"shared":shared,"observations":observations,"unique_context":unique_context,"iterations":self.trace.len(),"actual_node_ids":self.ptrs.len()})
    }
}
#[cfg(test)]
mod merge24_tests{
    use super::*;
    #[test]fn duplicate_identity_is_not_new_cross_evidence(){
        let mut l=Merge24::default();
        assert_eq!(l.observe(9,(1,0),(20,false),0.5),(0,0,0,None,None));
        assert_eq!(l.observe(9,(2,0),(20,false),0.5),(1,1,0,Some(0.5),None));
        assert_eq!(l.observe(9,(2,0),(21,false),-0.5),(1,1,0,Some(0.5),None));
        assert_eq!(l.observe(9,(1,0),(22,false),0.),(2,2,1,Some(0.),Some(-0.5)));
        assert_eq!(l.observe(9,(3,0),(23,false),0.25),(4,3,3,Some(0.125),Some(0.)));
    }
}
''')
a='    gomoku_b_steps:Mutex<Vec<Value>>,'
assert s.count(a)==1;s=s.replace(a,a+'\n    merge24:Option<Mutex<Merge24>>,merge24_mode:u8,')
a='gomoku_b_steps:Mutex::new(Vec::new()),'
assert s.count(a)==1;s=s.replace(a,'gomoku_b_steps:Mutex::new(Vec::new()),merge24:None,merge24_mode:0,')
a='    fn backup<G:GameState>(&self,root:&G,path:&[PathEdge<G::Move>],leaf:f32,terminal:bool){'
assert s.count(a)==1
s=s.replace(a,'''    pub fn with_merge24(mut self)->Self{
        assert!(self.mask==0&&self.vmode==0&&self.bmode==0&&self.proof_mode==0);
        self.merge24=Some(Mutex::new(Merge24::default()));self
    }
    pub fn set_merge24(&mut self,mode:u8){assert!(mode<=2&&self.merge24.is_some());self.merge24_mode=mode;}
    pub fn merge24_counts(&self)->Value{self.merge24.as_ref().map(|x|x.lock().unwrap().counts()).unwrap_or(Value::Null)}
    pub fn merge24_trace(&self)->Value{self.merge24.as_ref().map(|x|json!(x.lock().unwrap().trace)).unwrap_or(Value::Null)}
    fn backup24<G:GameState>(&self,root:&G,path:&[PathEdge<G::Move>],leaf:f32,terminal:bool){
        let mut l=self.merge24.as_ref().unwrap().lock().unwrap();
        let mut state=root.clone();let mut keys=Vec::with_capacity(path.len());let mut moves=Vec::with_capacity(path.len());
        for pe in path {
            let e=&pe.parent.read_edges()[pe.edge_idx];let parent=exact_key(&state);let mv=state.move_to_idx(e.mv);
            assert_eq!(state.tt_hash(),pe.parent.hash);
            state=state.apply_move(e.mv);
            let child=exact_key(&state);assert_eq!(state.tt_hash(),e.child.hash);
            l.identity(child,&*e.child as *const _ as usize);
            moves.push(mv);keys.push((child,(parent,mv)));
        }
        assert_eq!(state.is_terminal(),terminal);
        let leaf_key=(exact_key(&state),terminal);
        let mut raw=leaf;let mut y=leaf;let mut steps=Vec::with_capacity(path.len());
        for (i,pe) in path.iter().enumerate().rev(){
            let e=&pe.parent.read_edges()[pe.edge_idx];
            let (child,ctx)=keys[i];
            let (raw_n,unique_n,novel_n,raw_mean,novel_mean)=l.observe(child,ctx,leaf_key,raw);
            let incoming=y;
            if !terminal{
                if self.merge24_mode==1{if let Some(m)=raw_mean{y=math::blend(y,m,raw_n,false);}}
                else if self.merge24_mode==2{if let Some(m)=novel_mean{y=math::blend(y,m,novel_n as u64,false);}}
            }
            raw=-raw;y=-y;
            let (vv,vq)=pe.applied_vl;e.remove_vl(vv,vq);e.record_sample(y);
            e.child.record_incoming_sample(e as *const _ as usize,-y);
            atomic_f64_add(&pe.parent.w_total,y as f64);
            pe.parent.n_total.fetch_add(1,Ordering::AcqRel);
            steps.push(MergeStep24{raw_n,unique_n,novel_n,raw_mean,novel_mean,incoming,backup:y});
        }
        steps.reverse();
        l.trace.push(MergeTrace24{path:moves,leaf,terminal,mode:self.merge24_mode,steps});
    }
'''+a+'\n        if self.merge24.is_some(){self.backup24::<G>(root,path,leaf,terminal);return;}')
p.write_text(s)
p=Path('src/mcts_server/evidence_cpu_server.rs');s=p.read_text()
a='#[derive(Deserialize)]\n#[serde(deny_unknown_fields)]\nstruct Req'
assert s.count(a)==1
s=s.replace(a,'''// K24: identical shadow prefix followed by actual cross-context value transfer.
fn k24_run(engine:&MctsEngine<Connect4>,q:&Req,proof:&[(usize,f64,f64)])->Result<Value,String>{
    let base=q.arm=="K24BASE";let mut h=Hooks::new_modes(0,0,0,false);
    if !base{h=h.with_merge24();}
    let t=Instant::now();engine.four_axis_block(32,1,&h);
    let pre=engine.four_axis_snapshot();let pc=h.merge24_counts();
    let pre_nn=engine.evaluator_call_count();let prefix_us=t.elapsed().as_micros();
    let mut n:Vec<u64>=pre["edges"].as_array().unwrap().iter().map(|e|e["n"].as_u64().unwrap()).collect();
    n.sort_unstable();let bin=2*usize::from(!base&&pc["shared"].as_u64().unwrap_or(0)>0)+usize::from(n[n.len()-1]-n[n.len()-2]>=4);
    let mode=match q.arm.as_str(){"K24BASE"|"K24N"=>0,"K24M"=>1,"K24D"=>2,"K24LIVE"=>q.merge_table.as_ref().unwrap()[bin],_=>return Err("bad K24 arm".into())};
    if !base{h.set_merge24(mode);}
    let t=Instant::now();let mut visits=32;let cert=certified(proof);
    if cert.is_none(){while visits<512&&engine.evaluator_call_count()<pre_nn+64{
        engine.four_axis_block(1,1,&h);visits+=1;
    }}
    let nn=engine.evaluator_call_count()-pre_nn;assert!(nn<=64);
    let tail_us=t.elapsed().as_micros();
    let result=engine.four_axis_snapshot();let best=cert.unwrap_or(result["best"].as_u64().unwrap() as usize);
    Ok(json!({"prefix":pre,"prefix_counts":pc,"result":result,"best":best,"visits":visits,"incremental_nn":nn,
        "cause":if cert.is_some(){"CERTIFIED"}else if nn==64{"NN_TARGET"}else{"VISIT_CAP"},
        "mode":mode,"bin":bin,"counts":h.merge24_counts(),"trace":h.merge24_trace(),"proof":proof,"prefix_us":prefix_us,"tail_us":tail_us}))
}
'''+a)
a='    let k23=matches!(q.arm.as_str(),"K23PATH"|"K23LIVE");'
assert s.count(a)==1;s=s.replace(a,a+'\n    let k24=matches!(q.arm.as_str(),"K24BASE"|"K24N"|"K24M"|"K24D"|"K24LIVE");')
s=s.replace('(!k20&&!k21&&!k22&&!k23&&!matches!','(!k20&&!k21&&!k22&&!k23&&!k24&&!matches!').replace('(k20||k21||k22||k23)&&q.cap!=64','(k20||k21||k22||k23||k24)&&q.cap!=64')
a='    if k23 {\n        let proof=proof_intervals(&state);'
assert s.count(a)==1
s=s.replace(a,'''    if k24 {
        let proof=proof_intervals(&state);
        let mut out=k24_run(&engine,&q,&proof)?;
        let calls=engine.evaluator_call_count();drop(engine);
        out.as_object_mut().unwrap().extend(json!({"schema":"quartz_k24_c4_v1","history":q.history,"arm":q.arm,"cap":q.cap,"seed":q.seed,"legal":legal,"model_identity":q.model_identity,"cache_hit":hit,"NN_calls":calls,"whole_us":start.elapsed().as_micros()}).as_object().unwrap().clone());
        return Ok(out);
    }
'''+a)
s=s.replace('h2_table:Option<H2Policy>}','h2_table:Option<H2Policy>,#[serde(default)]merge_table:Option<[u8;4]>}')
s=s.replace('    let state=Connect4::from_history(&q.history)?;', '    if (q.arm=="K24LIVE")!=q.merge_table.is_some() || q.merge_table.as_ref().is_some_and(|t|t.iter().any(|&v|v>2)){return Err("bad K24 table".into());}\n    let state=Connect4::from_history(&q.history)?;')
p.write_text(s)
