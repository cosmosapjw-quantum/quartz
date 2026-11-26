// K32: proof facts stay separate from the raw NN receipt ledger.
struct K32{
    paths:Mutex<Vec<Vec<usize>>>, forced:Mutex<Vec<Value>>,
    checks:AtomicU64,replay:AtomicU64,reg_ns:AtomicU64,frontier_hits:AtomicU64,
}
impl K32{fn new()->Self{Self{paths:Mutex::new(Vec::new()),forced:Mutex::new(Vec::new()),checks:AtomicU64::new(0),replay:AtomicU64::new(0),reg_ns:AtomicU64::new(0),frontier_hits:AtomicU64::new(0)}}}
fn k32_fifo(g:&Graph,root:Key)->Option<Vec<usize>>{
    use std::collections::{BTreeSet,VecDeque};
    let mut seen=BTreeSet::from([root]);let mut q=VecDeque::from([(root,Vec::new())]);
    while let Some((k,p))=q.pop_front(){
        let (l,u)=g.interval(k);if l==u{continue;}
        if !g.registered(k){return Some(p);}
        let ok=g.allowed(k);
        if let Some(aa)=&g.nodes[&k].actions{for &(m,c) in aa{
            if ok.contains(&m)&&seen.insert(c){let mut t=p.clone();t.push(m);q.push_back((c,t));}
        }}
    }
    None
}
impl Hooks{
    pub fn with_k32(mut self)->Self{
        assert!(self.k25.is_some()&&self.proof_mode==0);
        self.k32=Some(K32::new());self.proof_mode=1;self.proof_priority=false;self
    }
    pub fn k32_init<G:GameState>(&self,s:&G){
        let k=self.k32.as_ref().unwrap();assert_eq!(self.proof_mode,1);
        let mut g=self.graph.lock().unwrap();self.register_locked(&mut g,s);
        for m in s.legal_moves(){k.replay.fetch_add(1,Ordering::Relaxed);self.register_locked(&mut g,&s.apply_move(m));}
    }
    pub fn k32_activate(&mut self){
        assert!(self.k32.is_some());assert_eq!(self.k25.as_ref().unwrap().active.load(Ordering::Acquire),0);
        self.proof_mode=2;self.proof_priority=false;
    }
    pub fn k32_forced<G:GameState>(&self,s:&G,mode:u8,depth:usize,rank:usize,mv:G::Move,before:usize,after:usize,visible:usize){
        if let Some(k)=&self.k32{
            k.forced.lock().unwrap().push(json!({"key":keyhex(exact_key(s)),"mode":mode,"depth":depth,"rank":rank,"move":s.move_to_idx(mv),"before":before,"after":after,"visible":visible,"hidden":rank>=before}));
        }
    }
    pub fn k32_report<G:GameState>(&self,s:&G,l:&LiteRoot,anchor:usize,dump:bool,paths:bool)->Value{
        let Some(k)=&self.k32 else{return Value::Null;};
        let vs:Vec<_>=l.moves[..l.len].iter().copied().zip(l.n[..l.len].iter().copied()).collect();
        let mut v=self.root_proof(s,anchor,&vs,dump);
        let aa=self.graph.lock().unwrap().actions(exact_key(s));
        v["risk"]=json!(super::proof_graph::risk_width(&aa));
        v["checks"]=json!(k.checks.load(Ordering::Relaxed));v["replay"]=json!(k.replay.load(Ordering::Relaxed));
        v["rule_work"]=json!(k.checks.load(Ordering::Relaxed)+k.replay.load(Ordering::Relaxed));
        v["reg_ns"]=json!(k.reg_ns.load(Ordering::Relaxed));v["frontier_hits"]=json!(k.frontier_hits.load(Ordering::Relaxed));
        v["forced"]=json!(*k.forced.lock().unwrap());
        v["paths"]=if paths{json!(*k.paths.lock().unwrap())}else{Value::Null};
        v
    }
}
impl<G:GameState> MctsEngine<G>{
    pub fn k32_rule_expand(&self,h:&Hooks,anchor:usize)->Value{
        let k=h.k32.as_ref().unwrap();let root=&self.root_state;let rk=exact_key(root);let l=self.four_axis_lite();
        let mut steps=Vec::new();
        for _ in 0..8{
            let t=std::time::Instant::now();
            let (path,score,scored,kind)={
                let g=h.graph.lock().unwrap();if g.witness(rk).is_some(){break;}
                let(f,n)=g.frontier(rk,16);
                if let Some(f)=f{(Some(f.path.iter().map(|x|x.1).collect::<Vec<_>>()),Some(f.score),n,"FRONTIER")}
                else{(k32_fifo(&g,rk),None,n,"FIFO_FALLBACK")}
            };
            let plan_ns=t.elapsed().as_nanos() as u64;
            let Some(path)=path else{break;};let mut s=root.clone();
            for &m in &path{
                assert!(!s.is_terminal());let mv=s.legal_moves().into_iter().find(|&mv|s.move_to_idx(mv)==m).unwrap();
                s=s.apply_move(mv);k.replay.fetch_add(1,Ordering::Relaxed);
            }
            let key=exact_key(&s);assert!(!s.is_terminal());
            {let mut g=h.graph.lock().unwrap();assert!(!g.registered(key)&&g.nodes.contains_key(&key));h.register_locked(&mut g,&s);}
            steps.push(json!({"path":path,"key":keyhex(key),"kind":kind,"score":score,"scored":scored,"plan_ns":plan_ns,"post":h.k32_report(root,&l,anchor,false,false)}));
        }
        json!(steps)
    }
    fn k32_prepare(&self,h:&mut Hooks,guide:u8)->Value{
        h.local_mode=0;h.local_frontier=None;h.local_remaining.store(0,Ordering::Release);
        if guide==0{return Value::Null;}
        assert!(h.k32.is_some()&&h.proof_mode==2&&(guide==2||guide==3));
        let t=std::time::Instant::now();let mut n=0;
        if guide==3{
            let(f,s)=h.graph.lock().unwrap().frontier(exact_key(&self.root_state),16);
            h.local_frontier=f;n=s;
        }
        h.local_mode=if h.local_frontier.is_some(){3}else{2};h.local_remaining.store(1,Ordering::Release);
        json!({"mode":h.local_mode,"scored":n,"plan_ns":t.elapsed().as_nanos() as u64,
          "target":h.local_frontier.as_ref().map(|f|json!({"key":keyhex(f.key),"path":f.path.iter().map(|x|x.1).collect::<Vec<_>>(),"score":f.score}))})
    }
    // One common resource pool plus optional V floor; certificate checks only at joins.
    pub fn four_axis_k32_budget(&self,target:u64,cap:u32,threads:usize,reserve:u64,guide:u8,cert_stop:bool,h:&mut Hooks)->Value{
        assert!(target>0&&target<=64&&cap>0&&reserve<target&&matches!(threads,1|4|8));
        assert!(reserve==0||h.vmode==1);
        assert!(h.proof_mode==0||h.k32.is_some());
        assert!(!cert_stop||h.proof_mode==2);
        {
            let k=h.k25.as_ref().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
            assert_eq!(k.budget26.load(Ordering::Acquire),u64::MAX);assert_eq!(k.refine27.load(Ordering::Acquire),u64::MAX);
            assert_eq!(k.floor28.swap(pack28(target-reserve,reserve),Ordering::AcqRel),u64::MAX);
        }
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        let before=self.root.n_total.load(Ordering::Acquire);let calls=self.evaluator_call_count();
        let base0=h.counts[7].load(Ordering::Acquire);let ext0=h.counts[8].load(Ordering::Acquire);
        let mut used=0u32;let mut released=0;let mut pre_base=0;let mut pre_extra=0;let mut borrowed=0;let mut protected=0;
        let mut pre_visits=0;let mut rounds=Vec::new();let mut certified=false;
        for phase in 0..2{
            loop{
                if cert_stop&&h.root_control(&self.root_state).0.is_some(){certified=true;break;}
                let shared=pools28(h.k25.as_ref().unwrap().floor28.load(Ordering::Acquire)).0;
                if used>=cap||shared==0{break;}
                let quota=(cap-used).min(8);let plan=self.k32_prepare(h,guide);let nn0=self.evaluator_call_count();
                let forced0=h.k32.as_ref().map_or(0,|k|k.forced.lock().unwrap().len());
                let tickets=AtomicU32::new(0);
                {
                    let hr:&Hooks=h;
                    let worker=||loop{
                        if !hr.k26_claim(1){break;}
                        if tickets.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|if n<quota{Some(n+1)}else{None}).is_err(){hr.k26_refund();break;}
                        hr.k25_enter();self.four_iterate(hr);hr.k25_leave();
                    };
                    if threads==1{worker();}else{rayon::scope(|s|{for _ in 0..threads{let w=&worker;s.spawn(move |_|w());}});}
                }
                assert_eq!(h.k25.as_ref().unwrap().active.load(Ordering::Acquire),0);
                let n=tickets.load(Ordering::Acquire);assert!(n>0&&n<=quota);used+=n;
                let(s,p)=pools28(h.k25.as_ref().unwrap().floor28.load(Ordering::Acquire));
                assert_eq!(self.evaluator_call_count()-calls+s+p,target);
                let w=if cert_stop{h.root_control(&self.root_state).0}else{None};
                rounds.push(json!({"phase":phase,"plan":plan,"visits":n,"nn":self.evaluator_call_count()-nn0,"shared":s,"protected":p,"witness":w,
                  "forced":h.k32.as_ref().map_or(0,|k|k.forced.lock().unwrap().len())-forced0}));
                if w.is_some(){certified=true;break;}
            }
            if phase==0{
                let(s,p)=pools28(h.k25.as_ref().unwrap().floor28.load(Ordering::Acquire));
                released=p;protected=reserve-p;pre_base=h.counts[7].load(Ordering::Acquire)-base0;pre_extra=h.counts[8].load(Ordering::Acquire)-ext0;pre_visits=used;
                borrowed=pre_extra-protected;assert_eq!(pre_base+borrowed+s,target-reserve);
                h.k25.as_ref().unwrap().floor28.store(pack28(s+p,0),Ordering::Release);
            }
            if certified{break;}
        }
        h.local_mode=0;h.local_frontier=None;h.local_remaining.store(0,Ordering::Release);
        let(free,p)=pools28(h.k25.as_ref().unwrap().floor28.swap(u64::MAX,Ordering::AcqRel));assert_eq!(p,0);
        let nn=self.evaluator_call_count()-calls;assert_eq!(target-free,nn);
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+used);
        assert_eq!(nn,h.counts[7].load(Ordering::Acquire)-base0+h.counts[8].load(Ordering::Acquire)-ext0);
        json!({"target":target,"cap":cap,"used_nn":nn,"unused":free,"visits":used,"rounds":rounds,
          "reserve":reserve,"released":released,"protected_base":pre_base,"protected_extra":protected,"protected_visits":pre_visits,"borrowed":borrowed,"pre_extra":pre_extra,
          "denied_refinement":h.k26_denied(),"cause":if certified{"CERTIFIED"}else if free==0{"NN_TARGET"}else{"VISIT_CAP"}})
    }
}
