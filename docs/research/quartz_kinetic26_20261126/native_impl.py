from pathlib import Path
p=Path('src/mcts/four_axis.rs');s=p.read_text()
def sub(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
sub('    width_realized:AtomicU64, work:[AtomicU64;4],','    width_realized:AtomicU64, work:[AtomicU64;4],\n    budget26:AtomicU64,denied26:AtomicU64,')
sub('    work:std::array::from_fn(|_|AtomicU64::new(0))}}}','    work:std::array::from_fn(|_|AtomicU64::new(0)),budget26:AtomicU64::new(u64::MAX),denied26:AtomicU64::new(0)}}}')
sub('        k.peak.store(0,Ordering::Relaxed);k.width_realized.store(0,Ordering::Relaxed);','        k.peak.store(0,Ordering::Relaxed);k.width_realized.store(0,Ordering::Relaxed);k.denied26.store(0,Ordering::Relaxed);')
sub('    pub fn k25_clock(&self)->Option<std::time::Instant>', '''    fn k26_limited(&self)->bool{
        self.k25.as_ref().is_some_and(|k|k.budget26.load(Ordering::Acquire)!=u64::MAX)
    }
    fn k26_claim(&self,n:u64)->bool{
        let k=self.k25.as_ref().unwrap();
        k.budget26.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_sub(n)).is_ok()
    }
    fn k26_refund(&self){
        if self.k26_limited(){self.k25.as_ref().unwrap().budget26.fetch_add(1,Ordering::AcqRel);}
    }
    pub fn k26_denied(&self)->u64{self.k25.as_ref().unwrap().denied26.load(Ordering::Acquire)}
    pub fn k25_clock(&self)->Option<std::time::Instant>''')
sub('            h.add(12,1);raw_base=sel.leaf_state.outcome();raw_base','            h.add(12,1);h.k26_refund();raw_base=sel.leaf_state.outcome();raw_base')
sub('                let k=if h.vmode==4{cand.len()}else{2};\n                let mut best=-1f32;', '''                let k=if h.vmode==4{cand.len()}else{2};
                let permitted=if h.k26_limited(){
                    let need=cand.iter().take(k).enumerate().filter(|(i,(mv,_))|known[*i].is_none()&&!sel.leaf_state.apply_move(*mv).is_terminal()).count() as u64;
                    h.k26_claim(need)
                }else{true};
                if permitted{
                let mut best=-1f32;''')
sub('                h.add(10,1);h.add(11,((v-ev.value).abs()>1e-7) as u64);ev.value=v;\n            }','''                h.add(10,1);h.add(11,((v-ev.value).abs()>1e-7) as u64);ev.value=v;
                }else{h.k25.as_ref().unwrap().denied26.fetch_add(1,Ordering::Relaxed);}
            }''')
sub('    pub fn four_axis_k25_block(&self,delta:u32,threads:usize,h:&Hooks){','''    // Hard evaluator-call budget. Each admitted iteration owns one base token;
    // terminal returns it, refinements reserve the full extra bundle atomically.
    // Only quiescent callers, fixed research hook, no proof/async path.
    pub fn four_axis_k26_budget(&self,target:u64,cap:u32,threads:usize,wave:bool,h:&Hooks)->Value{
        assert!(target>0&&cap>0&&matches!(threads,1|2|4|8));
        assert_eq!(h.proof_mode,0);assert!(h.vmode==0||h.vmode==1);
        let k=h.k25.as_ref().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        assert_eq!(k.budget26.swap(target,Ordering::AcqRel),u64::MAX);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        let before=self.root.n_total.load(Ordering::Acquire);let calls=self.evaluator_call_count();
        let mut used=0u32;let mut rounds=0u32;
        while used<cap&&k.budget26.load(Ordering::Acquire)>0{
            let quota=(cap-used).min(if wave{8}else{cap});let tickets=AtomicU32::new(0);
            let worker=||loop{
                if !h.k26_claim(1){break;}
                if tickets.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|if n<quota{Some(n+1)}else{None}).is_err(){h.k26_refund();break;}
                h.k25_enter();self.four_iterate(h);h.k25_leave();
            };
            if threads==1{worker();}else{rayon::scope(|scope|{for _ in 0..threads{let w=&worker;scope.spawn(move |_|w());}});}
            let n=tickets.load(Ordering::Acquire);assert!(n>0);used+=n;rounds+=1;
        }
        assert_eq!(k.active.load(Ordering::Acquire),0);
        let free=k.budget26.swap(u64::MAX,Ordering::AcqRel);
        let nn=self.evaluator_call_count()-calls;assert_eq!(target-free,nn);
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+used);
        json!({"target":target,"cap":cap,"used_nn":nn,"unused":free,"visits":used,"rounds":rounds,
            "denied_refinement":h.k26_denied(),"cause":if free==0{"NN_TARGET"}else{"VISIT_CAP"}})
    }
    pub fn k26_root_diversity(&mut self,seed:u64){
        let ca=self.root.candidates.get().unwrap();let prior:Vec<_>=ca.iter().map(|e|e.1).collect();
        self.root_noise=Some(compute_dirichlet_noise(prior.len(),&prior,&DirichletConfig{alpha:0.3,epsilon:0.25},Some(seed)));
    }
    pub fn four_axis_k25_block(&self,delta:u32,threads:usize,h:&Hooks){''')
p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text()
sub('mod k25_cpu_server;','mod k25_cpu_server;\n#[path = "mcts_server/k26_cpu_server.rs"]\nmod k26_cpu_server;')
sub('        if cmd == "research_connect4_k25_cpu" {','''        if cmd == "research_connect4_k26_cpu" {
            emit_stdout_json_value(&k26_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
        if cmd == "research_connect4_k25_cpu" {''')
p.write_text(s)
