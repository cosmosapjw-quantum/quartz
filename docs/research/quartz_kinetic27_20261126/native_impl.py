from pathlib import Path
p=Path('src/mcts/four_axis.rs');s=p.read_text()
def sub(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
sub('    budget26:AtomicU64,denied26:AtomicU64,','    budget26:AtomicU64,denied26:AtomicU64,refine27:AtomicU64,')
sub('budget26:AtomicU64::new(u64::MAX),denied26:AtomicU64::new(0)}}}', 'budget26:AtomicU64::new(u64::MAX),denied26:AtomicU64::new(0),refine27:AtomicU64::new(u64::MAX)}}}')
sub('    fn k26_refund(&self){','''    fn k27_claim_refine(&self,n:u64)->bool{
        let k=self.k25.as_ref().unwrap();
        if k.refine27.load(Ordering::Acquire)==u64::MAX{self.k26_claim(n)}
        else{k.refine27.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_sub(n)).is_ok()}
    }
    fn k26_refund(&self){''')
sub('                    h.k26_claim(need)','                    h.k27_claim_refine(need)')
sub('    pub fn k26_root_diversity(&mut self,seed:u64){','''    // Protected optional-call pool. Only a quiescent join transfers its
    // unused balance back into the common pool. r=0 is K26 common pool.
    pub fn four_axis_k27_budget(&self,target:u64,cap:u32,threads:usize,wave:bool,reserve:u64,h:&Hooks)->Value{
        assert!(target>0&&cap>0&&reserve<target&&matches!(threads,1|2|4|8));
        assert_eq!(h.proof_mode,0);assert!(h.vmode==0||h.vmode==1);assert!(reserve==0||h.vmode==1);
        let k=h.k25.as_ref().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        assert_eq!(k.budget26.swap(target-reserve,Ordering::AcqRel),u64::MAX);
        assert_eq!(k.refine27.swap(if reserve>0{reserve}else{u64::MAX},Ordering::AcqRel),u64::MAX);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        let before=self.root.n_total.load(Ordering::Acquire);let calls=self.evaluator_call_count();
        let base0=h.counts[7].load(Ordering::Acquire);let ext0=h.counts[8].load(Ordering::Acquire);
        let mut used=0u32;let mut rounds=0u32;let mut release=0;let mut protected_base=0;let mut protected_extra=0;let mut protected_visits=0;
        for phase in 0..(if reserve>0{2}else{1}){
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
            if phase==0&&reserve>0{
                release=k.refine27.swap(u64::MAX,Ordering::AcqRel);
                protected_base=h.counts[7].load(Ordering::Acquire)-base0;
                protected_extra=h.counts[8].load(Ordering::Acquire)-ext0;
                protected_visits=used;
                assert_eq!(protected_extra,reserve-release);
                assert_eq!(protected_base,target-reserve-k.budget26.load(Ordering::Acquire));
                assert_eq!(self.evaluator_call_count()-calls,protected_base+protected_extra);
                k.budget26.fetch_add(release,Ordering::AcqRel);
            }
        }
        assert_eq!(k.refine27.load(Ordering::Acquire),u64::MAX);
        let free=k.budget26.swap(u64::MAX,Ordering::AcqRel);
        let nn=self.evaluator_call_count()-calls;assert_eq!(target-free,nn);
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+used);
        json!({"target":target,"cap":cap,"used_nn":nn,"unused":free,"visits":used,"rounds":rounds,
            "reserve":reserve,"released":release,"protected_base":protected_base,"protected_extra":protected_extra,"protected_visits":protected_visits,
            "denied_refinement":h.k26_denied(),"cause":if free==0{"NN_TARGET"}else{"VISIT_CAP"}})
    }
    pub fn k26_root_diversity(&mut self,seed:u64){''')
p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text()
sub('mod k26_cpu_server;','mod k26_cpu_server;\n#[path = "mcts_server/k27_cpu_server.rs"]\nmod k27_cpu_server;')
sub('        if cmd == "research_connect4_k26_cpu" {','''        if cmd == "research_connect4_k27_cpu" {
            emit_stdout_json_value(&k27_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
        if cmd == "research_connect4_k26_cpu" {''')
p.write_text(s)
