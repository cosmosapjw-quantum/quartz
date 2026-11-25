from pathlib import Path
p=Path('src/mcts/four_axis.rs');s=p.read_text()
def sub(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
sub('pub const S:u8=1;pub const X:u8=2;pub const V:u8=4;pub const B:u8=8;', '''// K25: one ledger per isolated tree. Exact base-evaluator observations,
// distinct from refined/backed-up values. Receipt insertion and backup
// linearize under one lock. This is not posterior sample accounting.
type Leaf25=(Key,bool);
type Evidence25=std::collections::BTreeMap<Leaf25,f32>;
#[derive(Default)]
struct Ctx25{direct:Evidence25,received:Evidence25}
#[derive(Default)]
struct Ledger25{
    nodes:std::collections::BTreeMap<Key,std::collections::BTreeMap<(Key,usize),Ctx25>>,
    exact:Merge24,
    payload:Evidence25,
    events:u64,edges:u64,eligible:u64,novel:u64,blocked:u64,transferred:u64,
    applied:u64,changed:u64,repeat:u64,trace:Vec<serde_json::Value>,
}
fn compatible25(m:&mut Evidence25,k:Leaf25,v:f32){
    if let Some(old)=m.insert(k,v){assert_eq!(old,v,"conflicting deterministic raw evidence");}
}
impl Ledger25{
    fn observe(&mut self,child:Key,ctx:(Key,usize),leaf:Leaf25,raw:f32,deliver:bool)->(usize,usize,usize,Option<f64>){
        let contexts=self.nodes.entry(child).or_default();
        let mut other=Evidence25::new();
        for (k,v) in contexts.iter(){if *k!=ctx{for (&l,&x) in &v.direct{compatible25(&mut other,l,x);}}}
        other.remove(&leaf);
        let own=contexts.entry(ctx).or_default();
        if let Some(v)=own.received.get(&leaf){assert_eq!(*v,raw);}
        for (l,x) in &own.direct{if let Some(y)=other.remove(l){assert_eq!(y,*x);}}
        let eligible=other.len();
        for (l,x) in &own.received{if let Some(y)=other.remove(l){assert_eq!(y,*x);}}
        let novel=other.len();let blocked=eligible-novel;
        let mean=if novel>0{Some(other.values().map(|&x|x as f64).sum::<f64>()/novel as f64)}else{None};
        compatible25(&mut own.direct,leaf,raw);
        if deliver{for (l,v) in other{if own.received.contains_key(&l){self.repeat+=1;}compatible25(&mut own.received,l,v);self.transferred+=1;}}
        self.edges+=1;self.eligible+=eligible as u64;self.novel+=novel as u64;self.blocked+=blocked as u64;
        (eligible,novel,blocked,mean)
    }
    fn summary(&self)->Value{
        let received:usize=self.nodes.values().flat_map(|v|v.values()).map(|c|c.received.len()).sum();
        assert_eq!(received as u64,self.transferred);
        json!({"iterations":self.events,"edges":self.edges,"children":self.nodes.len(),
            "contexts":self.nodes.values().map(|v|v.len()).sum::<usize>(),
            "shared":self.nodes.values().filter(|v|v.len()>1).count(),
            "unique_base_leaf":self.payload.len(),"eligible":self.eligible,"novel":self.novel,
            "blocked_receipt":self.blocked,"transferred":self.transferred,"receipt_items":received,
            "applied":self.applied,"changed":self.changed,"repeated_delivery":self.repeat})
    }
}
struct K25{
    ledger:Mutex<Ledger25>, trace:bool, b:bool, active:AtomicU64,peak:AtomicU64,
    width_realized:AtomicU64, work:[AtomicU64;4],
}
impl K25{fn new(trace:bool)->Self{Self{ledger:Mutex::new(Ledger25::default()),trace,b:false,
    active:AtomicU64::new(0),peak:AtomicU64::new(0),width_realized:AtomicU64::new(0),
    work:std::array::from_fn(|_|AtomicU64::new(0))}}}
#[cfg(test)]
mod receipt25_tests{
    use super::*;
    #[test]fn once_per_context_even_across_direct_arrival(){
        let mut l=Ledger25::default();
        assert_eq!(l.observe(9,(1,0),(20,false),0.5,false),(0,0,0,None));
        assert_eq!(l.observe(9,(2,0),(21,false),-0.5,true),(1,1,0,Some(0.5)));
        assert_eq!(l.observe(9,(2,0),(22,false),0.,true),(1,0,1,None));
        assert_eq!(l.observe(9,(2,0),(20,false),0.5,true),(0,0,0,None));
        assert_eq!(l.observe(9,(2,0),(23,false),0.25,true),(0,0,0,None));
        assert_eq!(l.transferred,1);assert_eq!(l.repeat,0);
    }
}
pub const S:u8=1;pub const X:u8=2;pub const V:u8=4;pub const B:u8=8;''')
sub('    merge24:Option<Mutex<Merge24>>,merge24_mode:u8,','    merge24:Option<Mutex<Merge24>>,merge24_mode:u8,\n    k25:Option<K25>,')
sub('merge24:None,merge24_mode:0,','merge24:None,merge24_mode:0,k25:None,')
sub('    pub fn with_merge24(mut self)->Self{','''    pub fn with_k25(mut self,trace:bool)->Self{
        assert!(self.mask==0&&self.vmode==0&&self.bmode==0&&self.proof_mode==0);
        self.k25=Some(K25::new(trace));self
    }
    pub fn set_k25_mask(&mut self,m:u8){
        assert!(m<=15);
        let k=self.k25.as_mut().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        self.mask=m&3;self.vmode=if m&V!=0{1}else{0};k.b=m&B!=0;
        for c in &self.counts{c.store(0,Ordering::Relaxed);}
        self.workers.store(0,Ordering::Relaxed);
        k.peak.store(0,Ordering::Relaxed);k.width_realized.store(0,Ordering::Relaxed);
        for c in &k.work{c.store(0,Ordering::Relaxed);}
    }
    pub fn k25_clock(&self)->Option<std::time::Instant>{self.k25.as_ref().map(|_|std::time::Instant::now())}
    fn k25_time(&self,i:usize,t:Option<std::time::Instant>){
        if let (Some(k),Some(t))=(&self.k25,t){k.work[i].fetch_add(t.elapsed().as_nanos().min(u64::MAX as u128) as u64,Ordering::Relaxed);}
    }
    pub fn k25_width_observed(&self,base:usize,target:usize,old:usize,actual:usize,t:Option<std::time::Instant>){
        if let Some(k)=&self.k25{
            if self.mask&X!=0{k.width_realized.fetch_add(actual.min(target).saturating_sub(base.max(old)) as u64,Ordering::Relaxed);}
            self.k25_time(1,t);
        }
    }
    fn k25_enter(&self){
        let k=self.k25.as_ref().unwrap();
        let v=k.active.fetch_add(1,Ordering::AcqRel)+1;k.peak.fetch_max(v,Ordering::Relaxed);
    }
    fn k25_leave(&self){let k=self.k25.as_ref().unwrap();assert!(k.active.fetch_sub(1,Ordering::AcqRel)>0);}
    pub fn k25_report(&self,trace:bool)->Value{
        let k=self.k25.as_ref().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        let l=k.ledger.lock().unwrap();
        json!({"hooks":self.snapshot(),"ledger":l.summary(),"peak":k.peak.load(Ordering::Acquire),
            "width_realized":k.width_realized.load(Ordering::Acquire),
            "work_ns":k.work.iter().map(|v|v.load(Ordering::Acquire)).collect::<Vec<_>>(),
            "trace":if trace&&k.trace{json!(l.trace)}else{Value::Null}})
    }
    fn backup25<G:GameState>(&self,root:&G,path:&[PathEdge<G::Move>],raw_leaf:f32,value:f32,terminal:bool){
        let k=self.k25.as_ref().unwrap();let mut l=k.ledger.lock().unwrap();
        let mut state=root.clone();let mut keys=Vec::with_capacity(path.len());let mut moves=Vec::with_capacity(path.len());
        for pe in path{
            let e=&pe.parent.read_edges()[pe.edge_idx];let parent=exact_key(&state);let mv=state.move_to_idx(e.mv);
            assert_eq!(state.tt_hash(),pe.parent.hash);state=state.apply_move(e.mv);
            let child=exact_key(&state);assert_eq!(state.tt_hash(),e.child.hash);
            l.exact.identity(child,&*e.child as *const _ as usize);
            keys.push((child,(parent,mv)));moves.push(mv);
        }
        assert_eq!(state.is_terminal(),terminal);
        let leaf=(exact_key(&state),terminal);
        compatible25(&mut l.payload,leaf,raw_leaf);
        let mut raw=raw_leaf;let mut y=value;let mut steps=Vec::new();
        for (i,pe) in path.iter().enumerate().rev(){
            let e=&pe.parent.read_edges()[pe.edge_idx];let(child,ctx)=keys[i];
            let incoming=y;let deliver=k.b&&!terminal;
            let (eligible,novel,blocked,mean)=l.observe(child,ctx,leaf,raw,deliver);
            if deliver{if let Some(m)=mean{y=math::blend(y,m,novel as u64,false);l.applied+=1;l.changed+=u64::from((y-incoming).abs()>1e-7);}}
            raw=-raw;y=-y;
            let(vv,vq)=pe.applied_vl;e.remove_vl(vv,vq);e.record_sample(y);
            e.child.record_incoming_sample(e as *const _ as usize,-y);
            atomic_f64_add(&pe.parent.w_total,y as f64);pe.parent.n_total.fetch_add(1,Ordering::AcqRel);
            if k.trace{steps.push(json!({"eligible":eligible,"novel":novel,"blocked":blocked,"mean":mean,"incoming":incoming,"backup":y}));}
        }
        l.events+=1;
        if k.trace{steps.reverse();l.trace.push(json!({"path":moves,"raw":raw_leaf,"leaf":value,"terminal":terminal,"mask":self.mask|if self.vmode!=0{V}else{0}|if k.b{B}else{0},"steps":steps}));}
    }
    pub fn with_merge24(mut self)->Self{''')
sub('    fn four_iterate(&self,h:&Hooks){\n        let sel=', '    fn four_iterate(&self,h:&Hooks){\n        let select_start=h.k25_clock();\n        let sel=')
sub('            self.par_ctrl.should_reserve_virtual_loss(),&self.par_ctrl,h);\n        if let Some(t)=rayon::current_thread_index()', '            self.par_ctrl.should_reserve_virtual_loss(),&self.par_ctrl,h);\n        h.k25_time(0,select_start);\n        if let Some(t)=rayon::current_thread_index()')
sub('        let mut proved=terminal;\n        let known_interval=', '        let mut proved=terminal;let mut raw_base=0f32;\n        let known_interval=')
sub('            h.add(12,1);sel.leaf_state.outcome()', '            h.add(12,1);raw_base=sel.leaf_state.outcome();raw_base')
sub('            let mut ev=self.evaluator.evaluate(&sel.leaf_state);h.add(7,1);','            let eval_start=h.k25_clock();\n            let mut ev=self.evaluator.evaluate(&sel.leaf_state);h.add(7,1);raw_base=ev.value;')
sub('            expand_with_result(&sel.leaf,&sel.leaf_state,ev,&self.tt,self.config.pw.as_ref())\n        };\n        h.backup::<G>(&self.root_state,&sel.path,value,proved);', '''            h.k25_time(2,eval_start);let expand_start=h.k25_clock();
            let v=expand_with_result(&sel.leaf,&sel.leaf_state,ev,&self.tt,self.config.pw.as_ref());
            h.k25_time(1,expand_start);v
        };
        let backup_start=h.k25_clock();
        if h.k25.is_some(){assert_eq!(terminal,proved);h.backup25::<G>(&self.root_state,&sel.path,raw_base,value,terminal);}
        else{h.backup::<G>(&self.root_state,&sel.path,value,proved);}
        h.k25_time(3,backup_start);''')
sub('    pub fn four_axis_block(&self,delta:u32,threads:usize,h:&Hooks){','''    pub fn four_axis_k25_block(&self,delta:u32,threads:usize,h:&Hooks){
        assert!(threads==1||threads==2);assert!(delta>=threads as u32&&delta%threads as u32==0);
        let before=self.root.n_total.load(Ordering::Acquire);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        if threads==1{for _ in 0..delta{h.k25_enter();self.four_iterate(h);h.k25_leave();}}
        else{rayon::scope(|scope|{for _ in 0..threads{scope.spawn(move |_|{
            for _ in 0..delta/threads as u32{h.k25_enter();self.four_iterate(h);h.k25_leave();}
        });}});}
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+delta);
    }
    pub fn four_axis_block(&self,delta:u32,threads:usize,h:&Hooks){''')
p.write_text(s)
p=Path('src/mcts/select.rs');s=p.read_text()
sub('        let mut n_visible = axes.map_or(n_visible, |h|h.width(&cur_node,n_visible,depth));','        let base_visible=n_visible;\n        let mut n_visible = axes.map_or(n_visible, |h|h.width(&cur_node,n_visible,depth));')
sub('        let n_mat = cur_node.materialized_count();\n        if n_mat < n_visible {','        let n_mat = cur_node.materialized_count();\n        let mat_clock=if n_mat<n_visible{axes.and_then(|h|h.k25_clock())}else{None};\n        if n_mat < n_visible {')
sub('        let n_edges = n_visible.min(edges_full.len());\n        if n_edges == 0 {','        let n_edges = n_visible.min(edges_full.len());\n        if let Some(h)=axes{h.k25_width_observed(base_visible,n_visible,n_mat,n_edges,mat_clock);}\n        if n_edges == 0 {')
p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text()
sub('mod evidence_cpu_server;','mod evidence_cpu_server;\n#[path = "mcts_server/k25_cpu_server.rs"]\nmod k25_cpu_server;')
sub('        if cmd == "research_connect4_evidence_cpu" {','''        if cmd == "research_connect4_k25_cpu" {
            emit_stdout_json_value(&k25_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
        if cmd == "research_connect4_evidence_cpu" {''')
p.write_text(s)
