//! Research-only synchronous two-worker interventions on all four MCTS stages.
//! Standard search paths retain their existing semantics; no TT layout changes.
use super::*;
use super::four_axis_math::{self as math, Ledger};
use super::node::{PathEdge,atomic_f64_load,atomic_f64_add};
use serde_json::{json,Value};
use std::sync::Mutex;
pub const S:u8=1;pub const X:u8=2;pub const V:u8=4;pub const B:u8=8;
const NAMES:[&str;22]=["select_nodes","select_changed","select_deep","width_nodes","width_limited","width_mass_added","width_deep","base_evals","extra_evals","exact_child_evals","refinements","eval_changed","terminal_leaves","backup_edges","temporal_blends","merge_blends","backup_changed","terminal_edges","proofs","proof_win","proof_scans","proof_checks"];
pub struct Hooks{
    pub mask:u8,
    counts:[AtomicU64;22],
    vmode:u8,bmode:u8,trace:bool,
    leaf_paths:Mutex<Vec<Vec<usize>>>,workers:AtomicU64,
    ledger:Mutex<Ledger>,
}
impl Hooks{
    pub fn new(mask:u8)->Self{Self::new_modes(mask&3,if mask&V!=0{1}else{0},if mask&B!=0{4}else{0},false)}
    pub fn new_modes(mask:u8,vmode:u8,bmode:u8,trace:bool)->Self{
        assert!(mask<=3&&vmode<=4&&bmode<=4);
        Self{mask,vmode,bmode,trace,counts:std::array::from_fn(|_|AtomicU64::new(0)),ledger:Mutex::new(Ledger::default()),leaf_paths:Mutex::new(Vec::new()),workers:AtomicU64::new(0)}
    }
    fn add(&self,i:usize,n:u64){self.counts[i].fetch_add(n,Ordering::Relaxed);}
    pub fn width<M:Copy+Send+Sync+'static>(&self,node:&MctsNode<M>,base:usize,depth:usize)->usize{
        if self.mask&X==0{return base;}
        let p=node.candidates.get().expect("expanded node");
        let prior:Vec<_>=p.iter().map(|e|e.1).collect();
        let old=node.materialized_count();let w=math::mass_width(&prior,base,old);
        self.add(3,1);self.add(4,(w<p.len()) as u64);self.add(5,(w>base.max(old)) as u64);self.add(6,(depth>0) as u64);
        w
    }
    pub fn local_c<M:Copy+Send+Sync+'static>(&self,edges:&[super::node::MctsEdge<M>],base:f32,depth:usize)->f32{
        if self.mask&S==0{return base;}
        let ns:Vec<_>=edges.iter().map(|e|e.n.load(Ordering::Relaxed)).collect();
        let ps:Vec<_>=edges.iter().map(|e|e.p).collect();
        let c=math::local_c(base as f64,&ns,&ps) as f32;
        self.add(0,1);self.add(1,((c-base).abs()>1e-6) as u64);self.add(2,(depth>0) as u64);
        c
    }
    pub fn snapshot(&self)->Value{
        let mut o=serde_json::Map::new();
        for (i,n) in NAMES.iter().enumerate(){o.insert(n.to_string(),json!(self.counts[i].load(Ordering::Relaxed)));}
        let (nodes,contexts,shared,obs)=self.ledger.lock().unwrap().counts();
        o.insert("ledger_nodes".into(),json!(nodes));o.insert("ledger_contexts".into(),json!(contexts));
        o.insert("ledger_shared_children".into(),json!(shared));o.insert("ledger_observations".into(),json!(obs));
        o.insert("leaf_paths".into(),json!(*self.leaf_paths.lock().unwrap()));
        o.insert("worker_mask".into(),json!(self.workers.load(Ordering::Relaxed)));
        Value::Object(o)
    }
    fn backup<G:GameState>(&self,path:&[PathEdge<G::Move>],leaf:f32,terminal:bool){
        if self.bmode==0{backprop::<G>(path,leaf);return;}
        // All research B writers in this engine share this lock. Readers use
        // the existing weak atomic snapshots; this is NOT an atomic read tuple.
        let mut ledger=self.ledger.lock().unwrap();
        let mut raw=leaf;let mut y=leaf;
        for pe in path.iter().rev(){
            let e=&pe.parent.read_edges()[pe.edge_idx];
            let (own,other)=if self.bmode>=2{ledger.observe_split(e.child.hash,pe.parent.hash,pe.edge_idx,raw)}else{(None,None)};
            let old=match self.bmode{
                2=>own.map(|(m,n)|(m,n,false)),
                3=>other.map(|(m,n)|(m,n,true)),
                4=>other.map(|(m,n)|(m,n,true)).or_else(||own.map(|(m,n)|(m,n,false))),
                _=>None,
            };
            self.add(13,1);self.add(17,terminal as u64);
            if !terminal{
                if let Some((m,n,other))=old{
                    let z=math::blend(y,m,n,false);
                    self.add(if other{15}else{14},1);self.add(16,((z-y).abs()>1e-7) as u64);
                    y=z;
                }
            }
            raw=-raw;y=-y;
            let (vv,vq)=pe.applied_vl;e.remove_vl(vv,vq);
            let n0=e.n.fetch_add(1,Ordering::AcqRel) as f64;
            let w0=atomic_f64_load(&e.w);
            e.add_w(y);
            if n0>=1.{
                let mu=w0/n0;let mu1=(w0+y as f64)/(n0+1.);
                atomic_f64_add(&e.m2,(y as f64-mu)*(y as f64-mu1));
            }
            // No legacy repeated-edge-as-RTT counter here. Provenance is
            // the actual incoming-context ledger above.
            atomic_f64_add(&pe.parent.w_total,y as f64);
            pe.parent.n_total.fetch_add(1,Ordering::AcqRel);
        }
    }
}
impl<G:GameState> MctsEngine<G>{
    fn four_iterate(&self,h:&Hooks){
        let sel=super::select::select_axes(&self.root,&self.root_state,self.config.c_puct,
            self.root_noise.as_deref(),self.config.pw.as_ref(),self.resident_root_mask_words_snapshot(),
            self.config.max_depth,&self.tt,self.config.exact_terminal_value,self.config.fpu_reduction,
            self.par_ctrl.should_reserve_virtual_loss(),&self.par_ctrl,h);
        if let Some(t)=rayon::current_thread_index(){if t<64{h.workers.fetch_or(1u64<<t,Ordering::Relaxed);}}
        let terminal=sel.leaf_state.is_terminal();
        let mut proved=terminal;
        let value=if terminal{
            h.add(12,1);sel.leaf_state.outcome()
        }else{
            if h.trace{
                let path:Vec<_>=sel.path.iter().map(|pe|self.root_state.move_to_idx(pe.parent.read_edges()[pe.edge_idx].mv)).collect();
                let mut p=h.leaf_paths.lock().unwrap();if p.len()<4&&!p.contains(&path){p.push(path);}
            }
            let mut ev=self.evaluator.evaluate(&sel.leaf_state);h.add(7,1);
            let mut cand=if h.vmode>0{ev.policy.clone()}else{Vec::new()};
            cand.sort_by(|a,b|b.1.total_cmp(&a.1).then_with(||sel.leaf_state.move_to_idx(a.0).cmp(&sel.leaf_state.move_to_idx(b.0))));
            let mut known=vec![None;cand.len()];
            if h.vmode>=2{
                h.add(20,1);
                for (i,&(mv,_)) in cand.iter().enumerate(){
                    let child=sel.leaf_state.apply_move(mv);h.add(21,1);
                    if child.is_terminal(){known[i]=Some(-child.outcome());}
                }
                if let Some(v)=math::certificate(&known){
                    ev.value=v;proved=true;h.add(18,1);h.add(19,(v==1.) as u64);
                }
            }
            if !proved && matches!(h.vmode,1|3|4) && ev.value.abs()<=0.65 && cand.len()>=2{
                let k=if h.vmode==4{cand.len()}else{2};
                let mut best=-1f32;
                for (i,&(mv,_)) in cand.iter().take(k).enumerate(){
                    let a=if let Some(v)=known[i]{h.add(9,1);v}else{
                        let child=sel.leaf_state.apply_move(mv);
                        if child.is_terminal(){h.add(9,1);-child.outcome()}else{h.add(8,1);-self.evaluator.evaluate(&child).value}
                    };
                    best=best.max(a);
                }
                let v=(0.5*ev.value+0.5*best).clamp(-1.,1.);
                h.add(10,1);h.add(11,((v-ev.value).abs()>1e-7) as u64);ev.value=v;
            }
            expand_with_result(&sel.leaf,&sel.leaf_state,ev,&self.tt,self.config.pw.as_ref())
        };
        h.backup::<G>(&sel.path,value,proved);
    }
    // Scope: fresh, nonterminal alternating-turn acyclic game roots, fixed budget,
    // no legacy Quartz/policy/async/TT cap. Endpoint enforces this contract.
    pub fn run_four_axis(&self,visits:u32,threads:usize,h:&Hooks){
        assert!(threads==1||threads==2);
        assert!(self.config.quartz.is_none()&&self.config.search_policy.is_none()&&self.config.max_tt_size.is_none());
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        self.selection_telemetry.reset();
        let next=AtomicU32::new(self.root.n_total.load(Ordering::Relaxed));
        let chunk=fixed_budget_ticket_chunk(threads);
        rayon::scope(|scope|{
            for _ in 0..threads{
                let t=&next;
                scope.spawn(move |_|loop{
                    let lo=t.fetch_add(chunk,Ordering::Relaxed);
                    if lo>=visits{break;}
                    for _ in lo..lo.saturating_add(chunk).min(visits){self.four_iterate(h);}
                });
            }
        });
        assert_eq!(self.root.n_total.load(Ordering::Acquire),visits);
    }
    pub fn four_axis_block(&self,delta:u32,threads:usize,h:&Hooks){
        assert!(threads==1||threads==2);assert!(delta>=threads as u32&&delta%threads as u32==0);
        let before=self.root.n_total.load(Ordering::Acquire);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        rayon::scope(|scope|{for _ in 0..threads{scope.spawn(move |_|{
            for _ in 0..delta/threads as u32{self.four_iterate(h);}
        });}});
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+delta);
    }
    pub fn four_axis_snapshot(&self)->Value{
        let e:Vec<_>=self.root.read_edges().iter().map(|e|json!({"move":self.root_state.move_to_idx(e.mv),"n":e.n.load(Ordering::Acquire),"w":atomic_f64_load(&e.w),"m2":atomic_f64_load(&e.m2),"prior":e.p,"child_hash":format!("{:016x}",e.child.hash),"vl":e.virtual_losses.load(Ordering::Acquire),"vvalue":atomic_f64_load(&e.virtual_value)})).collect();
        let c:Vec<_>=self.root.candidates.get().map(|c|c.iter().map(|(mv,p)|json!([self.root_state.move_to_idx(*mv),p])).collect()).unwrap_or_default();
        json!({"root_hash":format!("{:016x}",self.root_state.tt_hash()),"root_n":self.root.n_total.load(Ordering::Acquire),"root_w":atomic_f64_load(&self.root.w_total),"candidates":c,"edges":e,"best":self.best_move().map(|x|self.root_state.move_to_idx(x)),"evaluator_calls":self.evaluator_call_count(),"tt_size":self.tt.size()})
    }
}
