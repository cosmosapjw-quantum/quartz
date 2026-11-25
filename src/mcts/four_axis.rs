//! Research-only synchronous two-worker interventions on all four MCTS stages.
//! Standard search paths retain their existing semantics; no TT layout changes.
use super::*;
use super::four_axis_math::{self as math, Ledger};
use super::node::{PathEdge,atomic_f64_load,atomic_f64_add};
use serde_json::{json,Value};
use std::sync::Mutex;
use super::proof_graph::{Graph,Key,I,Frontier};
use crate::games::connect4::Connect4;
fn exact_key<G:GameState>(s:&G)->Key{
    let c=(s as &dyn std::any::Any).downcast_ref::<Connect4>().expect("proof research is unrestricted Connect4 only");
    assert!(c.active_root_gate_mask().is_none());
    let mut k=if c.current_player()==1{1u128<<84}else{0};
    for (i,v) in c.board_flat().iter().enumerate(){k|=((*v+1) as u128)<<(2*i);}
    k
}
fn keyhex(k:Key)->String{format!("{k:022x}")}
pub struct ProofPlan{pub stop:bool,pub allowed:Vec<usize>}

pub const S:u8=1;pub const X:u8=2;pub const V:u8=4;pub const B:u8=8;
const NAMES:[&str;36]=["select_nodes","select_changed","select_deep","width_nodes","width_limited","width_mass_added","width_deep","base_evals","extra_evals","exact_child_evals","refinements","eval_changed","terminal_leaves","backup_edges","temporal_blends","merge_blends","backup_changed","terminal_edges","proofs","proof_win","proof_scans","proof_checks","graph_register","graph_rule_checks","graph_solved_skips","graph_filtered","graph_width_extra","graph_plans","graph_b_skip","local_claims","local_focus_edges","local_expand_edges","local_frontier_edges","local_frontier_hits","local_forced_deep","local_scored"];
pub struct Hooks{
    pub mask:u8,
    counts:[AtomicU64;36],
    graph:Mutex<Graph>,pub proof_mode:u8,proof_priority:bool,trace_b:bool,b_events:Mutex<Vec<Value>>,
    vmode:u8,bmode:u8,trace:bool,
    local_mode:u8,local_remaining:AtomicU64,local_focus:Option<usize>,local_frontier:Option<Frontier>,
    leaf_paths:Mutex<Vec<Vec<usize>>>,workers:AtomicU64,
    ledger:Mutex<Ledger>,
}
impl Hooks{
    pub fn new(mask:u8)->Self{Self::new_modes(mask&3,if mask&V!=0{1}else{0},if mask&B!=0{4}else{0},false)}
    pub fn new_modes(mask:u8,vmode:u8,bmode:u8,trace:bool)->Self{
        assert!(mask<=3&&vmode<=4&&bmode<=4);
        Self{mask,vmode,bmode,trace,local_mode:0,local_remaining:AtomicU64::new(0),local_focus:None,local_frontier:None,counts:std::array::from_fn(|_|AtomicU64::new(0)),ledger:Mutex::new(Ledger::default()),leaf_paths:Mutex::new(Vec::new()),workers:AtomicU64::new(0),graph:Mutex::new(Graph::default()),proof_mode:0,proof_priority:true,trace_b:false,b_events:Mutex::new(Vec::new())}
    }
    pub fn with_graph(mut self,mode:u8)->Self{assert!(mode<=2);self.proof_mode=mode;self}
    pub fn with_b_trace(mut self)->Self{self.trace_b=true;self}
    pub fn with_safe_priority(mut self)->Self{self.proof_priority=false;self}
    /// Only mutate between completed blocks, never while workers are active.
    pub fn configure_local(&mut self,mode:u8,v:u8,b:u8,priority:bool){
        assert!(mode<=3&&v<=4&&b<=4);self.local_mode=mode;self.vmode=v;self.bmode=b;self.proof_priority=priority;
    }
    pub fn claim_local(&self)->u8{
        if self.local_mode==0{return 0;}
        if self.local_remaining.fetch_update(Ordering::AcqRel,Ordering::Relaxed,|n|if n>0{Some(n-1)}else{None}).is_ok(){
            self.add(29,1);self.local_mode
        }else{0}
    }
    /// Apply only a currently safe unresolved action. Candidate rank is the
    /// actual materialization prefix index, not a new artificial move.
    pub fn local_override<G:GameState>(&self,mode:u8,s:&G,node:&MctsNode<G::Move>,allowed:&[usize],visible:usize,depth:usize,expanded:bool)->Option<(usize,usize)>{
        if mode==0||self.proof_mode!=2{return None;}
        if mode==1&&depth!=0||mode==2&&expanded{return None;}
        let k=exact_key(s);let g=self.graph.lock().unwrap();
        if g.witness(k).is_some()&&depth==0{return None;}
        let wanted=match mode{
            1=>self.local_focus?,
            2=>usize::MAX,
            3=>self.local_frontier.as_ref()?.path.iter().find(|x|x.0==k)?.1,
            _=>return None
        };
        let aa=g.actions(k);let safe=g.allowed(k);
        let c=node.candidates.get()?;
        c.iter().enumerate().find_map(|(i,(mv,_))|{
            let m=s.move_to_idx(*mv);
            if (if mode==2{i>=visible}else{m==wanted})&&allowed.contains(&m)&&safe.contains(&m)&&aa.iter().any(|&(j,(l,u))|j==m&&l<u){Some((i,m))}else{None}
        })
    }
    pub fn local_applied(&self,mode:u8,depth:usize){
        assert!((1..=3).contains(&mode));self.add(29+mode as usize,1);self.add(34,(depth>0) as u64);
    }

    /// Read exact root facts without constructing a full diagnostic JSON snapshot.
    pub fn root_control<G:GameState>(&self,s:&G)->(Option<usize>,u8,u8){
        if self.proof_mode==0{return (None,2,0);}
        let mut g=self.graph.lock().unwrap();let k=self.register_locked(&mut g,s);
        let a=g.actions(k);
        let risk=a.iter().map(|&(m,(l,_))|{
            let u=a.iter().filter(|x|x.0!=m).map(|x|x.1.1).max().unwrap_or(-1);
            (u-l).max(0) as u8
        }).min().unwrap();
        let best_l=a.iter().map(|x|x.1.0).max().unwrap();
        let h=a.iter().filter(|x|x.1.1>=best_l&&x.1.0<x.1.1).count() as u8;
        let w=g.witness(k);assert_eq!(w.is_some(),risk==0);(w,risk,h)
    }

    fn register_locked<G:GameState>(&self,g:&mut Graph,s:&G)->Key{
        let k=exact_key(s);if g.registered(k){return k;}
        self.add(22,1);
        if s.is_terminal(){g.terminal(k,s.outcome() as i8);return k;}
        let mut a=Vec::new();
        for mv in s.legal_moves(){
            let c=s.apply_move(mv);let ck=exact_key(&c);self.add(23,1);
            if c.is_terminal(){g.terminal(ck,c.outcome() as i8);}
            a.push((s.move_to_idx(mv),ck));
        }
        g.register(k,a);k
    }
    pub fn proof_register<G:GameState>(&self,s:&G)->Option<I>{
        if self.proof_mode==0{return None;}
        let mut g=self.graph.lock().unwrap();let k=self.register_locked(&mut g,s);Some(g.interval(k))
    }
    pub fn proof_plan<G:GameState>(&self,s:&G,depth:usize)->Option<ProofPlan>{
        if self.proof_mode==0{return None;}
        let mut g=self.graph.lock().unwrap();let k=self.register_locked(&mut g,s);
        if self.proof_mode==1{return None;}
        let iv=g.interval(k);self.add(27,1);
        if depth>0&&iv.0==iv.1{return Some(ProofPlan{stop:true,allowed:Vec::new()});}
        let all=g.actions(k);let mut allowed=g.allowed(k);let n0=allowed.len();
        if depth==0{
            if let Some(w)=g.witness(k){allowed=vec![w];}
        }
        if self.proof_priority&&allowed.len()>1{
            let unknown:Vec<_>=all.iter().filter(|x|allowed.contains(&x.0)&&x.1.0!=x.1.1).map(|x|x.0).collect();
            if !unknown.is_empty(){allowed=unknown;}
        }
        self.add(25,(all.len()-allowed.len()) as u64);
        assert!(!allowed.is_empty()&&allowed.len()<=n0);
        Some(ProofPlan{stop:false,allowed})
    }
    pub fn proof_width_extra(&self){self.add(26,1);}
    pub fn root_proof<G:GameState>(&self,s:&G,normal:usize,visits:&[(usize,u32)],dump:bool)->Value{
        if self.proof_mode==0{return Value::Null;}
        let mut g=self.graph.lock().unwrap();let k=self.register_locked(&mut g,s);
        let a=g.actions(k);let allowed=g.allowed(k);let witness=g.witness(k);
        let best=witness.unwrap_or_else(||if allowed.contains(&normal){normal}else{allowed.iter().copied().max_by_key(|m|(visits.iter().find(|x|x.0==*m).map_or(0,|x|x.1),std::cmp::Reverse(*m))).unwrap()});
        let mut obj=json!({"key":keyhex(k),"interval":g.interval(k),"actions":a,"allowed":allowed,"witness":witness,"best":best,"nodes":g.nodes.len(),"changes":g.changes,"notices":g.notices,"shared_notices":g.shared_notices,"registered":g.nodes.values().filter(|n|n.actions.is_some()||n.terminal.is_some()).count(),"solved":g.nodes.keys().filter(|&&x|{let i=g.interval(x);i.0==i.1}).count()});
        if dump{obj["graph"]=json!(g.nodes.iter().map(|(&key,n)|json!({"key":keyhex(key),"interval":g.interval(key),"terminal":n.terminal,"actions":n.actions.as_ref().map(|a|a.iter().map(|&(m,c)|(m,keyhex(c))).collect::<Vec<_>>())})).collect::<Vec<_>>());}
        obj
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
        o.insert("b_events".into(),json!(*self.b_events.lock().unwrap()));
        Value::Object(o)
    }
    fn backup<G:GameState>(&self,root:&G,path:&[PathEdge<G::Move>],leaf:f32,terminal:bool){
        if self.bmode==0{backprop::<G>(path,leaf);return;}
        // All research B writers in this engine share this lock. Readers use
        // the existing weak atomic snapshots; this is NOT an atomic read tuple.
        let mut exact_children=Vec::new();
        if self.proof_mode!=0&&self.bmode>=2&&!terminal{
            let mut state=root.clone();
            for pe in path{
                state=state.apply_move(pe.parent.read_edges()[pe.edge_idx].mv);
                exact_children.push(exact_key(&state));
            }
        }
        let mut ledger=self.ledger.lock().unwrap();
        let mut raw=leaf;let mut y=leaf;
        for (depth,pe) in path.iter().enumerate().rev(){
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
                    let unresolved=if exact_children.is_empty(){true}else{
                        let iv=self.graph.lock().unwrap().interval(exact_children[depth]);iv.0!=iv.1
                    };
                    if !unresolved{self.add(28,1);}
                    if unresolved{
                    if self.trace_b{
                        let mut ev=self.b_events.lock().unwrap();
                        if ev.len()<8{
                            let mut child=root.clone();let mut pth=Vec::new();let mut parent_key=0;
                            for (i,pe) in path.iter().take(depth+1).enumerate(){
                                if i==depth{parent_key=exact_key(&child);}
                                let mv=pe.parent.read_edges()[pe.edge_idx].mv;
                                pth.push(child.move_to_idx(mv));child=child.apply_move(mv);
                            }
                            let ck=keyhex(exact_key(&child));
                            if !ev.iter().any(|x|x["child_key"].as_str()==Some(ck.as_str())){
                                ev.push(json!({"path":pth,"child_key":ck,"parent_key":keyhex(parent_key),"child_hash":format!("{:016x}",e.child.hash),"parent_hash":format!("{:016x}",pe.parent.hash),"y":y,"m":m,"raw":raw,"n":n,"other":other}));
                            }
                        }
                    }
                    let z=math::blend(y,m,n,false);
                    self.add(if other{15}else{14},1);self.add(16,((z-y).abs()>1e-7) as u64);
                    y=z;
                    }
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
    pub fn prepare_local(&self,h:&mut Hooks)->Value{
        h.local_focus=None;h.local_frontier=None;
        h.local_remaining.store(if h.local_mode>0{2}else{0},Ordering::Release);
        if h.local_mode==0{return Value::Null;}
        if h.local_mode==2{return json!({"mode":"EXPAND","claims":2});}
        let k=exact_key(&self.root_state);let g=h.graph.lock().unwrap();
        if h.local_mode==1{
            let aa=g.actions(k);let allowed=g.allowed(k);
            let mut edges:Vec<_>=self.root.read_edges().iter().enumerate().filter_map(|(i,e)|{
                let m=self.root_state.move_to_idx(e.mv);let n=e.n.load(Ordering::Acquire);
                if n==0||!allowed.contains(&m){return None;}
                Some((i,m,n,atomic_f64_load(&e.w)/n as f64))
            }).collect();
            edges.sort_by_key(|x|(std::cmp::Reverse(x.2),x.0));
            let mut gap=None;
            if edges.len()>=2{
                let a=edges[0];let b=edges[1];let d=(a.3-b.3).abs();gap=Some(d);
                if g.witness(k).is_none()&&d<=0.25&&aa.iter().any(|&(m,(l,u))|m==b.1&&l<u){h.local_focus=Some(b.1);}
            }
            return json!({"mode":"FOCUS","claims":2,"target":h.local_focus,"gap":gap,"leaders":edges.into_iter().take(2).map(|(_,m,n,q)|(m,n,q)).collect::<Vec<_>>()});
        }
        let (f,n)=g.frontier(k,64);h.add(35,n as u64);h.local_frontier=f;
        json!({"mode":"FRONTIER","claims":2,"scored":n,"target":h.local_frontier.as_ref().map(|f|json!({"key":keyhex(f.key),"path":f.path.iter().map(|&(a,m,b)|(keyhex(a),m,keyhex(b))).collect::<Vec<_>>(),"gains":f.gains,"score":f.score}))})
    }
    fn four_iterate(&self,h:&Hooks){
        let sel=super::select::select_axes(&self.root,&self.root_state,self.config.c_puct,
            self.root_noise.as_deref(),self.config.pw.as_ref(),self.resident_root_mask_words_snapshot(),
            self.config.max_depth,&self.tt,self.config.exact_terminal_value,self.config.fpu_reduction,
            self.par_ctrl.should_reserve_virtual_loss(),&self.par_ctrl,h);
        if let Some(t)=rayon::current_thread_index(){if t<64{h.workers.fetch_or(1u64<<t,Ordering::Relaxed);}}
        if let Some(ref f)=h.local_frontier{if exact_key(&sel.leaf_state)==f.key{h.add(33,1);}}
        let terminal=sel.leaf_state.is_terminal();
        let mut proved=terminal;
        let known_interval=h.proof_register(&sel.leaf_state);
        let value=if terminal{
            h.add(12,1);sel.leaf_state.outcome()
        }else if h.proof_mode==2&&known_interval.is_some_and(|x|x.0==x.1){
            h.add(24,1);proved=true;known_interval.unwrap().0 as f32
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
        h.backup::<G>(&self.root_state,&sel.path,value,proved);
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
