//! Research-only synchronous two-worker interventions on all four MCTS stages.
//! Standard search paths retain their existing semantics; no TT layout changes.
use super::*;
use super::four_axis_math::{self as math, Ledger};
use super::node::{PathEdge,atomic_f64_load,atomic_f64_add};
use serde_json::{json,Value};
use std::sync::Mutex;
use super::proof_graph::{Graph,Key,I,Frontier};
use crate::games::connect4::Connect4;
use crate::games::gomoku15::Gomoku15;
fn exact_key<G:GameState>(s:&G)->Key{
    let c=(s as &dyn std::any::Any).downcast_ref::<Connect4>().expect("proof research is unrestricted Connect4 only");
    assert!(c.active_root_gate_mask().is_none());
    let mut k=if c.current_player()==1{1u128<<84}else{0};
    for (i,v) in c.board_flat().iter().enumerate(){k|=((*v+1) as u128)<<(2*i);}
    k
}
fn keyhex(k:Key)->String{format!("{k:022x}")}
pub struct ProofPlan{pub stop:bool,pub allowed:Vec<usize>}
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


// K25: one ledger per isolated tree. Exact base-evaluator observations,
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
// K28 atomic free-pool state: low32 shared, high32 protected.
// MAX is inactive; experimental budgets <=64 cannot alias the sentinel.
fn pack28(s:u64,p:u64)->u64{assert!(s<=u32::MAX as u64&&p<=u32::MAX as u64);s|(p<<32)}
fn pools28(v:u64)->(u64,u64){(v&0xffffffff,v>>32)}
fn claim28(v:u64,n:u64,optional:bool)->Option<u64>{
    assert_ne!(v,u64::MAX);
    let(s,p)=pools28(v);let h=if optional{p.min(n)}else{0};
    s.checked_sub(n-h).map(|s1|pack28(s1,p-h))
}
#[cfg(test)]
mod floor28_tests{
    use super::*;
    #[test]fn protected_floor_and_atomic_cross_pool(){
        let x=pack28(3,1);
        assert_eq!(claim28(x,2,true),Some(pack28(2,0)));
        assert_eq!(claim28(x,4,true),Some(0));
        assert_eq!(claim28(x,5,true),None);
        assert_eq!(claim28(pack28(0,2),1,false),None);
        assert_eq!(claim28(pack28(0,2),2,true),Some(0));
        assert_eq!(claim28(0,0,true),Some(0));
        assert_eq!(claim28(pack28(3,1),1,false),Some(pack28(2,1)));
    }
    #[test]fn two_pool_cas_never_overclaims(){
        let f=AtomicU64::new(pack28(4,4));
        let ok=AtomicU64::new(0);
        std::thread::scope(|sc|{for i in 0..16{let f=&f;let ok=&ok;
            sc.spawn(move||{let n=if i%2==0{1}else{2};
            if f.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|claim28(v,n,i%2==1)).is_ok(){ok.fetch_add(n,Ordering::AcqRel);}
        });}});
        let(s,p)=pools28(f.load(Ordering::Acquire));assert_eq!(s+p+ok.load(Ordering::Acquire),8);
    }
}
struct K25{
    ledger:Mutex<Ledger25>, trace:bool, b:bool, active:AtomicU64,peak:AtomicU64,
    width_realized:AtomicU64, work:[AtomicU64;4],
    budget26:AtomicU64,denied26:AtomicU64,refine27:AtomicU64,floor28:AtomicU64,
}
impl K25{fn new(trace:bool)->Self{Self{ledger:Mutex::new(Ledger25::default()),trace,b:false,
    active:AtomicU64::new(0),peak:AtomicU64::new(0),width_realized:AtomicU64::new(0),
    work:std::array::from_fn(|_|AtomicU64::new(0)),budget26:AtomicU64::new(u64::MAX),denied26:AtomicU64::new(0),refine27:AtomicU64::new(u64::MAX),floor28:AtomicU64::new(u64::MAX)}}}
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
pub const S:u8=1;pub const X:u8=2;pub const V:u8=4;pub const B:u8=8;
const NAMES:[&str;37]=["select_nodes","select_changed","select_deep","width_nodes","width_limited","width_mass_added","width_deep","base_evals","extra_evals","exact_child_evals","refinements","eval_changed","terminal_leaves","backup_edges","temporal_blends","merge_blends","backup_changed","terminal_edges","proofs","proof_win","proof_scans","proof_checks","graph_register","graph_rule_checks","graph_solved_skips","graph_filtered","graph_width_extra","graph_plans","graph_b_skip","local_claims","local_focus_edges","local_expand_edges","local_frontier_edges","local_frontier_hits","local_forced_deep","local_scored","graph_repeat_skips"];
pub struct Hooks{
    pub mask:u8,
    counts:[AtomicU64;37],
    graph:Mutex<Graph>,pub proof_mode:u8,proof_priority:bool,trace_b:bool,b_events:Mutex<Vec<Value>>,
    vmode:u8,bmode:u8,trace:bool,
    record_repeats:bool,solved_returns:Mutex<std::collections::BTreeSet<Key>>,
    local_mode:u8,local_remaining:AtomicU64,local_focus:Option<usize>,local_frontier:Option<Frontier>,
    tactical_root_floor:usize,tactical_root_first:Option<usize>,
    decision_force:Option<(usize,u32,u32)>,
    leaf_paths:Mutex<Vec<Vec<usize>>>,workers:AtomicU64,
    ledger:Mutex<Ledger>,
    gomoku_ledger:Option<Mutex<math::ContextLedger<[u64;8]>>>,
    gomoku_b_steps:Mutex<Vec<Value>>,
    merge24:Option<Mutex<Merge24>>,merge24_mode:u8,
    k25:Option<K25>,
}
impl Hooks{
    pub fn new(mask:u8)->Self{Self::new_modes(mask&3,if mask&V!=0{1}else{0},if mask&B!=0{4}else{0},false)}
    pub fn new_modes(mask:u8,vmode:u8,bmode:u8,trace:bool)->Self{
        assert!(mask<=3&&vmode<=4&&bmode<=5);
        Self{mask,vmode,bmode,trace,record_repeats:false,solved_returns:Mutex::new(std::collections::BTreeSet::new()),local_mode:0,local_remaining:AtomicU64::new(0),local_focus:None,local_frontier:None,tactical_root_floor:0,tactical_root_first:None,decision_force:None,counts:std::array::from_fn(|_|AtomicU64::new(0)),ledger:Mutex::new(Ledger::default()),gomoku_ledger:None,gomoku_b_steps:Mutex::new(Vec::new()),merge24:None,merge24_mode:0,k25:None,leaf_paths:Mutex::new(Vec::new()),workers:AtomicU64::new(0),graph:Mutex::new(Graph::default()),proof_mode:0,proof_priority:true,trace_b:false,b_events:Mutex::new(Vec::new())}
    }
    // Scope only exact unrestricted Freestyle15 board-only evaluator seam.
    pub fn with_gomoku_exact_merge(mut self)->Self{
        assert!(self.proof_mode==0&&!self.trace_b);
        self.gomoku_ledger=Some(Mutex::new(math::ContextLedger::default()));self
    }
    pub fn gomoku_backup_trace(&self)->Value{json!(*self.gomoku_b_steps.lock().unwrap())}
    pub fn with_graph(mut self,mode:u8)->Self{assert!(mode<=2);self.proof_mode=mode;self}
    pub fn with_b_trace(mut self)->Self{self.trace_b=true;self}
    pub fn with_safe_priority(mut self)->Self{self.proof_priority=false;self}
    pub fn with_repeat_trace(mut self)->Self{self.record_repeats=true;self}
    /// P3 research-only: preserve the original candidate order, expose its
    /// prefix through the target rank. First-action override is used only by
    /// the single-worker ladder endpoint before its first actual visit.
    pub fn with_tactical_root(mut self,floor:usize,first:Option<usize>)->Self{
        assert!(self.mask==0&&self.proof_mode==0&&self.local_mode==0&&floor<=225);
        self.tactical_root_floor=floor;self.tactical_root_first=first;self
    }
    pub fn tactical_first_action(&self,depth:usize,n:u32)->Option<usize>{
        if depth!=0{return None;}
        if let Some((mv,lo,hi))=self.decision_force{
            if (lo..hi).contains(&n){return Some(mv);}
        }
        if n==0{self.tactical_root_first}else{None}
    }
    /// One-worker research-only completed-block control. Do not use with graph
    /// filtering or other root constraints. Floor and force cleared explicitly.
    pub fn set_decision_force(&mut self,target:Option<(usize,usize)>,lo:u32,hi:u32){
        assert!(self.mask==0&&self.proof_mode==0&&self.local_mode==0&&self.tactical_root_first.is_none());
        if let Some((rank,mv))=target{
            assert!(rank<225&&lo<hi);self.tactical_root_floor=rank+1;self.decision_force=Some((mv,lo,hi));
        }else{self.tactical_root_floor=0;self.decision_force=None;}
    }
    /// Only mutate between completed blocks, never while workers are active.
    pub fn configure_local(&mut self,mode:u8,v:u8,b:u8,priority:bool){
        assert!(mode<=3&&v<=4&&b<=5);self.local_mode=mode;self.vmode=v;self.bmode=b;self.proof_priority=priority;
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
        let base=if depth==0{base.max(self.tactical_root_floor).min(node.candidate_count())}else{base};
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
        let (nodes,contexts,shared,obs)=if let Some(l)=&self.gomoku_ledger{l.lock().unwrap().counts()}else{self.ledger.lock().unwrap().counts()};
        o.insert("ledger_nodes".into(),json!(nodes));o.insert("ledger_contexts".into(),json!(contexts));
        o.insert("ledger_shared_children".into(),json!(shared));o.insert("ledger_observations".into(),json!(obs));
        o.insert("leaf_paths".into(),json!(*self.leaf_paths.lock().unwrap()));
        o.insert("worker_mask".into(),json!(self.workers.load(Ordering::Relaxed)));
        o.insert("b_events".into(),json!(*self.b_events.lock().unwrap()));
        Value::Object(o)
    }
    pub fn with_k25(mut self,trace:bool)->Self{
        assert!(self.mask==0&&self.vmode==0&&self.bmode==0&&self.proof_mode==0);
        self.k25=Some(K25::new(trace));self
    }
    pub fn set_k25_mask(&mut self,m:u8){
        assert!(m<=15);
        let k=self.k25.as_mut().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        self.mask=m&3;self.vmode=if m&V!=0{1}else{0};k.b=m&B!=0;
        for c in &self.counts{c.store(0,Ordering::Relaxed);}
        self.workers.store(0,Ordering::Relaxed);
        k.peak.store(0,Ordering::Relaxed);k.width_realized.store(0,Ordering::Relaxed);k.denied26.store(0,Ordering::Relaxed);
        for c in &k.work{c.store(0,Ordering::Relaxed);}
    }
    fn k26_limited(&self)->bool{
        self.k25.as_ref().is_some_and(|k|k.budget26.load(Ordering::Acquire)!=u64::MAX||k.floor28.load(Ordering::Acquire)!=u64::MAX)
    }
    fn k26_claim(&self,n:u64)->bool{
        let k=self.k25.as_ref().unwrap();
        if k.floor28.load(Ordering::Acquire)!=u64::MAX{
            return k.floor28.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|claim28(v,n,false)).is_ok();
        }
        k.budget26.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_sub(n)).is_ok()
    }
    fn k27_claim_refine(&self,n:u64)->bool{
        let k=self.k25.as_ref().unwrap();
        if k.floor28.load(Ordering::Acquire)!=u64::MAX{
            return k.floor28.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|claim28(v,n,true)).is_ok();
        }
        if k.refine27.load(Ordering::Acquire)==u64::MAX{self.k26_claim(n)}
        else{k.refine27.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_sub(n)).is_ok()}
    }
    fn k26_refund(&self){
        if self.k26_limited(){
            let k=self.k25.as_ref().unwrap();
            if k.floor28.load(Ordering::Acquire)!=u64::MAX{
                k.floor28.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|{let(s,p)=pools28(v);Some(pack28(s+1,p))}).unwrap();
            }else{k.budget26.fetch_add(1,Ordering::AcqRel);}
        }
    }
    pub fn k26_denied(&self)->u64{self.k25.as_ref().unwrap().denied26.load(Ordering::Acquire)}
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
    pub fn with_merge24(mut self)->Self{
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
    fn backup<G:GameState>(&self,root:&G,path:&[PathEdge<G::Move>],leaf:f32,terminal:bool){
        if self.merge24.is_some(){self.backup24::<G>(root,path,leaf,terminal);return;}
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
        let mut exact_gomoku=Vec::new();let mut moves=Vec::new();
        if self.gomoku_ledger.is_some(){
            assert!(self.proof_mode==0&&!self.trace_b&&self.bmode>=2);
            let mut state=root.clone();
            let key=|s:&G|{
                let g=(s as &dyn std::any::Any).downcast_ref::<Gomoku15>().expect("Gomoku board-only exact ledger");
                math::placement_key(g.board_as_12(),g.current_player())
            };
            for pe in path{
                let e=&pe.parent.read_edges()[pe.edge_idx];
                let parent=key(&state);
                assert_eq!(state.tt_hash(),pe.parent.hash);
                moves.push(state.move_to_idx(e.mv));
                state=state.apply_move(e.mv);
                assert_eq!(state.tt_hash(),e.child.hash);
                exact_gomoku.push((key(&state),parent));
            }
        }
        let mut ledger=self.ledger.lock().unwrap();
        let mut exact_ledger=self.gomoku_ledger.as_ref().map(|l|l.lock().unwrap());
        let mut raw=leaf;let mut y=leaf;let mut blends=0u64;
        for (depth,pe) in path.iter().enumerate().rev(){
            let e=&pe.parent.read_edges()[pe.edge_idx];
            let (own,other)=if let Some(l)=exact_ledger.as_mut(){
                let (child,parent)=exact_gomoku[depth];
                l.observe_split(child,parent,moves[depth],raw)
            }else if self.bmode>=2{ledger.observe_split(e.child.hash,pe.parent.hash,pe.edge_idx,raw)}else{(None,None)};
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
                    blends+=1;self.add(if other{15}else{14},1);self.add(16,((z-y).abs()>1e-7) as u64);
                    y=z;
                    }
                }
            }
            raw=-raw;y=-y;
            let (vv,vq)=pe.applied_vl;e.remove_vl(vv,vq);
            e.record_sample(y);
            // No legacy repeated-edge-as-RTT counter here. Provenance is
            // the actual incoming-context ledger above.
            atomic_f64_add(&pe.parent.w_total,y as f64);
            pe.parent.n_total.fetch_add(1,Ordering::AcqRel);
        }
        if exact_ledger.is_some(){
            self.gomoku_b_steps.lock().unwrap().push(json!({"path":moves,"leaf":leaf,"proved":terminal,"mode":self.bmode,"blends":blends}));
        }
    }
}
// K21: quiescent serial, at most seven legal actions. Accounting, not posterior samples.
#[derive(Clone,Debug,serde::Serialize)]
pub struct LiteRoot {
    pub len:usize,pub moves:[usize;7],pub n:[u32;7],pub w:[f64;7],
    pub m2:[f64;7],pub prior:[f64;7],pub visits:u32,pub nn:u64,pub best:usize,
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
        let select_start=h.k25_clock();
        let sel=super::select::select_axes(&self.root,&self.root_state,self.config.c_puct,
            self.root_noise.as_deref(),self.config.pw.as_ref(),self.resident_root_mask_words_snapshot(),
            self.config.max_depth,&self.tt,self.config.exact_terminal_value,self.config.fpu_reduction,
            self.par_ctrl.should_reserve_virtual_loss(),&self.par_ctrl,h);
        h.k25_time(0,select_start);
        if let Some(t)=rayon::current_thread_index(){if t<64{h.workers.fetch_or(1u64<<t,Ordering::Relaxed);}}
        if let Some(ref f)=h.local_frontier{if exact_key(&sel.leaf_state)==f.key{h.add(33,1);}}
        let terminal=sel.leaf_state.is_terminal();
        let mut proved=terminal;let mut raw_base=0f32;
        let known_interval=h.proof_register(&sel.leaf_state);
        let value=if terminal{
            h.add(12,1);h.k26_refund();raw_base=sel.leaf_state.outcome();raw_base
        }else if h.proof_mode==2&&known_interval.is_some_and(|x|x.0==x.1){
            h.add(24,1);
            if h.record_repeats&&!h.solved_returns.lock().unwrap().insert(exact_key(&sel.leaf_state)){h.add(36,1);}
            proved=true;known_interval.unwrap().0 as f32
        }else{
            if h.trace{
                let path:Vec<_>=sel.path.iter().map(|pe|self.root_state.move_to_idx(pe.parent.read_edges()[pe.edge_idx].mv)).collect();
                let mut p=h.leaf_paths.lock().unwrap();if p.len()<4&&!p.contains(&path){p.push(path);}
            }
            let eval_start=h.k25_clock();
            let mut ev=self.evaluator.evaluate(&sel.leaf_state);h.add(7,1);raw_base=ev.value;
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
                let permitted=if h.k26_limited(){
                    let need=cand.iter().take(k).enumerate().filter(|(i,(mv,_))|known[*i].is_none()&&!sel.leaf_state.apply_move(*mv).is_terminal()).count() as u64;
                    h.k27_claim_refine(need)
                }else{true};
                if permitted{
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
                }else{h.k25.as_ref().unwrap().denied26.fetch_add(1,Ordering::Relaxed);}
            }
            h.k25_time(2,eval_start);let expand_start=h.k25_clock();
            let v=expand_with_result(&sel.leaf,&sel.leaf_state,ev,&self.tt,self.config.pw.as_ref());
            h.k25_time(1,expand_start);v
        };
        let backup_start=h.k25_clock();
        if h.k25.is_some(){assert_eq!(terminal,proved);h.backup25::<G>(&self.root_state,&sel.path,raw_base,value,terminal);}
        else{h.backup::<G>(&self.root_state,&sel.path,value,proved);}
        h.k25_time(3,backup_start);
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
    // Hard evaluator-call budget. Each admitted iteration owns one base token;
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
    // Protected optional-call pool. Only a quiescent join transfers its
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
    // Floor rather than ceiling: V consumes its protected balance first,
    // then shared surplus in the same atomic operation. Base never borrows
    // protected balance before the quiescent release.
    pub fn four_axis_k28_floor(&self,target:u64,cap:u32,threads:usize,reserve:u64,h:&Hooks)->Value{
        assert!(target>0&&target<=64&&cap>0&&reserve>0&&reserve<target&&matches!(threads,1|2|4|8));
        assert_eq!(h.proof_mode,0);assert_eq!(h.vmode,1);
        let k=h.k25.as_ref().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        assert_eq!(k.budget26.load(Ordering::Acquire),u64::MAX);assert_eq!(k.refine27.load(Ordering::Acquire),u64::MAX);
        assert_eq!(k.floor28.swap(pack28(target-reserve,reserve),Ordering::AcqRel),u64::MAX);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        let before=self.root.n_total.load(Ordering::Acquire);let calls=self.evaluator_call_count();
        let base0=h.counts[7].load(Ordering::Acquire);let ext0=h.counts[8].load(Ordering::Acquire);
        let mut used=0u32;let mut rounds=0u32;let mut released=0;let mut pre_base=0;let mut pre_extra=0;let mut pre_visits=0;let mut borrowed=0;let mut protected=0;
        for phase in 0..2{
            while used<cap&&pools28(k.floor28.load(Ordering::Acquire)).0>0{
                let quota=cap-used;let tickets=AtomicU32::new(0);
                let worker=||loop{
                    if !h.k26_claim(1){break;}
                    if tickets.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|if n<quota{Some(n+1)}else{None}).is_err(){h.k26_refund();break;}
                    h.k25_enter();self.four_iterate(h);h.k25_leave();
                };
                if threads==1{worker();}else{rayon::scope(|scope|{for _ in 0..threads{let w=&worker;scope.spawn(move |_|w());}});}
                let n=tickets.load(Ordering::Acquire);assert!(n>0);used+=n;rounds+=1;
            }
            assert_eq!(k.active.load(Ordering::Acquire),0);
            if phase==0{
                let(s,p)=pools28(k.floor28.load(Ordering::Acquire));
                released=p;protected=reserve-p;
                pre_base=h.counts[7].load(Ordering::Acquire)-base0;pre_extra=h.counts[8].load(Ordering::Acquire)-ext0;
                pre_visits=used;borrowed=pre_extra-protected;
                assert_eq!(pre_base+borrowed+s,target-reserve);
                assert_eq!(self.evaluator_call_count()-calls,pre_base+pre_extra);
                k.floor28.store(pack28(s+p,0),Ordering::Release);
            }
        }
        let(free,p)=pools28(k.floor28.swap(u64::MAX,Ordering::AcqRel));assert_eq!(p,0);
        let nn=self.evaluator_call_count()-calls;assert_eq!(target-free,nn);
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+used);
        json!({"target":target,"cap":cap,"used_nn":nn,"unused":free,"visits":used,"rounds":rounds,
            "reserve":reserve,"released":released,"protected_base":pre_base,"protected_extra":protected,"protected_visits":pre_visits,
            "borrowed":borrowed,"pre_extra":pre_extra,"denied_refinement":h.k26_denied(),"cause":if free==0{"NN_TARGET"}else{"VISIT_CAP"}})
    }
    pub fn k26_root_diversity(&mut self,seed:u64){
        let ca=self.root.candidates.get().unwrap();let prior:Vec<_>=ca.iter().map(|e|e.1).collect();
        self.root_noise=Some(compute_dirichlet_noise(prior.len(),&prior,&DirichletConfig{alpha:0.3,epsilon:0.25},Some(seed)));
    }
    pub fn four_axis_k25_block(&self,delta:u32,threads:usize,h:&Hooks){
        assert!(threads==1||threads==2);assert!(delta>=threads as u32&&delta%threads as u32==0);
        let before=self.root.n_total.load(Ordering::Acquire);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        if threads==1{for _ in 0..delta{h.k25_enter();self.four_iterate(h);h.k25_leave();}}
        else{rayon::scope(|scope|{for _ in 0..threads{scope.spawn(move |_|{
            for _ in 0..delta/threads as u32{h.k25_enter();self.four_iterate(h);h.k25_leave();}
        });}});}
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+delta);
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
    /// Quiescent serial root, immutable complete candidate order, no root masks.
    /// Unmaterialized actions have zero visits. None means no competitor.
    pub fn vote_min_distance(&self)->Option<u32>{
        let mut n=vec![0u32;self.root.candidate_count()];
        for (i,e) in self.root.read_edges().iter().enumerate(){n[i]=e.n.load(Ordering::Acquire);}
        assert_eq!(n.iter().sum::<u32>(),self.root.n_total.load(Ordering::Acquire));
        let (_,d)=math::vote_reach(&n);d.into_iter().flatten().min()
    }
    pub fn four_axis_lite(&self)->LiteRoot{
        let ca=self.root.candidates.get().expect("initialized candidates");
        let len=ca.len();assert!((1..=7).contains(&len));
        let mut r=LiteRoot{len,moves:[0;7],n:[0;7],w:[0.;7],m2:[0.;7],prior:[0.;7],
            visits:self.root.n_total.load(Ordering::Acquire),nn:self.evaluator_call_count(),
            best:self.best_move().map(|m|self.root_state.move_to_idx(m)).expect("nonempty root")};
        for (i,(m,p)) in ca.iter().enumerate(){r.moves[i]=self.root_state.move_to_idx(*m);r.prior[i]=*p as f64;}
        for (i,e) in self.root.read_edges().iter().enumerate(){
            assert_eq!(r.moves[i],self.root_state.move_to_idx(e.mv));
            r.n[i]=e.n.load(Ordering::Acquire);r.w[i]=atomic_f64_load(&e.w);r.m2[i]=atomic_f64_load(&e.m2);
        }
        assert_eq!(r.n.iter().sum::<u32>(),r.visits);r
    }
    pub fn four_axis_snapshot(&self)->Value{
        let e:Vec<_>=self.root.read_edges().iter().map(|e|json!({"move":self.root_state.move_to_idx(e.mv),"n":e.n.load(Ordering::Acquire),"w":atomic_f64_load(&e.w),"m2":atomic_f64_load(&e.m2),"prior":e.p,"child_hash":format!("{:016x}",e.child.hash),"vl":e.virtual_losses.load(Ordering::Acquire),"vvalue":atomic_f64_load(&e.virtual_value)})).collect();
        let c:Vec<_>=self.root.candidates.get().map(|c|c.iter().map(|(mv,p)|json!([self.root_state.move_to_idx(*mv),p])).collect()).unwrap_or_default();
        json!({"root_hash":format!("{:016x}",self.root_state.tt_hash()),"root_n":self.root.n_total.load(Ordering::Acquire),"root_w":atomic_f64_load(&self.root.w_total),"candidates":c,"edges":e,"best":self.best_move().map(|x|self.root_state.move_to_idx(x)),"evaluator_calls":self.evaluator_call_count(),"tt_size":self.tt.size()})
    }
}
