//! Actual bounded rule-only expansion on a graph of observed MCTS paths.
//! No hypothetical solution is inserted. No NN or MCTS visit is fabricated.
use super::*;
use crate::mcts::proof_graph::{Graph,Key,risk_width};
use std::collections::{BTreeSet,VecDeque};
fn key(s:&Connect4)->Key{
    assert!(s.active_root_gate_mask().is_none());
    let mut k=if s.current_player()==1{1u128<<84}else{0};
    for (i,v) in s.board_flat().iter().enumerate(){k|=((*v+1) as u128)<<(2*i);}
    k
}
fn hex(k:Key)->String{format!("{k:022x}")}
#[derive(Default)]
struct Work{checks:u64,replay:u64,registered:u64,scored:u64}
fn register(g:&mut Graph,s:&Connect4,w:&mut Work){
    let k=key(s);if g.registered(k){return;}
    w.registered+=1;
    if s.is_terminal(){g.terminal(k,s.outcome() as i8);return;}
    let mut a=Vec::new();
    for m in s.legal_moves(){
        let c=s.apply_move(m);let ck=key(&c);w.checks+=1;
        if c.is_terminal(){if !g.registered(ck){w.registered+=1;}g.terminal(ck,c.outcome() as i8);}
        a.push((m,ck));
    }
    g.register(k,a);
}
fn fifo(g:&Graph,k:Key)->Option<Vec<usize>>{
    if g.witness(k).is_some(){return None;}
    let mut seen=BTreeSet::from([k]);let mut todo=VecDeque::from([(k,Vec::new())]);
    while let Some((k,p))=todo.pop_front(){
        let(l,u)=g.interval(k);if l==u{continue;}
        if !g.registered(k){return Some(p);}
        let allowed=g.allowed(k);
        if let Some(aa)=&g.nodes[&k].actions{
            for &(m,c) in aa{
                if allowed.contains(&m)&&seen.insert(c){let mut q=p.clone();q.push(m);todo.push_back((c,q));}
            }
        }
    }
    None
}
fn report(g:&Graph,root:Key,mid:&LiteRoot,anchor:usize,w:&Work,dump:bool)->Value{
    let a=g.actions(root);let allowed=g.allowed(root);let witness=g.witness(root);
    let best=witness.unwrap_or_else(||if allowed.contains(&anchor){anchor}else{
        allowed.iter().copied().max_by_key(|m|(mid.n[mid.moves[..mid.len].iter().position(|x|x==m).unwrap()],std::cmp::Reverse(*m))).unwrap()
    });
    let mut o=json!({"key":hex(root),"actions":a,"interval":g.interval(root),"allowed":allowed,"witness":witness,
        "best":best,"risk":risk_width(&a),"nodes":g.nodes.len(),"registered":w.registered,
        "checks":w.checks,"replay":w.replay,"rule_work":w.checks+w.replay,"scored":w.scored,
        "changes":g.changes,"notices":g.notices,"shared_notices":g.shared_notices});
    if dump{o["graph"]=json!(g.nodes.iter().map(|(&k,n)|json!({"key":hex(k),"interval":g.interval(k),"terminal":n.terminal,
       "actions":n.actions.as_ref().map(|a|a.iter().map(|&(m,c)|(m,hex(c))).collect::<Vec<_>>())})).collect::<Vec<_>>());}
    o
}
pub(super) fn execute(root:&Connect4,mid:&LiteRoot,anchor:usize,trace:&Value,arm:usize,dump:bool)->Value{
    assert!((6..=8).contains(&arm));let t=Instant::now();
    let mut g=Graph::default();let mut w=Work::default();let rk=key(root);
    register(&mut g,root,&mut w);
    for m in root.legal_moves(){w.replay+=1;register(&mut g,&root.apply_move(m),&mut w);}
    let paths:Vec<Vec<usize>>=trace.as_array().unwrap().iter().map(|z|z["path"].as_array().unwrap().iter().map(|m|m.as_u64().unwrap() as usize).collect()).collect();
    assert_eq!(paths.len(),mid.visits as usize);
    for p in &paths{
        let mut s=root.clone();
        for &m in p{assert!(!s.is_terminal()&&s.legal_moves().contains(&m));s=s.apply_move(m);w.replay+=1;register(&mut g,&s,&mut w);}
    }
    let build_us=us(t);let before=report(&g,rk,mid,anchor,&w,false);
    let mut steps=Vec::new();let t=Instant::now();
    if arm!=6{
        for _ in 0..8{
            if g.witness(rk).is_some(){break;}
            let t0=Instant::now();
            let (p,score,kind)=if arm==8{
                let(f,n)=g.frontier(rk,16);w.scored+=n as u64;
                if let Some(f)=f{(Some(f.path.iter().map(|x|x.1).collect::<Vec<_>>()),Some(f.score),"FRONTIER")}
                else{(fifo(&g,rk),None,"FIFO_FALLBACK")}
            }else{(fifo(&g,rk),None,"FIFO")};
            let plan_us=us(t0);let Some(path)=p else{break;};
            let mut s=root.clone();
            for &m in &path{assert!(!s.is_terminal()&&s.legal_moves().contains(&m));s=s.apply_move(m);w.replay+=1;}
            let k=key(&s);assert!(!g.registered(k)&&!s.is_terminal()&&g.nodes.contains_key(&k));
            register(&mut g,&s,&mut w);assert!(g.registered(k));
            let post=report(&g,rk,mid,anchor,&w,false);
            steps.push(json!({"path":path,"key":hex(k),"kind":kind,"hypothetical_score":score,"plan_us":plan_us,"post":post}));
        }
    }
    let expansion_us=us(t);let end=report(&g,rk,mid,anchor,&w,dump);
    json!({"arm":arm,"paths":paths,"before":before,"steps":steps,"end":end,"build_us":build_us,"expansion_us":expansion_us})
}
