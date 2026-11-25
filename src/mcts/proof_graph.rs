// Exact integer WDL proof graph, independent of statistical MCTS.
use std::collections::{BTreeMap,BTreeSet};
pub type Key=u128;
pub type I=(i8,i8);
pub const UNKNOWN:I=(-1,1);
#[derive(Clone,Debug,Default)]
pub struct Node {pub interval:Option<I>,pub actions:Option<Vec<(usize,Key)>>,pub parents:BTreeSet<Key>,pub terminal:Option<i8>}
#[derive(Default)]
pub struct Graph{pub nodes:BTreeMap<Key,Node>,pub changes:u64,pub notices:u64,pub shared_notices:u64}
impl Graph{
    pub fn interval(&self,k:Key)->I{self.nodes.get(&k).and_then(|n|n.interval).unwrap_or(UNKNOWN)}
    pub fn terminal(&mut self,k:Key,v:i8){
        assert!((-1..=1).contains(&v));let n=self.nodes.entry(k).or_default();
        assert!(n.actions.is_none());
        if let Some(old)=n.terminal{assert_eq!(old,v);return;}
        n.terminal=Some(v);self.propagate(k);
    }
    pub fn register(&mut self,k:Key,children:Vec<(usize,Key)>){
        assert!(!children.is_empty());assert_eq!(children.iter().map(|x|x.0).collect::<BTreeSet<_>>().len(),children.len());
        let n=self.nodes.entry(k).or_default();assert!(n.terminal.is_none());
        if let Some(ref old)=n.actions{assert_eq!(old,&children);return;}
        n.actions=Some(children.clone());
        for &(_,c) in &children{assert_ne!(c,k);self.nodes.entry(c).or_default().parents.insert(k);}
        self.propagate(k);
    }
    fn propagate(&mut self,k:Key){
        let mut todo=std::collections::VecDeque::from([k]);
        while let Some(k)=todo.pop_front(){
            let old=self.interval(k);
            let n=&self.nodes[&k];
            let new=if let Some(v)=n.terminal{(v,v)}else if let Some(ref a)=n.actions{
                (a.iter().map(|x|-self.interval(x.1).1).max().unwrap(),
                 a.iter().map(|x|-self.interval(x.1).0).max().unwrap())
            }else{UNKNOWN};
            assert!(old.0<=new.0&&new.0<=new.1&&new.1<=old.1);
            if new==old{continue;}
            let n=self.nodes.get_mut(&k).unwrap();n.interval=Some(new);self.changes+=1;
            let m=n.parents.len() as u64;self.notices+=m;if m>1{self.shared_notices+=m;}
            todo.extend(n.parents.iter().copied());
        }
    }
    pub fn actions(&self,k:Key)->Vec<(usize,I)>{self.nodes.get(&k).and_then(|n|n.actions.as_ref()).map(|a|a.iter().map(|&(m,c)|{let(l,u)=self.interval(c);(m,(-u,-l))}).collect()).unwrap_or_default()}
    pub fn witness(&self,k:Key)->Option<usize>{
        let a=self.actions(k);a.iter().find(|&&(m,(l,_))|a.iter().all(|&(j,(_,u))|j==m||l>=u)).map(|x|x.0)
    }
    pub fn allowed(&self,k:Key)->Vec<usize>{let a=self.actions(k);let l=a.iter().map(|x|x.1.0).max().unwrap_or(-1);a.iter().filter(|x|x.1.1>=l).map(|x|x.0).collect()}
    pub fn registered(&self,k:Key)->bool{self.nodes.get(&k).is_some_and(|n|n.actions.is_some()||n.terminal.is_some())}

}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn unknown_and_dominance(){
    let mut g=Graph::default();g.register(1,vec![(0,2),(1,3)]);
    g.terminal(2,1);
    assert_eq!(g.interval(1),UNKNOWN);
    assert_eq!(g.allowed(1),vec![0,1]); // both can be optimal if unknown equals loss
    g.terminal(3,0);
    assert_eq!(g.interval(1),(0,0));assert_eq!(g.witness(1),Some(1));assert_eq!(g.allowed(1),vec![1]);
 }
 #[test]fn graph_propagates_all_parents(){
    let mut g=Graph::default();
    g.register(10,vec![(0,20),(1,21)]);
    g.register(11,vec![(2,20),(3,22)]);
    g.terminal(20,-1);
    assert_eq!(g.interval(10),(1,1));assert_eq!(g.interval(11),(1,1));
    assert_eq!(g.witness(10),Some(0));assert_eq!(g.witness(11),Some(2));assert!(g.shared_notices>0);
    g.register(12,vec![(4,23)]);
    assert_eq!(g.interval(12),UNKNOWN);assert_eq!(g.witness(12),Some(4));
 }
}
