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
    pub fn terminal(&mut self,k:Key,v:i8){self.nodes.entry(k).or_default().terminal=Some(v);}
    pub fn register(&mut self,k:Key,children:Vec<(usize,Key)>){self.nodes.entry(k).or_default().actions=Some(children);}
    pub fn actions(&self,k:Key)->Vec<(usize,I)>{self.nodes.get(&k).and_then(|n|n.actions.as_ref()).map(|a|a.iter().map(|&(m,c)|{let(l,u)=self.interval(c);(m,(-u,-l))}).collect()).unwrap_or_default()}
    pub fn witness(&self,_k:Key)->Option<usize>{None}
    pub fn allowed(&self,k:Key)->Vec<usize>{self.actions(k).iter().map(|a|a.0).collect()}
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
