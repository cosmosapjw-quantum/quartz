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

// Local decision-sensitive proof frontier: all hypothetical values are read-only.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Frontier {
    pub key:Key, pub path:Vec<(Key,usize,Key)>,
    pub gains:Vec<(i8,u8,u32)>,pub score:(u8,u32),
}
pub fn risk_width(a:&[(usize,I)]) -> (u8,u32) {
    let d=a.iter().map(|&(m,(l,_))|(a.iter().filter(|x|x.0!=m).map(|x|x.1.1).max().unwrap_or(-1)-l).max(0) as u8).min().unwrap_or(0);
    (d,a.iter().map(|(_, (l,u))|(*u-*l) as u32).sum())
}
impl Graph {
    pub fn hypothetical(&self,root:Key,s:Key,v:i8)->(u8,u32){
        assert!(!self.registered(s));let iv=self.interval(s);assert!(iv.0<=v&&v<=iv.1);
        let mut ov=BTreeMap::new();ov.insert(s,(v,v));
        let mut todo=std::collections::VecDeque::new();todo.extend(self.nodes[&s].parents.iter().copied());
        while let Some(k)=todo.pop_front(){
            let old=ov.get(&k).copied().unwrap_or_else(||self.interval(k));
            let a=self.nodes[&k].actions.as_ref().unwrap();
            let new=(a.iter().map(|x|-ov.get(&x.1).copied().unwrap_or_else(||self.interval(x.1)).1).max().unwrap(),
                     a.iter().map(|x|-ov.get(&x.1).copied().unwrap_or_else(||self.interval(x.1)).0).max().unwrap());
            assert!(old.0<=new.0&&new.0<=new.1&&new.1<=old.1);
            if new!=old{ov.insert(k,new);todo.extend(self.nodes[&k].parents.iter().copied());}
        }
        let aa:Vec<_>=self.nodes[&root].actions.as_ref().unwrap().iter().map(|&(m,c)|{let(l,u)=ov.get(&c).copied().unwrap_or_else(||self.interval(c));(m,(-u,-l))}).collect();
        risk_width(&aa)
    }
    pub fn frontier(&self,root:Key,cap:usize)->(Option<Frontier>,usize){
        if self.witness(root).is_some()||cap==0||!self.registered(root){return(None,0);}
        let (d,w)=risk_width(&self.actions(root));let mut best:Option<Frontier>=None;let mut n=0;
        let mut seen=BTreeSet::from([root]);let mut todo=std::collections::VecDeque::from([(root,Vec::new())]);
        while let Some((k,path))=todo.pop_front(){
            let (l,u)=self.interval(k);if l==u{continue;}
            if !self.registered(k){
                n+=1;let gains:Vec<_>=(l..=u).map(|v|{let (dd,ww)=self.hypothetical(root,k,v);assert!(dd<=d&&ww<=w);(v,d-dd,w-ww)}).collect();
                let score=gains.iter().map(|&(_,x,y)|(x,y)).max().unwrap();
                if score>(0,0)&&best.as_ref().is_none_or(|f|score>f.score||(score==f.score&&path.len()<f.path.len())){
                    best=Some(Frontier{key:k,path,gains,score});
                }
                if n>=cap{break;}continue;
            }
            let allowed=self.allowed(k);
            if let Some(ref aa)=self.nodes[&k].actions{
                for &(m,c) in aa{
                    let iv=self.interval(c);
                    if allowed.contains(&m)&&iv.0<iv.1&&seen.insert(c){
                        let mut p=path.clone();p.push((k,m,c));todo.push_back((c,p));
                    }
                }
            }
        }
        (best,n)
    }
}
#[cfg(test)]
mod frontier_tests{
    use super::*;
    #[test]fn overlay_shared_child_and_action_width(){
        let mut g=Graph::default();
        g.register(1,vec![(0,2),(1,3)]);
        g.register(2,vec![(0,4)]);
        g.register(3,vec![(0,4)]);
        assert_eq!(risk_width(&g.actions(1)),(2,4));
        assert_eq!(g.hypothetical(1,4,0),(0,0));
        assert_eq!(g.interval(4),UNKNOWN);
        assert_eq!(risk_width(&g.actions(1)),(2,4));
        let (f,n)=g.frontier(1,64);
        assert_eq!(n,1);
        let f=f.unwrap();assert_eq!(f.key,4);assert_eq!(f.path,vec![(1,0,2),(2,0,4)]);assert_eq!(f.score,(2,4));
    }
    #[test]fn unchanged_root_value_and_certified_abstention(){
        let mut g=Graph::default();
        g.register(1,vec![(0,2),(1,3),(2,4)]);
        assert_eq!(g.hypothetical(1,2,1),(2,4));
        assert_eq!(g.frontier(1,1).1,1);
        g.terminal(2,-1);
        assert!(g.frontier(1,64).0.is_none());
    }
}
