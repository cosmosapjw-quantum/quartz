// Research-only four-axis numerical primitives. No game or engine dependency.
use std::collections::HashMap;
pub fn local_c(c:f64,counts:&[u32],priors:&[f32])->f64 {
    assert_eq!(counts.len(),priors.len());
    if counts.is_empty(){return c;}
    let eps=1e-4;
    let n=counts.iter().map(|&x|x as f64+eps).sum::<f64>();
    let p=priors.iter().map(|&x|(x as f64).max(0.)+eps).sum::<f64>();
    let d=counts.iter().zip(priors).map(|(&x,&y)|{
        let q=(x as f64+eps)/n;let a=((y as f64).max(0.)+eps)/p;
        q*(q/a).ln()
    }).sum::<f64>().max(0.);
    c*(1.+0.5*d/(1.+d))
}
pub fn mass_width(priors:&[f32],base:usize,old:usize)->usize {
    let total=priors.iter().map(|&p|p.max(0.) as f64).sum::<f64>();
    let mut acc=0.;let mut k=0;
    if total<=0.{k=(3*priors.len()+3)/4;}
    else {for &p in priors{acc+=p.max(0.) as f64;k+=1;if acc>=0.75*total{break;}}}
    base.max(old).max(k).min(priors.len())
}
pub fn blend(y:f32,m:f64,n:u64,terminal:bool)->f32 {
    if terminal||n==0{return y;}
    let a=0.25f64.min(n as f64/(n as f64+4.));
    ((1.-a)*y as f64+a*m).clamp(-1.,1.) as f32
}
#[derive(Default)]
pub struct Ledger { entries:HashMap<u64,HashMap<(u64,usize),(f64,u64)>> }
impl Ledger{
    pub fn observe(&mut self,child:u64,parent:u64,edge:usize,raw:f32)->Option<(f64,u64,bool)>{
        let e=self.entries.entry(child).or_default();
        let key=(parent,edge);
        // Stable accumulation order across HashMap seeds.
        let mut other:Vec<_>=e.iter().filter(|(k,_)|**k!=key).collect();
        other.sort_unstable_by_key(|(k,_)|**k);
        let (s,n)=other.iter().fold((0.,0u64),|(s,n),(_,v)|(s+v.0,n+v.1));
        let old=if n>0{Some((s/n as f64,n,true))}
            else{e.get(&key).map(|&(s,n)|(s/n as f64,n,false))};
        let a=e.entry(key).or_default();a.0+=raw as f64;a.1+=1;
        old
    }
    pub fn counts(&self)->(usize,usize,usize,u64){
        (self.entries.len(),self.entries.values().map(|v|v.len()).sum(),
         self.entries.values().filter(|v|v.len()>1).count(),
         self.entries.values().flat_map(|v|v.values()).map(|v|v.1).sum())
    }
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn surprise_and_width(){
    assert!((local_c(2.,&[1,1],&[0.5,0.5])-2.).abs()<1e-9);
    assert!(local_c(2.,&[9,1],&[0.1,0.9])>2.);
    assert_eq!(mass_width(&[0.4,0.3,0.2,0.1],1,0),3);
    assert_eq!(mass_width(&[0.4,0.3,0.2,0.1],1,4),4);
 }
 #[test]fn actual_incoming_provenance(){
    let mut l=Ledger::default();
    assert_eq!(l.observe(100,1,0,0.5),None);
    assert_eq!(l.observe(100,1,0,0.5),Some((0.5,1,false)));
    assert_eq!(l.observe(100,2,0,-0.5),Some((0.5,2,true)));
    assert_eq!(l.observe(100,1,0,0.0),Some((-0.5,1,true)));
    assert_eq!(l.observe(101,1,0,0.0),None);
    assert_eq!(blend(1.,-1.,4,false),0.5);
    assert_eq!(blend(1.,-1.,4,true),1.);
 }
}

// C04 red scaffold: certificate must not assert loss from partial coverage.
pub fn certificate(terminal_actions:&[Option<f32>])->Option<f32>{
    if terminal_actions.is_empty(){return None;}
    if terminal_actions.iter().any(|x|*x==Some(1.)){return Some(1.);}
    if terminal_actions.iter().all(Option::is_some){
        return terminal_actions.iter().filter_map(|&x|x).reduce(f32::max);
    }
    None
}
impl Ledger{
    pub fn observe_split(&mut self,child:u64,parent:u64,edge:usize,raw:f32)->(Option<(f64,u64)>,Option<(f64,u64)>){
        let e=self.entries.entry(child).or_default();let key=(parent,edge);
        let own=e.get(&key).map(|&(s,n)|(s/n as f64,n));
        let mut other:Vec<_>=e.iter().filter(|(k,_)|**k!=key).collect();
        other.sort_unstable_by_key(|(k,_)|**k);
        let (s,n)=other.iter().fold((0.,0u64),|(s,n),(_,v)|(s+v.0,n+v.1));
        let cross=if n>0{Some((s/n as f64,n))}else{None};
        let a=e.entry(key).or_default();a.0+=raw as f64;a.1+=1;
        (own,cross)
    }
}
#[cfg(test)]
mod mechanism_tests {
    use super::*;
    #[test]fn certificate_bounds(){
        assert_eq!(certificate(&[None,Some(1.),Some(-1.)]),Some(1.));
        assert_eq!(certificate(&[None,Some(-1.)]),None);
        assert_eq!(certificate(&[Some(0.),Some(-1.)]),Some(0.));
        assert_eq!(certificate(&[Some(-1.),Some(-1.)]),Some(-1.));
        assert_eq!(certificate(&[]),None);
    }
    #[test]fn separate_contexts(){
        let mut l=Ledger::default();
        assert_eq!(l.observe_split(9,1,0,0.5),(None,None));
        assert_eq!(l.observe_split(9,1,0,0.5),(Some((0.5,1)),None));
        assert_eq!(l.observe_split(9,2,0,-0.5),(None,Some((0.5,2))));
        assert_eq!(l.observe_split(9,1,0,0.),(Some((0.5,2)),Some((-0.5,1))));
        assert_eq!(l.counts(),(1,2,1,4));
    }
}
