// Research-only four-axis numerical primitives. No game or engine dependency.
use std::collections::HashMap;
pub fn local_c(c:f64,_counts:&[u32],_priors:&[f32])->f64 {c}
pub fn mass_width(_priors:&[f32],base:usize,old:usize)->usize {base.max(old)}
pub fn blend(y:f32,_m:f64,_n:u64,_terminal:bool)->f32 {y}
#[derive(Default)]
pub struct Ledger { entries:HashMap<u64,HashMap<(u64,usize),(f64,u64)>> }
impl Ledger{
    pub fn observe(&mut self,_child:u64,_parent:u64,_edge:usize,_raw:f32)->Option<(f64,u64,bool)>{None}
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
