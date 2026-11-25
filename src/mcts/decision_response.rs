// Research-only readout of observed forced counts (not a causal estimator).
pub fn subtract(n:&[u32],f:&[u32])->Vec<u32>{
 assert_eq!(n.len(),f.len());
 n.iter().zip(f).map(|(&a,&b)|a.checked_sub(b).expect("forced count exceeds actual")).collect()
}
pub fn leader(n:&[u32])->usize{
 assert!(n.iter().any(|&x|x>0));
 n.iter().enumerate().max_by_key(|&(i,v)|(*v,i)).unwrap().0
}
pub fn candidates(n:&[u32],visible:usize)->(usize,Option<usize>,Option<usize>){
 assert!(visible<=n.len());let l=leader(n);
 let rival=n.iter().enumerate().filter(|&(i,&v)|i!=l&&v>0).max_by_key(|&(i,v)|(*v,i)).map(|(i,_)|i).or_else(||(0..n.len()).find(|&i|i!=l));
 (l,rival,(visible<n.len()).then_some(visible))
}
pub fn injection(n:&[u32],f:&[u32])->(usize,usize,f64,f64){
 let sub=subtract(n,f);let raw=leader(n);let corrected=leader(&sub);
 let n0=n.iter().map(|&v|v as f64).sum::<f64>();let m=sub.iter().map(|&v|v as f64).sum::<f64>();
 let w=1.-m/n0;
 let tv=n.iter().zip(&sub).map(|(&a,&b)|(a as f64/n0-b as f64/m).abs()).sum::<f64>()/2.;
 assert!(tv<=w+1e-12);
 (raw,corrected,w,tv)
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn injection_and_candidate_semantics(){
   let (a,b,w,tv)=injection(&[12,20],&[0,16]);
   assert_eq!((a,b),(1,0));assert!((w-0.5).abs()<1e-12);assert!((tv-0.375).abs()<1e-12);
   assert_eq!(candidates(&[8,0,0],1),(0,Some(1),Some(1)));
   assert_eq!(candidates(&[4,4,0],2),(1,Some(0),Some(2)));
   assert_eq!(injection(&[3,3],&[0,0]),(1,1,0.,0.));
 }
 #[test]#[should_panic]fn reject_overcount(){subtract(&[2],&[3]);}
 #[test]fn forced_count_can_reverse_recommendation(){
   assert_eq!(subtract(&[12,20],&[0,16]),vec![12,4]);
 }
}
