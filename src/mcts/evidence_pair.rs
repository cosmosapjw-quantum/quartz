// Research-only, decision readout from actual value observations, not votes.
pub fn pulse_score(pre:f64,dn:u32,dw:f64)->f64{
    assert!(pre.is_finite()&&pre.abs()<=1.000001&&dw.is_finite()&&dw.abs()<=dn as f64+1e-6);
    (dw+4.*pre)/(dn as f64+4.)
}
pub fn score(n:u32,w:f64)->f64{
    assert!(w.is_finite()&&w.abs()<=n as f64+1e-6);
    w/(n as f64+4.)
}
pub fn choose(a:usize,b:usize,x:f64,y:f64)->usize{
    assert!(x.is_finite()&&y.is_finite());if y>x{b}else{a}
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn ties_and_prior(){
    assert_eq!(choose(4,2,0.1,0.1),4);
    assert_eq!(choose(4,2,-0.1,0.),2);
    assert_eq!(score(0,0.),0.);
    assert_eq!(score(4,4.),0.5);
 }
 #[test]fn responds_to_evidence_not_dose(){
  assert!((pulse_score(0.25,4,-4.)+0.375).abs()<1e-12);
  assert_eq!(pulse_score(-0.5,0,0.),-0.5);
 }
}
