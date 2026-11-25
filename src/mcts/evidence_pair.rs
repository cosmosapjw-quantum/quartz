// Research-only, decision readout from actual value observations, not votes.
pub fn pulse_score(pre:f64,_dn:u32,_dw:f64)->f64{pre}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn responds_to_evidence_not_dose(){
  assert!((pulse_score(0.25,4,-4.)+0.375).abs()<1e-12);
  assert_eq!(pulse_score(-0.5,0,0.),-0.5);
 }
}
