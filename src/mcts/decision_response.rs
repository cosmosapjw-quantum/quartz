// Research-only readout of observed forced counts (not a causal estimator).
pub fn subtract(n:&[u32],_f:&[u32])->Vec<u32>{n.to_vec()}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn forced_count_can_reverse_recommendation(){
   assert_eq!(subtract(&[12,20],&[0,16]),vec![12,4]);
 }
}
