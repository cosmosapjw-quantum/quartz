// Research-only finite occupation response.
pub fn flux(_before:&[u32],_after:&[u32],_prior:&[f64])->[f64;6]{[0.;6]}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn actual_block_reallocation(){
  let r=flux(&[8,0],&[8,8],&[0.5,0.5]);
  assert_eq!(r[0],0.5);
  assert_eq!(r[1],1.0);
  assert_eq!(r[2],0.5);
  assert!((r[3]-2f64.ln()).abs()<1e-12);
  assert!((r[4]+2f64.ln()).abs()<1e-12);
 }
}
