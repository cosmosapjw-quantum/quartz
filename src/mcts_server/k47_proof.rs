use serde_json::{json,Value};
pub(super) fn probe(_board:&[u8],_n:usize,_d:i32,_cap:u64)->Result<Value,String>{Err("missing Rust proof".into())}
#[cfg(test)]
mod tests{
 use super::*;
 #[test] fn proof_gate(){let mut b=vec![0;49];b[0]=1;b[1]=1;b[2]=1;b[42]=2;b[43]=2;b[48]=2;let r=probe(&b,7,7,4096).expect("native receipt");assert_eq!(r["gate"],3);assert_eq!(r["queries"],0);}
 #[test] fn cap_zero(){let mut b=vec![0;49];b[0]=1;b[7]=1;b[48]=2;b[47]=2;let r=probe(&b,7,7,0).expect("native receipt");assert_eq!(r["queries"],0);assert_eq!(r["cert"],Value::Null);}
}
