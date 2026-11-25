// Research-only finite occupation response.
pub const DIM:usize=21;
pub const PRICE:f64=0.002;
fn normalize(p:&[f64])->Vec<f64>{
    assert!(!p.is_empty()&&p.iter().all(|x|x.is_finite()&&*x>=0.));
    let s=p.iter().map(|v|v+1e-8).sum::<f64>();
    p.iter().map(|v|(v+1e-8)/s).collect()
}
fn kl(a:&[f64],b:&[f64])->f64{
    a.iter().zip(b).filter(|(x,_)|**x>0.).map(|(&x,&y)|x*(x/y).ln()).sum()
}
fn entropy(p:&[f64])->f64{-p.iter().filter(|&&x|x>0.).map(|&x|x*x.ln()).sum::<f64>()}
fn tv(a:&[f64],b:&[f64])->f64{0.5*a.iter().zip(b).map(|(&x,&y)|(x-y).abs()).sum::<f64>()}
fn occupation(n:&[u32])->Vec<f64>{
    let z=n.iter().map(|&x|x as f64).sum::<f64>();assert!(z>0.);
    n.iter().map(|&x|x as f64/z).collect()
}
/// gamma, TV(old,block), TV(old,new), weighted Jensen deficit, delta KL, delta H.
/// Counts are completed, before and after positive, same full legal action order.
pub fn flux(before:&[u32],after:&[u32],prior:&[f64])->[f64;6]{
    assert_eq!(before.len(),after.len());assert_eq!(before.len(),prior.len());
    let old=occupation(before);let new=occupation(after);
    let increments:Vec<_>=before.iter().zip(after).map(|(&a,&b)|b.checked_sub(a).expect("nonmonotone visits")).collect();
    let j=occupation(&increments);let pi=normalize(prior);
    let gamma=increments.iter().map(|&x|x as f64).sum::<f64>()/after.iter().map(|&x|x as f64).sum::<f64>();
    let mixing=(1.-gamma)*kl(&old,&new)+gamma*kl(&j,&new);
    let delta=kl(&new,&pi)-kl(&old,&pi);
    assert!((delta-gamma*(kl(&j,&pi)-kl(&old,&pi))+mixing).abs()<1e-10);
    let activity=tv(&old,&j);let move_tv=tv(&old,&new);
    assert!((move_tv-gamma*activity).abs()<1e-10);
    [gamma,activity,move_tv,mixing,delta,entropy(&new)-entropy(&old)]
}
pub fn features(prior:&[f64],n:&[u32],visible:usize,shared:u64,v0:f64,first:Option<usize>,gibbs:&[f64;8],f:&[f64;6],kind:usize)->[f64;DIM]{
    assert!(kind<3&&visible<=n.len()&&n.len()==prior.len());
    let pi=normalize(prior);let r=occupation(n);let total=n.iter().sum::<u32>() as f64;
    let l=n.iter().enumerate().max_by_key(|&(i,x)|(*x,i)).unwrap().0;
    let d=n.iter().enumerate().filter(|(i,_)|*i!=l).map(|(i,&v)|n[l]-v+u32::from(i<l)).min().unwrap_or(n[l]);
    let logm=(n.len() as f64).ln().max(1.);
    let scale=(-pi.iter().copied().fold(1.,f64::min).ln()).max(1.);
    let mut out=[0.;DIM];
    out[..8].copy_from_slice(&[1.,v0,r[l],d as f64/total,pi[visible..].iter().sum(),visible as f64/n.len() as f64,shared as f64/(1.+shared as f64),entropy(&pi)/logm]);
    if let Some(c)=first{assert!(c<5);out[8+c]=1.;}
    if kind==1 {out[13..].copy_from_slice(&[gibbs[0],gibbs[1],gibbs[2],gibbs[3],gibbs[4]/scale,gibbs[5]/logm,gibbs[6],gibbs[7]]);}
    if kind==2 {out[13..].copy_from_slice(&[kl(&r,&pi)/scale,entropy(&r)/logm,tv(&r,&pi),f[1],f[3]/std::f64::consts::LN_2,f[4]/scale,f[5]/logm,f[2]]);}
    assert!(out.iter().all(|x|x.is_finite()));out
}
pub fn choose(x:&[f64],gain:&[Vec<f64>],cost:&[Vec<f64>])->(usize,[f64;5],[f64;5],[f64;5]){
    assert!(x.len()==DIM&&gain.len()==5&&cost.len()==5);
    let g=std::array::from_fn(|i|x.iter().zip(&gain[i]).map(|(&a,&b)|a*b).sum::<f64>());
    let c=std::array::from_fn(|i|x.iter().zip(&cost[i]).map(|(&a,&b)|a*b).sum::<f64>());
    let s=std::array::from_fn(|i|g[i]-PRICE*c[i]);
    let mut best=0;
    for i in 1..5{if s[i]>s[best]+1e-12{best=i;}}
    (best,g,c,s)
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn score_and_relabel(){
    let f=flux(&[1,2,0],&[2,2,3],&[0.3,0.5,0.2]);
    let g=flux(&[0,1,2],&[3,2,2],&[0.2,0.3,0.5]);
    for (x,y) in f.iter().zip(g){assert!((x-y).abs()<1e-12);}
    let x=features(&[0.3,0.5,0.2],&[2,2,3],3,1,0.,Some(2),&[0.;8],&f,2);
    assert_eq!(x[10],1.);
    let mut gain=vec![vec![0.;DIM];5];let mut cost=gain.clone();
    assert_eq!(choose(&x,&gain,&cost).0,0);
    gain[2][0]=0.1;cost[2][0]=100.;
    assert_eq!(choose(&x,&gain,&cost).0,0);
    cost[2][0]=10.;assert_eq!(choose(&x,&gain,&cost).0,2);
 }
 #[test]fn actual_block_reallocation(){
  let r=flux(&[8,0],&[8,8],&[0.5,0.5]);
  assert_eq!(r[0],0.5);
  assert_eq!(r[1],1.0);
  assert_eq!(r[2],0.5);
  assert!((r[3]-2f64.ln()).abs()<1e-12);
  assert!((r[4]+2f64.ln()).abs()<1e-12);
 }
}
