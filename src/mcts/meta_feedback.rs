// Research-only, engine-independent two-level metacognitive controller.
pub const KAPPA:f64=4.;
pub const BETA:f64=4.;
pub const ETA:f64=8.;
pub const EPS:f64=1e-8;
pub fn update_memory(m:f64,y:f64)->f64 {
    assert!(m.is_finite()&&y.is_finite()&&m.abs()<=1.000000001&&y.abs()<=1.000000001);
    (m+y)*0.5
}
fn normalize(v:&mut [f64]) {
    let z=v.iter().sum::<f64>();
    assert!(z.is_finite()&&z>0.);
    for x in v{*x/=z;}
}
fn kl(a:&[f64],b:&[f64])->f64 {
    assert_eq!(a.len(),b.len());
    a.iter().zip(b).map(|(&x,&y)|x*(x/y).ln()).sum::<f64>()
}
#[derive(Clone,Debug)]
pub struct Monitor {
    pub n:Vec<u32>,pub u:Vec<f64>,pub q:Vec<f64>,
    pub phi:f64,pub uncertainty:f64,pub js:f64,pub hidden:f64,
    pub shared:u64,pub kl_prior:f64,pub entropy:f64,
}
pub fn monitor(prior:&[f64],n:&[u32],w:&[f64],visible:usize,shared:u64,v0:f64)->Monitor{
    assert!(!prior.is_empty()&&prior.len()==n.len()&&n.len()==w.len());
    assert!(visible<=n.len()&&v0.is_finite()&&v0.abs()<=1.);
    assert!(prior.iter().all(|&v|v.is_finite()&&v>=0.));
    let mut pi:Vec<_>=prior.iter().map(|&x|x+EPS).collect();normalize(&mut pi);
    let u:Vec<_>=n.iter().zip(w).map(|(&n,&w)|{
        assert!(w.is_finite()&&w.abs()<=n as f64+1e-6);
        (w+KAPPA*v0)/(n as f64+KAPPA)
    }).collect();
    let mut q:Vec<_>=pi.iter().zip(&u).map(|(&p,&v)|p*(BETA*v).exp()).collect();
    let phi=q.iter().sum::<f64>().ln()/BETA;normalize(&mut q);
    let mut r:Vec<_>=n.iter().map(|&x|x as f64+EPS).collect();normalize(&mut r);
    let mid:Vec<_>=r.iter().zip(&q).map(|(&a,&b)|(a+b)*0.5).collect();
    let js=(kl(&q,&mid)+kl(&r,&mid))*0.5;
    let uncertainty=q.iter().zip(n).map(|(&p,&x)|p/(x as f64+KAPPA)).sum();
    let hidden=q[visible..].iter().sum();
    let kl_prior=kl(&q,&pi);let entropy=-q.iter().map(|&x|x*x.ln()).sum::<f64>();
    Monitor{n:n.to_vec(),u,q,phi,uncertainty,js,hidden,shared,kl_prior,entropy}
}
pub fn policy(x:&Monitor,m:&[f64;5])->([f64;5],[f64;5]){
    let g=x.shared as f64/(1.+x.shared as f64);
    let mut omega=[1.,0.25+2.*x.js/std::f64::consts::LN_2,0.25+2.*x.hidden,0.25+4.*x.uncertainty,0.25+2.*g];
    normalize(&mut omega);
    let mut z=std::array::from_fn(|i|omega[i]*(ETA*m[i]).exp());
    normalize(&mut z);(omega,z)
}
// Fixed deterministic uniform stream; separate from MCTS PRNG.
pub fn uniform(seed:u64,step:usize)->f64{
    let mut z=seed.wrapping_add((step as u64+1).wrapping_mul(0x9e3779b97f4a7c15));
    z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);
    z^=z>>31;
    (z>>11) as f64/(1u64<<53) as f64
}
pub fn select(z:&[f64;5],u:f64)->usize{
    assert!((0. ..1.).contains(&u));
    let mut cum=0.;
    for (i,&p) in z.iter().enumerate(){cum+=p;if u<cum{return i;}}
    4
}
#[derive(Debug)]
pub struct Response{
    pub work:f64,pub reorganization:f64,pub delta_phi:f64,
    pub depletion:f64,pub transport:f64,pub delta_a:f64,pub y:f64,
}
pub fn response(a:&Monitor,b:&Monitor,nn:u32)->Response{
    assert_eq!(a.n.len(),b.n.len());
    assert!(a.n.iter().zip(&b.n).all(|(&x,&y)|x<=y));
    let work=a.q.iter().zip(a.u.iter().zip(&b.u)).map(|(&q,(&u,&v))|q*(v-u)).sum::<f64>();
    let reorganization=kl(&a.q,&b.q)/BETA;let delta_phi=b.phi-a.phi;
    let depletion=a.q.iter().zip(a.n.iter().zip(&b.n)).map(|(&q,(&n,&m))|q*(1./(n as f64+KAPPA)-1./(m as f64+KAPPA))).sum::<f64>();
    let transport=a.q.iter().zip(b.q.iter().zip(&b.n)).map(|(&q,(&p,&n))|(p-q)/(n as f64+KAPPA)).sum::<f64>();
    let delta_a=b.uncertainty-a.uncertainty;
    assert!((delta_phi-work-reorganization).abs()<1e-10&&depletion>=-1e-12&&(delta_a+depletion-transport).abs()<1e-10);
    let y=-4.*delta_a/(1.+nn as f64/16.);
    Response{work,reorganization,delta_phi,depletion,transport,delta_a,y}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_level_variational_and_response() {
        let p=[0.3,0.7];let a=monitor(&p,&[2,0],&[1.,0.],1,0,0.);
        let b=monitor(&p,&[2,2],&[1.,-2.],2,1,0.);
        let t=response(&a,&b,2);
        assert!(t.depletion>=0.&&t.reorganization>=-1e-12&&t.y.abs()<=1.);
        assert!((t.delta_phi-t.work-t.reorganization).abs()<1e-12);
        assert!((t.delta_a+t.depletion-t.transport).abs()<1e-12);
        let (o,z)=policy(&a,&[0.;5]);assert_eq!(o,z);
        let (_,z1)=policy(&a,&[0.,0.25,0.,0.,0.]);
        assert!(((z1[1]/z1[0])/(z[1]/z[0])-2f64.exp()).abs()<1e-12);
        assert_eq!(select(&[0.2;5],0.21),1);
        assert_eq!(uniform(17,0),uniform(17,0));
    }
    #[test]
    fn signed_prediction_error_changes_memory() {
        assert_eq!(update_memory(0.,0.5),0.25);
        assert_eq!(update_memory(0.25,-0.75),-0.25);
        assert_eq!(update_memory(1.,-1.),0.);
    }
}
