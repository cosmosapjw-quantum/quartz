//! Finite coherent joint empirical U,Z law with proof-compatible support.
use super::*;
use serde::Deserialize;
use serde_json::{json,Value};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Donor{f:Vec<f64>,u:Vec<u8>,z:[[usize;2];8],cost:[[u64;2];8]}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Model{k:usize,eps:f64,alpha:[f64;8],guard:[f64;8],donors:Vec<Donor>}
pub(super) fn load(path:&str,id:&str)->Result<Arc<Model>,String>{
    if id.len()!=64||!id.bytes().all(|x|x.is_ascii_hexdigit()){return Err("bad belief digest".into());}
    let p=std::fs::canonicalize(path).map_err(|e|e.to_string())?;
    let key=format!("{}|{}",p.display(),id);
    static CACHE:OnceLock<Mutex<HashMap<String,Arc<Model>>>>=OnceLock::new();
    let cache=CACHE.get_or_init(||Mutex::new(HashMap::new()));
    if let Some(m)=cache.lock().unwrap().get(&key){return Ok(m.clone());}
    let s=std::fs::read_to_string(p).map_err(|e|e.to_string())?;
    let m:Model=serde_json::from_str(&s).map_err(|e|e.to_string())?;
    if !matches!(m.k,0|32|96|256)||!m.eps.is_finite()||m.eps<=0.||m.eps>=1.||m.donors.is_empty()||m.donors.len()>10000||m.alpha.iter().any(|x|!x.is_finite()||*x<0.||*x>1.)||m.guard.iter().any(|x|!x.is_finite()||*x<0.||*x>2.){return Err("bad belief params".into());}
    for d in &m.donors{
        let n=d.u.len();
        if !(3..=7).contains(&n)||d.f.len()!=1+8*n||d.f.iter().any(|x|!x.is_finite()||x.abs()>10.)||
          d.u.iter().any(|&x|x>2)||d.z.iter().flatten().any(|&x|x>=2*n)||d.cost.iter().flatten().any(|&x|x>32){return Err("bad donor".into());}
    }
    let m=Arc::new(m);cache.lock().unwrap().insert(key,m.clone());Ok(m)
}
pub(super) fn argmax(x:&[f64])->usize{
    let mut a=0;for i in 1..x.len(){if x[i]>x[a]+1e-12{a=i;}}a
}
pub(super) struct Forecast{pub prior:usize,pub post:Vec<Vec<usize>>,pub arm:usize,pub priced:usize,pub report:Value}
impl Model{
    pub(super) fn predict(&self,f:&[f64],allowed:&[(u8,u8)])->Forecast{
        let l=allowed.len();let base:Vec<_>=allowed.iter().map(|&(a,b)|(a as f64+b as f64)/4.).collect();
        let mut ds:Vec<(usize,f64)>=self.donors.iter().enumerate().filter(|(_,d)|d.u.len()==l&&d.u.iter().zip(allowed).all(|(&u,&(lo,hi))|lo<=u&&u<=hi))
            .map(|(i,d)|(i,d.f.iter().zip(f).map(|(&a,&b)|(a-b)*(a-b)).sum::<f64>()/f.len() as f64)).collect();
        ds.sort_by(|a,b|a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        if self.k>0{ds.truncate(self.k);}
        let n=ds.len();let eps=if n==0{1.}else{self.eps};
        let raw:Vec<_>=ds.iter().map(|&(_,d)|if self.k==0{1.}else{1./(0.01+d)}).collect();
        let sum=raw.iter().sum::<f64>();let w:Vec<_>=raw.iter().map(|&v|v/sum*(1.-eps)).collect();
        let ess=if n==0{0.}else{(1.-eps).powi(2)/w.iter().map(|x|x*x).sum::<f64>()};
        let mut m:Vec<_>=base.iter().map(|v|v*eps).collect();
        for (&(i,_),&v) in ds.iter().zip(&w){for a in 0..l{m[a]+=v*self.donors[i].u[a] as f64/2.;}}
        let prior=0;
        let mut post=Vec::new();let mut reports=Vec::new();let mut gs=Vec::new();let mut nets=Vec::new();
        for c in 0..8{
            let zlen=2*l;let mut mass=vec![eps/zlen as f64;zlen];let mut j=vec![base.iter().map(|v|v*eps/zlen as f64).collect::<Vec<_>>();zlen];let mut cost=eps*32.;
            for (&(i,_),&v) in ds.iter().zip(&w){
                let d=&self.donors[i];for r in 0..2{
                    let z=d.z[c][r];let w=v/2.;mass[z]+=w;cost+=w*d.cost[c][r] as f64;
                    for a in 0..l{j[z][a]+=w*d.u[a] as f64/2.;}
                }
            }
            let alpha=self.alpha[c];let guard=self.guard[c];
            for z in 0..zlen{for a in 0..l{j[z][a]=alpha*j[z][a]+(1.-alpha)*mass[z]*m[a];}}
            let p:Vec<_>=j.iter().enumerate().map(|(z,row)|{
                let a=argmax(row);if n>0&&alpha>0.&&(row[a]-row[z/2])/mass[z]>guard+1e-12{a}else{z/2}
            }).collect();
            for a in 0..l{assert!((j.iter().map(|z|z[a]).sum::<f64>()-m[a]).abs()<1e-9);}
            let value=j.iter().map(|row|row[argmax(row)]).sum::<f64>();let g=(value-m[argmax(&m)]).max(0.);
            let d=(0..zlen).map(|z|j[z][z/2]).sum::<f64>()-m[0];
            let h=(0..zlen).map(|z|j[z][p[z]]).sum::<f64>()-m[0];let net=h-0.002*cost;
            gs.push(h);nets.push(net);post.push(p.clone());
            reports.push(json!({"mass":mass,"post_rank":p,"joint_moments":j,"G":g,"D":d,"H":h,"alpha":alpha,"guard":guard,"cost":cost,"net":net}));
        }
        let arm=if n==0{0}else{argmax(&gs)};
        let a=argmax(&nets);let priced=if n>0&&nets[a]>1e-12{a}else{8};
        Forecast{prior,post,arm,priced,report:json!({"prior":m,"prior_rank":prior,"donors":n,"ess":ess,"nearest":ds.first().map(|x|x.1),"arms":reports,"continue_arm":arm,"priced_arm":priced})}
    }
}
