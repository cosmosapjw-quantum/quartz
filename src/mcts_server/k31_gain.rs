//! Direct conditional native-gain and resource regression. No learned readout.
use super::*;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arm{mean:Vec<f64>,scale:Vec<f64>,coef:Vec<Vec<f64>>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Model{arms:Vec<Arm>}
pub(super) fn load(path:&str,id:&str)->Result<Arc<Model>,String>{
    if id.len()!=64||!id.bytes().all(|x|x.is_ascii_hexdigit()){return Err("bad gain identity".into());}
    let p=std::fs::canonicalize(path).map_err(|e|e.to_string())?;
    let key=format!("{}|{}",p.display(),id);
    static CACHE:OnceLock<Mutex<HashMap<String,Arc<Model>>>>=OnceLock::new();
    let cache=CACHE.get_or_init(||Mutex::new(HashMap::new()));
    if let Some(m)=cache.lock().unwrap().get(&key){return Ok(m.clone());}
    let m:Model=serde_json::from_str(&std::fs::read_to_string(p).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if m.arms.len()!=9{return Err("bad arm count".into());}
    for a in &m.arms{
        if a.mean.len()!=16||a.scale.len()!=16||a.coef.len()!=3||a.coef.iter().any(|x|x.len()!=17)||
           a.mean.iter().chain(a.scale.iter()).chain(a.coef.iter().flatten()).any(|x|!x.is_finite())||
           a.scale.iter().any(|x|*x<=0.){return Err("bad direct model".into());}
    }
    let m=Arc::new(m);cache.lock().unwrap().insert(key,m.clone());Ok(m)
}
pub(super) fn features(f:&[f64],pre:&LiteRoot,mid:&LiteRoot)->Vec<f64>{
    let l=(f.len()-1)/8;assert_eq!(l,mid.len);
    let v:Vec<_>=f[1..].chunks_exact(8).collect();
    let maxr=|i:usize|v[1..].iter().map(|x|x[i]).fold(f64::NEG_INFINITY,f64::max);
    let entropy=|i:usize| -v.iter().map(|x|{let p=x[i]*x[i];if p>0.{p*p.ln()}else{0.}}).sum::<f64>()/(l as f64).ln();
    vec![f[0],l as f64/7.,v[0][0]*v[0][0],maxr(0).powi(2),v[0][1].powi(2),maxr(1).powi(2),
        v[0][2],maxr(2),v[0][3],maxr(3),entropy(0),entropy(1),mid.nn as f64/41.,mid.visits as f64/136.,
        v.iter().filter(|x|x[7]==0.).count() as f64/l as f64,if pre.best==mid.best{1.}else{0.}]
}
pub(super) struct Forecast{pub fixed:usize,pub priced:usize,pub report:Value}
impl Model{
    pub(super) fn predict(&self,f:&[f64])->Forecast{
        assert_eq!(f.len(),16);let mut rows=Vec::new();let mut nets=Vec::new();
        for (i,a) in self.arms.iter().enumerate(){
            let v:Vec<_>=f.iter().zip(&a.mean).zip(&a.scale).map(|((&x,&m),&s)|(x-m)/s).collect();
            let y:Vec<_>=a.coef.iter().map(|c|c[0]+c[1..].iter().zip(&v).map(|(w,x)|w*x).sum::<f64>()).collect();
            let g=y[0].clamp(-1.,1.);let nn=if i<6{y[1].clamp(0.,32.)}else{0.};let rules=if i>=6{y[2].max(0.)}else{0.};
            let net=g-0.002*nn-0.00002*rules;nets.push(net);
            rows.push(json!({"g":g,"nn":nn,"rules":rules,"net":net}));
        }
        let best=|n:usize|{let mut a=0;for i in 1..n{if nets[i]>nets[a]+1e-12{a=i;}}a};
        let fixed=best(6);let p=best(9);let priced=if nets[p]>1e-12{p}else{9};
        Forecast{fixed,priced,report:json!({"arms":rows,"fixed_arm":fixed,"priced_arm":priced})}
    }
}
