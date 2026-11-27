// Opt-in serial 7x7 reference-PUCT substrate with external frozen NN evaluation.
// Not the generic parallel MctsEngine or training path.
use super::*;
use super::k47_proof::{self as pf,FULL,won};
use serde::Deserialize;
use serde_json::{json,Value};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:Vec<u16>,cap:u32,quartz:bool,#[serde(default)]trace:bool,#[serde(default)]atlas:bool,#[serde(default)]mask:Option<u8>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalReply{cmd:String,id:u32,a:u64,b:u64,p:i8,prior:Vec<f64>,value:f64}
struct External{calls:u32,trace:Vec<Value>,traced:bool}
impl External{
    fn eval(&mut self,a:u64,b:u64,p:i8,legal:&[u16])->Result<(Vec<f64>,f64),String>{
        let id=self.calls+1;
        emit_stdout_json_value(&json!({"type":"k47_eval","id":id,"a":a,"b":b,"p":p,"legal":legal}));
        let Some(ServerCommandInput::Json(s))=read_stdin_command() else{return Err("missing evaluator response".into())};
        let r:EvalReply=serde_json::from_str(&s).map_err(|e|e.to_string())?;
        if r.cmd!="k47_eval_reply"||r.id!=id||r.a!=a||r.b!=b||r.p!=p||r.prior.len()!=legal.len()||r.prior.iter().any(|x|!x.is_finite()||*x<0.0)||(r.prior.iter().sum::<f64>()-1.0).abs()>1e-9||!r.value.is_finite()||!(-1.0..=1.0).contains(&r.value){return Err("invalid evaluator response".into())}
        self.calls+=1;
        if self.traced{self.trace.push(json!({"a":a,"b":b,"p":p,"legal":legal,"prior":r.prior,"value":r.value}));}
        Ok((r.prior,r.value))
    }
}
struct Edge{m:u16,prior:f64,n:u32,w:f64,child:Option<Box<Node>>}
struct Node{a:u64,b:u64,p:i8,edges:Option<Vec<Edge>>}
impl Node{
    fn new(a:u64,b:u64,p:i8)->Self{Self{a,b,p,edges:None}}
    fn expand(&mut self,ev:&mut External)->Result<(f64,bool),String>{
        assert!(self.edges.is_none());
        if won(if self.p==1{self.b}else{self.a}){return Ok((-1.0,true))}
        if self.a|self.b==FULL{return Ok((0.0,true))}
        let legal:Vec<u16>=(0..49).filter(|&m|(self.a|self.b)>>m&1==0).collect();
        let (prior,v)=ev.eval(self.a,self.b,self.p,&legal)?;
        self.edges=Some(legal.into_iter().zip(prior).map(|(m,prior)|Edge{m,prior,n:0,w:0.0,child:None}).collect());
        Ok((v,false))
    }
    fn visit(&mut self,ev:&mut External)->Result<(f64,bool),String>{
        if self.edges.is_none(){return self.expand(ev)}
        let es=self.edges.as_mut().unwrap();
        let n=es.iter().map(|e|e.n).sum::<u32>();let f=(n.max(1) as f64).sqrt();
        let mut idx=0;let mut best=f64::NEG_INFINITY;
        for (i,e) in es.iter().enumerate(){
            let q=if e.n>0{e.w/e.n as f64}else{0.0};
            let s=q+(2.0*e.prior)*f/(1+e.n) as f64;
            if s>best{best=s;idx=i}
        }
        let e=&mut es[idx];let m=1u64<<e.m;
        let child=e.child.get_or_insert_with(||Box::new(if self.p==1{Node::new(self.a|m,self.b,-1)}else{Node::new(self.a,self.b|m,1)}));
        let (v,t)=child.visit(ev)?;let v=-v;e.n+=1;e.w+=v;Ok((v,t))
    }
}
fn leader(es:&[Edge])->usize{(0..es.len()).max_by_key(|&i|(es[i].n,std::cmp::Reverse(es[i].m))).unwrap()}
fn distance(es:&[Edge])->u32{let i=leader(es);es.iter().enumerate().filter(|(j,_)|*j!=i).map(|(_,e)|es[i].n-e.n+u32::from(e.m>es[i].m)).min().unwrap_or(1_000_000_000)}
pub(super) fn parse_history(h:&[u16],n:usize)->Result<Vec<u8>,String>{
    if !matches!(n,7|15)||h.len()>=n*n{return Err("invalid history length".into())}
    let mut b=vec![0;n*n];let k=if n==7{4}else{5};
    for (i,&mm) in h.iter().enumerate(){
        let m=mm as usize;if m>=n*n||b[m]!=0{return Err("illegal move".into())}let p=1+(i%2) as u8;b[m]=p;
        let x=(m%n) as i32;let y=(m/n) as i32;
        for (dx,dy) in [(1,0),(0,1),(1,1),(1,-1)]{
            let mut count=1;for sg in [-1,1]{for j in 1..k{
                let xx=x+sg*j*dx;let yy=y+sg*j*dy;
                if xx<0||xx>=n as i32||yy<0||yy>=n as i32||b[(yy*n as i32+xx) as usize]!=p{break}count+=1;
            }}
            if count>=k{return Err("terminal history".into())}
        }
    }
    Ok(b)
}
pub(super) fn oriented(b:&[u8],p:u8)->Vec<u8>{b.iter().map(|&x|if x==0{0}else if x==p{1}else{2}).collect()}
fn inner(line:&str)->Result<Value,String>{
    let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    let k48=q.cmd=="research_gomoku7_k48";
    if !(q.cmd=="research_gomoku7_k47"||k48)||(!q.quartz&&q.cap!=32)||(q.quartz&&q.cap!=20&&q.cap!=32)||(!k48&&(q.atlas||q.mask.is_some()))||(k48&&(!q.quartz||q.cap!=20||q.atlas==q.mask.is_some())){return Err("invalid K47/K48 NN request".into())}
    let started=Instant::now();let b=parse_history(&q.history,7)?;
    let p=1-2*(q.history.len()%2) as i8;let ob=oriented(&b,if p==1{1}else{2});
    let mut a=0;let mut bb=0;for (m,&x) in b.iter().enumerate(){if x==1{a|=1u64<<m}else if x==2{bb|=1u64<<m}}
    let t=Instant::now();
    let mut part=if q.quartz{pf::combined(&ob,7)?}else{Value::Null};
    let base=if q.quartz{part["cold"]["receipt"].clone()}else{pf::probe(&ob,7,7,0)?};
    let proof_ns=t.elapsed().as_nanos() as u64;
    let g=base["gate"].as_u64().map(|x|x as u16);
    let mut out=json!({"schema":"quartz_gomoku7_k47_v1","history":q.history,"cap":q.cap,"lock":q.quartz,"partial":Value::Null,"replacement":false,"original_move":Value::Null,"nn":0,"visits":0,"terminal_leaves":0,"votes":[],"distance":Value::Null,"prior_best":Value::Null,"proof_ns":proof_ns});
    if let Some(m)=g{
        let bd=base["bounds"].as_array().unwrap().iter().find(|r|r[0]==m).unwrap();
        let stone=if p==1{a}else{bb};let kind=if won(stone|1u64<<m){"WIN1"}else if bd[1]==1{"WIN3"}else if bd[1]==0{"DRAW"}else if bd[2]==-1{"LOSE2"}else{"UNKNOWN"};
        out["move"]=json!(m);out["kind"]=json!("CERTIFICATE");out["proof"]=json!(kind);out["original_move"]=json!(m);
    }else if q.quartz&&!part["cert"].is_null(){
        out["move"]=part["cert"].clone();out["kind"]=json!("PARTIAL47");out["proof"]=json!(if part["exact"]["complete"]==true{"EXACT42"}else{"INTERVAL"});
        out["partial"]=part;
    }else{
        let mut ev=External{calls:0,trace:vec![],traced:q.trace};let mut root=Node::new(a,bb,p);
        let (_,term)=root.expand(&mut ev)?;assert!(!term);
        let mut n=0;let mut terminal=0;
        let snap=|root:&Node,ev:&External,n:u32,terminal:u32,cap:u32|->Value{
            let es=root.edges.as_ref().unwrap();
            assert_eq!(es.iter().map(|e|e.n).sum::<u32>(),n);assert_eq!(ev.calls,1+n-terminal);
            let m=es[leader(es)].m;let votes:Vec<_>=es.iter().map(|e|(e.m,e.n)).collect();
            let mut pi=0;for i in 1..es.len(){if es[i].prior>es[pi].prior{pi=i}}
            let mv=if q.quartz{pf::replace(m,&votes,&part["bounds"],7)}else{m};
            json!({"move":mv,"original_move":m,"replacement":mv!=m,"kind":if n<cap{"VOTE_LOCK"}else{"CAP"},"proof":"UNRESOLVED","nn":ev.calls,"visits":n,"terminal_leaves":terminal,"votes":votes,"distance":distance(es),"prior_best":es[pi].m,"cap":cap})
        };
        let mut horizon=q.cap;
        loop{
            while n<horizon{
                if q.quartz&&distance(root.edges.as_ref().unwrap())>horizon-n{break}
                let (_,t)=root.visit(&mut ev)?;n+=1;terminal+=u32::from(t);
            }
            if !k48||horizon==32{break}
            let low=snap(&root,&ev,n,terminal,20);
            let z=usize::from(low["distance"].as_u64().unwrap()<=3)+2*usize::from(low["original_move"]!=low["prior_best"])+4*usize::from(part["bounds"].as_array().unwrap().iter().any(|r|r[2]==-1));
            let extend=k48_extend(q.atlas,q.mask,z);
            out["low"]=low;out["bin"]=json!(z);out["extended"]=json!(extend);out["low_wall_ns"]=json!(started.elapsed().as_nanos() as u64);
            if !extend{break}horizon=32;
        }
        let final_snap=snap(&root,&ev,n,terminal,horizon);
        let chosen=if q.atlas{out["high"]=final_snap;out["low"].clone()}else{final_snap};
        for (k,v) in chosen.as_object().unwrap(){out[k]=v.clone()}
        if k48{out["physical_nn"]=json!(ev.calls);out["mask"]=json!(q.mask);out["atlas"]=json!(q.atlas)}
        out["partial"]=if q.quartz{part.take()}else{Value::Null};if q.trace{out["eval_trace"]=json!(ev.trace)}
    }
    if q.trace&&out.get("eval_trace").is_none(){out["eval_trace"]=json!([])}
    out["whole_ns"]=json!(started.elapsed().as_nanos() as u64);Ok(out)
}
pub(super) fn handle(line:&str)->Value{inner(line).unwrap_or_else(|e|json!({"error":e}))}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeReq{cmd:String,history:Vec<u16>,n:usize,cap:u64,depth:i32,#[serde(default)]exact:bool}
pub(super) fn inspect(line:&str)->Value{
    let f=||->Result<Value,String>{
        let r:ProbeReq=serde_json::from_str(line).map_err(|e|e.to_string())?;if r.cmd!="research_k47_proof"||r.cap>1000000{return Err("invalid inspect".into())}
        let b=oriented(&parse_history(&r.history,r.n)?,1+(r.history.len()%2) as u8);
        if r.exact{
            if r.n!=7{return Err("exact only 7x7".into())}
            let mut a=0;let mut bb=0;for (i,&x) in b.iter().enumerate(){if x==1{a|=1u64<<i}else if x==2{bb|=1u64<<i}}pf::exact(a,bb,r.cap)
        }else{pf::probe(&b,r.n,r.depth,r.cap)}
    };f().unwrap_or_else(|e|json!({"error":e}))
}

fn k48_extend(atlas:bool,mask:Option<u8>,z:usize)->bool{atlas||mask.is_some_and(|m|(m>>z)&1==1)}
#[cfg(test)]
mod k48_tests {
    use super::*;
    #[test]fn k48_extension_contract(){
        assert!(k48_extend(true,None,3));
        assert!(k48_extend(false,Some(8),3));
        assert!(!k48_extend(false,Some(8),7));
        assert!(!k48_extend(false,None,0));
    }
}
