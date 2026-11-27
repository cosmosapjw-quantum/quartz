// K44/K42 exact root proof semantics, scoped to unrestricted 7x7/4 and 15x15/5.
use serde_json::{json,Value};
use std::sync::OnceLock;
use std::collections::HashMap;
#[derive(Clone,Copy,Default)]
struct Bits([u64;4]);
impl Bits {
    fn get(self,m:usize)->bool{self.0[m/64]>>(m%64)&1!=0}
    fn set(&mut self,m:usize,v:bool){let x=1u64<<(m%64);if v{self.0[m/64]|=x}else{self.0[m/64]&=!x}}
    fn any(self)->bool{self.0.iter().any(|&x|x!=0)}
    fn count(self)->u32{self.0.iter().map(|x|x.count_ones()).sum()}
    fn first(self)->usize{for (i,&x) in self.0.iter().enumerate(){if x!=0{return 64*i+x.trailing_zeros() as usize}}unreachable!()}
    fn or(self,b:Self)->Self{Self(std::array::from_fn(|i|self.0[i]|b.0[i]))}
    fn and(self,b:Self)->Self{Self(std::array::from_fn(|i|self.0[i]&b.0[i]))}
}
struct Geometry{n:usize,k:usize,lines:Vec<Vec<usize>>,inc:Vec<Vec<usize>>}
fn geometry(n:usize)->&'static Geometry{
    static G7:OnceLock<Geometry>=OnceLock::new();static G15:OnceLock<Geometry>=OnceLock::new();
    (if n==7{&G7}else{assert_eq!(n,15);&G15}).get_or_init(||{
        let k=if n==7{4}else{5};let mut lines=vec![];let mut inc=vec![vec![];n*n];
        for r in 0..n as i32{for c in 0..n as i32{for (dr,dc) in [(0,1),(1,0),(1,1),(1,-1)]{
            let rr=r+(k as i32-1)*dr;let cc=c+(k as i32-1)*dc;
            if !(0..n as i32).contains(&rr)||!(0..n as i32).contains(&cc){continue}
            let l:Vec<_>=(0..k as i32).map(|i|((r+i*dr)*n as i32+c+i*dc) as usize).collect();
            for &m in &l{inc[m].push(lines.len())}lines.push(l);
        }}}
        Geometry{n,k,lines,inc}
    })
}
struct Side{imm:Bits,adj:Vec<Bits>,ic:Vec<u8>,support:Vec<u8>,pc:Vec<u8>,total:i32,pairs:i32,wins:i32}
impl Side{fn new(n:usize)->Self{Self{imm:Bits::default(),adj:vec![Bits::default();n],ic:vec![0;n],support:vec![0;n],pc:vec![0;n*n],total:0,pairs:0,wins:0}}}
struct State{g:&'static Geometry,b:Vec<u8>,counts:Vec<[u8;2]>,s:[Side;2],empty:Bits,scans:u64}
fn adj(x:&mut u8,d:i32){let v=*x as i32+d;assert!((0..=20).contains(&v));*x=v as u8}
impl State{
    fn new(n:usize,b:&[u8])->Self{
        let g=geometry(n);let mut z=Self{g,b:b.to_vec(),counts:vec![[0;2];g.lines.len()],s:[Side::new(n*n),Side::new(n*n)],empty:Bits::default(),scans:0};
        for (m,&v) in b.iter().enumerate(){if v==0{z.empty.set(m,true)}}
        for (i,l) in g.lines.iter().enumerate(){z.scans+=1;for &m in l{if b[m]!=0{z.counts[i][b[m] as usize-1]+=1}}z.contrib(i,1)}
        z
    }
    fn contrib(&mut self,i:usize,d:i32){
        let c=self.counts[i];if c[0]>0&&c[1]>0{return}
        let p=usize::from(c[0]==0);let k=self.g.k;let n=c[p] as usize;
        if n==k{self.s[p].wins+=d;assert!(self.s[p].wins>=0);return}
        if n<k-2{return}
        let mut e=self.g.lines[i].iter().copied().filter(|&m|self.b[m]==0);let x=e.next().unwrap();
        let z=&mut self.s[p];z.total+=d;adj(&mut z.support[x],d);
        if n==k-1{assert!(e.next().is_none());adj(&mut z.ic[x],d);z.imm.set(x,z.ic[x]>0)}
        else{let y=e.next().unwrap();assert!(e.next().is_none());z.pairs+=d;adj(&mut z.support[y],d);
            let n2=self.g.n*self.g.n;let xy=x*n2+y;let yx=y*n2+x;
            adj(&mut z.pc[xy],d);adj(&mut z.pc[yx],d);z.adj[x].set(y,z.pc[xy]>0);z.adj[y].set(x,z.pc[yx]>0)
        }assert!(z.total>=0&&z.pairs>=0);
    }
    fn alter(&mut self,m:usize,p:usize,make:bool){
        assert_eq!(self.b[m],if make{0}else{p as u8+1});
        let inc=&self.g.inc[m];for &i in inc{self.scans+=1;self.contrib(i,-1)}
        self.b[m]=if make{p as u8+1}else{0};self.empty.set(m,!make);
        for &i in inc{if make{self.counts[i][p]+=1}else{self.counts[i][p]-=1}self.contrib(i,1)}
    }
    fn threats(&self,p:usize,m:usize)->Bits{let mut z=self.s[p].imm.or(self.s[p].adj[m]);z.set(m,false);z}
}
struct Search{st:State,cap:u64,queries:u64,checks:u64,skipped:u64,exhausted:bool}
impl Search{
    fn tick(&mut self)->bool{if self.queries==self.cap{self.exhausted=true;false}else{self.queries+=1;true}}
    fn after(&mut self,p:usize,m:usize,d:i32)->bool{
        self.checks+=1;if d<1{return false}if self.st.s[p].imm.get(m){return true}if d<3{return false}
        let mut o=self.st.s[1-p].imm;o.set(m,false);if o.any(){return false}
        let q=self.st.threats(p,m);if q.count()>=2{return true}if !q.any(){return false}
        let x=q.first();self.st.alter(m,p,true);self.st.alter(x,1-p,true);
        let ok=self.attack(p,d-2);self.st.alter(x,1-p,false);self.st.alter(m,p,false);ok
    }
    fn attack(&mut self,p:usize,d:i32)->bool{
        if d<=0||!self.tick(){return false}
        if self.st.s[1-p].wins>0||!self.st.empty.any(){return false}
        assert_eq!(self.st.s[p].wins,0);
        if self.st.s[p].imm.any(){return true}if d<3||self.st.s[1-p].imm.count()>1{return false}
        let mut e=self.st.empty;if self.st.s[1-p].imm.any(){e=e.and(self.st.s[1-p].imm)}
        for m in 0..self.st.g.n*self.st.g.n{if e.get(m)&&self.st.s[p].adj[m].any(){
            if self.after(p,m,d){return true}if self.exhausted{break}
        }}false
    }
}
fn eligible(bd:&[[i32;3]],i:usize)->bool{bd.iter().enumerate().all(|(j,r)|j==i||bd[i][1]>=r[2])}
fn cert(bd:&[[i32;3]])->Option<usize>{(0..bd.len()).find(|&i|eligible(bd,i)).map(|i|bd[i][0] as usize)}
pub(in crate::mcts_server) fn probe(board:&[u8],n:usize,d:i32,cap:u64)->Result<Value,String>{
    if !matches!(n,7|15)||board.len()!=n*n||board.iter().any(|&x|x>2)||!(3..=31).contains(&d){return Err("invalid proof input".into())}
    let mut s=Search{st:State::new(n,board),cap,queries:0,checks:0,skipped:0,exhausted:false};
    if !s.st.empty.any()||s.st.s.iter().any(|x|x.wins>0){return Err("terminal proof root".into())}
    let bin=usize::from(s.st.s[0].pairs>0)+2*usize::from(s.st.s[1].pairs>0);
    let mut bd=vec![];for m in 0..n*n{if s.st.empty.get(m){
        let mut o=s.st.s[1].imm;o.set(m,false);
        let v=if s.st.s[0].imm.get(m){[1,1]}else if s.st.empty.count()==1{[0,0]}else if o.any(){[-1,-1]}else if s.st.threats(0,m).count()>=2{[1,1]}else{[-1,1]};
        bd.push([m as i32,v[0],v[1]])
    }}
    let gate=if n==7{
        (0..bd.len()).filter(|&i|eligible(&bd,i)).min_by_key(|&i|{let m=bd[i][0] as usize;(-bd[i][1],!s.st.s[0].imm.get(m),m.div_euclid(7).abs_diff(3)+m.rem_euclid(7).abs_diff(3),m)}).map(|i|bd[i][0] as usize)
    }else if s.st.s[0].imm.any(){Some(s.st.s[0].imm.first())}
    else if let Some(r)=bd.iter().find(|r|r[1]==1){Some(r[0] as usize)}
    else if s.st.s[1].imm.count()==1{Some(s.st.s[1].imm.first())}
    else if bd.len()==1{Some(bd[0][0] as usize)}else{None};
    let mut c=gate.or_else(||cert(&bd));
    if c.is_none(){
        for i in 0..bd.len(){let m=bd[i][0] as usize;if bd[i][1]<bd[i][2]&&s.st.threats(0,m).any(){
            if !s.tick(){break}if s.after(0,m,d){bd[i][1]=1;bd[i][2]=1;c=cert(&bd);break}if s.exhausted{break}
        }}
        if c.is_none()&&!s.exhausted{
            for i in 0..bd.len(){let m=bd[i][0] as usize;if bd[i][1]<bd[i][2]{
                assert!(s.st.s[1].total>=s.st.s[1].support[m] as i32);
                if s.st.s[1].total==s.st.s[1].support[m] as i32{s.skipped+=1;continue}
                if !s.tick(){break}s.st.alter(m,0,true);let loss=s.attack(1,d-1);s.st.alter(m,0,false);
                if loss{bd[i][1]=-1;bd[i][2]=-1}c=cert(&bd);if c.is_some()||s.exhausted{break}
            }}
        }
    }
    Ok(json!({"gate":gate,"cert":c,"exhausted":s.exhausted,"bounds":bd,"queries":s.queries,"scans":s.st.scans,"checks":s.checks,"depth":d,"budget":cap,"bin":bin,"skipped":s.skipped,"mode":"F"}))
}
pub(in crate::mcts_server) fn masks()->&'static Vec<u64>{
    static M:OnceLock<Vec<u64>>=OnceLock::new();M.get_or_init(||geometry(7).lines.iter().map(|l|l.iter().map(|&m|1u64<<m).sum()).collect())
}
pub(in crate::mcts_server) const FULL:u64=(1u64<<49)-1;
pub(in crate::mcts_server) fn won(a:u64)->bool{masks().iter().any(|&l|a&l==l)}
fn imm(a:u64,b:u64)->u64{let mut x=0;for &l in masks(){if b&l==0&&(a&l).count_ones()==3{x|=l&!a}}x}
fn dead(a:u64,b:u64)->bool{!masks().iter().any(|&l|l&a==0||l&b==0)}
struct Exact{cap:u64,nodes:u64,hits:u64,memo:HashMap<(u64,u64),[i32;2]>}
impl Exact{
    fn p(&mut self,a:u64,b:u64,t:i32)->Result<bool,()>{
        if let Some(x)=self.memo.get(&(a,b)){if x[0]>=t{self.hits+=1;return Ok(true)}if x[1]<t{self.hits+=1;return Ok(false)}}
        if self.nodes>=self.cap{return Err(())}self.nodes+=1;
        let mut e=FULL^(a|b);let mut exact=None;let mut ans=false;
        if won(b){exact=Some(-1)}else if e==0{exact=Some(0)}else if imm(a,b)!=0{exact=Some(1)}else{
            let q=imm(b,a);if q&(q.wrapping_sub(1))!=0{exact=Some(-1)}else if q!=0{ans=!self.p(b,a|q,1-t)?}else if dead(a,b){exact=Some(0)}else{
                while e!=0{let m=e& e.wrapping_neg();e^=m;if !self.p(b,a|m,1-t)?{ans=true;break}}
            }
        }
        let z=self.memo.entry((a,b)).or_insert([-1,1]);
        if let Some(v)=exact{assert!(z[0]<=v&&v<=z[1]);*z=[v,v];ans=v>=t}else if ans{z[0]=z[0].max(t)}else{z[1]=z[1].min(t-1)}
        assert!(z[0]<=z[1]);Ok(ans)
    }
}
pub(in crate::mcts_server) fn exact(a:u64,b:u64,cap:u64)->Result<Value,String>{
    if a&b!=0||(a|b)&!FULL!=0||won(a)||won(b)||(FULL^(a|b)).count_ones()>16||a|b==FULL{return Err("invalid exact root".into())}
    let mut s=Exact{cap,nodes:0,hits:0,memo:HashMap::new()};
    let mut bd:Vec<[i32;3]>=(0..49).filter(|&m|(a|b)>>m&1==0).map(|m|[m,-1,1]).collect();
    let mut best=None;let mut value=-1;
    let work=(||->Result<(),()>{
        for t in [1,0]{
            for x in &mut bd{
                let ok=if x[1]>=t{true}else if x[2]<t{false}else{
                    let ok=!s.p(b,a|1u64<<x[0],1-t)?;if ok{x[1]=x[1].max(t)}else{x[2]=x[2].min(t-1)}ok
                };
                if ok{best=Some(x[0]);value=t;break}
            }if best.is_some(){break}
        }
        if best.is_none(){best=Some(bd[0][0]);value=-1}Ok(())
    })();
    let mut z=json!({"complete":work.is_ok(),"nodes":s.nodes,"hits":s.hits,"states":s.memo.len()});
    if work.is_ok(){let m=best.unwrap();assert!(bd.iter().all(|r|r[2]<=value&&(r[0]>=m||r[2]<value)));
        assert_eq!(bd.iter().find(|r|r[0]==m).unwrap(),&[m,value,value]);z["best"]=json!(m);z["value"]=json!(value);z["bounds"]=json!(bd)}
    Ok(z)
}
pub(in crate::mcts_server) fn combined(board:&[u8],n:usize)->Result<Value,String>{
    let start=std::time::Instant::now();let r=probe(board,n,7,4096)?;let cold_ns=start.elapsed().as_nanos() as u64;
    let mut bd:Vec<[i32;3]>=serde_json::from_value(r["bounds"].clone()).map_err(|e|e.to_string())?;
    let mut c=r["cert"].clone();let mut ex=Value::Null;
    if n==7&&bd.len()<=16&&c.is_null(){
        let mut a=0;let mut b=0;for (m,&p) in board.iter().enumerate(){if p==1{a|=1u64<<m}else if p==2{b|=1u64<<m}}
        let t=std::time::Instant::now();ex=exact(a,b,25000)?;ex["wall_ns"]=json!(t.elapsed().as_nanos() as u64);
        if ex["complete"]==true{c=ex["best"].clone();let x:Vec<[i32;3]>=serde_json::from_value(ex["bounds"].clone()).map_err(|e|e.to_string())?;
            for (p,q) in bd.iter_mut().zip(x){assert_eq!(p[0],q[0]);p[1]=p[1].max(q[1]);p[2]=p[2].min(q[2]);assert!(p[1]<=p[2])}
        }
    }
    Ok(json!({"cold":{"receipt":r,"wall_ns":cold_ns},"exact":ex,"cert":c,"bounds":bd,"wall_ns":start.elapsed().as_nanos() as u64}))
}
pub(in crate::mcts_server) fn replace(m:u16,votes:&[(u16,u32)],bounds:&Value,n:usize)->u16{
    let bad=|m:u16|bounds.as_array().unwrap().iter().any(|r|r[0]==m&&r[2]==-1);
    if !bad(m){return m}
    votes.iter().enumerate().filter(|(_,x)|!bad(x.0)).max_by_key(|(i,x)|(x.1,if n==15{*i as i32}else{-(x.0 as i32)})).map(|(_,x)|x.0).unwrap_or(m)
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test] fn proof_gate(){let mut b=vec![0;49];b[0]=1;b[1]=1;b[2]=1;b[42]=2;b[43]=2;b[48]=2;let r=probe(&b,7,7,4096).expect("native receipt");assert_eq!(r["gate"],3);assert_eq!(r["queries"],0);}
 #[test] fn cap_zero(){let mut b=vec![0;49];b[0]=1;b[7]=1;b[48]=2;b[47]=2;let r=probe(&b,7,7,0).expect("native receipt");assert_eq!(r["queries"],0);assert_eq!(r["cert"],Value::Null);}
}
