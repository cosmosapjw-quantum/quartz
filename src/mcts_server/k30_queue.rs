//! Persistent per-model broker; one quiescent search epoch at a time.
use super::*;
use std::sync::mpsc::{self,Sender,Receiver,SyncSender};
use std::time::Duration;
use std::collections::{BTreeMap,BTreeSet};
type E=EvalResult<usize>;
struct Request{state:Connect4,at:Instant,reply:SyncSender<E>}
struct Mode{width:usize,dedup:bool,specialized:bool,trace:bool}
enum Msg{Begin(Mode,SyncSender<bool>),Eval(Request),End(SyncSender<serde_json::Value>)}
fn key(s:&Connect4)->u128{
    assert!(s.active_root_gate_mask().is_none()&&!s.is_terminal());
    let mut k=if s.current_player()==1{1u128<<84}else{0};
    for (i,v) in s.board_flat().iter().enumerate(){k|=((*v+1) as u128)<<(2*i);}
    k
}
struct Broker{tx:Sender<Msg>,epochs:AtomicU64}
fn broker(model:Arc<ConvConnect4Evaluator>,path:&str,id:&str)->Arc<Broker>{
    static CACHE:OnceLock<Mutex<HashMap<String,Arc<Broker>>>>=OnceLock::new();
    let cache=CACHE.get_or_init(||Mutex::new(HashMap::new()));let mut c=cache.lock().unwrap();
    let k=format!("{}|{}",path,id);
    if let Some(b)=c.get(&k){return b.clone();}
    let(tx,rx)=mpsc::channel();std::thread::spawn(move||{
        while let Ok(msg)=rx.recv(){
            match msg{Msg::Begin(m,ack)=>{ack.send(true).unwrap();service(&model,&rx,m);},_=>panic!("message outside broker epoch")}
        }
    });
    let b=Arc::new(Broker{tx,epochs:AtomicU64::new(0)});c.insert(k,b.clone());b
}
pub(super) struct QueueEval{
    scalar:Arc<ConvConnect4Evaluator>,path:String,id:String,tx:Mutex<Option<Arc<Broker>>>,pending:AtomicU64,peak:AtomicU64,
}
impl QueueEval{
    pub(super) fn new(scalar:Arc<ConvConnect4Evaluator>,path:&str,id:&str)->Self{
        Self{scalar,path:path.to_string(),id:id.to_string(),tx:Mutex::new(None),pending:AtomicU64::new(0),peak:AtomicU64::new(0)}
    }
    pub(super) fn start(&self,width:usize,dedup:bool,specialized:bool,trace:bool)->u64{
        assert_eq!(self.pending.load(Ordering::Acquire),0);assert_eq!(width,8);
        let b=broker(self.scalar.clone(),&self.path,&self.id);let n=b.epochs.fetch_add(1,Ordering::Relaxed)+1;
        let(ack,rx)=mpsc::sync_channel(1);b.tx.send(Msg::Begin(Mode{width,dedup,specialized,trace},ack)).unwrap();assert!(rx.recv().unwrap());
        assert!(self.tx.lock().unwrap().replace(b).is_none());n
    }
    pub(super) fn finish(&self,epoch:u64)->serde_json::Value{
        assert_eq!(self.pending.load(Ordering::Acquire),0);
        let b=self.tx.lock().unwrap().take().unwrap();let(tx,rx)=mpsc::sync_channel(1);b.tx.send(Msg::End(tx)).unwrap();
        let mut r=rx.recv().unwrap();r["peak_pending"]=serde_json::json!(self.peak.load(Ordering::Acquire));r["epoch"]=serde_json::json!(epoch);r
    }
}
impl Evaluator<Connect4> for QueueEval{
    fn evaluate(&self,s:&Connect4)->E{
        let b=self.tx.lock().unwrap().clone();
        if let Some(b)=b{
            let n=self.pending.fetch_add(1,Ordering::AcqRel)+1;self.peak.fetch_max(n,Ordering::Relaxed);
            let(reply,rx)=mpsc::sync_channel(1);
            b.tx.send(Msg::Eval(Request{state:s.clone(),at:Instant::now(),reply})).unwrap();
            let r=rx.recv().unwrap();assert!(self.pending.fetch_sub(1,Ordering::AcqRel)>0);r
        }else{self.scalar.evaluate(s)}
    }
}
fn ns(t:Instant)->u64{t.elapsed().as_nanos().min(u64::MAX as u128) as u64}
fn service(model:&ConvConnect4Evaluator,rx:&Receiver<Msg>,m:Mode){
    let Mode{width,dedup,specialized,trace}=m;
    let(mut logical,mut physical,mut batches,mut wait_ns,mut compute_ns,mut latency_ns,mut parity,mut lanes)=(0u64,0u64,0u64,0u64,0u64,0u64,0u64,0u64);
    let mut lh=[0u64;9];let mut ph=[0u64;9];let mut kh=[0u64;9];let mut seen=BTreeSet::new();let mut records=Vec::new();
    let finish=loop{
        let first=match rx.recv().expect("persistent broker disconnected"){Msg::Eval(r)=>r,Msg::End(tx)=>break tx,_=>panic!("nested epoch")};
        let deadline=Instant::now()+Duration::from_micros(20);let mut batch=vec![first];
        while batch.len()<width{
            let now=Instant::now();if now>=deadline{break;}
            match rx.recv_timeout(deadline-now){Ok(Msg::Eval(x))=>batch.push(x),Err(mpsc::RecvTimeoutError::Timeout)=>break,_=>panic!("nonquiescent end")}
        }
        let keys:Vec<_>=batch.iter().map(|r|key(&r.state)).collect();
        let mut unique=Vec::new();let mut map=BTreeMap::new();let mut route=Vec::new();
        for (r,&k) in batch.iter().zip(&keys){
            seen.insert(k);
            let idx=if dedup{if let Some(&i)=map.get(&k){i}else{let i=unique.len();unique.push(r.state.clone());map.insert(k,i);i}}
                else{let i=unique.len();unique.push(r.state.clone());i};route.push(idx);
        }
        let wait:Vec<_>=batch.iter().map(|r|ns(r.at)).collect();wait_ns+=wait.iter().sum::<u64>();
        let start=Instant::now();let(values,w)=if specialized{model.evaluate_batch30(&unique)}else{(model.evaluate_batch29(&unique),8)};
        let elapsed=ns(start);compute_ns+=elapsed;assert_eq!(values.len(),unique.len());assert!(w>=unique.len()&&w<=8);
        if trace{for (u,v) in unique.iter().zip(&values){let a=model.evaluate(u);assert_eq!(a.value.to_bits(),v.value.to_bits());assert_eq!(a.policy,v.policy);parity+=1;}}
        logical+=batch.len() as u64;physical+=unique.len() as u64;batches+=1;lanes+=w as u64;
        lh[batch.len()]+=1;ph[unique.len()]+=1;kh[w]+=1;
        if trace{records.push(serde_json::json!({"keys":keys.iter().map(|k|format!("{k:022x}")).collect::<Vec<_>>(),"route":route,"physical":unique.len(),"kernel_width":w,"wait_ns":wait,"compute_ns":elapsed}));}
        for (r,i) in batch.into_iter().zip(route){latency_ns+=ns(r.at);r.reply.send(values[i].clone()).expect("one live receiver");}
    };
    finish.send(serde_json::json!({"logical":logical,"physical":physical,"coalesced":logical-physical,"distinct":seen.len(),"repeat_keys":logical-seen.len() as u64,
        "batches":batches,"logical_hist":lh,"physical_hist":ph,"kernel_hist":kh,"kernel_lanes":lanes,"specialized":specialized,"wait_ns":wait_ns,"compute_ns":compute_ns,"latency_ns":latency_ns,
        "width":width,"dedup":dedup,"parity_checks":parity,"trace":if trace{serde_json::json!(records)}else{serde_json::Value::Null}})).unwrap();
}
