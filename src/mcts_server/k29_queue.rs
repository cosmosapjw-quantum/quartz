//! Actual bounded live worker->CPU batch service. No persistent eval cache.
use super::*;
use std::sync::mpsc::{self,Sender,Receiver,SyncSender};
use std::thread::JoinHandle;
use std::time::Duration;
use std::collections::{BTreeMap,BTreeSet};
type E=EvalResult<usize>;
struct Request{state:Connect4,at:Instant,reply:SyncSender<E>}
fn key(s:&Connect4)->u128{
    assert!(s.active_root_gate_mask().is_none()&&!s.is_terminal());
    let mut k=if s.current_player()==1{1u128<<84}else{0};
    for (i,v) in s.board_flat().iter().enumerate(){k|=((*v+1) as u128)<<(2*i);}
    k
}
pub(super) struct QueueEval{
    scalar:Arc<ConvConnect4Evaluator>,
    tx:Mutex<Option<Sender<Request>>>,
    pending:AtomicU64,peak:AtomicU64,
}
impl QueueEval{
    pub(super) fn new(scalar:Arc<ConvConnect4Evaluator>)->Self{Self{scalar,tx:Mutex::new(None),pending:AtomicU64::new(0),peak:AtomicU64::new(0)}}
    pub(super) fn start(&self,width:usize,dedup:bool,trace:bool)->JoinHandle<serde_json::Value>{
        assert!(matches!(width,4|8));assert_eq!(self.pending.load(Ordering::Acquire),0);
        let(tx,rx)=mpsc::channel();assert!(self.tx.lock().unwrap().replace(tx).is_none());
        let model=self.scalar.clone();
        std::thread::spawn(move||service(model,rx,width,dedup,trace))
    }
    pub(super) fn finish(&self,j:JoinHandle<serde_json::Value>)->serde_json::Value{
        assert_eq!(self.pending.load(Ordering::Acquire),0);
        drop(self.tx.lock().unwrap().take().unwrap());
        let mut r=j.join().expect("CPU batch service");
        r["peak_pending"]=serde_json::json!(self.peak.load(Ordering::Acquire));r
    }
}
impl Evaluator<Connect4> for QueueEval{
    fn evaluate(&self,s:&Connect4)->E{
        let tx=self.tx.lock().unwrap().clone();
        if let Some(tx)=tx{
            let n=self.pending.fetch_add(1,Ordering::AcqRel)+1;self.peak.fetch_max(n,Ordering::Relaxed);
            let(reply,rx)=mpsc::sync_channel(1);
            tx.send(Request{state:s.clone(),at:Instant::now(),reply}).expect("queue sender");
            let r=rx.recv().expect("one reply per admitted request");
            assert!(self.pending.fetch_sub(1,Ordering::AcqRel)>0);
            r
        }else{self.scalar.evaluate(s)}
    }
}
fn ns(t:Instant)->u64{t.elapsed().as_nanos().min(u64::MAX as u128) as u64}
fn service(model:Arc<ConvConnect4Evaluator>,rx:Receiver<Request>,width:usize,dedup:bool,trace:bool)->serde_json::Value{
    let(mut logical,mut physical,mut batches,mut wait_ns,mut compute_ns,mut latency_ns,mut parity)=(0u64,0u64,0u64,0u64,0u64,0u64,0u64);
    let mut lh=[0u64;9];let mut ph=[0u64;9];let mut seen=BTreeSet::new();let mut records=Vec::new();
    while let Ok(first)=rx.recv(){
        let deadline=Instant::now()+Duration::from_micros(20);let mut batch=vec![first];
        while batch.len()<width {
            let now=Instant::now();if now>=deadline{break;}
            match rx.recv_timeout(deadline-now){Ok(x)=>batch.push(x),Err(_)=>break}
        }
        let keys:Vec<_>=batch.iter().map(|r|key(&r.state)).collect();
        let mut unique=Vec::new();let mut map=BTreeMap::new();let mut route=Vec::new();
        for (r,&k) in batch.iter().zip(&keys){
            seen.insert(k);
            let idx=if dedup{
                if let Some(&i)=map.get(&k){i}else{let i=unique.len();unique.push(r.state.clone());map.insert(k,i);i}
            }else{let i=unique.len();unique.push(r.state.clone());i};
            route.push(idx);
        }
        let wait:Vec<_>=batch.iter().map(|r|ns(r.at)).collect();
        wait_ns+=wait.iter().sum::<u64>();let start=Instant::now();
        let values=model.evaluate_batch29(&unique);let elapsed=ns(start);compute_ns+=elapsed;
        assert_eq!(values.len(),unique.len());
        // Diagnostic only; excluded from experimental physical count.
        if trace{
            for (u,v) in unique.iter().zip(&values){let a=model.evaluate(u);assert_eq!(a.value.to_bits(),v.value.to_bits());assert_eq!(a.policy,v.policy);parity+=1;}
        }
        logical+=batch.len() as u64;physical+=unique.len() as u64;batches+=1;
        lh[batch.len()]+=1;ph[unique.len()]+=1;
        if trace{
            records.push(serde_json::json!({"keys":keys.iter().map(|k|format!("{k:022x}")).collect::<Vec<_>>(),"route":route,"physical":unique.len(),"wait_ns":wait,"compute_ns":elapsed}));
        }
        for (r,i) in batch.into_iter().zip(route){
            latency_ns+=ns(r.at);r.reply.send(values[i].clone()).expect("one live receiver");
        }
    }
    serde_json::json!({"logical":logical,"physical":physical,"coalesced":logical-physical,"distinct":seen.len(),"repeat_keys":logical-seen.len() as u64,
        "batches":batches,"logical_hist":lh,"physical_hist":ph,"wait_ns":wait_ns,"compute_ns":compute_ns,"latency_ns":latency_ns,
        "width":width,"dedup":dedup,"parity_checks":parity,"trace":if trace{serde_json::json!(records)}else{serde_json::Value::Null}})
}
