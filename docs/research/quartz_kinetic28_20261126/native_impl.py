from pathlib import Path
p=Path('src/mcts/four_axis.rs');s=p.read_text()
def sub(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
sub('    budget26:AtomicU64,denied26:AtomicU64,refine27:AtomicU64,','    budget26:AtomicU64,denied26:AtomicU64,refine27:AtomicU64,floor28:AtomicU64,')
sub('refine27:AtomicU64::new(u64::MAX)}}}', 'refine27:AtomicU64::new(u64::MAX),floor28:AtomicU64::new(u64::MAX)}}}')
sub('struct K25{','''// K28 atomic free-pool state: low32 shared, high32 protected.
// MAX is inactive; experimental budgets <=64 cannot alias the sentinel.
fn pack28(s:u64,p:u64)->u64{assert!(s<=u32::MAX as u64&&p<=u32::MAX as u64);s|(p<<32)}
fn pools28(v:u64)->(u64,u64){(v&0xffffffff,v>>32)}
fn claim28(v:u64,n:u64,optional:bool)->Option<u64>{
    assert_ne!(v,u64::MAX);
    let(s,p)=pools28(v);let h=if optional{p.min(n)}else{0};
    s.checked_sub(n-h).map(|s1|pack28(s1,p-h))
}
#[cfg(test)]
mod floor28_tests{
    use super::*;
    #[test]fn protected_floor_and_atomic_cross_pool(){
        let x=pack28(3,1);
        assert_eq!(claim28(x,2,true),Some(pack28(2,0)));
        assert_eq!(claim28(x,4,true),Some(0));
        assert_eq!(claim28(x,5,true),None);
        assert_eq!(claim28(pack28(0,2),1,false),None);
        assert_eq!(claim28(pack28(0,2),2,true),Some(0));
        assert_eq!(claim28(0,0,true),Some(0));
        assert_eq!(claim28(pack28(3,1),1,false),Some(pack28(2,1)));
    }
    #[test]fn two_pool_cas_never_overclaims(){
        let f=AtomicU64::new(pack28(4,4));
        let ok=AtomicU64::new(0);
        std::thread::scope(|sc|{for i in 0..16{let f=&f;let ok=&ok;
            sc.spawn(move||{let n=if i%2==0{1}else{2};
            if f.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|claim28(v,n,i%2==1)).is_ok(){ok.fetch_add(n,Ordering::AcqRel);}
        });}});
        let(s,p)=pools28(f.load(Ordering::Acquire));assert_eq!(s+p+ok.load(Ordering::Acquire),8);
    }
}
struct K25{''')
sub('self.k25.as_ref().is_some_and(|k|k.budget26.load(Ordering::Acquire)!=u64::MAX)',
    'self.k25.as_ref().is_some_and(|k|k.budget26.load(Ordering::Acquire)!=u64::MAX||k.floor28.load(Ordering::Acquire)!=u64::MAX)')
sub('        k.budget26.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_sub(n)).is_ok()',
    '''        if k.floor28.load(Ordering::Acquire)!=u64::MAX{
            return k.floor28.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|claim28(v,n,false)).is_ok();
        }
        k.budget26.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|v.checked_sub(n)).is_ok()''')
sub('        if k.refine27.load(Ordering::Acquire)==u64::MAX{self.k26_claim(n)}',
    '''        if k.floor28.load(Ordering::Acquire)!=u64::MAX{
            return k.floor28.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|claim28(v,n,true)).is_ok();
        }
        if k.refine27.load(Ordering::Acquire)==u64::MAX{self.k26_claim(n)}''')
sub('        if self.k26_limited(){self.k25.as_ref().unwrap().budget26.fetch_add(1,Ordering::AcqRel);}',
    '''        if self.k26_limited(){
            let k=self.k25.as_ref().unwrap();
            if k.floor28.load(Ordering::Acquire)!=u64::MAX{
                k.floor28.fetch_update(Ordering::AcqRel,Ordering::Acquire,|v|{let(s,p)=pools28(v);Some(pack28(s+1,p))}).unwrap();
            }else{k.budget26.fetch_add(1,Ordering::AcqRel);}
        }''')
sub('    pub fn k26_root_diversity(&mut self,seed:u64){','''    // Floor rather than ceiling: V consumes its protected balance first,
    // then shared surplus in the same atomic operation. Base never borrows
    // protected balance before the quiescent release.
    pub fn four_axis_k28_floor(&self,target:u64,cap:u32,threads:usize,reserve:u64,h:&Hooks)->Value{
        assert!(target>0&&target<=64&&cap>0&&reserve>0&&reserve<target&&matches!(threads,1|2|4|8));
        assert_eq!(h.proof_mode,0);assert_eq!(h.vmode,1);
        let k=h.k25.as_ref().unwrap();assert_eq!(k.active.load(Ordering::Acquire),0);
        assert_eq!(k.budget26.load(Ordering::Acquire),u64::MAX);assert_eq!(k.refine27.load(Ordering::Acquire),u64::MAX);
        assert_eq!(k.floor28.swap(pack28(target-reserve,reserve),Ordering::AcqRel),u64::MAX);
        self.par_ctrl.set_n_threads(threads as u32);self.par_ctrl.reset_for_search();
        let before=self.root.n_total.load(Ordering::Acquire);let calls=self.evaluator_call_count();
        let base0=h.counts[7].load(Ordering::Acquire);let ext0=h.counts[8].load(Ordering::Acquire);
        let mut used=0u32;let mut rounds=0u32;let mut released=0;let mut pre_base=0;let mut pre_extra=0;let mut pre_visits=0;let mut borrowed=0;let mut protected=0;
        for phase in 0..2{
            while used<cap&&pools28(k.floor28.load(Ordering::Acquire)).0>0{
                let quota=cap-used;let tickets=AtomicU32::new(0);
                let worker=||loop{
                    if !h.k26_claim(1){break;}
                    if tickets.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|if n<quota{Some(n+1)}else{None}).is_err(){h.k26_refund();break;}
                    h.k25_enter();self.four_iterate(h);h.k25_leave();
                };
                if threads==1{worker();}else{rayon::scope(|scope|{for _ in 0..threads{let w=&worker;scope.spawn(move |_|w());}});}
                let n=tickets.load(Ordering::Acquire);assert!(n>0);used+=n;rounds+=1;
            }
            assert_eq!(k.active.load(Ordering::Acquire),0);
            if phase==0{
                let(s,p)=pools28(k.floor28.load(Ordering::Acquire));
                released=p;protected=reserve-p;
                pre_base=h.counts[7].load(Ordering::Acquire)-base0;pre_extra=h.counts[8].load(Ordering::Acquire)-ext0;
                pre_visits=used;borrowed=pre_extra-protected;
                assert_eq!(pre_base+borrowed+s,target-reserve);
                assert_eq!(self.evaluator_call_count()-calls,pre_base+pre_extra);
                k.floor28.store(pack28(s+p,0),Ordering::Release);
            }
        }
        let(free,p)=pools28(k.floor28.swap(u64::MAX,Ordering::AcqRel));assert_eq!(p,0);
        let nn=self.evaluator_call_count()-calls;assert_eq!(target-free,nn);
        assert_eq!(self.root.n_total.load(Ordering::Acquire),before+used);
        json!({"target":target,"cap":cap,"used_nn":nn,"unused":free,"visits":used,"rounds":rounds,
            "reserve":reserve,"released":released,"protected_base":pre_base,"protected_extra":protected,"protected_visits":pre_visits,
            "borrowed":borrowed,"pre_extra":pre_extra,"denied_refinement":h.k26_denied(),"cause":if free==0{"NN_TARGET"}else{"VISIT_CAP"}})
    }
    pub fn k26_root_diversity(&mut self,seed:u64){''')
p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text()
sub('mod k27_cpu_server;','mod k27_cpu_server;\n#[path = "mcts_server/k28_cpu_server.rs"]\nmod k28_cpu_server;')
sub('        if cmd == "research_connect4_k27_cpu" {','''        if cmd == "research_connect4_k28_cpu" {
            emit_stdout_json_value(&k28_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
        if cmd == "research_connect4_k27_cpu" {''')
p.write_text(s)
p=Path('src/mcts_server/k27_cpu_server.rs');s=p.read_text().replace('K27','K28').replace('k27_cpu','k28_cpu').replace('quartz_k27_c4_v1','quartz_k28_c4_v1')
a=s.index('// mask, workers');b=s.index('#[derive(Deserialize)]',a)
s=s[:a]+'''// mask, workers, mode (0 common,1 split,2 floor), reserve.
const ACTIONS:[(u8,usize,u8,u64);20]=[
 (0,1,0,0),(0,2,0,0),(0,4,0,0),(0,8,0,0),
 (15,1,0,0),(15,2,0,0),(15,4,0,0),(15,8,0,0),
 (15,1,1,4),(15,2,1,4),(15,4,1,4),(15,8,1,4),
 (15,1,2,4),(15,2,2,4),(15,4,2,4),(15,8,2,4),
 (4,8,2,4),(11,4,0,0),(15,8,2,2),(15,4,2,2)];
const STOP:usize=20;
'''+s[b:]
s=s.replace('second:[[usize;8];15]','second:[[usize;8];20]').replace('matches!(q.budget,8|32)','matches!(q.budget,8|16|32|64)')
s=s.replace('p.iter().any(|&a|a>=ACTIONS.len())','p[0]>=ACTIONS.len()||p[1]>STOP')
s=s.replace('t.first.iter().chain(t.second.iter().flatten()).any(|&a|a>=ACTIONS.len())','t.first.iter().any(|&a|a>=ACTIONS.len())||t.second.iter().flatten().any(|&a|a>STOP)')
s=s.replace('chosen[j]=a;let(mask,workers,wave,reserve)=ACTIONS[a];let t=Instant::now();h.set_k25_mask(mask);','''chosen[j]=a;if a==STOP{assert_eq!(j,1);break;}
        let(mask,workers,mode,reserve)=ACTIONS[a];let t=Instant::now();h.set_k25_mask(mask);''')
s=s.replace('let b=e.four_axis_k27_budget(q.budget,(q.budget*4) as u32,workers,wave,reserve,&h);',
'''let mut b=if mode==2{e.four_axis_k28_floor(q.budget,(q.budget*4) as u32,workers,reserve,&h)}
        else{e.four_axis_k27_budget(q.budget,(q.budget*4) as u32,workers,false,reserve,&h)};
        if mode!=2{b["borrowed"]=json!(0);b["pre_extra"]=b["protected_extra"].clone();}''')
s=s.replace('"wave":wave','"mode":mode')
s=s.replace('let best=cert.unwrap_or(end.best);','let mid_best=cert.unwrap_or(stages[0]["state"]["best"].as_u64().unwrap() as usize);let best=cert.unwrap_or(end.best);')
s=s.replace('"chosen":chosen,','"chosen":chosen,"mid_best":mid_best,"stopped":chosen[1]==STOP,')
Path('src/mcts_server/k28_cpu_server.rs').write_text(s)
