from pathlib import Path
p=Path('src/mcts/four_axis.rs');s=p.read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
rep('    k25:Option<K25>,','    k25:Option<K25>,\n    k32:Option<K32>,')
rep('merge24_mode:0,k25:None,','merge24_mode:0,k25:None,k32:None,')
rep('if (if mode==2{i>=visible}else{m==wanted})','if (if mode==2{i>=if self.k32.is_some(){visible.max(node.materialized_count())}else{visible}}else{m==wanted})')
rep('if depth>0&&iv.0==iv.1{return Some(ProofPlan','if depth>0&&iv.0==iv.1&&self.k32.is_none(){return Some(ProofPlan')
rep('}else if h.proof_mode==2&&known_interval.is_some_and','}else if h.proof_mode==2&&h.k32.is_none()&&known_interval.is_some_and')
rep('if let Some(ref f)=h.local_frontier{if exact_key(&sel.leaf_state)==f.key{h.add(33,1);}}','if let Some(ref f)=h.local_frontier{if exact_key(&sel.leaf_state)==f.key{h.add(33,1);if let Some(k)=&h.k32{k.frontier_hits.fetch_add(1,Ordering::Relaxed);}}}')
rep('        self.add(22,1);\n        if s.is_terminal(){g.terminal(k,s.outcome() as i8);return k;}','        self.add(22,1);\n        let clock=self.k32.as_ref().map(|_|std::time::Instant::now());\n        if s.is_terminal(){g.terminal(k,s.outcome() as i8);return k;}')
rep('let ck=exact_key(&c);self.add(23,1);','let ck=exact_key(&c);self.add(23,1);if let Some(k)=&self.k32{k.checks.fetch_add(1,Ordering::Relaxed);}')
rep('        g.register(k,a);k','        g.register(k,a);if let (Some(k),Some(t))=(&self.k32,clock){k.reg_ns.fetch_add(t.elapsed().as_nanos() as u64,Ordering::Relaxed);}k')
rep('        l.events+=1;\n        if k.trace{steps.reverse();','        l.events+=1;\n        if let Some(p)=&self.k32{p.paths.lock().unwrap().push(moves.clone());}\n        if k.trace{steps.reverse();')
rep('        assert_eq!(h.proof_mode,0);assert!(h.vmode==0||h.vmode==1);assert!(reserve==0||h.vmode==1);','        assert!(h.proof_mode==0||(h.proof_mode==1&&h.k32.is_some()));assert!(h.vmode==0||h.vmode==1);assert!(reserve==0||h.vmode==1);')
s+='\n'+Path(__file__).with_name('native_methods.rs').read_text()
p.write_text(s)
p=Path('src/mcts/select.rs');s=p.read_text();a='            axes.unwrap().local_applied(local_claim,depth);';assert s.count(a)==1
s=s.replace(a,a+'\n            axes.unwrap().k32_forced(cur_state,local_claim,depth,best_idx,edges[best_idx].mv,n_mat,n_edges,base_visible);');p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text();a='mod k31_cpu_server;';assert s.count(a)==1
s=s.replace(a,a+'\n#[path = "mcts_server/k32_cpu_server.rs"]\nmod k32_cpu_server;')
a='        if cmd == "research_connect4_k31_cpu" {';assert s.count(a)==1
s=s.replace(a,'''        if cmd == "research_connect4_k32_cpu" {
            emit_stdout_json_value(&k32_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
'''+a);p.write_text(s)
s=Path('src/mcts_server/k31_cpu_server.rs').read_text()
s=s.replace('research_connect4_k31_cpu','research_connect4_k32_cpu').replace('quartz_k31_c4_v1','quartz_k32_c4_v1').replace('K31','K32')
a=s.index('#[path="k31_gain.rs"]');b=s.index('#[derive(Deserialize)]',a)
s=s[:a]+'#[path="k31_proof.rs"] mod replay31;\n'+s[b:]
a=s.index('    #[serde(default)]arm:');b=s.index('fn us(',a)
s=s[:a]+'''    arm:usize,#[serde(default)]trace:bool}
'''+s[b:]
a=s.index('    if q.cmd!=');b=s.index('    if rayon::',a)
s=s[:a]+'''    if q.cmd!="research_connect4_k32_cpu"||q.arm>10{return Err("bad K32 contract".into());}
'''+s[b:]
a=s.index('    let belief_model=');b=s.index('    let ev=',a);s=s[:a]+s[b:]
s=s.replace('let e=MctsEngine::new(root.clone(),ev.clone(),cfg);let mut h=Hooks::new_modes(0,0,0,false).with_k25(true);','''let e=MctsEngine::new(root.clone(),ev.clone(),cfg);let arm=q.arm;
    let graph=(2..=8).contains(&arm);let mut h=Hooks::new_modes(0,0,0,false).with_k25(q.trace||arm==10);
    if graph{h=h.with_k32();h.k32_init(&root);}''')
a=s.index('    let df=gain::');b=s.index('    drop(e);',a)
s=s[:a]+'''    let mg=h.k32_report(&root,&mid,anchor,false,false);
    let mut batch=Value::Null;let mut tail=Value::Null;let mut replay=Value::Null;let mut rule_steps=Value::Null;let mut tail_us=0;let mut queue_setup_us=0;let mut queue_join_us=0;
    if arm==1{
        h.set_k25_mask(0);let t=Instant::now();tail=e.four_axis_k27_budget(32,128,1,false,0,&h);tail_us=us(t);
    }else if arm==3{
        let t=Instant::now();rule_steps=e.k32_rule_expand(&h,anchor);tail_us=us(t);
    }else if (4..=9).contains(&arm){
        let full=arm>=7;let worker=if full{8}else{1};let guide=match arm{5=>2,6|7|8=>3,_=>0};
        h.set_k25_mask(if full{15}else{0});if graph{h.k32_activate();}
        let t=Instant::now();let j=if arm>=8{Some(ev.start(8,true,true,q.trace))}else{None};queue_setup_us=us(t);
        let t=Instant::now();tail=e.four_axis_k32_budget(32,128,worker,if full{4}else{0},guide,graph,&mut h);tail_us=us(t);
        let t=Instant::now();if let Some(j)=j{batch=ev.finish(j);}queue_join_us=us(t);
    }else if arm==10{
        let t=Instant::now();let r=h.k25_report(true);replay=replay31::execute(&root,&mid,anchor,&r["trace"],8,q.trace);tail_us=us(t);
    }
    let end=e.four_axis_lite();let eh=h.k25_report(q.trace);let snap=e.four_axis_snapshot();
    let native=cert.unwrap_or(end.best);
    let eg=h.k32_report(&root,&end,native,q.trace,true);
    let best=if graph{eg["best"].as_u64().unwrap() as usize}else if arm==10{replay["end"]["best"].as_u64().unwrap() as usize}else{native};
    let rules=if graph{eg["rule_work"].as_u64().unwrap()}else if arm==10{replay["end"]["rule_work"].as_u64().unwrap()}else{0};
    let logical=end.nn;let physical=logical-batch["coalesced"].as_u64().unwrap_or(0);
    if !batch.is_null(){assert_eq!(batch["logical"],logical-mid.nn);}
'''+s[b:]
a=s.index('    Ok(json!(');b=s.index('\n}',a)
s=s[:a]+'''    Ok(json!({"schema":"quartz_k32_c4_v1","history":q.history,"model_identity":q.model_identity,"arm":arm,"legal":legal,"proof":proof,
        "prefix":pre,"prefix_hooks":ph,"first":first,"mid":mid,"mid_hooks":mh,"mid_best":anchor,"rank":rank,"features":f,"allowed":allowed,
        "tail":tail,"end":end,"end_hooks":eh,"snapshot":snap,"native_best":native,"best":best,"mid_graph":mg,"end_graph":eg,"rule_steps":rule_steps,"replay_graph":replay,"rule_work":rules,
        "NN_calls":logical,"physical_calls":physical,"visits":end.visits,"queue":batch,
        "setup_us":setup_us,"prefix_us":prefix_us,"first_us":first_us,"tail_us":tail_us,"queue_setup_us":queue_setup_us,"queue_join_us":queue_join_us,"whole_us":us(start)}))
'''+s[b:]
Path('src/mcts_server/k32_cpu_server.rs').write_text(s)
