from pathlib import Path
p=Path('src/mcts_server.rs');s=p.read_text()
a='mod k30_cpu_server;';assert s.count(a)==1
s=s.replace(a,a+'\n#[path = "mcts_server/k31_cpu_server.rs"]\nmod k31_cpu_server;')
a='        if cmd == "research_connect4_k30_cpu" {';assert s.count(a)==1
s=s.replace(a,'''        if cmd == "research_connect4_k31_cpu" {
            emit_stdout_json_value(&k31_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
'''+a);p.write_text(s)
s=Path('src/mcts_server/k30_cpu_server.rs').read_text().replace('research_connect4_k30_cpu','research_connect4_k31_cpu').replace('quartz_k30_c4_v1','quartz_k31_c4_v1').replace('K30','K31')
s=s.replace('#[path="k30_belief.rs"] mod belief;','#[path="k31_gain.rs"] mod gain;\n#[path="k31_proof.rs"] mod proof31;')
a=s.index('const ARMS:');b=s.index('#[derive(Deserialize)]',a)
s=s[:a]+'''const ARMS:[(u8,usize,usize,bool,u64);6]=[
    (0,1,0,false,0),(0,4,0,false,0),(0,8,0,false,0),
    (0,8,8,true,0),(15,8,0,false,4),(15,8,8,true,4)];
'''+s[b:]
s=s.replace('q.arm.is_some_and(|a|a>8)','q.arm.is_some_and(|a|a>9)').replace('matches!(p.as_str(),"prior"|"continue"|"priced")','matches!(p.as_str(),"continue"|"priced")')
s=s.replace('belief::load(','gain::load(')
# traces are collected internally for all arms, only serialized externally on diagnostic or proof paths
s=s.replace('.with_k25(q.trace)','.with_k25(true)')
a=s.index('    let t=Instant::now();let prediction=');b=s.index('    let logical=end.nn;',a)
s=s[:a]+'''    let df=gain::features(&f,&pre,&mid);
    let t=Instant::now();let prediction=belief_model.as_ref().map(|m|m.predict(&df));
    let arm=if let Some(a)=q.arm{a}else{
        let p=prediction.as_ref().unwrap();match q.policy.as_deref().unwrap(){"continue"=>p.fixed,"priced"=>p.priced,_=>unreachable!()}
    };
    let control_us=us(t);let mut batch=Value::Null;let mut tail=Value::Null;let mut graph=Value::Null;let mut tail_us=0;let mut queue_setup_us=0;let mut queue_join_us=0;
    if arm<6{
        let(mask,workers,width,dedup,reserve)=ARMS[arm];h.set_k25_mask(mask);
        let t=Instant::now();let j=if width>0{Some(ev.start(width,dedup,true,q.trace))}else{None};queue_setup_us=us(t);
        let t=Instant::now();tail=if reserve>0{e.four_axis_k28_floor(32,128,workers,reserve,&h)}
            else{e.four_axis_k27_budget(32,128,workers,false,0,&h)};tail_us=us(t);
        let t=Instant::now();if let Some(j)=j{batch=ev.finish(j);}queue_join_us=us(t);
    }else if arm<9{
        let t=Instant::now();let report=h.k25_report(true);
        graph=proof31::execute(&root,&mid,anchor,&report["trace"],arm,q.trace);tail_us=us(t);
    }
    let end=e.four_axis_lite();let eh=h.k25_report(q.trace);let snap=e.four_axis_snapshot();
    let native=cert.unwrap_or(end.best);
    let best=if graph.is_null(){native}else{graph["end"]["best"].as_u64().unwrap() as usize};
    let rules=graph["end"]["rule_work"].as_u64().unwrap_or(0);
'''+s[b:]
s=s.replace('if arm==8{assert_eq!','if arm>=6{assert_eq!')
s=s.replace('"native_best":native,"z":z,"response":response,"gap_flag":flag,"best":best','"native_best":native,"best":best,"direct_features":df,"proof_graph":graph,"rule_work":rules')
Path('src/mcts_server/k31_cpu_server.rs').write_text(s)
