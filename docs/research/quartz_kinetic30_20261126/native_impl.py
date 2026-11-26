from pathlib import Path
p=Path('src/games/connect4_conv.rs');s=p.read_text()
a=s.index('    pub fn evaluate_batch29');b=s.index('    fn finish29',a)
t=s[a:b].replace('pub fn evaluate_batch29(&self, states:', 'fn batch30<const W: usize>(&self, states:').replace('states.len()<=8','states.len()<=W').replace('[0f32;8]','[0f32;W]').replace('conv_bias[oc];8','conv_bias[oc];W').replace('head_bias[i];8','head_bias[i];W').replace('0..8','0..W')
# output head count must stay eight (outer array); per-lane width W.
assert '[[0f32;W];8]' in t
t+='''    pub fn evaluate_batch30(&self, states: &[Connect4]) -> (Vec<EvalResult<usize>>,usize) {
        match states.len(){1=>(self.batch30::<1>(states),1),2=>(self.batch30::<2>(states),2),3|4=>(self.batch30::<4>(states),4),5..=8=>(self.batch30::<8>(states),8),_=>panic!("invalid batch30 size")}
    }
'''
p.write_text(s[:b]+t+s[b:])
p=Path('src/mcts_server.rs');s=p.read_text()
a='mod k29_cpu_server;';assert s.count(a)==1
s=s.replace(a,a+'\n#[path = "mcts_server/k30_cpu_server.rs"]\nmod k30_cpu_server;')
a='        if cmd == "research_connect4_k29_cpu" {';assert s.count(a)==1
s=s.replace(a,'''        if cmd == "research_connect4_k30_cpu" {
            emit_stdout_json_value(&k30_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
'''+a)
p.write_text(s)
p=Path('src/mcts_server/k29_belief.rs');s=p.read_text()
s=s.replace('struct Model{k:usize,eps:f64,donors:Vec<Donor>}','struct Model{k:usize,eps:f64,alpha:[f64;8],guard:[f64;8],donors:Vec<Donor>}')
s=s.replace('d.z.iter().flatten().any(|&x|x>=n)','d.z.iter().flatten().any(|&x|x>=2*n)')
s=s.replace('m.donors.len()>10000','m.donors.len()>10000||m.alpha.iter().any(|x|!x.is_finite()||*x<0.||*x>1.)||m.guard.iter().any(|x|!x.is_finite()||*x<0.||*x>2.)')
s=s.replace('let prior=if n==0{0}else{argmax(&m)};','let prior=0;')
s=s.replace('let mut mass=vec![eps/l as f64;l];let mut j=vec![base.iter().map(|v|v*eps/l as f64).collect::<Vec<_>>();l];','let zlen=2*l;let mut mass=vec![eps/zlen as f64;zlen];let mut j=vec![base.iter().map(|v|v*eps/zlen as f64).collect::<Vec<_>>();zlen];')
a=s.index('            let p:Vec<_>=j.iter()');b=s.index('        }\n        let arm=',a)
s=s[:a]+'''            let alpha=self.alpha[c];let guard=self.guard[c];
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
'''+s[b:]
Path('src/mcts_server/k30_belief.rs').write_text(s)
p=Path('src/mcts_server/k29_cpu_server.rs');s=p.read_text().replace('k29_queue.rs','k30_queue.rs').replace('k29_belief.rs','k30_belief.rs').replace('research_connect4_k29_cpu','research_connect4_k30_cpu').replace('quartz_k29_c4_v1','quartz_k30_c4_v1').replace('K29','K30')
# Arm3 padded queue8, 4 specialized queue8, 5 dedup specialized. Same direct/full arms.
s=s.replace('(0,4,4,false,0)','(0,8,8,false,0)')
s=s.replace('let ev=Arc::new(queue::QueueEval::new(model));','let ev=Arc::new(queue::QueueEval::new(model,&q.model_path,&q.model_identity));')
s=s.replace('Some(ev.start(width,dedup,q.trace))','Some(ev.start(width,dedup,arm!=3,q.trace))')
s=s.replace('let best=if let Some(p)=&prediction{cert.unwrap_or(rank[if arm==8{p.prior}else{p.post[arm][z]}])}else{native};','''let gap=|r:&LiteRoot|{
        let x:Vec<_>=rank.iter().map(|mv|{let i=r.moves[..r.len].iter().position(|m|m==mv).unwrap();0.5+0.5*r.w[i]/(r.n[i] as f64+4.)}).collect();
        x[0]-x[1..].iter().copied().fold(f64::NEG_INFINITY,f64::max)
    };
    let flag=usize::from(gap(&end)>gap(&mid)+1e-12);let response=2*z+flag;
    let best=if let Some(p)=&prediction{cert.unwrap_or(rank[if arm==8{p.prior}else{p.post[arm][response]}])}else{native};''')
s=s.replace('"native_best":native,"z":z,"best":best','"native_best":native,"z":z,"response":response,"gap_flag":flag,"best":best')
Path('src/mcts_server/k30_cpu_server.rs').write_text(s)
