from pathlib import Path
p=Path('src/games/connect4_conv.rs');s=p.read_text()
needle='\n}\n\nimpl Evaluator<Connect4> for ConvConnect4Evaluator'
assert s.count(needle)==1
s=s.replace(needle,'''
    /// True CPU multi-board kernel: same per-lane accumulation order as
    /// scalar inference, with batch lanes contiguous in the inner loop.
    pub fn evaluate_batch29(&self, states: &[Connect4]) -> Vec<EvalResult<usize>> {
        assert!(!states.is_empty() && states.len()<=8);
        assert!(states.iter().all(|s| !s.is_terminal()));
        let n=states.len();
        let mut input=[[0f32;8];126];
        for (b,s) in states.iter().enumerate() {
            let a=s.encode_planes();assert_eq!(a.len(),126);
            for i in 0..126 {input[i][b]=a[i];}
        }
        let mut hidden=[[0f32;8];672];
        for oc in 0..16 {
            for row in 0..6i32 {
                for col in 0..7i32 {
                    let mut v=[self.weights.conv_bias[oc];8];
                    for ic in 0..3 {
                        for dy in 0..3 {
                            let yp=row+dy as i32-1;if !(0..6).contains(&yp){continue;}
                            for dx in 0..3 {
                                let xp=col+dx as i32-1;if !(0..7).contains(&xp){continue;}
                                let w=self.weights.conv_weight[oc][ic][dy][dx];
                                let a=input[ic*42+yp as usize*7+xp as usize];
                                for b in 0..8 {v[b]+=w*a[b];}
                            }
                        }
                    }
                    for b in 0..8{v[b]=v[b].max(0.0);}
                    hidden[oc*42+row as usize*7+col as usize]=v;
                }
            }
        }
        let mut output=[[0f32;8];8];
        for (i,dense) in self.weights.head_weight.iter().enumerate() {
            let mut v=[self.weights.head_bias[i];8];
            for (&w,a) in dense.iter().zip(hidden.iter()){
                for b in 0..8{v[b]+=w*a[b];}
            }
            output[i]=v;
        }
        (0..n).map(|b|{
            let mut logits=[0f32;7];for i in 0..7{logits[i]=output[i][b];}
            self.finish29(&states[b],logits,output[7][b].tanh())
        }).collect()
    }
    fn finish29(&self,state:&Connect4,logits:[f32;7],value:f32)->EvalResult<usize>{
        let legal=state.legal_moves();
        let max_logit=legal.iter().map(|&c|logits[c]).fold(f32::NEG_INFINITY,f32::max);
        let weights=legal.iter().map(|&c|(logits[c]-max_logit).exp()).collect::<Vec<_>>();
        let z=weights.iter().sum::<f32>();
        if !z.is_finite()||z<=0.0{return state.uniform_eval(value);}
        EvalResult{policy:legal.iter().zip(weights.iter()).map(|(&col,&p)|(col,p/z)).collect(),value}
    }
}

impl Evaluator<Connect4> for ConvConnect4Evaluator''')
# Keep scalar path untouched. Batch uses identical masking/normalization.
p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text()
a='mod k28_cpu_server;';assert s.count(a)==1
s=s.replace(a,a+'\n#[path = "mcts_server/k29_cpu_server.rs"]\nmod k29_cpu_server;')
a='        if cmd == "research_connect4_k28_cpu" {';assert s.count(a)==1
s=s.replace(a,'''        if cmd == "research_connect4_k29_cpu" {
            emit_stdout_json_value(&k29_cpu_server::handle(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
'''+a)
p.write_text(s)
