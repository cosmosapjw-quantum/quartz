//! COG05 research-only policy-only (13-feature) Gomoku15 fair-wall-CPU MCTS.
 //! No Rapfi/teacher labels may enter online search. All root edges remain materialized.
use super::*;
use crate::games::gomoku15::{Gomoku15, GomokuVariant};
use crate::mcts::node::ArenaRef;
use std::path::Path;

const D: usize = 13;
const DIRS: [(i32, i32); 4] = [(1,0),(0,1),(1,1),(1,-1)];

fn features(board: &[u8;225], side:u8, action:u16) -> [f32;D] {
    let a=action as usize;
    let x=(a%15) as i32;
    let y=(a/15) as i32;
    let opp=3-side;
    assert_eq!(board[a],0,"features only on legal candidates");
    let mut r1=[0u32;2];
    let mut r2=[0u32;2];
    for dy in -2i32..=2 {
      for dx in -2i32..=2 {
        if dx==0 && dy==0 {continue;}
        let xx=x+dx;let yy=y+dy;
        if xx<0||yy<0||xx>=15||yy>=15{continue;}
        let stone=board[yy as usize*15+xx as usize];
        if stone==0 {continue;}
        let slot=if stone==side {0} else {1};
        let r=dx.abs().max(dy.abs());
        if r==1{r1[slot]+=1;}else if r==2{r2[slot]+=1;}
      }
    }
    let mut z=[0.0f32;D];
    z[0]=1.;
    z[1]=1.-(x-7).abs().max((y-7).abs()) as f32/7.;
    z[2]=1.-((x-7).abs()+(y-7).abs()) as f32/14.;
    z[3]=r1[0] as f32/8.;
    z[4]=r1[1] as f32/8.;
    z[5]=r2[0] as f32/16.;
    z[6]=r2[1] as f32/16.;
    for (part,target) in [(0,side),(1,opp)] {
        let mut best_run=1u32;
        let mut best_window=0u32;
        for (dx,dy) in DIRS {
            let mut run=1u32;
            for sign in [-1i32,1i32] {
                let mut xx=x+sign*dx;let mut yy=y+sign*dy;
                while (0..15).contains(&xx)&&(0..15).contains(&yy)
                 &&board[yy as usize*15+xx as usize]==target {
                    run+=1;xx+=sign*dx;yy+=sign*dy;
                }
            }
            best_run=best_run.max(run);
            for start in -4i32..=0 {
                let mut count=1u32;
                let mut valid=true;
                for j in 0..5i32 {
                    let xx=x+(start+j)*dx;
                    let yy=y+(start+j)*dy;
                    if xx<0||yy<0||xx>=15||yy>=15 {valid=false;break;}
                    if xx==x&&yy==y {continue;}
                    let stone=board[yy as usize*15+xx as usize];
                    if stone==3-target {valid=false;break;}
                    if stone==target {count+=1;}
                }
                if valid {best_window=best_window.max(count);}
            }
        }
        z[7+part]=best_run.min(5) as f32/5.;
        z[9+part]=best_window as f32/5.;
        z[11+part]=if best_run>=5 {1.}else{0.};
    }
    z
}

#[derive(Clone)]
pub(super) struct Policy13 { theta:[f32;D] }
impl Policy13 {
    fn score(&self, board:&[u8;225], side:u8, a:u16)->f32 {
        features(board,side,a).iter().zip(self.theta.iter()).map(|(x,w)|x*w).sum()
    }
}
impl Evaluator<Gomoku15> for Policy13 {
    fn evaluate(&self, state:&Gomoku15)->EvalResult<u16> {
        let legal=state.legal_moves();
        if legal.is_empty(){return EvalResult{policy:vec![],value:0.};}
        let side=if state.current_player()==1 {1u8}else{2u8};
        let board=state.board_as_12();
        let scores:Vec<f32>=legal.iter().map(|&a|self.score(board,side,a)).collect();
        let m=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let exps:Vec<f32>=scores.iter().map(|v|(v-m).exp()).collect();
        let sum: f32=exps.iter().sum();
        if !sum.is_finite() || sum<=0. {return EvalResult::uniform(&legal,0.);}
        EvalResult{policy:legal.into_iter().zip(exps.into_iter().map(|e|e/sum)).collect(),value:0.}
    }
}
pub(super) fn load_model(path:&str)->Result<Policy13,String> {
    let v:serde_json::Value=serde_json::from_slice(&std::fs::read(Path::new(path))
         .map_err(|e|format!("model file read: {e}"))?)
         .map_err(|e|format!("model JSON: {e}"))?;
    if v["schema"]!="quartz_cog05_loglinear_policy13_v1"
       ||v["trained_value_head"]!=false
       ||v["heldout_test_teacher_labels_used"]!=false {
        return Err("model lacks policy-only and strict no-test-label contract".into());
    }
    let Some(arr)=v["theta"].as_array() else {return Err("theta missing".into());};
    if arr.len()!=D{return Err("13 coefficients required".into());}
    let mut theta=[0f32;D];
    for (i,q) in arr.iter().enumerate(){
        let Some(z)=q.as_f64() else{return Err("non-numeric coefficient".into());};
        if !z.is_finite()||z.abs()>500. {return Err("nonfinite / unbounded coefficient".into());}
        theta[i]=z as f32;
    }
    Ok(Policy13{theta})
}
fn root_edges(engine:&MctsEngine<Gomoku15>)->Vec<serde_json::Value>{
    engine.root.edge_snapshot(engine.root.materialized_count()).iter()
        .map(|e|serde_json::json!([e.mv,e.n,e.p,e.w,e.m2])).collect()
}
fn root_edge_visits(engine:&MctsEngine<Gomoku15>)->u32 {
    engine.root.edge_snapshot(engine.root.materialized_count()).iter().map(|e|e.n).sum()
}

pub(super) fn handle_research_gomoku15_prior_cpu(line:&str)->serde_json::Value {
    let q:serde_json::Value=match serde_json::from_str(line){
        Ok(q)=>q,Err(e)=>return serde_json::json!({"error":format!("bad prior request: {e}")})
    };
    for forbidden in ["teacher_move","teacher_WDL","Rapfi_move","oracle",
                      "exact_action_scores","labels","policy_target"] {
        if q.get(forbidden).is_some(){return serde_json::json!({"error":"teacher input forbidden in online search"});}
    }
    if q["rule"]!="freestyle" {
        return serde_json::json!({"error":"COG05 Freestyle15 rule required"});
    }
    let Some(h)=q["history"].as_array() else {
        return serde_json::json!({"error":"sequential 225-cell history array required"});
    };
    let mut game=Gomoku15::new(GomokuVariant::Freestyle);
    let mut hist=Vec::with_capacity(h.len());
    for x in h {
        let Some(a)=x.as_u64() else {return serde_json::json!({"error":"invalid source move"});};
        if a>=225||!game.legal_moves().contains(&(a as u16)) {
            return serde_json::json!({"error":"illegal or duplicate source move"});
        }
        game=game.apply_move(a as u16);
        hist.push(a as u16);
        if game.is_terminal() {
            return serde_json::json!({"error":"source terminated before decision"});
        }
    }
    let legal=game.legal_moves();
    if legal.len()<16 {return serde_json::json!({"error":"outside fixed >=16 legal window"});}
    let side=if game.current_player()==1 {1u8}else{2u8};
    if q["cmd"]=="research_gomoku15_prior_features" {
        let Some(a)=q["action"].as_u64() else {
            return serde_json::json!({"error":"legal inspection action required"});
        };
        if a>=225||!legal.contains(&(a as u16)){
            return serde_json::json!({"error":"nonlegal inspection action"});
        }
        return serde_json::json!({"rule":"freestyle","action":a,
         "features":features(game.board_as_12(),side,a as u16),
         "legal_count":legal.len(),"value_model":"none"});
    }
    if q["cmd"]!="research_gomoku15_prior_cpu" {
        return serde_json::json!({"error":"unknown command"});
    }
    let Some(arm)=q["arm"].as_str() else {return serde_json::json!({"error":"arm required"});};
    if !matches!(arm,"learned_gated"|"learned_all_legal"|"uniform_all_legal") {
        return serde_json::json!({"error":"unknown arm"});
    }
    let seed=q["seed"].as_u64().unwrap_or(17);
    if seed!=17 {return serde_json::json!({"error":"unregistered MCTS seed"});}
    let budget=q["cpu_budget_us"].as_u64().unwrap_or(20_000);
    if budget!=20_000 {return serde_json::json!({"error":"CPU budget locked at 20000us"});}
    let selected_model=if arm=="uniform_all_legal"{None}else {
        let Some(path)=q["model_path"].as_str() else {
            return serde_json::json!({"error":"trained policy file path required"});
        };
        match load_model(path){Ok(x)=>Some(x),Err(e)=>return serde_json::json!({"error":e})}
    };

    let t0=Instant::now(); // before policy and engine; after frozen model file load
    let eval:Arc<dyn Evaluator<Gomoku15>>=match &selected_model {
        Some(x)=>Arc::new(x.clone()),None=>Arc::new(UniformEval),
    };
    let initial_prior=eval.evaluate(&game);
    let mut ordered=legal.clone();
    let prior_lookup:HashMap<u16,f32>=initial_prior.policy.iter().copied().collect();
    ordered.sort_by(|a,b| {
        let pa=prior_lookup.get(a).copied().unwrap_or(0.);
        let pb=prior_lookup.get(b).copied().unwrap_or(0.);
        pb.total_cmp(&pa).then(a.cmp(b))
    });
    let first16:Vec<u16>=ordered.iter().take(16).copied().collect();
    let omitted:Vec<u16>=ordered.iter().skip(16).copied().collect();
    let gated=arm=="learned_gated";
    let mut cfg=MctsConfig::evaluation(2.0).with_seed(seed);
    cfg.root_forced_win=false;
    cfg.dirichlet=None;
    cfg.temperature=0.;
    let mut engine=MctsEngine::new(game.clone(),eval,cfg);
    if engine.root.candidate_count()!=legal.len()||engine.root.materialized_count()!=legal.len(){
        return serde_json::json!({"error":"full original root edges not materialized"});
    }
    let root_ptr=ArenaRef::as_ptr(&engine.root);
    let root_hash=engine.root.hash;
    let original_prior=engine.root.edge_priors_snapshot(legal.len());
    if gated {
        if let Err(e)=engine.set_resident_root_allowed_moves(&first16) {
            return serde_json::json!({"error":format!("initial attention: {e}")});
        }
    }
    let first=engine.run(&mut FixedIterations::new(8));
    if first.root_visits!=8 || root_edge_visits(&engine)!=8 {
        return serde_json::json!({"error":"first8 searches did not conserve root visits"});
    }
    let before=root_edges(&engine);
    let calls_before=engine.evaluator_call_count();
    let mut suggested=Vec::<u16>::new();
    let mut reason="full_all_legal";
    if gated {
        reason="no_rule_witness";
        let board=game.board_as_12();
        // First: wins by current player, then mandatory blocks of opponent instant win.
        let mut wins=Vec::new();
        let mut blocks=Vec::new();
        for &a in &omitted {
            let f=features(board,side,a);
            if f[11]>=1. {wins.push(a);}
            else if f[12]>=1. {blocks.push(a);}
        }
        suggested.extend(wins.iter().copied().take(4));
        if suggested.len()<4 {
            suggested.extend(blocks.iter().copied().take(4-suggested.len()));
        }
        if !wins.is_empty(){reason="omitted_immediate_win";}
        else if !blocks.is_empty(){reason="omitted_immediate_block";}
        let mut active=first16.clone();
        active.extend(suggested.iter().copied());
        if let Err(e)=engine.set_resident_root_allowed_moves(&active){
            return serde_json::json!({"error":format!("expand same root: {e}")});
        }
    }
    let unchanged=root_edges(&engine)==before
        &&engine.root.edge_priors_snapshot(legal.len())==original_prior
        &&engine.root.hash==root_hash
        &&ArenaRef::as_ptr(&engine.root)==root_ptr
        &&engine.evaluator_call_count()==calls_before;
    if !unchanged{return serde_json::json!({"error":"root priors or N/W/M2 mutated at reactivation"});}
    let mut visits=8u32;
    let mut chunks=0u32;
    // FixedIterations is total-root-visits; a bounded one-visit quantum near deadline.
    while t0.elapsed().as_micros()<(budget as u128) && visits<2048 {
        let next=visits+1;
        let result=engine.run(&mut FixedIterations::new(next));
        if result.root_visits!=next || root_edge_visits(&engine)!=next {
            return serde_json::json!({"error":"MCTS continuation visit counter mismatch"});
        }
        visits=next;
        chunks+=1;
    }
    let elapsed=t0.elapsed().as_micros();
    let Some(best)=engine.best_move() else {
        return serde_json::json!({"error":"completed root lacks decision"});
    };
    if !legal.contains(&best) {
        return serde_json::json!({"error":"search returned illegal action"});
    }
    serde_json::json!({
        "schema":"quartz_cog05_trained_prior_real_cpu_v1",
        "rule":"freestyle","arm":arm,"history":hist,"seed":seed,
        "source_model":"trained_policy13_value0" ,
        "model_loaded":selected_model.is_some(),
        "trained_value_head":false,
        "full_original_root_edges":legal.len(),
        "legal_actions":legal,
        "initial_ranked_moves":ordered,
        "initial_candidates":if gated {first16.clone()}else{legal.clone()},
        "reopened_candidates":suggested,
        "reopen_reason":reason,
        "root_prior_probabilities":initial_prior.policy,
        "first8_root_visits":first.root_visits,
        "final_root_visits":visits,
        "continuation_chunks":chunks,
        "first8_original_edge_stats":before,
        "first8_evaluator_calls":calls_before,
        "root_stats_and_priors_unchanged_on_activation":unchanged,
        "same_root_hash_preserved":engine.root.hash==root_hash,
        "same_root_ptr_preserved":ArenaRef::as_ptr(&engine.root)==root_ptr,
        "root_edge_visits_sum":root_edge_visits(&engine),
        "root_mask_after":engine.resident_root_mask_words_snapshot(),
        "final_legal_move":best,
        "selected_move_exact_terminal_win":game.is_winning_move(best),
        "target_cpu_budget_us":budget,
        "total_actual_cpu_wall_us":elapsed,
        "budget_includes_policy_and_search":true,
        "budget_excludes_model_load_and_output_serialization":true,
        "hit_2048_search_cap":visits==2048,
        "evaluator_calls":engine.evaluator_call_count(),
        "no_online_teacher_labels":true
    })
}

#[cfg(test)]
mod cog05_tests {
    use super::*;
    #[test]
    fn policy13_legal_normalization_and_no_value_head() {
        let b=[0u8;225];
        let f0=features(&b,1,0);
        let f224=features(&b,1,224);
        assert_eq!(f0,f224);
        assert_eq!(f0.len(),13);
        assert_eq!(f0[1],0.);
        let eval=Policy13{theta:[0.;13]};
        let game=Gomoku15::new(GomokuVariant::Freestyle);
        let p=eval.evaluate(&game);
        assert_eq!(p.policy.len(),225);
        assert!(p.policy.iter().all(|(_,p)| *p>0.&&p.is_finite()));
        assert!((p.policy.iter().map(|(_,p)|p).sum::<f32>()-1.).abs()<1e-4);
        assert_eq!(p.value,0.);
        let req=serde_json::json!({"cmd":"research_gomoku15_prior_cpu",
              "rule":"freestyle","history":[31,160,32,177,33,190,34,207],
              "arm":"learned_gated","teacher_move":30});
        assert!(handle_research_gomoku15_prior_cpu(&req.to_string()).get("error").is_some());
    }
    #[test]
    fn feature_inspection_accepts_index224_and_rejects_duplicate() {
        let q=serde_json::json!({"cmd":"research_gomoku15_prior_features",
                   "rule":"freestyle","history":[31,160],"action":224});
        let r=handle_research_gomoku15_prior_cpu(&q.to_string());
        assert_eq!(r["features"].as_array().unwrap().len(),13);
        let bad=serde_json::json!({"cmd":"research_gomoku15_prior_features",
                   "rule":"freestyle","history":[31,31],"action":224});
        assert!(handle_research_gomoku15_prior_cpu(&bad.to_string()).get("error").is_some());
    }
}
