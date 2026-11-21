//! C16 bounded nonterminal depth3 fork witness and physically matched native MCTS.
 //! Board-only classic Freestyle proof, NOT external engine oracle and NOT a value net.
use super::*;
use super::prior15_cpu_server::load_model;
use crate::games::gomoku15::{Gomoku15,GomokuVariant};
use crate::mcts::node::ArenaRef;

const AXES:[(i32,i32);4]=[(1,0),(0,1),(1,1),(1,-1)];

#[derive(Clone,Debug)]
struct ForkCertificate {
    action:u16,
    endpoints:[u16;2],
}
fn win_if_placed(board:&[u8;225],side:u8,a:u16)->bool{
    let a=a as usize;
    if board[a]!=0{return false;}
    let x=(a%15) as i32;
    let y=(a/15) as i32;
    for (dx,dy) in AXES {
       let mut count=1;
       for sign in [-1,1] {
         let mut xx=x+sign*dx;
         let mut yy=y+sign*dy;
         while (0..15).contains(&xx)&&(0..15).contains(&yy)
            &&board[yy as usize*15+xx as usize]==side {
            count+=1;
            xx+=sign*dx;yy+=sign*dy;
         }
       }
       if count>=5{return true;}
    }
    false
}
fn counterfactual_immediate_wins(board:&[u8;225],side:u8)->Vec<u16>{
   (0..225).filter_map(|i|{
      let a=i as u16;
      if board[i]==0&&win_if_placed(board,side,a){Some(a)}else{None}
   }).collect()
}
fn certificate_for(board:&[u8;225],side:u8,a:u16)->Option<ForkCertificate>{
    if board[a as usize]!=0||win_if_placed(board,side,a){return None;}
    let mut after=*board;
    after[a as usize]=side;
    let x=(a%15) as i32;
    let y=(a/15) as i32;
    let mut wins=Vec::with_capacity(4);
    for (dx,dy) in AXES {
      for sign in [-1,1] {
        for dist in 1..=4 {
          let xx=x+sign*dist*dx;
          let yy=y+sign*dist*dy;
          if !(0..15).contains(&xx)||!(0..15).contains(&yy){continue;}
          let c=(yy*15+xx) as u16;
          if win_if_placed(&after,side,c) && !wins.contains(&c) {wins.push(c);}
        }
      }
    }
    wins.sort_unstable();
    if wins.len()<2{return None;}
    Some(ForkCertificate{action:a,endpoints:[wins[0],wins[1]]})
}
fn collect_certificates(board:&[u8;225],side:u8,ordered:&[u16],cap:usize)
 ->(bool,Vec<ForkCertificate>,usize){
    // The pure-board opponent query must not use current_player legality.
    if !counterfactual_immediate_wins(board,side).is_empty()
        ||!counterfactual_immediate_wins(board,3-side).is_empty() {
        return(false,Vec::new(),0);
    }
    let mut certs=Vec::new();
    let mut probes=0;
    for &a in ordered {
        probes+=1;
        if let Some(cert)=certificate_for(board,side,a){
           certs.push(cert);
           if certs.len()>=cap{break;}
        }
    }
    (true,certs,probes)
}
fn edge_stats(engine:&MctsEngine<Gomoku15>)->Vec<serde_json::Value>{
    engine.root.edge_snapshot(engine.root.materialized_count()).iter()
      .map(|e|serde_json::json!([e.mv,e.n,e.p,e.w,e.m2])).collect()
}
fn visit_sum(engine:&MctsEngine<Gomoku15>)->u32{
    engine.root.edge_snapshot(engine.root.materialized_count()).iter().map(|e|e.n).sum()
}
fn rebuild_source(q:&serde_json::Value)->Result<(Gomoku15,Vec<u16>),String>{
    if q["rule"]!="freestyle"{return Err("C16 requires 15x15 Freestyle".into());}
    let Some(rows)=q["history"].as_array()else{return Err("sequential history array required".into());};
    let mut game=Gomoku15::new(GomokuVariant::Freestyle);
    let mut h=Vec::with_capacity(rows.len());
    for item in rows {
        let Some(a)=item.as_u64()else{return Err("history contains a noninteger move".into());};
        if a>=225||!game.legal_moves().contains(&(a as u16)){
            return Err("out-of-board, duplicate or illegal source move".into());
        }
        game=game.apply_move(a as u16);
        h.push(a as u16);
        if game.is_terminal(){return Err("source history already terminal".into());}
    }
    Ok((game,h))
}
fn certificates_json(rows:&[ForkCertificate])->Vec<serde_json::Value>{
    rows.iter().map(|x|serde_json::json!({
        "action":x.action,"endpoints":x.endpoints,"nonterminal":true,
        "guarantee":"forced_win_within_next_self_turn_under_freestyle_no_capture"
    })).collect()
}
pub(super) fn handle_research_gomoku15_fork_cpu(line:&str)->serde_json::Value{
    let q:serde_json::Value=match serde_json::from_str(line){
        Ok(q)=>q,Err(e)=>return serde_json::json!({"error":format!("bad C16 JSON:{e}")})
    };
    for key in ["teacher_move","teacher_WDL","Rapfi_move","exact_action_scores",
                "policy_target","oracle","labels","winning_action_label"] {
       if q.get(key).is_some(){return serde_json::json!({"error":"teacher/oracle input forbidden online"});}
    }
    let (game,h)=match rebuild_source(&q){
        Ok(v)=>v,Err(e)=>return serde_json::json!({"error":e})
    };
    let legal=game.legal_moves();
    let side=if game.current_player()==1 {1u8}else{2u8};
    if q["cmd"]=="research_gomoku15_fork_inspect" {
        let (eligible,all,probed)=collect_certificates(
          game.board_as_12(),side,&legal,legal.len());
        return serde_json::json!({
          "schema":"quartz_cog06_source_rule_fork_inspection_v1",
          "history":h,"rule":"freestyle","side":side,"legal_count":legal.len(),
          "no_existing_immediate_win_either_side":eligible,
          "inspected_action_count":probed,
          "certificates_all":certificates_json(&all),
          "no_external_oracle":true
        });
    }
    if q["cmd"]!="research_gomoku15_fork_cpu" {
        return serde_json::json!({"error":"unregistered C16 command"});
    }
    if legal.len()<16{return serde_json::json!({"error":"outside >=16 legal move window"});}
    let Some(arm)=q["arm"].as_str()else{return serde_json::json!({"error":"arm required"});};
    if !matches!(arm,"learned_all_legal"|"learned_fork_mask"|"learned_fork_proof_choice"
       |"learned_sham_mask"|"learned_top16"){
        return serde_json::json!({"error":"unregistered C16 intervention"});
    }
    if q["seed"].as_u64().unwrap_or(17)!=17{return serde_json::json!({"error":"frozen seed must be17"});}
    if q["cpu_budget_us"].as_u64().unwrap_or(20_000)!=20_000{
        return serde_json::json!({"error":"20,000us frozen C16 CPU budget required"});
    }
    let Some(path)=q["model_path"].as_str()else{
        return serde_json::json!({"error":"frozen trained policy model_path required"});
    };
    // Model file parse deliberately outside the active MCTS decision clock, as in C15.
    let model=match load_model(path){
        Ok(v)=>v,Err(e)=>return serde_json::json!({"error":e})
    };
    let started=Instant::now();
    let eval:Arc<dyn Evaluator<Gomoku15>>=Arc::new(model);
    let original=eval.evaluate(&game);
    let prior:HashMap<u16,f32>=original.policy.iter().copied().collect();
    if original.policy.len()!=legal.len(){
       return serde_json::json!({"error":"trained full-legal prior missing actions"});
    }
    let mut order=legal.clone();
    order.sort_by(|a,b|
        prior.get(b).copied().unwrap_or(0.).total_cmp(
             &prior.get(a).copied().unwrap_or(0.)).then(a.cmp(b))
    );
    let top16:Vec<u16>=order.iter().take(16).copied().collect();
    let omitted:Vec<u16>=order.iter().skip(16).copied().collect();
    let gated=arm!="learned_all_legal";
    let mut cfg=MctsConfig::evaluation(2.0).with_seed(17);
    cfg.root_forced_win=false;
    cfg.dirichlet=None;
    cfg.temperature=0.;
    let mut engine=MctsEngine::new(game.clone(),eval,cfg);
    if engine.root.candidate_count()!=legal.len()||
        engine.root.materialized_count()!=legal.len() {
        return serde_json::json!({"error":"full original root candidate count mismatch"});
    }
    let ptr=ArenaRef::as_ptr(&engine.root);
    let before_hash=engine.root.hash;
    let complete_priors=engine.root.edge_priors_snapshot(legal.len());
    if gated {
        if let Err(e)=engine.set_resident_root_allowed_moves(&top16) {
           return serde_json::json!({"error":format!("initial gated mask: {e}")});
        }
    }
    let first=engine.run(&mut FixedIterations::new(8));
    if first.root_visits!=8||visit_sum(&engine)!=8 {
        return serde_json::json!({"error":"initial 8 visits not conserved"});
    }
    let first_edges=edge_stats(&engine);
    let first_calls=engine.evaluator_call_count();
    let proof_start=Instant::now();
    let mut proven=Vec::new();
    let mut scanned_count=0usize;
    let mut precondition=false;
    if matches!(arm,"learned_fork_mask"|"learned_fork_proof_choice"|"learned_sham_mask"){
        let (ok,certs,probes)=collect_certificates(game.board_as_12(),side,&omitted,4);
        precondition=ok;proven=certs;scanned_count=probes;
    }
    let mut added=Vec::<u16>::new();
    if matches!(arm,"learned_fork_mask"|"learned_fork_proof_choice"){
        added.extend(proven.iter().map(|w|w.action));
    } else if arm=="learned_sham_mask" {
        // Same cardinality of random-looking NOT-certified omitted actions.
        // Scan is source-only; no oracle, no external finite teacher.
        let mut unproven=Vec::new();
        for &a in &omitted {
            if precondition && certificate_for(game.board_as_12(),side,a).is_some(){continue;}
            unproven.push(a);
        }
        unproven.sort_by_key(|a| {
            let mut x=17u64^(*a as u64);
            for &v in &h {x^=v as u64;x=x.wrapping_mul(0x100000001b3);}
            x
        });
        added.extend(unproven.into_iter().take(proven.len()));
    }
    let witness_us=proof_start.elapsed().as_micros();
    let mut active=if gated {top16.clone()}else{legal.clone()};
    for &a in &added {
        if !active.contains(&a){active.push(a);}
    }
    if gated {
        if let Err(e)=engine.set_resident_root_allowed_moves(&active){
            return serde_json::json!({"error":format!("existing-root activation: {e}")});
        }
    }
    let before_continuation=edge_stats(&engine);
    let stats_preserved=ptr==ArenaRef::as_ptr(&engine.root)
        &&before_hash==engine.root.hash
        &&first_edges==before_continuation
        &&complete_priors==engine.root.edge_priors_snapshot(legal.len())
        &&engine.evaluator_call_count()==first_calls
        &&visit_sum(&engine)==8;
    if !stats_preserved{return serde_json::json!({"error":"root N/W/M2/P, pointer or evaluator calls changed on mask activation"});}
    let mut n=8u32;
    while started.elapsed().as_micros()<20_000 && n<2048 {
        n+=1;
        let progress=engine.run(&mut FixedIterations::new(n));
        if progress.root_visits!=n||visit_sum(&engine)!=n{
          return serde_json::json!({"error":"MCTS continuation root count mismatch"});
        }
    }
    let elapsed=started.elapsed().as_micros();
    let Some(mcts_action)=engine.best_move()else{
       return serde_json::json!({"error":"real MCTS produced no best action"});
    };
    let overridden=arm=="learned_fork_proof_choice"&&!proven.is_empty();
    let choice=if overridden{proven[0].action}else{mcts_action};
    if !legal.contains(&choice) || !active.contains(&choice){
       return serde_json::json!({"error":"final action not in root legally active set"});
    }
    let certificate_selected=proven.iter().any(|cert|cert.action==choice);
    serde_json::json!({
        "schema":"quartz_cog06_real_native_fork15_v1",
        "history":h,"rule":"freestyle","arm":arm,
        "source_prior":"C15_frozen_trained_policy13_value0",
        "model_loaded":true,"trained_value_head":false,
        "original_full_root_legal_count":legal.len(),
        "legal_actions":legal,
        "initial_ranked_actions":order,
        "initial_candidates":if gated {top16}else{legal.clone()},
        "added_actions":added,
        "after_active_actions":active,
        "original_full_prior":original.policy,
        "first8_root_visits":first.root_visits,
        "first8_edge_stats":first_edges,
        "first8_evaluator_calls":first_calls,
        "source_rule_preconditions_verified":precondition,
        "candidate_inspections":scanned_count,
        "certificates_omitted":certificates_json(&proven),
        "witness_wall_us":witness_us,
        "same_root_N_W_M2_P_preserved_on_activation":stats_preserved,
        "same_root_ptr_hash":ArenaRef::as_ptr(&engine.root)==ptr&&engine.root.hash==before_hash,
        "root_mask_after":engine.resident_root_mask_words_snapshot(),
        "final_root_visits":n,"root_edge_visit_sum":visit_sum(&engine),
        "final_mcts_move":mcts_action,"final_legal_move":choice,
        "explicit_proof_choice_override":overridden,
        "selected_is_certified_fork":certificate_selected,
        "selected_is_immediate_win":game.is_winning_move(choice),
        "total_actual_cpu_wall_us":elapsed,
        "target_cpu_budget_us":20000,
        "hit_search_cap":n==2048,
        "evaluator_calls":engine.evaluator_call_count(),
        "teacher_oracle_never_online":true,
        "budget_excludes_model_file_load_output_serialization":true
    })
}
#[cfg(test)]
mod c16_tests {
    use super::*;
    #[test]
    fn c16_nonterminal_two_endpoints_returns_valid_certificate() {
        let req=serde_json::json!({
            "cmd":"research_gomoku15_fork_inspect",
            "rule":"freestyle",
            "history":[31,160,32,177,33,190],
        });
        let got=handle_research_gomoku15_fork_cpu(&req.to_string());
        let items=got["certificates_all"].as_array().expect("certificate list missing");
        let witness=items.iter().find(|x|x["action"]==34)
            .expect("legal nonterminal fork action 34 absent");
        assert_eq!(witness["endpoints"],serde_json::json!([30,35]));
        assert_eq!(witness["nonterminal"],true);
    }
    #[test]
    fn c16_existing_opponent_immediate_win_disables_unconditional_certificate() {
        // White has a ready 4, black still has a 3 setup.
        let q=serde_json::json!({
          "cmd":"research_gomoku15_fork_inspect","rule":"freestyle",
          "history":[31,160,32,161,33,162,70,163]});
        let out=handle_research_gomoku15_fork_cpu(&q.to_string());
        assert_eq!(out["no_existing_immediate_win_either_side"],false);
        assert_eq!(out["certificates_all"].as_array().unwrap().len(),0);
    }
    #[test]
    fn c16_teacher_input_and_duplicate_history_rejected() {
        let mut q=serde_json::json!({
          "cmd":"research_gomoku15_fork_inspect","rule":"freestyle",
          "history":[31,160,32,177,33,190]});
        q["teacher_move"]=serde_json::json!(34);
        assert!(handle_research_gomoku15_fork_cpu(&q.to_string()).get("error").is_some());
        q.as_object_mut().unwrap().remove("teacher_move");
        q["history"]=serde_json::json!([31,31]);
        assert!(handle_research_gomoku15_fork_cpu(&q.to_string()).get("error").is_some());
    }
}
