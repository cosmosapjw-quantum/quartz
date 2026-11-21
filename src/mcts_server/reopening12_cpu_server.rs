//! Research-only masked root candidate reopening endpoint, CPU real Rust MCTS.
//! Every arm uses two INDEPENDENT tree searches 8+16, never a false claim of
//! same-tree expansion; CPU records root setups, full legal prior and probe.
use super::*;

pub(super) fn handle_research_connect4_reopen_cpu(line: &str) -> serde_json::Value {
    let started = Instant::now();
    let req: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error":format!("invalid COG03 JSON: {e}")}),
    };
    let Some(history) = req.get("history").and_then(|x| x.as_str()) else {
        return serde_json::json!({"error":"missing Connect4 history"});
    };
    let state = match Connect4::from_history(history) {
        Ok(s) if !s.is_terminal() => s,
        Ok(_) => return serde_json::json!({"error":"terminal COG03 root"}),
        Err(e) => return serde_json::json!({"error":e}),
    };
    let legal = state.legal_moves();
    if legal.is_empty() {
        return serde_json::json!({"error":"no legal Connect4 root actions"});
    }
    let Some(arm) = req.get("arm").and_then(|x| x.as_str()) else {
        return serde_json::json!({"error":"missing research reopening arm"});
    };
    if !matches!(
        arm,
        "closed_sham"
            | "evidence_reopen"
            | "random_reopen"
            | "always_full_reopen"
            | "plain_full"
            | "closed_no_probe"
    ) {
        return serde_json::json!({"error":"unknown COG03 experimental arm"});
    }
    let mode = req
        .get("eval_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("uniform");
    let mode_label = match mode {
        "uniform" => "uniform_no_neural_network",
        "tiny_mlp" => "trained_tiny_mlp_cpu",
        "conv3x3" => "trained_conv3x3_cpu",
        _ => return serde_json::json!({"error":"unknown COG03 evaluator"}),
    };
    let seed = req.get("seed").and_then(|v| v.as_u64()).unwrap_or(17);
    let reopening_seed = req
        .get("reopen_seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(202611210201);
    if req.get("seed").is_some() && req.get("seed").and_then(|v| v.as_u64()).is_none() {
        return serde_json::json!({"error":"seed must be nonnegative integer"});
    }
    if req.get("reopen_seed").is_some() && req.get("reopen_seed").and_then(|v| v.as_u64()).is_none()
    {
        return serde_json::json!({"error":"reopen_seed must be nonnegative integer"});
    }
    // Explicitly reject external oracle/posthoc labels in online requests.
    if req.get("exact_action_scores").is_some() || req.get("oracle").is_some() {
        return serde_json::json!({"error":"no oracle values in live reopening endpoint"});
    }
    let cache = req
        .get("cache_model")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let load_t = Instant::now();
    let (eval, hit) = match model_for_root(
        mode,
        req.get("model_path").and_then(|v| v.as_str()),
        req.get("model_identity").and_then(|v| v.as_str()),
        cache,
    ) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error":format!("model load: {e}")}),
    };
    let model_load_us = load_t.elapsed().as_micros();
    // A real full-legal neural inference determines initial attention. This is
    // additional measured CPU work; no external exact game solver is called.
    let prior_t = Instant::now();
    let original_priors = eval.evaluate(&state);
    let prior_us = prior_t.elapsed().as_micros();
    let ordered = match ordered_legal_prior(&legal, &original_priors.policy) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error":format!("root prior: {e}")}),
    };
    let initial: Vec<usize> = ordered.iter().take(TOP_K).map(|(a, _)| *a).collect();
    let omitted: Vec<usize> = ordered.iter().skip(TOP_K).map(|(a, _)| *a).collect();
    let full_rank: Vec<usize> = ordered.iter().map(|(a, _)| *a).collect();
    let first_allowed = if arm == "plain_full" {
        legal.clone()
    } else {
        initial.clone()
    };

    let mut cfg = MctsConfig::evaluation(2.0).with_seed(seed);
    cfg.root_forced_win = false;
    cfg.dirichlet = None;
    cfg.temperature = 0.0;
    let first_game = if first_allowed.len() == legal.len() {
        state.clone()
    } else {
        match state.clone().with_root_candidates(&first_allowed) {
            Ok(s) => s,
            Err(e) => return serde_json::json!({"error":format!("first root gate: {e}")}),
        }
    };
    let first_setup_t = Instant::now();
    let engine1 = MctsEngine::new(first_game, eval.clone(), cfg.clone());
    let first_setup_us = first_setup_t.elapsed().as_micros();
    let first_hash = format!("{:016x}", engine1.root_state().tt_hash());
    let first_start = Instant::now();
    let s1 = engine1.run(&mut FixedIterations::new(8));
    let first_search_us = first_start.elapsed().as_micros();
    if s1.root_visits != 8 {
        return serde_json::json!({"error":"first MCTS incomplete","visits":s1.root_visits});
    }
    let Some(first_best) = engine1.best_move() else {
        return serde_json::json!({"error":"no first root action"});
    };
    let first_pi = engine1.pi_target(1.0);
    let first_calls = engine1.evaluator_call_count();
    let fixation_share = first_pi.iter().map(|x| x.1).fold(0f32, f32::max);

    let probe_started = Instant::now();
    let probe = if arm == "closed_no_probe" {
        None
    } else {
        // "plain_full" is an unmasked control; only project its leader to
        // original top3 for the dummy tactical-probe clock overhead.
        let leader = if initial.contains(&first_best) {
            first_best
        } else {
            initial[0]
        };
        match tactical_probe(&state, leader, &initial, &full_rank, fixation_share) {
            Ok(q) => Some(q),
            Err(e) => return serde_json::json!({"error":format!("one-ply tactical probe: {e}")}),
        }
    };
    let probe_us = probe_started.elapsed().as_micros();
    let evidence = probe
        .as_ref()
        .map(|x| x.suggested_reopen.clone())
        .unwrap_or_default();
    let proposed = match arm {
        "evidence_reopen" => evidence.clone(),
        "random_reopen" => random_reopen(&omitted, evidence.len(), reopening_seed, history),
        "always_full_reopen" => omitted.clone(),
        _ => Vec::new(),
    };
    let mut second_allowed = if arm == "plain_full" {
        legal.clone()
    } else {
        initial.clone()
    };
    for mv in &proposed {
        if !second_allowed.contains(mv) {
            second_allowed.push(*mv);
        }
    }
    // Genuine fresh root engine. The original tree cannot currently insert
    // new root edges. Count both root expansions and all 8+16 MCTS iterations.
    let second_game = if second_allowed.len() == legal.len() {
        state.clone()
    } else {
        match state.clone().with_root_candidates(&second_allowed) {
            Ok(s) => s,
            Err(e) => return serde_json::json!({"error":format!("fresh root gate: {e}")}),
        }
    };
    let second_setup_t = Instant::now();
    let engine2 = MctsEngine::new(second_game, eval.clone(), cfg);
    let second_setup_us = second_setup_t.elapsed().as_micros();
    let second_hash = format!("{:016x}", engine2.root_state().tt_hash());
    let second_start = Instant::now();
    let s2 = engine2.run(&mut FixedIterations::new(16));
    let second_search_us = second_start.elapsed().as_micros();
    if s2.root_visits != 16 {
        return serde_json::json!({"error":"fresh second MCTS incomplete","visits":s2.root_visits});
    }
    let Some(final_best) = engine2.best_move() else {
        return serde_json::json!({"error":"no reopened root action"});
    };
    let second_pi = engine2.pi_target(1.0);
    let second_calls = engine2.evaluator_call_count();
    let total_request_us = started.elapsed().as_micros();
    let decision_components_us =
        prior_us + first_setup_us + first_search_us + probe_us + second_setup_us + second_search_us;
    serde_json::json!({
        "schema":"quartz_cog03_reopening12_actual_cpu_root_restart_v1",
        "history":history,"board_bottom_up":state.board_flat().to_vec(),
        "side_to_move":state.current_player(),"ply":state.ply_count(),
        "seed":seed,"reopen_seed":reopening_seed,
        "arm":arm,"eval_mode":mode_label,
        "cache_model":cache,"model_cache_hit":hit,
        "legal_columns_0based":legal,
        "unmasked_evaluator_prior_rank":ordered,
        "unmasked_initial_evaluator_value":original_priors.value,
        "initial_candidates":initial,"initially_omitted":omitted,
        "final_candidates":second_allowed,"actually_added_candidates":proposed,
        "tactical_probe":probe.as_ref().map(|p|serde_json::json!({
            "fixation_share":p.fixation_share,"fixation_gate":p.fixation_gate,
            "initial_best_reply_vulnerable":p.initial_best_reply_vulnerable,
            "excluded_immediate_wins":p.original_omitted_immediate_wins,
            "excluded_safe_actions":p.original_omitted_safe_moves,
            "suggested_reopen":p.suggested_reopen,
            "trigger":p.trigger,"reason":p.reason
        })),
        "first_search":{
            "root_visits":8,"best_action":first_best,"root_hash":first_hash,
            "root_policy":first_pi,"root_evaluator_calls":first_calls,
            "setup_wall_us":first_setup_us,"search_wall_us":first_search_us,
        },
        "second_search":{
            "root_visits":16,"best_action":final_best,"root_hash":second_hash,
            "root_policy":second_pi,"root_evaluator_calls":second_calls,
            "setup_wall_us":second_setup_us,"search_wall_us":second_search_us,
        },
        "model_load_wall_us":model_load_us,
        "prior_inference_wall_us":prior_us,"probe_wall_us":probe_us,
        "sum_both_root_search_wall_us":first_search_us+second_search_us,
        "total_decision_components_wall_us":decision_components_us,
        "total_decision_wall_us_without_model_load":total_request_us.saturating_sub(model_load_us),
        "total_request_wall_us_including_model_load":total_request_us,
        "total_evaluator_calls_with_prior":1+first_calls+second_calls,
        "no_online_exact_game_oracle":true,
        "MCTS_restart_instead_of_same_tree_reopening":true
    })
}
