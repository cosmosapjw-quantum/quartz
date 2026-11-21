//! Actual same-root resident candidate reactivation for CPU Connect Four.
//!
//! All 7 legal root edges are allocated before search; opt-in gate controls
//! selection at depth 0 only. At checkpoint eight, change only the monotone
//! gate, without deleting/retiming any edge's N/W/M2 or evaluator calls.
//! This is ACTIVE EDGE REACTIVATION, not late arena slab insertion.
use super::*;
use std::sync::atomic::Ordering;

fn edge_state(engine: &MctsEngine<Connect4>) -> Vec<serde_json::Value> {
    engine
        .root
        .edge_snapshot(engine.root.materialized_count())
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "action":e.mv,"prior":e.p,"visits":e.n,
                "w":e.w,"m2":e.m2
            })
        })
        .collect()
}

fn sum_root_visits(engine: &MctsEngine<Connect4>) -> u32 {
    engine
        .root
        .edge_snapshot(engine.root.materialized_count())
        .iter()
        .map(|e| e.n)
        .sum()
}

pub(super) fn handle_research_connect4_resident_cpu(line: &str) -> serde_json::Value {
    let started = Instant::now();
    let req: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({"error":format!("invalid resident research JSON: {e}")})
        }
    };
    if req.get("oracle").is_some()
        || req.get("exact_action_scores").is_some()
        || req.get("teacher_WDL").is_some()
    {
        return serde_json::json!({"error":"external exact WDL is forbidden inside live resident controller"});
    }
    let Some(history) = req.get("history").and_then(|x| x.as_str()) else {
        return serde_json::json!({"error":"missing original Connect4 root history"});
    };
    let game = match Connect4::from_history(history) {
        Ok(g) if !g.is_terminal() => g,
        Ok(_) => return serde_json::json!({"error":"cannot search terminal root"}),
        Err(e) => return serde_json::json!({"error":e}),
    };
    let legal = game.legal_moves();
    let Some(arm) = req.get("arm").and_then(|x| x.as_str()) else {
        return serde_json::json!({"error":"missing registered resident research arm"});
    };
    if !matches!(
        arm,
        "resident_closed_sham"
            | "resident_evidence"
            | "resident_random_1"
            | "resident_random_2"
            | "resident_random_3"
            | "resident_all_after8"
            | "resident_full24"
            | "resident_closed_no_probe"
    ) {
        return serde_json::json!({"error":"unknown resident-13 experimental arm"});
    }
    let seed = match req.get("seed") {
        None => 17,
        Some(v) => match v.as_u64() {
            Some(n) => n,
            None => return serde_json::json!({"error":"invalid MCTS search seed"}),
        },
    };
    let mode = req
        .get("eval_mode")
        .and_then(|x| x.as_str())
        .unwrap_or("uniform");
    let mode_label = match mode {
        "uniform" => "uniform_no_neural_network",
        "tiny_mlp" => "trained_tiny_mlp_cpu",
        "conv3x3" => "trained_conv3x3_cpu",
        _ => return serde_json::json!({"error":"invalid trained CPU evaluator mode"}),
    };
    let cache = req
        .get("cache_model")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let load_t = Instant::now();
    let (evaluator, hit) = match model_for_root(
        mode,
        req.get("model_path").and_then(|x| x.as_str()),
        req.get("model_identity").and_then(|x| x.as_str()),
        cache,
    ) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error":format!("model source: {e}")}),
    };
    let load_us = load_t.elapsed().as_micros();

    // A legal full-root source evaluator prior is computed before the gate
    // for ALL arms, including all-legal control. It is physically timed.
    let prior_t = Instant::now();
    let p0 = evaluator.evaluate(&game);
    let prior_us = prior_t.elapsed().as_micros();
    let ranked = match ordered_legal_prior(&legal, &p0.policy) {
        Ok(x) => x,
        Err(e) => return serde_json::json!({"error":format!("original full root prior: {e}")}),
    };
    let initial: Vec<usize> = ranked.iter().take(TOP_K).map(|(c, _)| *c).collect();
    let omitted: Vec<usize> = ranked.iter().skip(TOP_K).map(|(c, _)| *c).collect();
    let full_rank: Vec<usize> = ranked.iter().map(|(c, _)| *c).collect();

    let mut cfg = MctsConfig::evaluation(2.0).with_seed(seed);
    cfg.root_forced_win = false;
    cfg.dirichlet = None;
    cfg.temperature = 0.;
    // Unlike the previous experiment, this is the ORIGINAL fully legal
    // game, with a single immutable root/TT hash for the entire search.
    let root_setup_t = Instant::now();
    let mut engine = MctsEngine::new(game.clone(), evaluator, cfg);
    let setup_us = root_setup_t.elapsed().as_micros();
    if engine.root.candidate_count() != legal.len()
        || engine.root.materialized_count() != legal.len()
    {
        return serde_json::json!({"error":"original full legal root edge slab was not materialized"});
    }
    let original_hash = engine.root.hash;
    let original_ptr = crate::mcts::node::ArenaRef::as_ptr(&engine.root);
    let baseline_edges = edge_state(&engine);
    let mut init_mask_us = 0;
    let first_active = if arm == "resident_full24" {
        legal.clone()
    } else {
        let t = Instant::now();
        if let Err(e) = engine.set_resident_root_allowed_moves(&initial) {
            return serde_json::json!({"error":format!("initial resident root gate: {e}")});
        }
        init_mask_us = t.elapsed().as_micros();
        initial.clone()
    };

    let search1_t = Instant::now();
    let first_stats = engine.run(&mut FixedIterations::new(8));
    let first_us = search1_t.elapsed().as_micros();
    if first_stats.root_visits != 8 || sum_root_visits(&engine) != 8 {
        return serde_json::json!({"error":"first resident MCTS root visit invariant failed"});
    }
    let Some(first_best) = engine.best_move() else {
        return serde_json::json!({"error":"first root action missing"});
    };
    let first_visit_policy = engine.pi_target(1.);
    let first_edges = edge_state(&engine);
    let first_eval_calls = engine.evaluator_call_count();
    let first_mask = engine.resident_root_mask_snapshot();
    if !first_active.contains(&first_best) {
        return serde_json::json!({"error":"initial masked root selected an inactive move"});
    }
    // A true game-rule-only tactical probe, including all omitted legal
    // columns. No exact solver, future visit or learned posthoc label used.
    let probe_t = Instant::now();
    let probe = if arm == "resident_closed_no_probe" {
        None
    } else {
        let leader = if initial.contains(&first_best) {
            first_best
        } else {
            initial[0]
        };
        let max_share = first_visit_policy.iter().map(|x| x.1).fold(0_f32, f32::max);
        match tactical_probe(&game, leader, &initial, &full_rank, max_share) {
            Ok(p) => Some(p),
            Err(e) => return serde_json::json!({"error":format!("current one-ply evidence: {e}")}),
        }
    };
    let probe_us = probe_t.elapsed().as_micros();
    let evidence = probe
        .as_ref()
        .map(|p| p.suggested_reopen.clone())
        .unwrap_or_default();
    let additions: Vec<usize> = match arm {
        "resident_evidence" => evidence.clone(),
        "resident_random_1" => random_reopen(&omitted, evidence.len(), 202611210301, history),
        "resident_random_2" => random_reopen(&omitted, evidence.len(), 202611210302, history),
        "resident_random_3" => random_reopen(&omitted, evidence.len(), 202611210303, history),
        "resident_all_after8" => omitted.clone(),
        _ => Vec::new(),
    };
    let mut final_active = if arm == "resident_full24" {
        legal.clone()
    } else {
        initial.clone()
    };
    for &c in &additions {
        if !final_active.contains(&c) {
            final_active.push(c)
        }
    }
    let mask_t = Instant::now();
    if arm != "resident_full24" {
        if let Err(e) = engine.set_resident_root_allowed_moves(&final_active) {
            return serde_json::json!({"error":format!("resident root unpruning: {e}")});
        }
    }
    let activation_us = mask_t.elapsed().as_micros();
    let after_activation_edges = edge_state(&engine);
    let old_counters_and_priors_equal = first_edges == after_activation_edges;
    let old_evaluator_calls_equal = engine.evaluator_call_count() == first_eval_calls;
    let same_node = crate::mcts::node::ArenaRef::as_ptr(&engine.root) == original_ptr;
    let same_hash =
        engine.root.hash == original_hash && engine.root_state().tt_hash() == original_hash;
    let after_activation_visits = engine.root.n_total.load(Ordering::Acquire);
    if !same_node
        || !same_hash
        || !old_counters_and_priors_equal
        || !old_evaluator_calls_equal
        || after_activation_visits != 8
    {
        return serde_json::json!({"error":"resident activation mutated old root identity or statistics"});
    }
    if additions.iter().any(|a| {
        first_edges.iter().any(|edge| {
            edge["action"].as_u64() == Some(*a as u64) && edge["visits"].as_u64() != Some(0)
        })
    }) {
        return serde_json::json!({"error":"activated root action already had a masked-period visit"});
    }
    let hash_after = engine.root.hash;
    // This runs SIXTEEN additional iterations on the SAME root node and
    // preserves every child/path/TT from the completed first eight.
    let search2_t = Instant::now();
    let second_stats = engine.run(&mut FixedIterations::new(24));
    let second_us = search2_t.elapsed().as_micros();
    if second_stats.root_visits != 24
        || second_stats.iterations != 16
        || sum_root_visits(&engine) != 24
    {
        return serde_json::json!({"error":"resident continuation did not add exactly sixteen root visits"});
    }
    let Some(second_best) = engine.best_move() else {
        return serde_json::json!({"error":"resident second root action missing"});
    };
    if !final_active.contains(&second_best) {
        return serde_json::json!({"error":"final selected root action not activated"});
    }
    let second_pi = engine.pi_target(1.);
    let after_edges = edge_state(&engine);
    let n_evals = engine.evaluator_call_count();
    let end_to_end = started.elapsed().as_micros();
    let components =
        prior_us + setup_us + init_mask_us + first_us + probe_us + activation_us + second_us;
    serde_json::json!({
        "schema":"quartz_cog03_resident13_actual_same_tree_root_activation_v1",
        "history":history,"ply":game.ply_count(),
        "board_bottom_up":game.board_flat().to_vec(),
        "side_to_move":game.current_player(),
        "seed":seed,"eval_mode":mode_label,"arm":arm,
        "cache_model":cache,"model_cache_hit":hit,
        "legal_columns_0based":legal,
        "full_unmasked_prior":ranked,
        "prior_source_value":p0.value,
        "initial_candidates":initial,"initially_omitted":omitted,
        "first_active_candidates":first_active,
        "final_candidates":final_active,
        "actually_added_candidates":additions,
        "first_root_action_mask":first_mask,
        "final_root_action_mask":engine.resident_root_mask_snapshot(),
        "root_hash_before":format!("{:016x}",original_hash),
        "root_hash_after":format!("{:016x}",hash_after),
        "resident_same_root_node":same_node,
        "prior_and_edge_stats_identical_immediately_after_activation":old_counters_and_priors_equal,
        "root_evaluator_calls_identical_during_activation":old_evaluator_calls_equal,
        "original_full_root_edges":baseline_edges.len(),
        "first_search":{
            "root_visits":first_stats.root_visits,
            "best_action":first_best,
            "root_policy":first_visit_policy,
            "edge_visit_sum":first_edges.iter().map(|e|e["visits"].as_u64().unwrap_or(0)).sum::<u64>(),
            "edge_stats":first_edges,
            "root_evaluator_calls":first_eval_calls,
            "search_wall_us":first_us
        },
        "activation_edge_stats":after_activation_edges,
        "root_visits_immediately_after_activation":after_activation_visits,
        "tactical_probe":probe.as_ref().map(|p|serde_json::json!({
            "trigger":p.trigger,"reason":p.reason,
            "fixation_share":p.fixation_share,
            "fixation_gate":p.fixation_gate,
            "original_omitted_immediate_wins":p.original_omitted_immediate_wins,
            "original_omitted_safe_moves":p.original_omitted_safe_moves,
            "initial_best_reply_vulnerable":p.initial_best_reply_vulnerable,
            "suggested_reopen":p.suggested_reopen
        })),
        "second_search":{
            "root_visits":second_stats.root_visits,
            "additional_root_visits":second_stats.iterations,
            "best_action":second_best,
            "root_policy":second_pi,
            "edge_visit_sum":after_edges.iter().map(|e|e["visits"].as_u64().unwrap_or(0)).sum::<u64>(),
            "edge_stats":after_edges,
            "root_evaluator_calls":n_evals,
            "search_wall_us":second_us
        },
        "cost":{
            "model_load_wall_us":load_us,
            "prior_source_eval_wall_us":prior_us,
            "full_root_setup_wall_us":setup_us,
            "initial_mask_wall_us":init_mask_us,
            "first_eight_search_wall_us":first_us,
            "one_ply_probe_wall_us":probe_us,
            "activation_mask_wall_us":activation_us,
            "next_sixteen_search_wall_us":second_us,
            "measured_components_wall_us":components,
            "two_stage_pure_search_wall_us":first_us+second_us
        },
        "total_wall_us_without_model_load":end_to_end.saturating_sub(load_us),
        "total_wall_us_including_model_load":end_to_end,
        "total_evaluator_calls_including_prior_source":n_evals+1,
        "original_full_root_prior_preserved":true,
        "game_rules_and_child_legality_unmodified":true,
        "no_external_exact_oracle_in_runtime":true,
        "single_tree_root_activation_not_late_edge_insertion":true
    })
}
