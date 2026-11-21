//! Research-only 15x15 Freestyle MCTS true same-root high-branch experiment.
//! Rapfi NNUE is external posthoc teacher: NO Rapfi output enters this server.
//! The original evaluator is Uniform (not pretrained); source top16 geometry
//! is an artificial attention prior and is disclosed to consumers.
use super::*;
use crate::games::gomoku15::{Gomoku15, GomokuVariant};
use crate::mcts::node::ArenaRef;
use std::sync::atomic::Ordering;

fn root_edges(engine: &MctsEngine<Gomoku15>) -> Vec<serde_json::Value> {
    engine
        .root
        .edge_snapshot(engine.root.materialized_count())
        .iter()
        .map(|e| serde_json::json!([e.mv, e.n, e.p, e.w, e.m2]))
        .collect()
}
fn edge_sum(engine: &MctsEngine<Gomoku15>) -> u32 {
    engine
        .root
        .edge_snapshot(engine.root.materialized_count())
        .iter()
        .map(|e| e.n)
        .sum()
}
fn area_rank(a: u16) -> (i32, i32, u16) {
    let x = (a % 15) as i32;
    let y = (a / 15) as i32;
    (
        (x - 7).abs().max((y - 7).abs()),
        (x - 7).abs() + (y - 7).abs(),
        a,
    )
}
fn opp_immediate(game: &Gomoku15, a: u16) -> bool {
    if !game.legal_moves().contains(&a) {
        return false;
    }
    let next = game.apply_move(a);
    if next.is_terminal() {
        return false;
    }
    next.legal_moves()
        .iter()
        .copied()
        .any(|m| next.is_winning_move(m))
}
fn proposed_random(omitted: &[u16], n: usize, seed: u64, hist: &[u16]) -> Vec<u16> {
    let mut result = omitted.to_vec();
    result.sort_by_key(|col| {
        let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
        for b in hist.iter().map(|x| *x as u8).chain([*col as u8]) {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        hash
    });
    result.into_iter().take(n).collect()
}

pub(super) fn handle_research_gomoku15_highbranch_cpu(line: &str) -> serde_json::Value {
    let t0 = Instant::now();
    let q: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error":format!("bad highbranch14 JSON: {e}")}),
    };
    if [
        "oracle",
        "teacher_move",
        "teacher_WDL",
        "Rapfi_move",
        "exact_action_scores",
    ]
    .iter()
    .any(|key| q.get(*key).is_some())
    {
        return serde_json::json!({"error":"external teacher or oracle forbidden in online MCTS"});
    }
    if q.get("rule").and_then(|v| v.as_str()) != Some("freestyle") {
        return serde_json::json!({"error":"C14 online science is Freestyle15 only, not Renju"});
    }
    let Some(history) = q.get("history").and_then(|v| v.as_array()) else {
        return serde_json::json!({"error":"missing sequential flat 15x15 move history"});
    };
    let Some(arm) = q.get("arm").and_then(|v| v.as_str()) else {
        return serde_json::json!({"error":"missing experimental arm"});
    };
    if !matches!(
        arm,
        "closed"
            | "evidence"
            | "random_1"
            | "random_2"
            | "random_3"
            | "all_after8"
            | "full32"
            | "closed_no_probe"
    ) {
        return serde_json::json!({"error":"invalid registered C14 attention controller"});
    }
    let mut game = Gomoku15::new(GomokuVariant::Freestyle);
    let mut hist = Vec::with_capacity(history.len());
    for v in history {
        let Some(a) = v.as_u64() else {
            return serde_json::json!({"error":"non-integer original game move"});
        };
        if a >= 225 || !game.legal_moves().contains(&(a as u16)) {
            return serde_json::json!({"error":"out-of-board or illegal source move"});
        }
        game = game.apply_move(a as u16);
        hist.push(a as u16);
        if game.is_terminal() {
            return serde_json::json!({"error":"source history already ended the game"});
        }
    }
    let legal = game.legal_moves();
    if legal.len() < 16 {
        return serde_json::json!({"error":"C14 source outside the frozen high-branch window"});
    }
    let first_seed = q.get("seed").and_then(|v| v.as_u64()).unwrap_or(17);
    if first_seed != 17 {
        return serde_json::json!({"error":"nonfrozen C14 MCTS seed"});
    }
    let prior_t = Instant::now();
    let source_evaluator = UniformEval;
    let original = source_evaluator.evaluate(&game);
    let prior_us = prior_t.elapsed().as_micros();
    if original.policy.len() != legal.len() {
        return serde_json::json!({"error":"original full legal prior missing action"});
    }
    let mut ordered = legal.clone();
    ordered.sort_by_key(|a| area_rank(*a));
    let initial: Vec<u16> = ordered.iter().take(16).copied().collect();
    let omitted: Vec<u16> = ordered.iter().skip(16).copied().collect();
    let initial_mask = if arm == "full32" {
        legal.clone()
    } else {
        initial.clone()
    };

    let mut cfg = MctsConfig::evaluation(2.0).with_seed(first_seed);
    cfg.root_forced_win = false;
    cfg.dirichlet = None;
    cfg.temperature = 0.0;
    let setup_t = Instant::now();
    let mut engine = MctsEngine::new(game.clone(), Arc::new(UniformEval), cfg);
    let setup_us = setup_t.elapsed().as_micros();
    if engine.root.candidate_count() != legal.len()
        || engine.root.materialized_count() != legal.len()
    {
        return serde_json::json!({"error":"root full original 15x15 slab was not allocated"});
    }
    let pointer = ArenaRef::as_ptr(&engine.root);
    let original_hash = engine.root.hash;
    let original_prior = engine.root.edge_priors_snapshot(legal.len());
    let mask_t = Instant::now();
    if arm != "full32" {
        if let Err(e) = engine.set_resident_root_allowed_moves(&initial) {
            return serde_json::json!({"error":format!("original root attention: {e}")});
        }
    }
    let first_mask_us = mask_t.elapsed().as_micros();
    let mask_initial = engine.resident_root_mask_words_snapshot();
    let search1_t = Instant::now();
    let first = engine.run(&mut FixedIterations::new(8));
    let first_us = search1_t.elapsed().as_micros();
    if first.root_visits != 8 || edge_sum(&engine) != 8 {
        return serde_json::json!({"error":"first8 root visits failed"});
    }
    let Some(first_best) = engine.best_move() else {
        return serde_json::json!({"error":"no first root decision"});
    };
    if !initial_mask.contains(&first_best) {
        return serde_json::json!({"error":"disabled first-root candidate was selected"});
    }
    let first_policy = engine.pi_target(1.);
    let first_edges = root_edges(&engine);
    let first_calls = engine.evaluator_call_count();
    let share = first_policy.iter().fold(0_f32, |n, (_, p)| n.max(*p));

    let probe_t = Instant::now();
    let mut reason = "no_probe";
    let mut suggested = Vec::new();
    let mut witness_wins = Vec::new();
    let mut initial_best_vulnerable = false;
    if arm != "closed_no_probe" {
        witness_wins = omitted
            .iter()
            .copied()
            .filter(|a| game.is_winning_move(*a))
            .collect();
        if !witness_wins.is_empty() {
            suggested = witness_wins.iter().copied().take(4).collect();
            reason = "omitted_legal_immediate_five";
        } else {
            let leader = if initial.contains(&first_best) {
                first_best
            } else {
                initial[0]
            };
            initial_best_vulnerable = opp_immediate(&game, leader);
            if share >= 0.5 && initial_best_vulnerable {
                for &a in &omitted {
                    if suggested.len() == 4 {
                        break;
                    }
                    if !opp_immediate(&game, a) {
                        suggested.push(a)
                    }
                }
                if !suggested.is_empty() {
                    reason = "oneply_safe_after_fixation"
                } else {
                    reason = "no_safe_omitted_action"
                }
            } else {
                reason = "no_tactical_witness"
            }
        }
    }
    let probe_us = probe_t.elapsed().as_micros();
    let added = match arm {
        "evidence" => suggested.clone(),
        "random_1" => proposed_random(&omitted, suggested.len(), 202611210411, &hist),
        "random_2" => proposed_random(&omitted, suggested.len(), 202611210412, &hist),
        "random_3" => proposed_random(&omitted, suggested.len(), 202611210413, &hist),
        "all_after8" => omitted.clone(),
        _ => Vec::new(),
    };
    let mut final_mask_actions = initial_mask.clone();
    for &a in &added {
        if !final_mask_actions.contains(&a) {
            final_mask_actions.push(a)
        }
    }
    let activation_t = Instant::now();
    if arm != "full32" {
        if let Err(e) = engine.set_resident_root_allowed_moves(&final_mask_actions) {
            return serde_json::json!({"error":format!("resident C14 activation: {e}")});
        }
    }
    let activation_us = activation_t.elapsed().as_micros();
    let after_edges = root_edges(&engine);
    let after_mask = engine.resident_root_mask_words_snapshot();
    let same_stats = (first_edges == after_edges);
    let same_root = ArenaRef::as_ptr(&engine.root) == pointer
        && engine.root.hash == original_hash
        && engine.root_state().tt_hash() == original_hash;
    let same_evals = engine.evaluator_call_count() == first_calls;
    if !same_root
        || !same_stats
        || !same_evals
        || edge_sum(&engine) != 8
        || engine.root.edge_priors_snapshot(legal.len()) != original_prior
    {
        return serde_json::json!({"error":"C14 root state or edge statistics changed on activation"});
    }
    let search2_t = Instant::now();
    let second = engine.run(&mut FixedIterations::new(32));
    let second_us = search2_t.elapsed().as_micros();
    let Some(final_move) = engine.best_move() else {
        return serde_json::json!({"error":"no completed 32-visit final action"});
    };
    if second.root_visits != 32
        || second.iterations != 24
        || edge_sum(&engine) != 32
        || !final_mask_actions.contains(&final_move)
    {
        return serde_json::json!({"error":"32-visit same-tree continuation failed"});
    }
    let second_edges = root_edges(&engine);
    let final_calls = engine.evaluator_call_count();
    let elapsed = t0.elapsed().as_micros();
    serde_json::json!({
      "schema":"quartz_cog04_highbranch14_real_same_tree_freestyle_v1",
      "history":hist,"rule":"freestyle","side_to_move":game.current_player(),
      "root_full_original_legal_count":legal.len(),
      "legal_actions":legal,"source_prior_type":"UniformEval_with_separate_center_rank_gate",
      "prior_source_model_value":original.value,
      "original_uniform_prior":original.policy,
      "prior_center_rank":ordered,
      "initial_candidates":initial,"initial_active_actions":initial_mask,
      "added_candidates":added,"final_active_actions":final_mask_actions,
      "root_mask_initial":mask_initial,"root_mask_final":after_mask,
      "root_hash_before":format!("{original_hash:016x}"),
      "root_hash_after":format!("{:016x}",engine.root.hash),
      "same_root_identity_verified":same_root,
      "edge_stats_preserved_at_activation":same_stats,
      "no_root_evaluator_call_on_mask_change":same_evals,
      "root_prior_preserved":true,
      "first_stage":{
        "root_visits":first.root_visits,"best_action":first_best,
        "root_policy":first_policy,"edge_stats":first_edges,
        "evaluator_calls":first_calls,"search_wall_us":first_us,
        "edge_visit_sum":8
      },
      "activation_edge_stats":after_edges,
      "tactical_probe":{
        "reason":reason,"first_root_visit_share":share,
        "initial_best_vulnerable_to_oneply_reply":initial_best_vulnerable,
        "omitted_legal_immediate_wins":witness_wins,
        "suggested_reopen":suggested
      },
      "second_stage":{
        "root_visits":second.root_visits,
        "additional_root_visits":second.iterations,
        "best_action":final_move,
        "edge_visit_sum":32,"evaluator_calls":final_calls,
        "edge_stats":second_edges,"search_wall_us":second_us
      },
      "final_legal_action":final_move,
      "cost":{
        "original_prior_wall_us":prior_us,
        "full_root_setup_wall_us":setup_us,
        "first_mask_wall_us":first_mask_us,
        "search8_wall_us":first_us,
        "tactical_probe_wall_us":probe_us,
        "activation_wall_us":activation_us,
        "search24_extra_wall_us":second_us,
        "pure_MCTS_search_wall_us":first_us+second_us
      },
      "total_wall_us_without_output_serialization":elapsed,
      "total_evaluator_calls_including_manual_original_prior":final_calls+1,
      "no_Rapfi_teacher_data_online":true,
      "original_complete_root_edges_preallocated":true,
      "root_full_not_late_edge_insertion":true
    })
}
