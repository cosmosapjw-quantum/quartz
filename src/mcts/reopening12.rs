//! Research-only Connect4 root candidate ordering and tactical re-opening.
//!
//! Operational analogue of candidate fixation, NOT human Einstellung and NOT
//! perfect-game inference. All evidence uses immediate legal game operations,
//! never Pascal Pons exact external oracle scores or future MCTS rollouts.
use crate::game::GameState;
use crate::games::connect4::Connect4;

pub const TOP_K: usize = 3;
pub const MAX_REOPEN: usize = 2;
pub const FIXATION_MIN_SHARE: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct TacticalProbe {
    pub fixation_share: f32,
    pub fixation_gate: bool,
    pub initial_best_reply_vulnerable: bool,
    pub original_omitted_immediate_wins: Vec<usize>,
    pub original_omitted_safe_moves: Vec<usize>,
    pub suggested_reopen: Vec<usize>,
    pub trigger: bool,
    pub reason: &'static str,
}

/// Sort original UNMASKED legal root prior, descending; exact ties favor
/// center column (3) and then ascending column. Never inspect oracle WDL.
pub fn ordered_legal_prior(
    legal: &[usize],
    original_evaluator_prior: &[(usize, f32)],
) -> Result<Vec<(usize, f32)>, String> {
    if legal.is_empty() {
        return Err("no legal root moves".into());
    }
    let mut seen = [false; 7];
    let mut rank = Vec::with_capacity(legal.len());
    for &(c, p) in original_evaluator_prior {
        if c >= 7 || !legal.contains(&c) || seen[c] || !p.is_finite() || p < 0.0 {
            return Err("malformed original evaluator root prior".into());
        }
        seen[c] = true;
        rank.push((c, p));
    }
    if rank.len() != legal.len() || legal.iter().any(|c| !seen[*c]) {
        return Err("original model omitted a legal root prior".into());
    }
    if rank.iter().map(|(_, p)| *p as f64).sum::<f64>() <= 0. {
        return Err("all evaluator root priors zero".into());
    }
    rank.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then((a.0 as i32 - 3).abs().cmp(&(b.0 as i32 - 3).abs()))
            .then(a.0.cmp(&b.0))
    });
    Ok(rank)
}

pub fn admits_immediate_opponent_win(state: &Connect4, col: usize) -> bool {
    if !state.legal_moves().contains(&col) {
        return false;
    }
    let after = state.apply_move(col);
    if after.is_terminal() {
        return false;
    }
    after
        .legal_moves()
        .iter()
        .copied()
        .any(|reply| after.is_winning_move(reply))
}

pub fn tactical_probe(
    state: &Connect4,
    first_best: usize,
    initial: &[usize],
    ranked: &[usize],
    first_visit_share: f32,
) -> Result<TacticalProbe, String> {
    if initial.is_empty()
        || !initial.contains(&first_best)
        || !first_visit_share.is_finite()
        || !(0.0..=1.0).contains(&first_visit_share)
    {
        return Err("invalid masked first-root choice or observed fixation share".into());
    }
    let legal = state.legal_moves();
    if !initial.iter().all(|c| legal.contains(c)) || !ranked.iter().all(|c| legal.contains(c)) {
        return Err("masked root candidates differ from original legal game".into());
    }
    let omitted: Vec<usize> = ranked
        .iter()
        .copied()
        .filter(|c| !initial.contains(c))
        .collect();
    let vulnerable = admits_immediate_opponent_win(state, first_best);
    let mut wins = Vec::new();
    let mut safe = Vec::new();
    for &col in &omitted {
        if state.is_winning_move(col) {
            wins.push(col)
        } else if !admits_immediate_opponent_win(state, col) {
            safe.push(col)
        }
    }
    let fixation = first_visit_share >= FIXATION_MIN_SHARE;
    let (suggested, reason) = if !fixation {
        (Vec::new(), "fixation_threshold_not_reached")
    } else if !wins.is_empty() {
        (
            wins.iter().copied().take(MAX_REOPEN).collect(),
            "excluded_immediate_win",
        )
    } else if vulnerable && !safe.is_empty() {
        (
            safe.iter().copied().take(MAX_REOPEN).collect(),
            "leader_allows_immediate_opponent_win",
        )
    } else {
        (Vec::new(), "no_tactical_reopening_evidence")
    };
    Ok(TacticalProbe {
        fixation_share: first_visit_share,
        fixation_gate: fixation,
        initial_best_reply_vulnerable: vulnerable,
        original_omitted_immediate_wins: wins,
        original_omitted_safe_moves: safe,
        trigger: !suggested.is_empty(),
        suggested_reopen: suggested,
        reason,
    })
}

/// Deterministic oracle-blind hash for random-reopening control; same number
/// of added root candidates as actual tactical proposal, never outcome labels.
pub fn random_reopen(omitted: &[usize], number: usize, seed: u64, history: &str) -> Vec<usize> {
    let mut ranked = omitted.to_vec();
    ranked.sort_by_key(|col| {
        let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
        for c in history.bytes().chain([*col as u8]) {
            hash ^= c as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        hash
    });
    ranked.into_iter().take(number).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_ties_rank_center_3_2_4_and_include_all() {
        let p = vec![(0, 1. / 7.); 7];
        let p: Vec<(usize, f32)> = p
            .into_iter()
            .enumerate()
            .map(|(c, (_, v))| (c, v))
            .collect();
        let ranks = ordered_legal_prior(&[0, 1, 2, 3, 4, 5, 6], &p).unwrap();
        assert_eq!(
            ranks.iter().take(3).map(|(c, _)| *c).collect::<Vec<_>>(),
            vec![3, 2, 4]
        );
        assert!(ordered_legal_prior(&[0, 1, 2], &[(0, 0.5), (1, 0.5)]).is_err());
    }
    #[test]
    fn actual_connect4_omitted_vertical_winning_action_triggered_by_legal_probe() {
        let root = Connect4::from_history("121213").unwrap();
        assert!(root.is_winning_move(0));
        let initial = [3, 2, 4];
        let ranked = [3, 2, 4, 1, 5, 0, 6];
        let p = tactical_probe(&root, 3, &initial, &ranked, 0.75).unwrap();
        assert!(p.trigger);
        assert_eq!(p.reason, "excluded_immediate_win");
        assert_eq!(p.suggested_reopen, vec![0]);
        let not_fixated = tactical_probe(&root, 3, &initial, &ranked, 0.1).unwrap();
        assert!(!not_fixated.trigger);
    }
    #[test]
    fn oracle_blind_random_ranks_are_deterministic_and_limited() {
        let a = random_reopen(&[0, 1, 5, 6], 2, 17, "121213");
        let b = random_reopen(&[0, 1, 5, 6], 2, 17, "121213");
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|v| [0, 1, 5, 6].contains(v)));
    }
}
