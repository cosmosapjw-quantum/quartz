import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVAL_MODULES = [
    _load_module("quartz_eval_module", "quartz/evaluation.py"),
]


class ToyGame:
    def __init__(self):
        self._board = [0, 0, 0, 0]
        self._player = 0
        self._terminal = False
        self._outcome = None

    def clone(self):
        other = ToyGame()
        other._board = self._board[:]
        other._player = self._player
        other._terminal = self._terminal
        other._outcome = self._outcome
        return other

    def apply_move(self, action):
        marker = 1 if self._player == 0 else -1
        self._board[action] = marker
        if not any(v == 0 for v in self._board):
            black_score = self._board[0] + self._board[2]
            white_score = -(self._board[1] + self._board[3])
            if black_score > white_score:
                self._outcome = 1.0
            elif black_score < white_score:
                self._outcome = -1.0
            else:
                self._outcome = 0.0
            self._terminal = True
        self._player = 1 - self._player

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def current_player(self):
        return self._player

    def legal_moves(self):
        if self._terminal:
            return []
        return [idx for idx, value in enumerate(self._board) if value == 0]


class FirstMoveEngine:
    def __init__(self, name: str):
        self._name = name

    def select_move(self, state):
        return state.legal_moves()[0], {"time_used_ms": 0, "simulations": 1}

    def reset(self):
        return None

    def name(self):
        return self._name


class LastMoveEngine:
    def __init__(self, name: str):
        self._name = name

    def select_move(self, state):
        return state.legal_moves()[-1], {"time_used_ms": 0, "simulations": 1}

    def reset(self):
        return None

    def name(self):
        return self._name


class WinBiasGame:
    def __init__(self):
        self._player = 0
        self._terminal = False
        self._outcome = None
        self._score = 0
        self._moves = 0

    def clone(self):
        other = WinBiasGame()
        other._player = self._player
        other._terminal = self._terminal
        other._outcome = self._outcome
        other._score = self._score
        other._moves = self._moves
        return other

    def apply_move(self, action):
        bonus = 1 if action == 0 else -1
        if self._player == 0:
            self._score += bonus
        else:
            self._score -= bonus
        self._moves += 1
        if self._moves >= 2:
            if self._score > 0:
                self._outcome = 1.0
            elif self._score < 0:
                self._outcome = -1.0
            else:
                self._outcome = 0.0
            self._terminal = True
        self._player = 1 - self._player

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def current_player(self):
        return self._player

    def legal_moves(self):
        return [] if self._terminal else [0, 1]


class PreferZeroEngine:
    def __init__(self, name: str):
        self._name = name

    def select_move(self, state):
        return 0, {"time_used_ms": 0, "simulations": 1}

    def reset(self):
        return None

    def name(self):
        return self._name


class PreferOneEngine:
    def __init__(self, name: str):
        self._name = name

    def select_move(self, state):
        return 1, {"time_used_ms": 0, "simulations": 1}

    def reset(self):
        return None

    def name(self):
        return self._name


class BatchedFirstMoveEngine(FirstMoveEngine):
    def __init__(self, name: str):
        super().__init__(name)
        self.batch_calls = 0

    def select_moves_batch(self, states):
        self.batch_calls += 1
        return [
            (state.legal_moves()[0], {"time_used_ms": 0, "simulations": 1})
            for state in states
        ]


class BatchedLastMoveEngine(LastMoveEngine):
    def __init__(self, name: str):
        super().__init__(name)
        self.batch_calls = 0

    def select_moves_batch(self, states):
        self.batch_calls += 1
        return [
            (state.legal_moves()[-1], {"time_used_ms": 0, "simulations": 1})
            for state in states
        ]


class FailingBatchedEngine(FirstMoveEngine):
    def select_moves_batch(self, states):
        raise RuntimeError("boom")


class SharedTallyEngine(FirstMoveEngine):
    def __init__(self, name: str, tally_cls):
        super().__init__(name)
        self.shared_calls = 0
        self._tally_cls = tally_cls

    def play_match_tally_against(
        self,
        opponent,
        game_factory,
        opening_book,
        num_games,
        color_swap=True,
        logger=None,
        max_moves=500,
        seed=None,
    ):
        self.shared_calls += 1
        return self._tally_cls(
            engine_name=self.name(),
            opponent_name=opponent.name(),
            wins=num_games,
            draws=0,
            losses=0,
            errors=0,
            voids=0,
            total=num_games,
        )


def test_play_match_tally_matches_reference():
    for module in EVAL_MODULES:
        runner = module.MatchRunner(ToyGame, seed=7, max_moves=10)
        eng_a = FirstMoveEngine("candidate")
        eng_b = LastMoveEngine("champion")
        records = runner.play_match(eng_a, eng_b, 12, color_swap=True)
        ref = module.tally_match(records, eng_a.name())

        runner = module.MatchRunner(ToyGame, seed=7, max_moves=10)
        fast = runner.play_match_tally(eng_a, eng_b, 12, color_swap=True)

        assert (fast.wins, fast.draws, fast.losses, fast.errors, fast.total) == (
            ref.wins,
            ref.draws,
            ref.losses,
            ref.errors,
            ref.total,
        )
        assert fast.score_rate == ref.score_rate


def test_weighted_glicko_matches_repeated_terms():
    for module in EVAL_MODULES:
        params = module.RatingParams()
        mu_a, phi_a = module.Glicko2.to_g2(1560.0, 82.0, params)
        mu_b, phi_b = module.Glicko2.to_g2(1495.0, 95.0, params)
        g_b = module.Glicko2.g(phi_b)
        e_b = module.Glicko2.E(mu_a, mu_b, phi_b)
        repeated = (
            [(g_b, e_b, 1.0)] * 11 + [(g_b, e_b, 0.5)] * 3 + [(g_b, e_b, 0.0)] * 6
        )
        weighted = [(g_b, e_b, 1.0, 11), (g_b, e_b, 0.5, 3), (g_b, e_b, 0.0, 6)]

        ref = module.Glicko2.update(mu_a, phi_a, 0.06, repeated, params.tau)
        fast = module.Glicko2.update_weighted(mu_a, phi_a, 0.06, weighted, params.tau)

        for lhs, rhs in zip(ref, fast):
            assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def test_game_record_json_uses_move_count():
    for module in EVAL_MODULES:
        record = module.GameRecord(
            game_id="g0001",
            engine_black="a",
            engine_white="b",
            outcome="draw",
            score_black=0.5,
            move_count=17,
            total_time_ms=12.3,
            moves=[],
        )
        payload = record.to_jsonl()
        assert '"moves": 17' in payload


def test_game_record_json_includes_search_manifest_hashes():
    for module in EVAL_MODULES:
        record = module.GameRecord(
            game_id="g0002",
            engine_black="a",
            engine_white="b",
            outcome="draw",
            score_black=0.5,
            move_count=9,
            total_time_ms=8.4,
            moves=[],
            search_manifest_hash="deadbeefcafefeed",
            search_manifest_hash_black="deadbeefcafefeed",
            search_manifest_hash_white="deadbeefcafefeed",
        )
        payload = record.to_jsonl()
        assert '"search_manifest_hash": "deadbeefcafefeed"' in payload
        assert '"search_manifest_hash_black": "deadbeefcafefeed"' in payload
        assert '"search_manifest_hash_white": "deadbeefcafefeed"' in payload


def test_parallel_evaluator_matches_sequential():
    for module in EVAL_MODULES:
        seq = module.TrainingEvaluator(
            config=module.EvalConfig(
                num_games=12,
                promotion_threshold=0.5,
                confidence=0.95,
                sanity_check_interval=99,
                parallel_workers=1,
            )
        )
        par = module.TrainingEvaluator(
            config=module.EvalConfig(
                num_games=12,
                promotion_threshold=0.5,
                confidence=0.95,
                sanity_check_interval=99,
                parallel_workers=3,
            )
        )

        seq_candidate = FirstMoveEngine("candidate")
        seq_champion = LastMoveEngine("champion")
        par_candidate = FirstMoveEngine("candidate")
        par_champion = LastMoveEngine("champion")

        seq_result = seq.evaluate_checkpoint(
            candidate=seq_candidate,
            champion=seq_champion,
            game_factory=ToyGame,
            candidate_id="candidate",
            generation=1,
        )
        par_result = par.evaluate_checkpoint(
            candidate=par_candidate,
            champion=par_champion,
            game_factory=ToyGame,
            candidate_id="candidate",
            generation=1,
            candidate_factory=lambda: FirstMoveEngine("candidate"),
            champion_factory=lambda: LastMoveEngine("champion"),
        )

        assert seq_result.tally == par_result.tally
        assert seq_result.promotion == par_result.promotion
        assert seq_result.elo == par_result.elo
        assert seq_result.published == par_result.published


def test_batched_match_tally_matches_reference_and_uses_batch_api():
    for module in EVAL_MODULES:
        runner = module.MatchRunner(ToyGame, seed=7, max_moves=10)
        batched_a = BatchedFirstMoveEngine("candidate")
        batched_b = BatchedLastMoveEngine("champion")
        tally = runner.play_match_tally_batched(
            batched_a, batched_b, 12, color_swap=True
        )

        runner = module.MatchRunner(ToyGame, seed=7, max_moves=10)
        records = runner.play_match(
            FirstMoveEngine("candidate"),
            LastMoveEngine("champion"),
            12,
            color_swap=True,
        )
        ref = module.tally_match(records, "candidate")

        assert (tally.wins, tally.draws, tally.losses, tally.errors, tally.total) == (
            ref.wins,
            ref.draws,
            ref.losses,
            ref.errors,
            ref.total,
        )
        assert batched_a.batch_calls > 0
        assert batched_b.batch_calls > 0


def test_batched_match_tally_prefers_shared_engine_path():
    for module in EVAL_MODULES:
        runner = module.MatchRunner(ToyGame, seed=7, max_moves=10)
        candidate = SharedTallyEngine("candidate", module.MatchTally)
        champion = SharedTallyEngine("champion", module.MatchTally)

        tally = runner.play_match_tally_batched(candidate, champion, 6, color_swap=True)

        assert tally.wins == 6
        assert tally.total == 6
        assert candidate.shared_calls == 1
        assert champion.shared_calls == 0


def test_training_evaluator_uses_model_ids_for_ladder_and_monotonic_published_elo():
    for module in EVAL_MODULES:
        evaluator = module.TrainingEvaluator(
            config=module.EvalConfig(
                num_games=40,
                promotion_threshold=0.5,
                confidence=0.95,
                sanity_check_interval=99,
            )
        )

        first = evaluator.evaluate_checkpoint(
            candidate=PreferZeroEngine("candidate_alias_1"),
            champion=PreferOneEngine("champion"),
            game_factory=WinBiasGame,
            candidate_id="gen_1",
            generation=1,
        )
        second = evaluator.evaluate_checkpoint(
            candidate=PreferZeroEngine("candidate_alias_2"),
            champion=PreferOneEngine("champion"),
            game_factory=WinBiasGame,
            candidate_id="gen_2",
            generation=2,
        )

        assert first.promotion["verdict"] == "promote"
        assert second.promotion["verdict"] == "promote"
        assert "champion" not in evaluator.ladder.players
        assert "gen_0" in evaluator.ladder.players
        assert "gen_1" in evaluator.ladder.players
        assert "gen_2" in evaluator.ladder.players
        assert second.champion == "gen_1"
        assert second.published["candidate_abs"] >= first.published["candidate_abs"]


def test_delta_elo_uses_smoothed_head_to_head_estimate():
    for module in EVAL_MODULES:
        evaluator = module.TrainingEvaluator(
            config=module.EvalConfig(
                num_games=40,
                promotion_threshold=0.5,
                confidence=0.95,
                sanity_check_interval=99,
            )
        )

        result = evaluator.evaluate_checkpoint(
            candidate=PreferZeroEngine("candidate_alias"),
            champion=PreferOneEngine("champion"),
            game_factory=WinBiasGame,
            candidate_id="gen_1",
            generation=1,
        )

        expected = round(module.estimate_match_delta_elo(40, 0, 0), 1)
        assert result.elo["delta"] == expected
        assert result.elo["delta"] < result.elo["ladder_delta"]


def test_zero_scored_batched_eval_is_marked_invalid_and_does_not_mutate_ladder():
    for module in EVAL_MODULES:
        evaluator = module.TrainingEvaluator(
            config=module.EvalConfig(
                num_games=4,
                promotion_threshold=0.5,
                confidence=0.95,
                sanity_check_interval=99,
                parallel_workers=1,
            )
        )
        period_before = evaluator.ladder.period
        summary_before = evaluator.ladder.summary()

        result = evaluator.evaluate_checkpoint(
            candidate=FailingBatchedEngine("candidate"),
            champion=FailingBatchedEngine("champion"),
            game_factory=ToyGame,
            candidate_id="candidate",
            generation=1,
        )

        assert result.valid_eval is False
        assert "zero scored games" in (result.invalid_reason or "")
        assert result.tally["scored"] == 0
        assert result.tally["errors"] == 4
        assert result.published is None
        assert evaluator.ladder.period == period_before
        assert evaluator.ladder.summary() == summary_before


def test_rating_ladder_stays_finite_on_lopsided_large_match():
    for module in EVAL_MODULES:
        ladder = module.RatingLadder()
        ladder.record_match("candidate", "champion", 191, 0, 9)
        cand = ladder.get("candidate")
        champ = ladder.get("champion")

        assert math.isfinite(cand.mu_elo)
        assert math.isfinite(champ.mu_elo)
        assert abs(cand.mu_elo) < 10000
        assert abs(champ.mu_elo) < 10000


def test_training_evaluator_does_not_explode_after_multiple_extreme_promotions():
    for module in EVAL_MODULES:
        evaluator = module.TrainingEvaluator(
            config=module.EvalConfig(
                num_games=200,
                promotion_threshold=0.55,
                confidence=0.95,
                sanity_check_interval=99,
            )
        )

        first = evaluator.evaluate_checkpoint(
            candidate=PreferZeroEngine("candidate_alias_1"),
            champion=PreferOneEngine("champion"),
            game_factory=WinBiasGame,
            candidate_id="gen_1",
            generation=1,
        )
        second = evaluator.evaluate_checkpoint(
            candidate=PreferZeroEngine("candidate_alias_2"),
            champion=PreferOneEngine("champion"),
            game_factory=WinBiasGame,
            candidate_id="gen_2",
            generation=2,
        )

        assert math.isfinite(first.published["candidate_abs"])
        assert math.isfinite(second.published["candidate_abs"])
        assert abs(first.published["candidate_abs"]) < 10000
        assert abs(second.published["candidate_abs"]) < 10000


def test_champion_tracker_round_trips_current_champion_state(tmp_path):
    for module in EVAL_MODULES:
        path = tmp_path / f"{module.__name__}_champion.json"
        tracker = module.ChampionTracker(save_path=path, bridge_size=2)
        promo = module.PromotionResult(
            module.PromotionVerdict.PROMOTE,
            0.65,
            (0.55, 0.75),
            0.55,
            0.01,
            True,
            200,
            "ok",
        )

        tracker.try_promote("gen_5", 5, promo, published_elo=123.4)
        loaded = module.ChampionTracker.load(path, bridge_size=2)

        assert loaded.champion.model_id == "gen_5"
        assert loaded.champion.generation == 5
        assert loaded.champion.elo == 123.4
        assert loaded.bridge == ["gen_0"]


def test_match_runner_can_advance_engine_driven_game_states():
    for module in EVAL_MODULES:

        class EngineDrivenGame:
            def __init__(self):
                self.turn = 0
                self.terminal = False
                self.outcome = None

            def clone(self):
                g = EngineDrivenGame()
                g.turn = self.turn
                g.terminal = self.terminal
                g.outcome = self.outcome
                return g

            def apply_move(self, action):
                raise AssertionError("engine metadata should handle transitions")

            def apply_engine_meta(self, action, meta):
                if meta.get("terminal", False):
                    self.terminal = True
                    self.outcome = meta.get("outcome_for_black", 0.0)
                    return True
                self.turn += 1
                return True

            def is_terminal(self):
                return self.terminal

            def outcome_for_black(self):
                return self.outcome

            def current_player(self):
                return self.turn % 2

            def legal_moves(self):
                return [] if self.terminal else [0]

        class ScriptedEngine:
            def __init__(self, name, script):
                self._name = name
                self._script = list(script)

            def select_move(self, state):
                return self._script.pop(0)

            def reset(self):
                pass

            def name(self):
                return self._name

        runner = module.MatchRunner(EngineDrivenGame, max_moves=8)
        black = ScriptedEngine("black", [(0, {"result_fen": "after_black"})])
        white = ScriptedEngine(
            "white", [(0, {"terminal": True, "outcome_for_black": -1.0})]
        )
        rec = runner.play_game(black, white, "g0000", collect_moves=False)

        assert rec.outcome == "white_win"
        assert rec.is_void is False


def test_match_runner_tracks_void_games_without_counting_engine_errors():
    for module in EVAL_MODULES:

        class VoidGame:
            def __init__(self):
                self.turn = 0
                self.terminal = False

            def clone(self):
                other = VoidGame()
                other.turn = self.turn
                other.terminal = self.terminal
                return other

            def apply_move(self, action):
                self.terminal = True

            def is_terminal(self):
                return self.terminal

            def outcome_for_black(self):
                return None

            def is_void_result(self):
                return self.terminal

            def current_player(self):
                return self.turn

            def legal_moves(self):
                return [] if self.terminal else [0]

        runner = module.MatchRunner(VoidGame, max_moves=4)
        rec = runner.play_game(
            FirstMoveEngine("candidate"), LastMoveEngine("champion"), "g0001"
        )
        tally = module.tally_match([rec], "candidate")

        assert rec.outcome == "void"
        assert rec.is_void is True
        assert rec.error is None
        assert tally.errors == 0
        assert tally.voids == 1
        assert tally.scored == 0
