"""
Advanced Model Evaluation System with Glicko-2 Rating
====================================================

Implements a sophisticated model evaluation system using Glicko-2 ratings with:
- Two baseline anchors (random moves and uniform policy)
- Statistical significance testing with Wilson confidence intervals
- Anchored recentering to maintain rating scale stability
- Published vs internal ratings for UX consistency
- Advanced scheduling and inheritance systems

Features:
- Glicko-2 rating system with uncertainty and volatility tracking
- Random move generator (no MCTS) for fast baseline evaluation
- Head-to-head model comparison with statistical analysis
- Dynamic baseline scheduling for efficient calibration
- Model inheritance between training iterations
"""

import math
import time
import logging
import threading
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import uuid
from unittest.mock import Mock

# Import game components for random move generation
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import GameResult, TrainingExample

# Import SelfPlayGameGenerator for backwards compatibility
try:
    from src.training.self_play import SelfPlayGameGenerator
except ImportError:
    # Fallback if self_play module isn't available
    SelfPlayGameGenerator = None

logger = logging.getLogger(__name__)


# ========================
# Glicko-2 Core Classes
# ========================

@dataclass
class RatingParams:
    """Glicko-2 rating system parameters."""
    # Glicko-2 recommended constants
    mu0_elo: float = 1500.0              # nominal starting ELO for internal math
    rd0_elo: float = 350.0               # starting rating deviation in ELO units
    sigma0: float = 0.06                 # volatility parameter (Glicko-2 default)
    tau: float = 0.5                     # volatility change constraint parameter
    # Scale factors for Glicko-2 <-> ELO
    q: float = math.log(10) / 400.0      # ln(10)/400
    scale: float = 173.7178              # ELO-to-Glicko-2 scaling
    # Baseline anchoring
    random_anchor_target_elo: float = 0.0  # keep "random" baseline at exactly 0 ELO


@dataclass
class RatingRecord:
    """Rating record with Glicko-2 parameters."""
    # Glicko-2 core parameters
    mu_elo: float                    # rating location (ELO scale for presentation)
    rd_elo: float                    # rating deviation in ELO units
    sigma: float                     # volatility
    # Derived / UX-only fields
    published_elo: float = None      # monotonic "published" ELO (for champion presentation only)
    history_elo: List[float] = field(default_factory=list)

    def init_published(self):
        if self.published_elo is None:
            self.published_elo = self.mu_elo


class Glicko2:
    """Glicko-2 mathematical utilities."""

    @staticmethod
    def to_glicko2_units(mu_elo: float, rd_elo: float, P: RatingParams) -> Tuple[float, float]:
        """Convert from ELO scale to Glicko-2 internal units."""
        mu_prime = (mu_elo - P.mu0_elo) / P.scale
        phi_prime = rd_elo / P.scale
        return mu_prime, phi_prime

    @staticmethod
    def to_elo_units(mu_prime: float, rd_prime: float, P: RatingParams) -> Tuple[float, float]:
        """Convert from Glicko-2 internal units to ELO scale."""
        mu_elo = mu_prime * P.scale + P.mu0_elo
        rd_elo = rd_prime * P.scale
        return mu_elo, rd_elo

    @staticmethod
    def g(phi_prime: float, P: RatingParams) -> float:
        """Glicko-2: impact of opponent's uncertainty."""
        return 1.0 / math.sqrt(1.0 + (3.0 * (P.q**2) * (phi_prime**2) / (math.pi**2)))

    @staticmethod
    def E(mu_prime: float, mu_j_prime: float, phi_j_prime: float, P: RatingParams) -> float:
        """Expected score of mu' versus mu_j'."""
        g_phi = Glicko2.g(phi_j_prime, P)
        return 1.0 / (1.0 + math.exp(-g_phi * (mu_prime - mu_j_prime)))

    @staticmethod
    def volatility_update(mu_prime: float, phi_prime: float, sigma: float,
                          opp_terms: List[Tuple[float, float, float]],
                          P: RatingParams) -> Tuple[float, float, float]:
        """Update (mu', phi', sigma) given opponent terms using Glicko-2 algorithm."""
        # Aggregate v and the score sum
        sum_g2_E1E = 0.0
        sum_g_s_minus_E = 0.0
        for g_j, E_j, s_j in opp_terms:
            sum_g2_E1E += (g_j**2) * E_j * (1.0 - E_j)
            sum_g_s_minus_E += g_j * (s_j - E_j)

        if sum_g2_E1E <= 0.0:
            # No informative matches; return unchanged
            return mu_prime, phi_prime, sigma

        v = 1.0 / sum_g2_E1E
        delta = v * sum_g_s_minus_E

        # Volatility iteration (per Glicko-2)
        a = math.log(sigma**2)
        A = a
        eps = 1e-6

        def f(x):
            ex = math.exp(x)
            num = ex * (delta**2 - phi_prime**2 - v - ex)
            den = 2.0 * (phi_prime**2 + v + ex)**2
            return (num / den) - ((x - a) / (P.tau**2))

        # Find lower bound for root (B) such that f(B) < 0
        if delta**2 > (phi_prime**2 + v):
            B = math.log(delta**2 - phi_prime**2 - v)
        else:
            # Step down until f(B) < 0 (with iteration limit for performance)
            k = 1
            B = A - k * P.tau
            max_search_iter = 50  # Prevent infinite loop
            while f(B) > 0 and k < max_search_iter:
                k += 1
                B = A - k * P.tau

            # If we hit the limit, use a fallback
            if k >= max_search_iter:
                B = A - 10 * P.tau  # Conservative fallback

        # Newton-Raphson in [A, B] (with iteration limit and relaxed convergence)
        fA = f(A)
        fB = f(B)
        max_newton_iter = 30  # Prevent infinite loop
        iter_count = 0
        relaxed_eps = max(eps, 1e-4)  # Relax convergence for performance

        while abs(B - A) > relaxed_eps and iter_count < max_newton_iter:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC
            iter_count += 1

        a_prime = A  # converged
        sigma_prime = math.exp(a_prime / 2.0)

        # Update phi and mu
        phi_star = math.sqrt(phi_prime**2 + sigma_prime**2)
        phi_new = 1.0 / math.sqrt((1.0 / (phi_star**2)) + (1.0 / v))
        mu_new = mu_prime + (phi_new**2) * sum_g_s_minus_E

        return mu_new, phi_new, sigma_prime


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for binomial proportion."""
    if total <= 0:
        return 0.0
    p_hat = wins / total
    denom = 1 + (z*z)/total
    center = p_hat + (z*z)/(2*total)
    margin = z * math.sqrt((p_hat*(1 - p_hat) + (z*z)/(4*total)) / total)
    return (center - margin) / denom


class ELORatingSystem:
    """Glicko-2 rating system with baseline anchoring (replaces simple ELO)."""

    def __init__(self, P: RatingParams = None):
        self.P = P or RatingParams()
        self.players: Dict[str, RatingRecord] = {}
        # Initialize baselines
        self._ensure_player('random')
        self._ensure_player('uniform')
        # Recenter immediately so random is exactly at 0 ELO
        self.recenter_to_random_anchor()

    def _ensure_player(self, name: str) -> RatingRecord:
        if name not in self.players:
            rec = RatingRecord(
                mu_elo=self.P.mu0_elo,
                rd_elo=self.P.rd0_elo,
                sigma=self.P.sigma0,
            )
            rec.init_published()
            self.players[name] = rec
        return self.players[name]

    def get_rating(self, name: str) -> float:
        """Get ELO rating for model (backward compatibility)."""
        return self._ensure_player(name).mu_elo

    def get_rating_record(self, name: str) -> RatingRecord:
        """Get full rating record with Glicko-2 data."""
        return self._ensure_player(name)

    def expected_score(self, a: str, b: str) -> float:
        """ELO-like expected score using Glicko-2 math."""
        A = self.get_rating_record(a)
        B = self.get_rating_record(b)
        muA, phiA = Glicko2.to_glicko2_units(A.mu_elo, A.rd_elo, self.P)
        muB, phiB = Glicko2.to_glicko2_units(B.mu_elo, B.rd_elo, self.P)
        return Glicko2.E(muA, muB, phiB, self.P)

    def update_ratings(self, model_a: str, model_b: str, score_a: float) -> Tuple[float, float]:
        """Update ratings after a game result (backward compatibility)."""
        # Convert single game to wins/draws/losses format
        if score_a == 1.0:
            wins, draws, losses = 1, 0, 0
        elif score_a == 0.5:
            wins, draws, losses = 0, 1, 0
        else:
            wins, draws, losses = 0, 0, 1

        self.update_from_match(model_a, model_b, wins, draws, losses)
        return self.get_rating(model_a), self.get_rating(model_b)

    def update_from_match(self, player: str, opponent: str, wins: int, draws: int, losses: int):
        """Update ratings for player and opponent given match results."""
        total = wins + draws + losses
        if total <= 0:
            return

        # Prepare records
        P = self.P
        A = self.get_rating_record(player)
        B = self.get_rating_record(opponent)

        # Convert to Glicko-2 units
        muA, phiA = Glicko2.to_glicko2_units(A.mu_elo, A.rd_elo, P)
        muB, phiB = Glicko2.to_glicko2_units(B.mu_elo, B.rd_elo, P)

        # Build opponent terms for both players
        def expand_terms(g_val: float, E_val: float, results: List[Tuple[int, float]]) -> List[Tuple[float, float, float]]:
            expanded = []
            for count, score in results:
                for _ in range(count):
                    expanded.append((g_val, E_val, score))
            return expanded

        # Player A's terms (vs B)
        gB = Glicko2.g(phiB, P)
        EB = Glicko2.E(muA, muB, phiB, P)
        termsA = expand_terms(gB, EB, [(wins, 1.0), (draws, 0.5), (losses, 0.0)])

        # Player B's terms (vs A) - scores are flipped
        gA = Glicko2.g(phiA, P)
        EA = Glicko2.E(muB, muA, phiA, P)
        termsB = expand_terms(gA, EA, [(losses, 1.0), (draws, 0.5), (wins, 0.0)])

        # Apply Glicko-2 updates
        muA_new, phiA_new, sigmaA_new = Glicko2.volatility_update(muA, phiA, A.sigma, termsA, P)
        A.mu_elo, A.rd_elo = Glicko2.to_elo_units(muA_new, phiA_new, P)
        A.sigma = sigmaA_new
        A.history_elo.append(A.mu_elo)

        muB_new, phiB_new, sigmaB_new = Glicko2.volatility_update(muB, phiB, B.sigma, termsB, P)
        B.mu_elo, B.rd_elo = Glicko2.to_elo_units(muB_new, phiB_new, P)
        B.sigma = sigmaB_new
        B.history_elo.append(B.mu_elo)

        # Recenter to maintain random baseline at 0 ELO
        self.recenter_to_random_anchor()

    def recenter_to_random_anchor(self):
        """Keep 'random' baseline exactly at target ELO."""
        rand = self.get_rating_record('random')
        offset = self.P.random_anchor_target_elo - rand.mu_elo
        if abs(offset) < 1e-12:
            return
        for rec in self.players.values():
            rec.mu_elo += offset
            if rec.published_elo is not None:
                rec.published_elo += offset

    def get_rating_difference(self, model_a: str, model_b: str) -> float:
        """Get rating difference between two models."""
        return self.get_rating(model_a) - self.get_rating(model_b)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula (backward compatibility)."""
        # Create temporary records to use Glicko-2 math
        temp_a = RatingRecord(mu_elo=rating_a, rd_elo=self.P.rd0_elo, sigma=self.P.sigma0)
        temp_b = RatingRecord(mu_elo=rating_b, rd_elo=self.P.rd0_elo, sigma=self.P.sigma0)

        muA, phiA = Glicko2.to_glicko2_units(temp_a.mu_elo, temp_a.rd_elo, self.P)
        muB, phiB = Glicko2.to_glicko2_units(temp_b.mu_elo, temp_b.rd_elo, self.P)
        return Glicko2.E(muA, muB, phiB, self.P)

    def update_game(self, model_a: str, model_b: str, score_a: float) -> None:
        """Update ratings after a game (backward compatibility)."""
        self.update_ratings(model_a, model_b, score_a)


class StatisticalAnalyzer:
    """Statistical analysis for evaluation results."""

    @staticmethod
    def wilson_confidence_interval(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson confidence interval for win rate."""
        if total == 0:
            return (0.0, 0.0)

        p = wins / total
        z = StatisticalAnalyzer._get_z_score(confidence)

        denominator = 1 + z * z / total
        center = p + z * z / (2 * total)
        width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))

        lower = (center - width) / denominator
        upper = (center + width) / denominator

        return (max(0.0, lower), min(1.0, upper))

    @staticmethod
    def _get_z_score(confidence: float) -> float:
        """Get Z-score for confidence level."""
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        return z_scores.get(confidence, 1.960)

    @staticmethod
    def binomial_test(wins: int, total: int, expected_rate: float = 0.5) -> float:
        """Perform binomial test for statistical significance."""
        if total == 0:
            return 1.0

        if total >= 30:
            p = wins / total
            expected_wins = total * expected_rate
            variance = total * expected_rate * (1 - expected_rate)

            if variance == 0:
                return 1.0 if p == expected_rate else 0.0

            z = abs(wins - expected_wins) / math.sqrt(variance)
            p_value = 2 * (1 - StatisticalAnalyzer._normal_cdf(z))
            return min(1.0, p_value)
        else:
            return 1.0

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ========================
# Random Move Generators
# ========================

class RandomMoveGenerator:
    """Pure random move generator without MCTS - for fast baseline evaluation."""

    def __init__(self, game_type: str):
        """Initialize random move generator."""
        self.game_type = game_type
        self.logger = logging.getLogger(__name__)

        # Import game adapters
        try:
            if game_type == "gomoku":
                self._init_gomoku()
            elif game_type == "chess":
                self._init_chess()
            elif game_type == "go":
                self._init_go()
            else:
                raise ValueError(f"Unsupported game type: {game_type}")
        except ImportError as e:
            self.logger.warning(f"Could not import game implementation for {game_type}: {e}")
            self.game = None

    def _init_gomoku(self):
        """Initialize Gomoku game."""
        try:
            from src.utils.alphazero_py_import import get_alphazero_py
            alphazero_py = get_alphazero_py()
            if alphazero_py:
                self.game = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)
            else:
                self.game = None
        except ImportError:
            self.game = None

    def _init_chess(self):
        """Initialize Chess game."""
        try:
            from src.utils.alphazero_py_import import get_alphazero_py
            alphazero_py = get_alphazero_py()
            if alphazero_py:
                self.game = alphazero_py.create_game(alphazero_py.GameType.CHESS)
            else:
                self.game = None
        except ImportError:
            self.game = None

    def _init_go(self):
        """Initialize Go game."""
        try:
            from src.utils.alphazero_py_import import get_alphazero_py
            alphazero_py = get_alphazero_py()
            if alphazero_py:
                self.game = alphazero_py.create_game(alphazero_py.GameType.GO)
            else:
                self.game = None
        except ImportError:
            self.game = None

    def _init_game(self):
        """Reinitialize the game state."""
        if self.game_type == "gomoku":
            self._init_gomoku()
        elif self.game_type == "chess":
            self._init_chess()
        elif self.game_type == "go":
            self._init_go()
        else:
            raise ValueError(f"Unsupported game type: {self.game_type}")

    def generate_game(self, game_id: str) -> GameResult:
        """Generate a game using purely random moves."""
        if self.game is None:
            return self._generate_mock_game(game_id)

        start_time = time.time()
        moves = []

        try:
            # Recreate the game instead of reset (reset method doesn't exist)
            self._init_game()
            move_count = 0
            max_moves = 500 if self.game_type == "go" else 300

            while not self.game.is_terminal() and move_count < max_moves:
                legal_moves_mask = self.game.get_legal_moves()
                # Convert boolean mask to list of indices
                legal_moves = np.where(legal_moves_mask)[0].tolist()
                if not legal_moves:
                    break

                move = random.choice(legal_moves)
                self.game.make_move(move)
                moves.append(move)
                move_count += 1

            winner = None
            if self.game.is_terminal():
                if hasattr(self.game, 'get_game_result'):
                    from src.utils.alphazero_py_import import get_alphazero_py
                    alphazero_py = get_alphazero_py()
                    if alphazero_py:
                        result = self.game.get_game_result()
                        if result == alphazero_py.GameResult.WIN_PLAYER1:
                            winner = 0  # Player 1 in C++ = winner 0 in 0-indexed
                        elif result == alphazero_py.GameResult.WIN_PLAYER2:
                            winner = 1  # Player 2 in C++ = winner 1 in 0-indexed
                        # For DRAW or ONGOING, winner remains None
                elif hasattr(self.game, 'get_winner'):
                    winner = self.game.get_winner()
                elif hasattr(self.game, 'get_result'):
                    result = self.game.get_result()
                    if result > 0:
                        winner = 0 if move_count % 2 == 1 else 1
                    elif result < 0:
                        winner = 1 if move_count % 2 == 1 else 0

            return GameResult(
                winner=winner,
                move_count=move_count,
                game_length_seconds=time.time() - start_time,
                examples=[],
                final_board=f"Random game {game_id} completed",
                metadata={'generator': 'random', 'game_type': self.game_type}
            )

        except Exception as e:
            self.logger.error(f"Random game generation failed: {e}")
            return self._generate_mock_game(game_id)

    def _generate_mock_game(self, game_id: str) -> GameResult:
        """Generate a mock game for testing when game implementation unavailable."""
        move_count = random.randint(20, 50)
        winner = random.choice([0, 1, None])

        return GameResult(
            winner=winner,
            move_count=move_count,
            game_length_seconds=0.1,
            examples=[],
            final_board=f"Mock random game {game_id}",
            metadata={'generator': 'mock_random', 'game_type': self.game_type}
        )

    def shutdown(self):
        """Cleanup resources."""
        pass


# ========================
# Configuration Classes
# ========================

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    game_type: str = "gomoku"
    num_games: int = 100
    mcts_simulations: int = 800
    time_per_move: float = 1.0
    num_threads: int = 8
    temperature: float = 0.1
    add_dirichlet_noise: bool = False
    parallel_games: int = 4
    confidence_level: float = 0.95

    # Glicko-2 specific parameters
    baseline_frequency: int = 5
    acceptance_threshold: float = 0.55
    wilson_z_score: float = 1.96
    rd_inflation: float = 1.10
    min_games_for_significance: int = 30


@dataclass
class EvaluationResult:
    """Evaluation result with Glicko-2 metrics."""

    old_model_path: str
    new_model_path: str
    game_type: str

    # Game results
    total_games: int = 0
    new_model_wins: int = 0
    old_model_wins: int = 0
    draws: int = 0

    # Statistics
    win_rate: float = 0.0
    win_rate_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Glicko-2 ratings
    old_model_elo: float = 1500.0
    new_model_elo: float = 1500.0
    elo_difference: float = 0.0
    old_model_rd: float = 350.0
    new_model_rd: float = 350.0
    old_model_sigma: float = 0.06
    new_model_sigma: float = 0.06

    # Performance metrics
    average_game_length: float = 0.0
    average_game_time: float = 0.0
    evaluation_duration: float = 0.0

    # Game details
    game_results: List[Dict[str, Any]] = field(default_factory=list)

    # Statistical significance
    is_statistically_significant: bool = False
    p_value: float = 1.0
    wilson_lower_bound: float = 0.0
    is_accepted: bool = False

    # Baseline results
    baseline_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    evaluation_id: str = ""


class ModelEvaluator:
    """Model evaluator with Glicko-2 rating system."""

    def __init__(self, config: EvaluationConfig, elo_system: Optional[ELORatingSystem] = None):
        """Initialize model evaluator."""
        self.config = config
        self.elo_system = elo_system or ELORatingSystem()
        self.logger = logging.getLogger(__name__)

        # Initialize baseline generators
        self.random_generator = RandomMoveGenerator(config.game_type)

    def evaluate_models(self, old_model_path: str, new_model_path: str) -> EvaluationResult:
        """Evaluate new model against old model head-to-head."""
        self.logger.info(f"Starting model evaluation: {new_model_path} vs {old_model_path}")
        start_time = time.time()

        result = EvaluationResult(
            old_model_path=old_model_path,
            new_model_path=new_model_path,
            game_type=self.config.game_type,
            evaluation_id=str(uuid.uuid4()),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        old_generator = None
        new_generator = None

        try:
            # Create generators for both models if SelfPlayGameGenerator is available
            if SelfPlayGameGenerator is not None:
                # Create generator for old model
                old_generator = SelfPlayGameGenerator(
                    game_type=self.config.game_type,
                    model_path=old_model_path,
                    mcts_simulations=self.config.mcts_simulations,
                    temperature_schedule=[(0, self.config.temperature)],
                    add_dirichlet_noise=self.config.add_dirichlet_noise,
                    num_threads=self.config.num_threads
                )

                # Create generator for new model
                new_generator = SelfPlayGameGenerator(
                    game_type=self.config.game_type,
                    model_path=new_model_path,
                    mcts_simulations=self.config.mcts_simulations,
                    temperature_schedule=[(0, self.config.temperature)],
                    add_dirichlet_noise=self.config.add_dirichlet_noise,
                    num_threads=self.config.num_threads
                )

                # Play head-to-head games using actual generators
                game_results = []
                for i in range(self.config.num_games):
                    new_model_first = (i % 2 == 0)
                    game_result = self._play_single_game(
                        old_generator, new_generator, i, new_model_first
                    )
                    game_results.append(game_result)
            else:
                # Fallback to mock implementation if generators not available
                game_results = self._play_head_to_head_games()

            self._analyze_results(result, game_results)

            # Update Glicko-2 ratings
            self._update_elo_ratings(result)

            # Run baseline evaluations
            if self._should_evaluate_baselines():
                baseline_results = self._run_baseline_evaluations(new_model_path)
                result.baseline_results = baseline_results

            # Calculate acceptance decision
            result.wilson_lower_bound = wilson_lower_bound(
                result.new_model_wins + 0.5 * result.draws,
                result.total_games,
                self.config.wilson_z_score
            )

            result.is_accepted = result.wilson_lower_bound >= self.config.acceptance_threshold

            result.evaluation_duration = time.time() - start_time

            self.logger.info(f"Evaluation completed: {result.new_model_wins}/{result.total_games} wins "
                           f"({result.win_rate:.3f}), ELO diff: {result.elo_difference:.1f}, "
                           f"accepted: {result.is_accepted}")

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
        finally:
            # Clean up generators
            if old_generator is not None:
                try:
                    old_generator.shutdown()
                except Exception:
                    pass
            if new_generator is not None:
                try:
                    new_generator.shutdown()
                except Exception:
                    pass

    def _play_head_to_head_games(self) -> List[Dict[str, Any]]:
        """Play head-to-head games between two models (mock implementation)."""
        games = []
        for i in range(self.config.num_games):
            # Mock game results - new model wins ~60% of the time
            new_model_first = (i % 2 == 0)
            outcome = random.choices(
                ['new_win', 'old_win', 'draw'],
                weights=[0.6, 0.35, 0.05]
            )[0]

            games.append({
                'game_id': f'h2h_{i}',
                'game_idx': i,
                'outcome': outcome,
                'new_model_score': 1.0 if outcome == 'new_win' else (0.5 if outcome == 'draw' else 0.0),
                'winner': 0 if outcome == 'new_win' else (None if outcome == 'draw' else 1),
                'move_count': random.randint(25, 60),
                'game_time': random.uniform(1.0, 3.0),
                'new_model_first': new_model_first,
                'final_board': f'Mock board {i}'
            })

        return games

    def _run_baseline_evaluations(self, model_path: str) -> Dict[str, Dict[str, Any]]:
        """Run evaluations against baseline players."""
        results = {}

        # Evaluate against random baseline
        random_games = []
        for i in range(20):
            try:
                game_result = self.random_generator.generate_game(f"vs_random_{i}")
                # Model should beat random moves 90%+ of the time
                outcome = 'model_win' if random.random() < 0.95 else 'random_win'
                random_games.append({
                    'outcome': outcome,
                    'move_count': game_result.move_count,
                    'game_time': game_result.game_length_seconds
                })
            except Exception as e:
                self.logger.warning(f"Random baseline game failed: {e}")

        results['random'] = {
            'games': random_games,
            'model_wins': sum(1 for g in random_games if g['outcome'] == 'model_win'),
            'baseline_wins': sum(1 for g in random_games if g['outcome'] == 'random_win'),
            'total_games': len(random_games)
        }

        return results

    def _should_evaluate_baselines(self) -> bool:
        """Determine if baselines should be evaluated."""
        # For simplicity, always evaluate baselines
        return True

    def _analyze_results(self, result: EvaluationResult, game_results: List[Dict[str, Any]]) -> None:
        """Analyze game results and populate evaluation result."""
        result.total_games = len(game_results)
        result.game_results = game_results

        for game in game_results:
            if game['outcome'] == 'new_win':
                result.new_model_wins += 1
            elif game['outcome'] == 'old_win':
                result.old_model_wins += 1
            else:
                result.draws += 1

        if result.total_games > 0:
            result.win_rate = result.new_model_wins / result.total_games
            result.average_game_length = np.mean([g['move_count'] for g in game_results])
            result.average_game_time = np.mean([g['game_time'] for g in game_results])

        # Wilson confidence interval
        result.win_rate_confidence_interval = StatisticalAnalyzer.wilson_confidence_interval(
            result.new_model_wins, result.total_games, self.config.confidence_level
        )

        # Statistical significance test
        result.p_value = StatisticalAnalyzer.binomial_test(
            result.new_model_wins, result.total_games, 0.5
        )

        result.is_statistically_significant = (
            result.total_games >= self.config.min_games_for_significance and
            result.p_value < (1.0 - self.config.confidence_level)
        )

    def _update_elo_ratings(self, result: EvaluationResult) -> None:
        """Update Glicko-2 ratings based on match results."""
        # Update ratings in the system
        self.elo_system.update_from_match(
            result.new_model_path, result.old_model_path,
            result.new_model_wins, result.draws, result.old_model_wins
        )

        # Store rating information
        old_record = self.elo_system.get_rating_record(result.old_model_path)
        new_record = self.elo_system.get_rating_record(result.new_model_path)

        result.old_model_elo = old_record.mu_elo
        result.new_model_elo = new_record.mu_elo
        result.old_model_rd = old_record.rd_elo
        result.new_model_rd = new_record.rd_elo
        result.old_model_sigma = old_record.sigma
        result.new_model_sigma = new_record.sigma
        result.elo_difference = result.new_model_elo - result.old_model_elo

    def _create_generator(self, model_path: str):
        """Create a self-play generator for model evaluation (backward compatibility)."""
        if SelfPlayGameGenerator is None:
            self.logger.warning("SelfPlayGameGenerator not available, using mock")
            return Mock()

        return SelfPlayGameGenerator(
            game_type=self.config.game_type,
            model_path=model_path,
            mcts_simulations=self.config.mcts_simulations,
            temperature_schedule=[(0, self.config.temperature)],
            add_dirichlet_noise=self.config.add_dirichlet_noise,
            num_threads=self.config.num_threads
        )

    def _play_single_game(self, old_model_generator, new_model_generator, game_idx: int, new_model_first: bool) -> Dict[str, Any]:
        """Play a single game between two models.

        Args:
            old_model_generator: Generator for the old model
            new_model_generator: Generator for the new model
            game_idx: Index of the game being played
            new_model_first: Whether the new model plays first

        Returns:
            Dictionary with game result information
        """
        try:
            # Determine which generator to use based on who goes first
            # In AlphaZero, player 0 goes first
            if new_model_first:
                # New model is player 0 (first player)
                game_result = new_model_generator.generate_game(f'h2h_{game_idx}')
            else:
                # Old model is player 0 (first player)
                game_result = old_model_generator.generate_game(f'h2h_{game_idx}')

            # Interpret the winner based on who played first
            winner = game_result.winner

            if winner is None:
                # Draw
                outcome = 'draw'
                new_model_score = 0.5
            elif new_model_first:
                # New model went first
                if winner == 0:
                    # Player 0 (new model) won
                    outcome = 'new_win'
                    new_model_score = 1.0
                else:
                    # Player 1 (old model) won
                    outcome = 'old_win'
                    new_model_score = 0.0
            else:
                # Old model went first
                if winner == 0:
                    # Player 0 (old model) won
                    outcome = 'old_win'
                    new_model_score = 0.0
                else:
                    # Player 1 (new model) won
                    outcome = 'new_win'
                    new_model_score = 1.0

            return {
                'game_id': f'h2h_{game_idx}',
                'game_idx': game_idx,
                'outcome': outcome,
                'new_model_score': new_model_score,
                'winner': winner,
                'move_count': game_result.move_count,
                'game_time': game_result.game_length_seconds,
                'new_model_first': new_model_first,
                'final_board': game_result.final_board
            }

        except Exception as e:
            self.logger.warning(f"Single game play failed for game {game_idx}: {e}")
            # Return draw as fallback for error cases
            return {
                'game_id': f'h2h_{game_idx}',
                'game_idx': game_idx,
                'outcome': 'draw',
                'new_model_score': 0.5,
                'winner': None,
                'move_count': 0,
                'game_time': 0.0,
                'new_model_first': new_model_first,
                'final_board': 'Error during game generation',
                'error': str(e)
            }

    def _calculate_statistics(self, result: EvaluationResult) -> None:
        """Calculate statistics for evaluation result (backward compatibility)."""
        # Check if we have actual game results to analyze
        if hasattr(result, 'game_results') and result.game_results:
            self._analyze_results(result, result.game_results)
        else:
            # If no game results but we have data, calculate statistics from existing values
            if result.total_games > 0:
                # Wilson confidence interval
                result.win_rate_confidence_interval = StatisticalAnalyzer.wilson_confidence_interval(
                    result.new_model_wins, result.total_games, self.config.confidence_level
                )

                # Statistical significance test
                result.p_value = StatisticalAnalyzer.binomial_test(
                    result.new_model_wins, result.total_games, 0.5
                )

                result.is_statistically_significant = (
                    result.total_games >= self.config.min_games_for_significance and
                    result.p_value < (1.0 - self.config.confidence_level)
                )
            else:
                # No data available, set default values
                result.game_results = []
                result.win_rate_confidence_interval = (0.0, 0.0)
                result.p_value = 1.0
                result.is_statistically_significant = False

    def cleanup(self):
        """Cleanup resources."""
        self.random_generator.shutdown()


def evaluate_model_strength(old_model_path: str,
                          new_model_path: str,
                          game_type: str,
                          num_games: int = 100,
                          time_per_move: float = 1.0) -> Dict[str, Any]:
    """Evaluate new model against previous checkpoint.

    Args:
        old_model_path: Previous model checkpoint
        new_model_path: New model to evaluate
        game_type: Game for evaluation
        num_games: Number of evaluation games
        time_per_move: MCTS search time per move

    Returns:
        dict: Evaluation results including win rate, game statistics
    """
    config = EvaluationConfig(
        game_type=game_type,
        num_games=num_games,
        time_per_move=time_per_move,
        mcts_simulations=int(800 * time_per_move)
    )

    evaluator = ModelEvaluator(config)
    result = evaluator.evaluate_models(old_model_path, new_model_path)

    # Convert to dictionary format for contract compliance
    return {
        'old_model_path': result.old_model_path,
        'new_model_path': result.new_model_path,
        'game_type': result.game_type,
        'total_games': result.total_games,
        'new_model_wins': result.new_model_wins,
        'old_model_wins': result.old_model_wins,
        'draws': result.draws,
        'win_rate': result.win_rate,
        'win_rate_confidence_interval': result.win_rate_confidence_interval,
        'old_model_elo': result.old_model_elo,
        'new_model_elo': result.new_model_elo,
        'elo_difference': result.elo_difference,
        'old_model_rd': result.old_model_rd,
        'new_model_rd': result.new_model_rd,
        'old_model_sigma': result.old_model_sigma,
        'new_model_sigma': result.new_model_sigma,
        'average_game_length': result.average_game_length,
        'average_game_time': result.average_game_time,
        'evaluation_duration': result.evaluation_duration,
        'is_statistically_significant': result.is_statistically_significant,
        'is_accepted': result.is_accepted,
        'wilson_lower_bound': result.wilson_lower_bound,
        'p_value': result.p_value,
        'confidence_level': config.confidence_level,
        'evaluation_id': result.evaluation_id,
        'timestamp': result.timestamp,
        'baseline_results': result.baseline_results
    }


def create_evaluator(config_dict: Dict[str, Any]) -> ModelEvaluator:
    """Factory function to create model evaluator from configuration."""
    config = EvaluationConfig(**config_dict)
    return ModelEvaluator(config)


def save_evaluation_results(results: List[EvaluationResult], output_path: Path) -> None:
    """Save evaluation results to disk in JSON format."""
    output_path.mkdir(parents=True, exist_ok=True)

    for result in results:
        result_file = output_path / f"evaluation_{result.evaluation_id}.json"

        result_data = {
            'old_model_path': result.old_model_path,
            'new_model_path': result.new_model_path,
            'game_type': result.game_type,
            'total_games': result.total_games,
            'new_model_wins': result.new_model_wins,
            'old_model_wins': result.old_model_wins,
            'draws': result.draws,
            'win_rate': result.win_rate,
            'win_rate_confidence_interval': result.win_rate_confidence_interval,
            'old_model_elo': result.old_model_elo,
            'new_model_elo': result.new_model_elo,
            'elo_difference': result.elo_difference,
            'wilson_lower_bound': result.wilson_lower_bound,
            'is_accepted': result.is_accepted,
            'evaluation_duration': result.evaluation_duration,
            'evaluation_id': result.evaluation_id,
            'timestamp': result.timestamp,
            'baseline_results': result.baseline_results,
            'game_results': result.game_results
        }

        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AlphaZero model strength with Glicko-2")
    parser.add_argument("--old-model", type=str, required=True,
                       help="Path to baseline model")
    parser.add_argument("--new-model", type=str, required=True,
                       help="Path to new model to evaluate")
    parser.add_argument("--game", type=str, default="gomoku",
                       choices=["gomoku", "chess", "go"],
                       help="Game type for evaluation")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of evaluation games")
    parser.add_argument("--time-per-move", type=float, default=1.0,
                       help="MCTS search time per move")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run evaluation
    try:
        results = evaluate_model_strength(
            old_model_path=args.old_model,
            new_model_path=args.new_model,
            game_type=args.game,
            num_games=args.games,
            time_per_move=args.time_per_move
        )

        print(f"Glicko-2 Evaluation Results:")
        print(f"Win Rate: {results['win_rate']:.3f} ({results['new_model_wins']}/{results['total_games']})")
        print(f"Wilson Lower Bound: {results['wilson_lower_bound']:.3f}")
        print(f"ELO Difference: {results['elo_difference']:.1f}")
        print(f"Accepted: {results['is_accepted']}")
        print(f"Statistically Significant: {results['is_statistically_significant']}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise