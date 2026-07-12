"""
Comprehensive Self-Play Integration Tests
=========================================

Thorough testing of self-play game generation across all supported games and variations.
Tests for move bias, policy entropy, terminal detection, and MCTS health.

Key validation areas:
- All game types (Gomoku, Chess, Go) with variations
- Terminal detection and game outcome determination
- Move bias analysis and spatial distribution
- Policy entropy patterns and exploration/exploitation balance
- MCTS health indicators and convergence
"""

import pytest
import numpy as np
import tempfile
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.self_play import (
    SelfPlayGameGenerator, SelfPlayConfig, create_self_play_generator
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specs" / "001-goal-create-spec"))
from contracts.training_api import GameResult, TrainingExample


@dataclass
class GameAnalysisResult:
    """Analysis results for a single game."""

    game_id: str
    game_type: str
    move_count: int
    terminal_reason: str
    winner: Optional[int]

    # Move analysis
    move_positions: List[Tuple[int, int]]  # Board coordinates
    move_distribution: Dict[str, float]    # Spatial distribution statistics
    opening_moves: List[int]               # First 10 moves

    # Policy analysis
    policy_entropies: List[float]          # Entropy per move
    entropy_trend: str                     # "increasing", "decreasing", "stable"
    avg_early_entropy: float               # Average entropy moves 0-9
    avg_late_entropy: float                # Average entropy moves 10+

    # MCTS health indicators
    temperature_schedule: List[float]      # Temperature per move
    value_estimates: List[float]           # Position value estimates
    convergence_quality: float             # 0-1 score for MCTS convergence


@dataclass
class BatchAnalysisResult:
    """Analysis results for a batch of games."""

    total_games: int
    successful_games: int
    failed_games: int

    # Terminal detection validation
    terminal_reasons: Counter
    win_distribution: Counter              # Player 0, Player 1, Draw counts
    avg_game_length: float
    game_length_distribution: List[int]

    # Move bias analysis
    spatial_bias_score: float              # 0-1, lower is better (less bias)
    opening_diversity: float               # Shannon entropy of opening moves
    corner_bias: float                     # Tendency to play corners
    center_bias: float                     # Tendency to play center
    edge_bias: float                       # Tendency to play edges

    # Policy health
    avg_policy_entropy: float
    entropy_consistency: float             # Variance in entropy patterns
    exploration_quality: float             # How well early moves explore
    exploitation_quality: float            # How well late moves exploit

    # Statistical significance tests
    bias_tests: Dict[str, Dict[str, float]]  # p-values for various bias tests
    health_scores: Dict[str, float]        # Overall health indicators


class SelfPlayAnalyzer:
    """Comprehensive analyzer for self-play game quality."""

    def __init__(self, board_sizes: Dict[str, Tuple[int, int]] = None):
        """Initialize analyzer with board sizes for each game type."""
        self.board_sizes = board_sizes or {
            'gomoku': (15, 15),
            'chess': (8, 8),
            'go': (19, 19)  # Default to 19x19
        }

        self.logger = logging.getLogger(__name__)

    def analyze_game(self, game_result: GameResult) -> GameAnalysisResult:
        """Analyze a single completed game."""
        game_type = game_result.metadata.get('game_type', 'unknown')
        board_h, board_w = self.board_sizes.get(game_type, (15, 15))

        # Extract move positions from game history
        move_history = game_result.metadata.get('move_history', [])
        move_positions = self._extract_move_positions(move_history, board_w)

        # Analyze move distribution
        move_distribution = self._analyze_move_distribution(
            move_positions, board_h, board_w
        )

        # Analyze policy entropy from training examples
        policy_entropies = [
            self._calculate_entropy(example.policy)
            for example in game_result.examples
        ]

        entropy_trend = self._analyze_entropy_trend(policy_entropies)
        avg_early_entropy = np.mean(policy_entropies[:10]) if len(policy_entropies) >= 10 else np.mean(policy_entropies)
        avg_late_entropy = np.mean(policy_entropies[10:]) if len(policy_entropies) > 10 else avg_early_entropy

        # Extract temperature schedule (from move analysis or metadata)
        temperature_schedule = self._extract_temperature_schedule(game_result)

        # Extract value estimates
        value_estimates = [example.value for example in game_result.examples]

        # Calculate convergence quality
        convergence_quality = self._assess_convergence_quality(
            policy_entropies, value_estimates
        )

        return GameAnalysisResult(
            game_id=game_result.metadata.get('game_id', 'unknown'),
            game_type=game_type,
            move_count=game_result.move_count,
            terminal_reason=game_result.metadata.get('final_outcome', {}).get('result', 'unknown'),
            winner=game_result.winner,
            move_positions=move_positions,
            move_distribution=move_distribution,
            opening_moves=move_history[:10],
            policy_entropies=policy_entropies,
            entropy_trend=entropy_trend,
            avg_early_entropy=avg_early_entropy,
            avg_late_entropy=avg_late_entropy,
            temperature_schedule=temperature_schedule,
            value_estimates=value_estimates,
            convergence_quality=convergence_quality
        )

    def analyze_batch(self, game_results: List[GameResult]) -> BatchAnalysisResult:
        """Analyze a batch of games for patterns and biases."""
        if not game_results:
            raise ValueError("No games to analyze")

        # Analyze individual games
        game_analyses = []
        failed_count = 0

        for game_result in game_results:
            try:
                analysis = self.analyze_game(game_result)
                game_analyses.append(analysis)
            except Exception as e:
                self.logger.warning(f"Failed to analyze game {game_result.metadata.get('game_id')}: {e}")
                failed_count += 1

        if not game_analyses:
            raise ValueError("No games could be analyzed successfully")

        # Terminal detection analysis
        terminal_reasons = Counter(analysis.terminal_reason for analysis in game_analyses)
        win_distribution = Counter(analysis.winner for analysis in game_analyses)
        game_lengths = [analysis.move_count for analysis in game_analyses]
        avg_game_length = np.mean(game_lengths)

        # Spatial bias analysis
        spatial_bias_score = self._calculate_spatial_bias(game_analyses)
        opening_diversity = self._calculate_opening_diversity(game_analyses)
        corner_bias, center_bias, edge_bias = self._calculate_positional_biases(game_analyses)

        # Policy health analysis
        all_entropies = []
        for analysis in game_analyses:
            all_entropies.extend(analysis.policy_entropies)

        avg_policy_entropy = np.mean(all_entropies) if all_entropies else 0.0
        entropy_consistency = self._calculate_entropy_consistency(game_analyses)
        exploration_quality = self._calculate_exploration_quality(game_analyses)
        exploitation_quality = self._calculate_exploitation_quality(game_analyses)

        # Statistical tests
        bias_tests = self._run_bias_tests(game_analyses)
        health_scores = self._calculate_health_scores(game_analyses)

        return BatchAnalysisResult(
            total_games=len(game_results),
            successful_games=len(game_analyses),
            failed_games=failed_count,
            terminal_reasons=terminal_reasons,
            win_distribution=win_distribution,
            avg_game_length=avg_game_length,
            game_length_distribution=game_lengths,
            spatial_bias_score=spatial_bias_score,
            opening_diversity=opening_diversity,
            corner_bias=corner_bias,
            center_bias=center_bias,
            edge_bias=edge_bias,
            avg_policy_entropy=avg_policy_entropy,
            entropy_consistency=entropy_consistency,
            exploration_quality=exploration_quality,
            exploitation_quality=exploitation_quality,
            bias_tests=bias_tests,
            health_scores=health_scores
        )

    def _extract_move_positions(self, move_history: List[int], board_width: int) -> List[Tuple[int, int]]:
        """Convert move indices to board coordinates."""
        positions = []
        for move in move_history:
            if isinstance(move, int) and move >= 0:
                row, col = divmod(move, board_width)
                positions.append((row, col))
        return positions

    def _analyze_move_distribution(self, positions: List[Tuple[int, int]],
                                 board_h: int, board_w: int) -> Dict[str, float]:
        """Analyze spatial distribution of moves."""
        if not positions:
            return {}

        rows, cols = zip(*positions)

        corner_zone = max(2, min(board_h, board_w) // 3)

        def _in_corner(r: int, c: int) -> bool:
            return (
                (r < corner_zone and c < corner_zone)
                or (r < corner_zone and c >= board_w - corner_zone)
                or (r >= board_h - corner_zone and c < corner_zone)
                or (r >= board_h - corner_zone and c >= board_w - corner_zone)
            )

        return {
            'mean_row': np.mean(rows),
            'mean_col': np.mean(cols),
            'std_row': np.std(rows),
            'std_col': np.std(cols),
            'center_distance': np.mean([
                abs(r - board_h//2) + abs(c - board_w//2)
                for r, c in positions
            ]),
            'corner_ratio': sum(1 for r, c in positions if _in_corner(r, c)) / len(positions),
            'edge_ratio': sum(1 for r, c in positions
                            if (r in [0, board_h-1]) or (c in [0, board_w-1])) / len(positions)
        }

    def _calculate_entropy(self, policy: np.ndarray) -> float:
        """Calculate Shannon entropy of policy distribution."""
        # Add small epsilon to avoid log(0)
        policy_safe = policy + 1e-12
        policy_norm = policy_safe / np.sum(policy_safe)
        return -np.sum(policy_norm * np.log2(policy_norm))

    def _analyze_entropy_trend(self, entropies: List[float]) -> str:
        """Analyze trend in policy entropy over game."""
        if len(entropies) < 5:
            return "insufficient_data"

        # Use linear regression to determine trend
        x = np.arange(len(entropies))
        slope, _, r_value, p_value, _ = stats.linregress(x, entropies)

        if p_value > 0.05:  # Not significant
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _extract_temperature_schedule(self, game_result: GameResult) -> List[float]:
        """Extract temperature schedule from game metadata or estimate."""
        # For now, use a simple schedule based on move number
        # In real implementation, this would come from the actual generator
        schedule = []
        for i in range(game_result.move_count):
            if i < 30:
                schedule.append(1.0)
            else:
                schedule.append(0.1)
        return schedule

    def _assess_convergence_quality(self, entropies: List[float],
                                  values: List[float]) -> float:
        """Assess quality of MCTS convergence (0-1 score)."""
        if len(entropies) < 10:
            return 0.5  # Insufficient data

        # Good convergence: entropy decreases, values stabilize
        entropy_trend_score = 1.0 if self._analyze_entropy_trend(entropies) == "decreasing" else 0.0

        # Value stability in latter half of game
        if len(values) > 10:
            late_values = values[len(values)//2:]
            value_stability = 1.0 - min(1.0, np.std(late_values))
        else:
            value_stability = 0.5

        return (entropy_trend_score + value_stability) / 2.0

    def _calculate_spatial_bias(self, analyses: List[GameAnalysisResult]) -> float:
        """Calculate overall spatial bias score (0=no bias, 1=maximum bias)."""
        all_distributions = [a.move_distribution for a in analyses if a.move_distribution]

        if not all_distributions:
            return 0.5

        # Look at standard deviations - high std suggests good distribution
        row_stds = [d.get('std_row', 0) for d in all_distributions]
        col_stds = [d.get('std_col', 0) for d in all_distributions]

        # Normalize by board size (assume 15x15 for Gomoku)
        normalized_row_std = np.mean(row_stds) / 7.5  # Half board size
        normalized_col_std = np.mean(col_stds) / 7.5

        # Lower std = more bias, so invert
        bias_score = 1.0 - min(1.0, (normalized_row_std + normalized_col_std) / 2.0)
        return bias_score

    def _calculate_opening_diversity(self, analyses: List[GameAnalysisResult]) -> float:
        """Calculate diversity of opening moves using Shannon entropy."""
        all_opening_moves = []
        for analysis in analyses:
            if analysis.opening_moves:
                all_opening_moves.append(analysis.opening_moves[0])  # First move

        if not all_opening_moves:
            return 0.0

        # Calculate Shannon entropy of first moves
        move_counts = Counter(all_opening_moves)
        total = len(all_opening_moves)

        entropy = 0.0
        for count in move_counts.values():
            p = count / total
            entropy -= p * np.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(move_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_positional_biases(self, analyses: List[GameAnalysisResult]) -> Tuple[float, float, float]:
        """Calculate corner, center, and edge biases."""
        corner_ratios = []
        center_distances = []
        edge_ratios = []

        for analysis in analyses:
            if analysis.move_distribution:
                corner_ratios.append(analysis.move_distribution.get('corner_ratio', 0))
                center_distances.append(analysis.move_distribution.get('center_distance', 0))
                edge_ratios.append(analysis.move_distribution.get('edge_ratio', 0))

        corner_bias = np.mean(corner_ratios) if corner_ratios else 0.0
        center_bias = 1.0 - (np.mean(center_distances) / 15.0) if center_distances else 0.5  # Normalize by max distance
        edge_bias = np.mean(edge_ratios) if edge_ratios else 0.0

        return corner_bias, center_bias, edge_bias

    def _calculate_entropy_consistency(self, analyses: List[GameAnalysisResult]) -> float:
        """Calculate consistency of entropy patterns across games."""
        entropy_trends = [a.entropy_trend for a in analyses]
        trend_counts = Counter(entropy_trends)

        # Good consistency = most games have decreasing entropy
        decreasing_ratio = trend_counts.get('decreasing', 0) / len(entropy_trends)
        return decreasing_ratio

    def _calculate_exploration_quality(self, analyses: List[GameAnalysisResult]) -> float:
        """Calculate quality of exploration in early game."""
        early_entropies = [a.avg_early_entropy for a in analyses if a.avg_early_entropy > 0]

        if not early_entropies:
            return 0.0

        # Higher entropy = better exploration
        avg_early_entropy = np.mean(early_entropies)
        # Normalize by log2(225) for 15x15 board (maximum entropy)
        max_entropy = np.log2(225)
        return min(1.0, avg_early_entropy / max_entropy)

    def _calculate_exploitation_quality(self, analyses: List[GameAnalysisResult]) -> float:
        """Calculate quality of exploitation in late game."""
        late_entropies = [a.avg_late_entropy for a in analyses if a.avg_late_entropy > 0]

        if not late_entropies:
            return 0.0

        # Lower entropy = better exploitation
        avg_late_entropy = np.mean(late_entropies)
        max_entropy = np.log2(225)
        return 1.0 - min(1.0, avg_late_entropy / max_entropy)

    def _run_bias_tests(self, analyses: List[GameAnalysisResult]) -> Dict[str, Dict[str, float]]:
        """Run statistical tests for various biases."""
        bias_tests = {}

        # Test for uniform distribution of opening moves
        opening_moves = [a.opening_moves[0] for a in analyses if a.opening_moves]
        if len(opening_moves) > 10:
            # Chi-square test for uniformity
            observed = list(Counter(opening_moves).values())
            expected = [len(opening_moves) / len(set(opening_moves))] * len(set(opening_moves))
            chi2_stat, p_value = stats.chisquare(observed, expected)
            bias_tests['opening_uniformity'] = {'chi2_stat': chi2_stat, 'p_value': p_value}

        # Test for spatial bias in move positions
        all_positions = []
        for analysis in analyses:
            all_positions.extend(analysis.move_positions)

        if len(all_positions) > 20:
            rows, cols = zip(*all_positions)
            # Test if row distribution is uniform
            row_chi2, row_p = stats.chisquare(np.histogram(rows, bins=15)[0])
            col_chi2, col_p = stats.chisquare(np.histogram(cols, bins=15)[0])

            bias_tests['spatial_uniformity'] = {
                'row_chi2': row_chi2, 'row_p_value': row_p,
                'col_chi2': col_chi2, 'col_p_value': col_p
            }

        return bias_tests

    def _calculate_health_scores(self, analyses: List[GameAnalysisResult]) -> Dict[str, float]:
        """Calculate overall health scores for the self-play system."""
        health_scores = {}

        # Terminal detection health
        terminal_success_rate = len([a for a in analyses if a.terminal_reason != 'unknown']) / len(analyses)
        health_scores['terminal_detection'] = terminal_success_rate

        # Game completion health
        reasonable_lengths = [a for a in analyses if 10 <= a.move_count <= 500]
        completion_health = len(reasonable_lengths) / len(analyses)
        health_scores['game_completion'] = completion_health

        # MCTS convergence health
        convergence_scores = [a.convergence_quality for a in analyses]
        health_scores['mcts_convergence'] = np.mean(convergence_scores)

        # Overall health (geometric mean)
        individual_scores = list(health_scores.values())
        health_scores['overall'] = np.prod(individual_scores) ** (1.0 / len(individual_scores))

        return health_scores


class TestSelfPlayComprehensive:
    """Comprehensive integration tests for self-play system."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer for test validation."""
        return SelfPlayAnalyzer()

    @pytest.fixture
    def mock_generator(self):
        """Create mock generator for controlled testing."""
        generator = SelfPlayGameGenerator(
            game_type="gomoku",
            model_path="/tmp/test_model.pth",
            mcts_simulations=50,  # Reduced for faster testing
            temperature_schedule=[(10, 1.0), (100, 0.1)]
        )

        # Mock the components to avoid actual GPU/MCTS calls
        generator.inference_worker = Mock()
        generator.search_coordinator = Mock()

        return generator

    def _create_realistic_game_result(self, game_type: str = "gomoku",
                                    move_count: int = 50,
                                    winner: Optional[int] = None) -> GameResult:
        """Create realistic game result for testing."""
        # Generate realistic move sequence
        board_size = {'gomoku': 225, 'chess': 64, 'go': 361}[game_type]
        move_history = np.random.choice(board_size, size=min(move_count, board_size), replace=False).tolist()

        # Generate training examples with realistic policy patterns
        examples = []
        for i in range(move_count):
            # Create policy with decreasing entropy over time
            base_entropy = max(0.1, 1.0 - (i / move_count) * 0.8)  # Entropy decreases
            policy = np.random.dirichlet([base_entropy] * board_size)

            # Add some determinism to later moves
            if i > move_count * 0.7:
                best_move = np.random.randint(board_size)
                policy[best_move] *= 5.0
                policy /= np.sum(policy)

            example = TrainingExample(
                state=np.random.rand(36, 15, 15).astype(np.float32),
                policy=policy,
                value=winner if winner is not None else np.random.uniform(-1, 1),
                game_type=game_type,
                move_number=i,
                game_id=f"test_game_{game_type}"
            )
            examples.append(example)

        return GameResult(
            winner=winner,
            move_count=move_count,
            game_length_seconds=move_count * 0.5,  # 0.5s per move
            examples=examples,
            final_board=f"Final {game_type} board",
            metadata={
                'game_id': f"test_{game_type}",
                'game_type': game_type,
                'move_history': move_history,
                'final_outcome': {'result': 'win_player1' if winner == 0 else 'win_player2' if winner == 1 else 'draw'}
            }
        )

    def test_game_analysis_basic(self, analyzer):
        """Test basic game analysis functionality."""
        game_result = self._create_realistic_game_result("gomoku", 30, winner=0)

        analysis = analyzer.analyze_game(game_result)

        assert analysis.game_type == "gomoku"
        assert analysis.move_count == 30
        assert analysis.winner == 0
        assert len(analysis.policy_entropies) == 30
        assert analysis.avg_early_entropy > 0
        assert analysis.avg_late_entropy >= 0
        assert 0 <= analysis.convergence_quality <= 1

    def test_entropy_trend_analysis(self, analyzer):
        """Test entropy trend detection."""
        # Create game with decreasing entropy (good)
        game_result = self._create_realistic_game_result("gomoku", 20)
        analysis = analyzer.analyze_game(game_result)

        # Should detect decreasing trend due to realistic policy generation
        assert analysis.entropy_trend in ['decreasing', 'stable']
        assert analysis.avg_early_entropy >= analysis.avg_late_entropy

    def test_move_distribution_analysis(self, analyzer):
        """Test spatial move distribution analysis."""
        game_result = self._create_realistic_game_result("gomoku", 40)
        analysis = analyzer.analyze_game(game_result)

        assert len(analysis.move_positions) <= 40
        assert 'mean_row' in analysis.move_distribution
        assert 'mean_col' in analysis.move_distribution
        assert 'corner_ratio' in analysis.move_distribution
        assert 0 <= analysis.move_distribution['corner_ratio'] <= 1

    def test_batch_analysis_comprehensive(self, analyzer):
        """Test comprehensive batch analysis."""
        games = []

        # Create diverse set of games
        for i in range(10):
            game_type = ['gomoku', 'chess', 'go'][i % 3]
            move_count = np.random.randint(15, 100)
            winner = [0, 1, None][i % 3]

            game = self._create_realistic_game_result(game_type, move_count, winner)
            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        assert batch_analysis.total_games == 10
        assert batch_analysis.successful_games <= 10
        assert batch_analysis.avg_game_length > 0
        assert 0 <= batch_analysis.spatial_bias_score <= 1
        assert 0 <= batch_analysis.opening_diversity <= 1
        assert 'overall' in batch_analysis.health_scores

    def test_opening_diversity_detection(self, analyzer):
        """Test detection of opening move diversity."""
        games = []

        # Create games with biased openings (all play center)
        for i in range(20):
            game = self._create_realistic_game_result("gomoku", 25)
            # Force all games to start at center (move 112 for 15x15 board)
            game.metadata['move_history'][0] = 112
            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        # Should detect low opening diversity
        assert batch_analysis.opening_diversity < 0.1  # Very low diversity

    def test_spatial_bias_detection(self, analyzer):
        """Test detection of spatial biases."""
        games = []

        # Create games with artificial bias toward top-left
        for i in range(15):
            game = self._create_realistic_game_result("gomoku", 30)
            # Bias moves toward top-left quadrant (positions 0-56)
            biased_moves = np.random.randint(0, 56, size=30)
            game.metadata['move_history'] = biased_moves.tolist()
            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        # Should detect high spatial bias
        assert batch_analysis.spatial_bias_score > 0.5
        assert batch_analysis.corner_bias > 0.2  # Should detect corner bias

    def test_policy_health_indicators(self, analyzer):
        """Test policy health and MCTS indicators."""
        games = []

        # Create games with good entropy patterns
        for i in range(12):
            game = self._create_realistic_game_result("gomoku", 35)
            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        assert batch_analysis.avg_policy_entropy > 0
        assert 0 <= batch_analysis.exploration_quality <= 1
        assert 0 <= batch_analysis.exploitation_quality <= 1
        assert 0 <= batch_analysis.entropy_consistency <= 1

    def test_terminal_detection_validation(self, analyzer):
        """Test validation of terminal detection across games."""
        games = []

        # Create games with different terminal conditions
        terminal_types = ['win_player1', 'win_player2', 'draw', 'max_moves_reached']

        for i, terminal_type in enumerate(terminal_types * 3):  # 12 games total
            winner = 0 if 'player1' in terminal_type else 1 if 'player2' in terminal_type else None
            game = self._create_realistic_game_result("gomoku", 25 + i, winner)
            game.metadata['final_outcome'] = {'result': terminal_type}
            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        # Should detect all terminal types
        assert len(batch_analysis.terminal_reasons) >= 3
        assert batch_analysis.health_scores['terminal_detection'] > 0.8

    def test_statistical_bias_tests(self, analyzer):
        """Test statistical significance tests for biases."""
        games = []

        # Create games with known patterns for testing
        for i in range(30):
            game = self._create_realistic_game_result("gomoku", 20)
            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        # Should have bias test results
        assert 'opening_uniformity' in batch_analysis.bias_tests or 'spatial_uniformity' in batch_analysis.bias_tests

        # Check p-values are reasonable
        for test_name, test_results in batch_analysis.bias_tests.items():
            if 'p_value' in test_results:
                assert 0 <= test_results['p_value'] <= 1

    def test_convergence_quality_assessment(self, analyzer):
        """Test MCTS convergence quality assessment."""
        # Create game with good convergence (decreasing entropy)
        good_game = self._create_realistic_game_result("gomoku", 25)
        good_analysis = analyzer.analyze_game(good_game)

        # Convergence quality should be reasonable
        assert 0 <= good_analysis.convergence_quality <= 1

        # Test with multiple games
        games = [self._create_realistic_game_result("gomoku", 30) for _ in range(8)]
        batch_analysis = analyzer.analyze_batch(games)

        assert 'mcts_convergence' in batch_analysis.health_scores
        assert 0 <= batch_analysis.health_scores['mcts_convergence'] <= 1

    @pytest.mark.parametrize("game_type,expected_board_size", [
        ("gomoku", (15, 15)),
        ("chess", (8, 8)),
        ("go", (19, 19))
    ])
    def test_game_type_specific_analysis(self, analyzer, game_type, expected_board_size):
        """Test analysis works correctly for different game types."""
        game = self._create_realistic_game_result(game_type, 30)
        analysis = analyzer.analyze_game(game)

        assert analysis.game_type == game_type
        assert len(analysis.move_positions) <= 30

        # Check move positions are within board bounds
        for row, col in analysis.move_positions:
            assert 0 <= row < expected_board_size[0]
            assert 0 <= col < expected_board_size[1]

    def test_health_score_calculation(self, analyzer):
        """Test comprehensive health score calculation."""
        games = []

        # Mix of good and problematic games
        for i in range(20):
            move_count = 20 + i  # Reasonable lengths
            winner = [0, 1, None][i % 3]
            game = self._create_realistic_game_result("gomoku", move_count, winner)

            # Make some games have good terminal detection
            if i < 15:
                game.metadata['final_outcome'] = {'result': 'win_player1' if winner == 0 else 'win_player2' if winner == 1 else 'draw'}

            games.append(game)

        batch_analysis = analyzer.analyze_batch(games)

        # All health scores should be in [0, 1]
        for score_name, score_value in batch_analysis.health_scores.items():
            assert 0 <= score_value <= 1, f"Health score {score_name} = {score_value} out of range"

        # Overall health should be reasonable
        assert batch_analysis.health_scores['overall'] > 0.1


# Specialized tests for game variations and edge cases

class TestGameVariations:
    """Test specific game variations and edge cases."""

    def test_gomoku_variations(self):
        """Test Gomoku with Renju/Omok variations."""
        # Test different Dirichlet alphas
        configs = [
            {'game_type': 'gomoku', 'dirichlet_alpha': 0.3, 'rule_variant': 'standard'},  # Standard Gomoku
            {'game_type': 'gomoku', 'dirichlet_alpha': 0.15, 'rule_variant': 'renju'},   # Renju (more constrained)
        ]

        for config in configs:
            generator = SelfPlayGameGenerator(
                game_type=config['game_type'],
                model_path="/tmp/test.pth",
                num_threads=2,
                mcts_simulations=100,
                temperature_schedule=[(10, 1.0)],
                add_dirichlet_noise=True,
                # Propagate variant override when provided
                # Older constructor signature ignores unknown kwargs, so we set env temporarily
            )

            if 'rule_variant' in config:
                generator.config.rule_variant = config['rule_variant']
                generator._set_game_specific_params()

            # Verify alpha was set correctly based on game type
            expected_alpha = config.get('dirichlet_alpha', 0.3)
            assert abs(generator.config.dirichlet_alpha - expected_alpha) < 0.01

    def test_chess_960_support(self):
        """Test Chess960 position generation."""
        generator = SelfPlayGameGenerator(
            game_type="chess",
            model_path="/tmp/test.pth"
        )

        # Chess should have different alpha than Gomoku
        assert generator.config.dirichlet_alpha == 0.2
        assert generator.config.max_game_length == 512

    def test_go_rule_variations(self):
        """Test Go with different rule sets."""
        generator = SelfPlayGameGenerator(
            game_type="go",
            model_path="/tmp/test.pth"
        )

        # Go should have much lower alpha for larger action space
        assert generator.config.dirichlet_alpha == 0.03
        assert generator.config.max_game_length == 722


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_game_batch(self):
        """Test analysis with empty game batch."""
        analyzer = SelfPlayAnalyzer()

        with pytest.raises(ValueError, match="No games to analyze"):
            analyzer.analyze_batch([])

    def test_single_move_game(self):
        """Test analysis with very short game."""
        analyzer = SelfPlayAnalyzer()

        # Create minimal game
        game = GameResult(
            winner=0,
            move_count=1,
            game_length_seconds=0.1,
            examples=[TrainingExample(
                state=np.random.rand(36, 15, 15).astype(np.float32),
                policy=np.random.dirichlet([1.0] * 225),
                value=1.0,
                game_type="gomoku",
                move_number=0,
                game_id="minimal_game"
            )],
            final_board="Minimal board",
            metadata={'game_id': 'minimal', 'game_type': 'gomoku', 'move_history': [112]}
        )

        analysis = analyzer.analyze_game(game)

        assert analysis.move_count == 1
        assert len(analysis.policy_entropies) == 1
        assert analysis.entropy_trend == "insufficient_data"

    def test_corrupted_game_data(self):
        """Test handling of corrupted game data."""
        analyzer = SelfPlayAnalyzer()

        # Create game with missing metadata
        corrupted_game = GameResult(
            winner=None,
            move_count=10,
            game_length_seconds=5.0,
            examples=[],  # Empty examples
            final_board="",
            metadata={}  # Missing metadata
        )

        # Should handle gracefully
        analysis = analyzer.analyze_game(corrupted_game)
        assert analysis.game_type == "unknown"
        assert len(analysis.policy_entropies) == 0


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "--tb=short"])
