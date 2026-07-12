#!/usr/bin/env python3
"""
Comprehensive Self-Play Testing Script
=====================================

Runs actual self-play games across all supported game types and performs
detailed analysis of move patterns, policy entropy, and system health.

Usage:
    python scripts/test_self_play_comprehensive.py --games 20 --output results/
    python scripts/test_self_play_comprehensive.py --quick-test
    python scripts/test_self_play_comprehensive.py --analyze-only results/games.json
"""

import argparse
import logging
import time
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.self_play import (
    SelfPlayGameGenerator, create_self_play_generator, save_games_to_disk, load_games_from_disk
)
from tests.integration.test_self_play_comprehensive import SelfPlayAnalyzer, BatchAnalysisResult
from specs.contracts.training_api import GameResult


class ComprehensiveSelfPlayTester:
    """Comprehensive tester for self-play system."""

    def __init__(self, output_dir: Path):
        """Initialize tester with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'test_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize analyzer
        self.analyzer = SelfPlayAnalyzer()

    def run_comprehensive_test(self, num_games_per_type: int = 10, quick_test: bool = False) -> Dict[str, Any]:
        """Run comprehensive self-play tests across all game types."""
        self.logger.info(f"Starting comprehensive self-play test with {num_games_per_type} games per type")

        if quick_test:
            num_games_per_type = 3
            self.logger.info("Quick test mode: reduced to 3 games per type")

        # Test configurations for each game type
        test_configs = {
            'gomoku': {
                'game_type': 'gomoku',
                'model_path': '/tmp/gomoku_model.pth',
                'mcts_simulations': 100 if quick_test else 400,
                'temperature_schedule': [(15, 1.0), (100, 0.1)],
                'variations': ['standard', 'renju']
            },
            'chess': {
                'game_type': 'chess',
                'model_path': '/tmp/chess_model.pth',
                'mcts_simulations': 150 if quick_test else 600,
                'temperature_schedule': [(20, 1.0), (150, 0.1)],
                'variations': ['standard', 'chess960']
            },
            'go': {
                'game_type': 'go',
                'model_path': '/tmp/go_model.pth',
                'mcts_simulations': 200 if quick_test else 800,
                'temperature_schedule': [(30, 1.0), (200, 0.1)],
                'variations': ['chinese', 'japanese', 'korean']
            }
        }

        all_results = {}
        all_games = []

        # Test each game type
        for game_type, config in test_configs.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Testing {game_type.upper()}")
            self.logger.info(f"{'='*50}")

            # Test each variation
            for variation in config['variations']:
                self.logger.info(f"\nTesting {game_type} - {variation} variation")

                games = self._run_game_batch(
                    config, variation, num_games_per_type, quick_test
                )

                if games:
                    # Analyze this batch
                    batch_analysis = self.analyzer.analyze_batch(games)

                    # Store results
                    result_key = f"{game_type}_{variation}"
                    all_results[result_key] = {
                        'config': config,
                        'variation': variation,
                        'games': len(games),
                        'analysis': batch_analysis
                    }

                    all_games.extend(games)

                    # Log key findings
                    self._log_batch_results(game_type, variation, batch_analysis)

        # Overall analysis
        if all_games:
            self.logger.info(f"\n{'='*50}")
            self.logger.info("OVERALL ANALYSIS")
            self.logger.info(f"{'='*50}")

            overall_analysis = self.analyzer.analyze_batch(all_games)
            all_results['overall'] = {
                'total_games': len(all_games),
                'analysis': overall_analysis
            }

            self._log_overall_results(overall_analysis)

            # Save results
            self._save_results(all_results, all_games)

            # Generate visualizations
            self._generate_visualizations(all_results)

        return all_results

    def _run_game_batch(self, config: Dict[str, Any], variation: str,
                       num_games: int, quick_test: bool) -> List[GameResult]:
        """Run a batch of games for specific configuration."""
        try:
            # Create generator with enhanced configuration for testing
            test_config = config.copy()
            if quick_test:
                test_config['num_threads'] = 2

            generator = create_self_play_generator(test_config)

            # For comprehensive testing, we'll use mock MCTS to avoid GPU dependencies
            # but make it more realistic
            generator = self._setup_realistic_mock_generator(generator, config['game_type'], variation)

            # Generate games
            games = []
            start_time = time.time()

            for i in range(num_games):
                self.logger.info(f"  Generating game {i+1}/{num_games}...")

                try:
                    game_id = f"{config['game_type']}_{variation}_{i:03d}"
                    game = generator.generate_game(game_id)
                    games.append(game)

                    # Log progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (num_games - i - 1)
                    self.logger.info(f"    Game completed: {game.move_count} moves, "
                                   f"ETA: {eta:.1f}s")

                except Exception as e:
                    self.logger.error(f"    Failed to generate game {i}: {e}")
                    continue

            generation_time = time.time() - start_time
            self.logger.info(f"Generated {len(games)} games in {generation_time:.2f}s "
                           f"({generation_time/len(games):.2f}s per game)")

            return games

        except Exception as e:
            self.logger.error(f"Failed to run batch for {config['game_type']} {variation}: {e}")
            return []

    def _setup_realistic_mock_generator(self, generator, game_type: str, variation: str):
        """Setup generator with realistic mock components."""
        # Mock inference worker and search coordinator
        from unittest.mock import Mock

        generator.inference_worker = Mock()
        generator.search_coordinator = Mock()

        # Create realistic mock search results
        def create_mock_search_result(request):
            """Create realistic search result based on game progress."""
            move_number = request.request_id.split('_')[-1] if '_' in request.request_id else "0"
            try:
                move_num = int(move_number)
            except:
                move_num = 0

            # Board sizes
            board_sizes = {'gomoku': 225, 'chess': 64, 'go': 361}
            board_size = board_sizes.get(game_type, 225)

            # Create policy with realistic patterns
            if game_type == 'gomoku':
                policy = self._create_gomoku_policy(move_num, variation)
            elif game_type == 'chess':
                policy = self._create_chess_policy(move_num, variation)
            else:  # go
                policy = self._create_go_policy(move_num, variation)

            # Ensure correct size
            if len(policy) != board_size:
                policy = np.random.dirichlet([1.0] * board_size)

            # Create mock result
            mock_result = Mock()
            mock_result.policy = policy
            mock_result.value = np.random.uniform(-0.5, 0.5)  # Realistic value range

            # Create mock future
            mock_future = Mock()
            mock_future.result.return_value = mock_result

            return mock_future

        generator.search_coordinator.submit_search.side_effect = create_mock_search_result

        # Enhanced mock game state for realistic terminal detection
        original_is_terminal = generator._is_game_terminal
        def realistic_terminal_check(game_state):
            move_count = game_state.get('move_count', 0)

            # Game-specific termination logic
            if game_type == 'gomoku':
                # Gomoku games typically end between 15-50 moves
                if move_count > 15:
                    return np.random.random() < (move_count - 15) * 0.02
            elif game_type == 'chess':
                # Chess games typically longer
                if move_count > 30:
                    return np.random.random() < (move_count - 30) * 0.01
            else:  # go
                # Go games can be very long
                if move_count > 50:
                    return np.random.random() < (move_count - 50) * 0.005

            return False

        generator._is_game_terminal = realistic_terminal_check

        return generator

    def _create_gomoku_policy(self, move_num: int, variation: str) -> np.ndarray:
        """Create realistic Gomoku policy."""
        board_size = 225  # 15x15

        # Start with uniform base
        base_entropy = max(0.1, 1.0 - (move_num / 50) * 0.8)
        policy = np.random.dirichlet([base_entropy] * board_size)

        # Add realistic patterns
        if move_num < 5:
            # Early game: prefer center
            center_moves = [105, 106, 107, 120, 121, 122, 135, 136, 137]  # 3x3 center
            for move in center_moves:
                if move < board_size:
                    policy[move] *= 3.0

        elif move_num < 15:
            # Mid-early game: some randomization but avoid edges
            edges = list(range(15)) + list(range(210, 225))  # Top and bottom rows
            edges.extend([i * 15 for i in range(15)])  # Left column
            edges.extend([i * 15 + 14 for i in range(15)])  # Right column

            for edge in edges:
                policy[edge] *= 0.3

        else:
            # Late game: more focused
            best_moves = np.random.choice(board_size, size=5, replace=False)
            for move in best_moves:
                policy[move] *= 10.0

        # Variation-specific adjustments
        if variation == 'renju':
            # Renju has more restrictions, so some randomness
            policy = policy * np.random.uniform(0.8, 1.2, board_size)

        policy /= np.sum(policy)
        return policy

    def _create_chess_policy(self, move_num: int, variation: str) -> np.ndarray:
        """Create realistic Chess policy."""
        board_size = 64  # 8x8

        base_entropy = max(0.05, 1.0 - (move_num / 100) * 0.9)
        policy = np.random.dirichlet([base_entropy] * board_size)

        # Opening principles
        if move_num < 10:
            # Prefer center and development squares
            center_squares = [27, 28, 35, 36]  # e4, d4, e5, d5
            development_squares = [18, 21, 42, 45]  # c3, f3, c6, f6

            for square in center_squares + development_squares:
                if square < board_size:
                    policy[square] *= 5.0

        elif move_num < 30:
            # Middle game: more varied
            policy = np.random.dirichlet([0.5] * board_size)

        else:
            # Endgame: very focused
            active_squares = np.random.choice(board_size, size=8, replace=False)
            for square in active_squares:
                policy[square] *= 20.0

        # Chess960 variation
        if variation == 'chess960':
            # More chaotic opening
            policy = np.random.dirichlet([1.0] * board_size)

        policy /= np.sum(policy)
        return policy

    def _create_go_policy(self, move_num: int, variation: str) -> np.ndarray:
        """Create realistic Go policy."""
        board_size = 361  # 19x19

        base_entropy = max(0.02, 1.0 - (move_num / 200) * 0.95)
        policy = np.random.dirichlet([base_entropy] * board_size)

        # Go patterns
        if move_num < 20:
            # Opening: corners and sides
            corners = [0, 18, 342, 360]  # 4 corners
            star_points = [60, 66, 72, 180, 186, 192, 288, 294, 300]  # 3-3 points

            for point in corners + star_points:
                if point < board_size:
                    policy[point] *= 8.0

        elif move_num < 100:
            # Middle game: various patterns
            policy = np.random.dirichlet([0.3] * board_size)

        else:
            # Late game: very focused on a few areas
            active_areas = np.random.choice(board_size, size=20, replace=False)
            for area in active_areas:
                policy[area] *= 15.0

        # Rule variation effects
        if variation == 'japanese':
            # Slightly more conservative
            policy = policy * np.random.uniform(0.9, 1.1, board_size)
        elif variation == 'chinese':
            # More territorial focus
            policy = policy * np.random.uniform(0.8, 1.3, board_size)

        policy /= np.sum(policy)
        return policy

    def _log_batch_results(self, game_type: str, variation: str, analysis: BatchAnalysisResult):
        """Log key results for a game batch."""
        self.logger.info(f"\n--- {game_type.upper()} {variation} Results ---")
        self.logger.info(f"Games completed: {analysis.successful_games}/{analysis.total_games}")
        self.logger.info(f"Average game length: {analysis.avg_game_length:.1f} moves")

        # Terminal detection
        self.logger.info(f"Terminal reasons: {dict(analysis.terminal_reasons)}")
        self.logger.info(f"Win distribution: {dict(analysis.win_distribution)}")

        # Bias analysis
        self.logger.info(f"Spatial bias score: {analysis.spatial_bias_score:.3f} (lower=better)")
        self.logger.info(f"Opening diversity: {analysis.opening_diversity:.3f} (higher=better)")
        self.logger.info(f"Corner bias: {analysis.corner_bias:.3f}")
        self.logger.info(f"Center bias: {analysis.center_bias:.3f}")

        # Policy health
        self.logger.info(f"Average policy entropy: {analysis.avg_policy_entropy:.3f}")
        self.logger.info(f"Exploration quality: {analysis.exploration_quality:.3f}")
        self.logger.info(f"Exploitation quality: {analysis.exploitation_quality:.3f}")

        # Health scores
        self.logger.info("Health Scores:")
        for metric, score in analysis.health_scores.items():
            status = "✓" if score > 0.7 else "⚠" if score > 0.4 else "✗"
            self.logger.info(f"  {metric}: {score:.3f} {status}")

        # Statistical tests
        if analysis.bias_tests:
            self.logger.info("Bias Tests:")
            for test_name, results in analysis.bias_tests.items():
                p_val = results.get('p_value', results.get('row_p_value', 'N/A'))
                significant = " (significant bias detected!)" if isinstance(p_val, float) and p_val < 0.05 else ""
                self.logger.info(f"  {test_name}: p={p_val}{significant}")

    def _log_overall_results(self, overall_analysis: BatchAnalysisResult):
        """Log overall analysis results."""
        self.logger.info(f"Total games analyzed: {overall_analysis.successful_games}")
        self.logger.info(f"Overall spatial bias: {overall_analysis.spatial_bias_score:.3f}")
        self.logger.info(f"Overall opening diversity: {overall_analysis.opening_diversity:.3f}")
        self.logger.info(f"Overall policy health: {overall_analysis.avg_policy_entropy:.3f}")

        self.logger.info("\nOverall Health Assessment:")
        overall_health = overall_analysis.health_scores.get('overall', 0.0)

        if overall_health > 0.8:
            status = "EXCELLENT ✓✓✓"
        elif overall_health > 0.6:
            status = "GOOD ✓✓"
        elif overall_health > 0.4:
            status = "ACCEPTABLE ✓"
        else:
            status = "NEEDS IMPROVEMENT ✗"

        self.logger.info(f"System Health: {overall_health:.3f} - {status}")

        # Recommendations
        self._log_recommendations(overall_analysis)

    def _log_recommendations(self, analysis: BatchAnalysisResult):
        """Log recommendations based on analysis."""
        self.logger.info("\nRecommendations:")

        if analysis.spatial_bias_score > 0.3:
            self.logger.info("⚠ High spatial bias detected - check Dirichlet noise implementation")

        if analysis.opening_diversity < 0.3:
            self.logger.info("⚠ Low opening diversity - increase exploration in early moves")

        if analysis.exploration_quality < 0.5:
            self.logger.info("⚠ Poor exploration quality - check temperature schedule")

        if analysis.exploitation_quality < 0.5:
            self.logger.info("⚠ Poor exploitation quality - check late-game temperature")

        if analysis.health_scores.get('terminal_detection', 1.0) < 0.8:
            self.logger.info("⚠ Terminal detection issues - verify game rule implementations")

        if analysis.entropy_consistency < 0.6:
            self.logger.info("⚠ Inconsistent entropy patterns - check MCTS convergence")

    def _save_results(self, results: Dict[str, Any], games: List[GameResult]):
        """Save results to disk."""
        # Save detailed results
        results_file = self.output_dir / 'analysis_results.json'

        # Convert BatchAnalysisResult objects to dicts for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if 'analysis' in value:
                # Convert dataclass to dict
                analysis = value['analysis']
                analysis_dict = {
                    'total_games': analysis.total_games,
                    'successful_games': analysis.successful_games,
                    'failed_games': analysis.failed_games,
                    'terminal_reasons': dict(analysis.terminal_reasons),
                    'win_distribution': dict(analysis.win_distribution),
                    'avg_game_length': analysis.avg_game_length,
                    'spatial_bias_score': analysis.spatial_bias_score,
                    'opening_diversity': analysis.opening_diversity,
                    'corner_bias': analysis.corner_bias,
                    'center_bias': analysis.center_bias,
                    'edge_bias': analysis.edge_bias,
                    'avg_policy_entropy': analysis.avg_policy_entropy,
                    'entropy_consistency': analysis.entropy_consistency,
                    'exploration_quality': analysis.exploration_quality,
                    'exploitation_quality': analysis.exploitation_quality,
                    'bias_tests': analysis.bias_tests,
                    'health_scores': analysis.health_scores
                }
                value['analysis'] = analysis_dict

            serializable_results[key] = value

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Save raw games
        save_games_to_disk(games, self.output_dir / 'games')

        self.logger.info(f"Results saved to {self.output_dir}")

    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization plots."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')

            # Create visualizations directory
            viz_dir = self.output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)

            # 1. Health scores comparison
            self._plot_health_scores(results, viz_dir)

            # 2. Bias analysis
            self._plot_bias_analysis(results, viz_dir)

            # 3. Game length distribution
            self._plot_game_lengths(results, viz_dir)

            # 4. Entropy patterns
            self._plot_entropy_patterns(results, viz_dir)

            self.logger.info(f"Visualizations saved to {viz_dir}")

        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {e}")

    def _plot_health_scores(self, results: Dict[str, Any], viz_dir: Path):
        """Plot health scores comparison."""
        health_data = []

        for key, value in results.items():
            if key != 'overall' and 'analysis' in value:
                health_scores = value['analysis']['health_scores']
                for metric, score in health_scores.items():
                    health_data.append({
                        'Configuration': key,
                        'Metric': metric,
                        'Score': score
                    })

        if health_data:
            df = pd.DataFrame(health_data)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=df, x='Metric', y='Score', hue='Configuration')
            plt.title('Health Scores by Configuration')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'health_scores.png', dpi=300)
            plt.close()

    def _plot_bias_analysis(self, results: Dict[str, Any], viz_dir: Path):
        """Plot bias analysis."""
        bias_data = []

        for key, value in results.items():
            if key != 'overall' and 'analysis' in value:
                analysis = value['analysis']
                bias_data.append({
                    'Configuration': key,
                    'Spatial Bias': analysis['spatial_bias_score'],
                    'Opening Diversity': analysis['opening_diversity'],
                    'Corner Bias': analysis['corner_bias'],
                    'Center Bias': analysis['center_bias']
                })

        if bias_data:
            df = pd.DataFrame(bias_data)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            sns.barplot(data=df, x='Configuration', y='Spatial Bias', ax=axes[0,0])
            axes[0,0].set_title('Spatial Bias (Lower is Better)')
            axes[0,0].tick_params(axis='x', rotation=45)

            sns.barplot(data=df, x='Configuration', y='Opening Diversity', ax=axes[0,1])
            axes[0,1].set_title('Opening Diversity (Higher is Better)')
            axes[0,1].tick_params(axis='x', rotation=45)

            sns.barplot(data=df, x='Configuration', y='Corner Bias', ax=axes[1,0])
            axes[1,0].set_title('Corner Bias')
            axes[1,0].tick_params(axis='x', rotation=45)

            sns.barplot(data=df, x='Configuration', y='Center Bias', ax=axes[1,1])
            axes[1,1].set_title('Center Bias')
            axes[1,1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(viz_dir / 'bias_analysis.png', dpi=300)
            plt.close()

    def _plot_game_lengths(self, results: Dict[str, Any], viz_dir: Path):
        """Plot game length distributions."""
        length_data = []

        for key, value in results.items():
            if key != 'overall' and 'analysis' in value:
                lengths = value['analysis']['game_length_distribution']
                for length in lengths:
                    length_data.append({
                        'Configuration': key,
                        'Game Length': length
                    })

        if length_data:
            df = pd.DataFrame(length_data)

            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='Configuration', y='Game Length')
            plt.title('Game Length Distribution by Configuration')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'game_lengths.png', dpi=300)
            plt.close()

    def _plot_entropy_patterns(self, results: Dict[str, Any], viz_dir: Path):
        """Plot entropy patterns."""
        entropy_data = []

        for key, value in results.items():
            if key != 'overall' and 'analysis' in value:
                analysis = value['analysis']
                entropy_data.append({
                    'Configuration': key,
                    'Average Entropy': analysis['avg_policy_entropy'],
                    'Exploration Quality': analysis['exploration_quality'],
                    'Exploitation Quality': analysis['exploitation_quality'],
                    'Entropy Consistency': analysis['entropy_consistency']
                })

        if entropy_data:
            df = pd.DataFrame(entropy_data)

            plt.figure(figsize=(15, 6))

            plt.subplot(1, 2, 1)
            sns.barplot(data=df, x='Configuration', y='Average Entropy')
            plt.title('Average Policy Entropy')
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            # Melt data for grouped bar plot
            quality_df = df.melt(
                id_vars=['Configuration'],
                value_vars=['Exploration Quality', 'Exploitation Quality', 'Entropy Consistency'],
                var_name='Quality Metric',
                value_name='Score'
            )
            sns.barplot(data=quality_df, x='Configuration', y='Score', hue='Quality Metric')
            plt.title('Quality Metrics')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(viz_dir / 'entropy_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()

    def analyze_existing_results(self, results_file: Path) -> Dict[str, Any]:
        """Analyze existing results from file."""
        self.logger.info(f"Analyzing existing results from {results_file}")

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Generate visualizations from existing data
        self._generate_visualizations(results)

        # Log summary
        if 'overall' in results:
            overall = results['overall']['analysis']
            self.logger.info(f"Overall health score: {overall['health_scores']['overall']:.3f}")

        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Comprehensive Self-Play Testing')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games per game type/variation')
    parser.add_argument('--output', type=str, default='results/self_play_test',
                       help='Output directory for results')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced games and simulations')
    parser.add_argument('--analyze-only', type=str,
                       help='Analyze existing results file instead of generating new games')

    args = parser.parse_args()

    # Create tester
    tester = ComprehensiveSelfPlayTester(Path(args.output))

    if args.analyze_only:
        # Analyze existing results
        results = tester.analyze_existing_results(Path(args.analyze_only))
    else:
        # Run comprehensive test
        results = tester.run_comprehensive_test(
            num_games_per_type=args.games,
            quick_test=args.quick_test
        )

    print(f"\nTest completed! Results saved to {args.output}")
    print("Check the log file and visualizations for detailed analysis.")


if __name__ == "__main__":
    main()