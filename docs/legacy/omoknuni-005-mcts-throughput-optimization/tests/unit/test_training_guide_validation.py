"""
Unit Tests for Training Guide Validation
========================================

Validates that the training guide meets all acceptance criteria from T048.
"""

import pytest
import re
from pathlib import Path


@pytest.fixture
def training_guide_content():
    """Load training guide content."""
    guide_path = Path("docs/training_guide.md")
    assert guide_path.exists(), "Training guide must exist"
    return guide_path.read_text()


def test_training_guide_exists():
    """Test that training guide file exists at correct path."""
    guide_path = Path("docs/training_guide.md")
    assert guide_path.exists(), "Training guide must exist at docs/training_guide.md"
    assert guide_path.is_file(), "Training guide must be a file"


def test_comprehensive_structure(training_guide_content):
    """Test that training guide has comprehensive structure with all required sections."""
    required_sections = [
        "Quick Start",
        "Hyperparameter Recommendations",
        "Game-Specific Settings",
        "Training Process",
        "Performance Optimization",
        "Monitoring & Evaluation",
        "Troubleshooting",
        "Advanced Configuration",
        "Expected Performance Metrics"
    ]

    for section in required_sections:
        assert section in training_guide_content, f"Training guide must include {section} section"


def test_game_specific_content(training_guide_content):
    """Test that training guide includes specific settings for all supported games."""
    # Check for Gomoku-specific content
    assert "gomoku" in training_guide_content.lower()
    assert "15x15" in training_guide_content
    assert "5-in-a-row" in training_guide_content
    assert "48 hours" in training_guide_content or "48h" in training_guide_content

    # Check for Chess-specific content
    assert "chess" in training_guide_content.lower()
    assert "8x8" in training_guide_content
    assert "Chess960" in training_guide_content
    assert "1 week" in training_guide_content

    # Check for Go-specific content
    assert "go" in training_guide_content.lower()
    assert "9x9" in training_guide_content
    assert "19x19" in training_guide_content


def test_hyperparameter_recommendations(training_guide_content):
    """Test that comprehensive hyperparameter recommendations are included."""
    # Core MCTS parameters
    assert "simulations" in training_guide_content.lower()
    assert "exploration_constant" in training_guide_content.lower() or "puct" in training_guide_content.lower()
    assert "temperature" in training_guide_content.lower()

    # Neural network parameters
    assert "learning_rate" in training_guide_content.lower()
    assert "batch_size" in training_guide_content.lower()
    assert "weight_decay" in training_guide_content.lower()

    # Game-specific Dirichlet alpha values
    assert "dirichlet_alpha" in training_guide_content.lower()
    assert "0.3" in training_guide_content  # Gomoku
    assert "0.2" in training_guide_content  # Chess
    assert "0.03" in training_guide_content  # Go


def test_troubleshooting_coverage(training_guide_content):
    """Test that comprehensive troubleshooting section is included."""
    troubleshooting_topics = [
        "Training Loss Not Decreasing",
        "GPU Memory Errors",
        "Low GPU Utilization",
        "Training Instability",
        "Memory Leaks",
        "Performance Regression",
        "Thread Safety"
    ]

    for topic in troubleshooting_topics:
        # Check if topic or similar variation exists
        found = any(keyword in training_guide_content for keyword in [
            topic,
            topic.lower(),
            topic.replace(" ", "_").lower()
        ])
        assert found, f"Troubleshooting section must cover {topic}"


def test_performance_targets(training_guide_content):
    """Test that expected performance metrics are documented."""
    # Simulation speed targets
    assert "30,000" in training_guide_content or "30000" in training_guide_content
    assert "40,000" in training_guide_content or "40000" in training_guide_content

    # GPU utilization targets
    assert "80-92%" in training_guide_content or "80%" in training_guide_content

    # Memory targets
    assert "<1GB" in training_guide_content or "1GB" in training_guide_content

    # Self-play generation targets
    assert "games/hour" in training_guide_content.lower() or "games per hour" in training_guide_content.lower()


def test_optimization_guidelines(training_guide_content):
    """Test that performance optimization guidelines are comprehensive."""
    optimization_areas = [
        "GPU Utilization",
        "CPU Threading",
        "Memory Management",
        "Batch Size",
        "Virtual Loss"
    ]

    for area in optimization_areas:
        assert area.lower() in training_guide_content.lower(), f"Must include optimization for {area}"

    # Check for tuning scripts references
    assert "tune_threads.py" in training_guide_content
    assert "tune_batch_size.py" in training_guide_content
    assert "tune_virtual_loss.py" in training_guide_content


def test_monitoring_evaluation(training_guide_content):
    """Test that monitoring and evaluation guidance is comprehensive."""
    monitoring_topics = [
        "TensorBoard",
        "Glicko-2",
        "metrics",
        "evaluation"
    ]

    for topic in monitoring_topics:
        assert topic.lower() in training_guide_content.lower(), f"Must include guidance for {topic}"


def test_quick_start_section(training_guide_content):
    """Test that quick start section provides actionable guidance."""
    # Check for Docker and bare metal instructions
    assert "docker-compose" in training_guide_content.lower()
    assert "source venv/bin/activate" in training_guide_content or "venv" in training_guide_content

    # Check for build instructions
    assert "pip install" in training_guide_content
    assert "requirements.txt" in training_guide_content


def test_validation_commands(training_guide_content):
    """Test that validation commands are provided for verifying setup."""
    validation_patterns = [
        r"python.*validate.*\.py",
        r"python.*test.*\.py",
        r"pytest.*test.*\.py"
    ]

    has_validation = any(re.search(pattern, training_guide_content, re.IGNORECASE)
                        for pattern in validation_patterns)
    assert has_validation, "Training guide must include validation commands"


def test_expected_timelines(training_guide_content):
    """Test that realistic training timelines are provided."""
    # Check for specific timeline information
    assert "timeline" in training_guide_content.lower() or "progression" in training_guide_content.lower()

    # Check for milestone tracking
    assert "milestone" in training_guide_content.lower() or "hour" in training_guide_content.lower()


def test_comprehensive_coverage(training_guide_content):
    """Test that training guide is comprehensive (content length and depth)."""
    # Should be substantial content (at least 500 lines)
    lines = training_guide_content.split('\n')
    assert len(lines) >= 500, "Training guide should be comprehensive (500+ lines)"

    # Should have good section coverage (25+ sections/subsections)
    section_count = len([line for line in lines if line.startswith('##')])
    assert section_count >= 25, f"Training guide should have comprehensive sections (25+), found {section_count}"

    # Should have substantial content (not just headers)
    content_lines = [line for line in lines if line.strip() and not line.startswith('#')]
    assert len(content_lines) >= 400, "Training guide should have substantial content"


@pytest.mark.integration
def test_training_guide_enables_target_performance():
    """Integration test validating that the guide content supports achieving target performance."""
    guide_path = Path("docs/training_guide.md")
    content = guide_path.read_text()

    # Verify superhuman performance targets are documented
    assert "superhuman" in content.lower()

    # Verify hardware-specific optimizations are included
    assert "RTX 3060 Ti" in content
    assert "Ryzen 5900X" in content

    # Verify all game types have complete training guidance
    games = ["gomoku", "chess", "go"]
    for game in games:
        assert f"game_type: \"{game}\"" in content, f"Must have complete YAML config for {game}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])