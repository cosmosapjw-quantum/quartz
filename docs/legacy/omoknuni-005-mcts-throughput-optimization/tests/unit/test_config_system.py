"""
Unit tests for the configuration management system.

Tests configuration loading, validation, environment overrides,
and integration with the AlphaZero engine components.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from src.utils.config import (
    ConfigManager, AlphaZeroConfig, MCTSConfig, NeuralNetworkConfig,
    TrainingConfig, GameConfig, SystemConfig, ConfigurationError,
    load_config, get_config, get_config_manager
)


class TestConfigurationDataClasses:
    """Test configuration dataclass creation and validation."""

    def test_mcts_config_defaults(self):
        """Test MCTSConfig default values."""
        config = MCTSConfig()

        assert config.simulations == 800
        assert config.exploration_constant == 1.0
        assert config.virtual_loss == 1.0
        assert config.threads == 8
        assert config.max_tree_size_mb == 1024
        assert config.batch_size_min == 32
        assert config.batch_size_max == 64

    def test_neural_network_config_defaults(self):
        """Test NeuralNetworkConfig default values."""
        config = NeuralNetworkConfig()

        assert config.channels == 256
        assert config.blocks == 20
        assert config.se_ratio == 0.25
        assert config.learning_rate == 0.001
        assert config.use_mixed_precision is True
        assert config.device == "cuda"

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()

        assert config.self_play_games_per_iteration == 50
        assert config.parallel_self_play_games == 4
        assert config.training_steps_per_iteration == 1000
        assert config.batch_size == 512
        assert config.experience_buffer_size == 1_000_000
        assert config.lr_schedule == "cosine"

    def test_game_config_defaults(self):
        """Test GameConfig default values."""
        config = GameConfig()

        assert config.game_type == "gomoku"
        assert config.board_size == 15
        assert config.win_condition == 5
        assert config.rule_variant == "standard"
        assert config.history_length == 8
        assert "gomoku" in config.dirichlet_alpha_overrides

    def test_system_config_defaults(self):
        """Test SystemConfig default values."""
        config = SystemConfig()

        assert config.log_level == "INFO"
        assert config.enable_tensorboard is True
        assert config.max_memory_gb == 32.0
        assert config.model_dir == "models"

    def test_alphazero_config_creation(self):
        """Test AlphaZeroConfig creation with sub-configurations."""
        config = AlphaZeroConfig()

        assert isinstance(config.mcts, MCTSConfig)
        assert isinstance(config.neural_network, NeuralNetworkConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.game, GameConfig)
        assert isinstance(config.system, SystemConfig)
        assert config.config_version == "1.0"

    def test_config_with_custom_values(self):
        """Test configuration creation with custom values."""
        mcts_config = MCTSConfig(simulations=1600, threads=12)
        config = AlphaZeroConfig(mcts=mcts_config)

        assert config.mcts.simulations == 1600
        assert config.mcts.threads == 12
        assert config.mcts.exploration_constant == 1.0  # Default value


class TestConfigManager:
    """Test ConfigManager functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_config_file(self, config_dict: dict) -> str:
        """Create a test configuration file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return self.config_path

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert manager.config_path == "config/default.yaml"
        assert manager._config is None

        custom_manager = ConfigManager("custom/path.yaml")
        assert custom_manager.config_path == "custom/path.yaml"

    def test_load_basic_config(self):
        """Test loading basic configuration."""
        config_dict = {
            "config_version": "1.0",
            "mcts": {"simulations": 1200, "threads": 10},
            "neural_network": {"channels": 128, "blocks": 15},
            "training": {"batch_size": 256},
            "game": {"game_type": "chess"},
            "system": {"log_level": "DEBUG"}
        }

        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        config = manager.load_config()

        assert config.config_version == "1.0"
        assert config.mcts.simulations == 1200
        assert config.mcts.threads == 10
        assert config.neural_network.channels == 128
        assert config.training.batch_size == 256
        assert config.game.game_type == "chess"
        assert config.system.log_level == "DEBUG"

    def test_load_config_with_missing_sections(self):
        """Test loading configuration with missing sections uses defaults."""
        config_dict = {
            "mcts": {"simulations": 1600}
        }

        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        config = manager.load_config()

        # Specified values
        assert config.mcts.simulations == 1600

        # Default values from missing sections
        assert config.mcts.threads == 8  # Default
        assert config.neural_network.channels == 256  # Default
        assert config.training.batch_size == 512  # Default

    def test_load_nonexistent_config_file(self):
        """Test loading non-existent configuration file raises error."""
        manager = ConfigManager("nonexistent/path.yaml")

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "Configuration file not found" in str(exc_info.value)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with open(self.config_path, 'w') as f:
            f.write("invalid: yaml: content: {")

        manager = ConfigManager(self.config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "Failed to parse YAML file" in str(exc_info.value)

    def test_get_config_before_loading(self):
        """Test get_config before loading raises error."""
        manager = ConfigManager()

        with pytest.raises(ConfigurationError) as exc_info:
            manager.get_config()

        assert "Configuration not loaded" in str(exc_info.value)

    def test_get_config_after_loading(self):
        """Test get_config after loading returns same instance."""
        config_dict = {"mcts": {"simulations": 1000}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        config1 = manager.load_config()
        config2 = manager.get_config()

        assert config1 is config2
        assert config2.mcts.simulations == 1000

    def test_save_config(self):
        """Test saving configuration to file."""
        config = AlphaZeroConfig()
        config.mcts.simulations = 1500
        config.neural_network.channels = 512

        output_path = os.path.join(self.temp_dir, "saved_config.yaml")
        manager = ConfigManager()
        manager.save_config(config, output_path)

        # Verify file was created and contains expected data
        assert os.path.exists(output_path)

        with open(output_path, 'r') as f:
            saved_dict = yaml.safe_load(f)

        assert saved_dict['mcts']['simulations'] == 1500
        assert saved_dict['neural_network']['channels'] == 512

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config_dict = {
            "mcts": {"simulations": 800, "threads": 8},
            "neural_network": {"learning_rate": 0.001}
        }

        config_path = self.create_test_config_file(config_dict)

        # Set environment variables
        env_vars = {
            "ALPHAZERO_MCTS_SIMULATIONS": "1600",
            "ALPHAZERO_MCTS_THREADS": "12",
            "ALPHAZERO_NEURAL_NETWORK_LEARNING_RATE": "0.01"
        }

        with patch.dict(os.environ, env_vars):
            manager = ConfigManager(config_path)
            config = manager.load_config()

        # Check overrides were applied
        assert config.mcts.simulations == 1600
        assert config.mcts.threads == 12
        assert config.neural_network.learning_rate == 0.01

    def test_environment_overrides_type_conversion(self):
        """Test environment variable type conversion."""
        config_dict = {
            "neural_network": {"use_mixed_precision": True},
            "mcts": {"exploration_constant": 1.0}
        }

        config_path = self.create_test_config_file(config_dict)

        env_vars = {
            "ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION": "false",
            "ALPHAZERO_MCTS_EXPLORATION_CONSTANT": "2.5"
        }

        with patch.dict(os.environ, env_vars):
            manager = ConfigManager(config_path)
            config = manager.load_config()

        assert config.neural_network.use_mixed_precision is False
        assert config.mcts.exploration_constant == 2.5

    def test_environment_overrides_invalid_conversion(self):
        """Test environment variable override with invalid type conversion."""
        config_dict = {"mcts": {"simulations": 800}}
        config_path = self.create_test_config_file(config_dict)

        env_vars = {"ALPHAZERO_MCTS_SIMULATIONS": "not_a_number"}

        with patch.dict(os.environ, env_vars):
            manager = ConfigManager(config_path)
            # Should not raise error, should log warning and use original value
            config = manager.load_config()

        assert config.mcts.simulations == 800  # Original value


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_config_file(self, config_dict: dict) -> str:
        """Create a test configuration file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return self.config_path

    def test_valid_configuration_passes(self):
        """Test that valid configuration passes validation."""
        config_dict = {
            "mcts": {"simulations": 800, "exploration_constant": 1.0, "threads": 8},
            "neural_network": {"channels": 256, "blocks": 20, "learning_rate": 0.001},
            "training": {"batch_size": 512, "experience_buffer_size": 100000, "min_experience_size": 1000},
            "game": {"game_type": "gomoku"},
            "system": {"log_level": "INFO"}
        }

        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        # Should not raise exception
        config = manager.load_config()
        assert config is not None

    def test_invalid_mcts_simulations(self):
        """Test validation fails for invalid MCTS simulations."""
        config_dict = {"mcts": {"simulations": 0}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "simulations must be positive" in str(exc_info.value)

    def test_invalid_mcts_exploration_constant(self):
        """Test validation fails for negative exploration constant."""
        config_dict = {"mcts": {"exploration_constant": -1.0}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "exploration constant must be non-negative" in str(exc_info.value)

    def test_invalid_thread_count(self):
        """Test validation fails for invalid thread count."""
        config_dict = {"mcts": {"threads": 0}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "threads must be between 1 and 32" in str(exc_info.value)

    def test_invalid_neural_network_parameters(self):
        """Test validation fails for invalid neural network parameters."""
        config_dict = {"neural_network": {"channels": 0, "blocks": 0, "learning_rate": -0.1}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError):
            manager.load_config()

    def test_invalid_game_type(self):
        """Test validation fails for unsupported game type."""
        config_dict = {"game": {"game_type": "invalid_game"}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "Unsupported game type" in str(exc_info.value)

    def test_invalid_log_level(self):
        """Test validation fails for invalid log level."""
        config_dict = {"system": {"log_level": "INVALID"}}
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "Invalid log level" in str(exc_info.value)

    def test_invalid_experience_buffer_size(self):
        """Test validation fails for invalid experience buffer configuration."""
        config_dict = {
            "training": {
                "experience_buffer_size": 1000,
                "min_experience_size": 2000  # Larger than buffer size
            }
        }
        config_path = self.create_test_config_file(config_dict)
        manager = ConfigManager(config_path)

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()

        assert "buffer size must be >= min_experience_size" in str(exc_info.value)


class TestGlobalConfigurationFunctions:
    """Test global configuration management functions."""

    def setup_method(self):
        """Setup test environment."""
        # Reset global state
        import src.utils.config
        src.utils.config._config_manager = None

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

        # Reset global state
        import src.utils.config
        src.utils.config._config_manager = None

    def create_test_config_file(self, config_dict: dict) -> str:
        """Create a test configuration file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return self.config_path

    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2

    def test_load_config_convenience_function(self):
        """Test load_config convenience function."""
        config_dict = {"mcts": {"simulations": 1200}}
        config_path = self.create_test_config_file(config_dict)

        config = load_config(config_path)

        assert config.mcts.simulations == 1200
        assert isinstance(config, AlphaZeroConfig)

    def test_get_config_convenience_function(self):
        """Test get_config convenience function."""
        config_dict = {"mcts": {"simulations": 1300}}
        config_path = self.create_test_config_file(config_dict)

        # Load config first
        load_config(config_path)

        # Now get config should work
        config = get_config()
        assert config.mcts.simulations == 1300

    def test_get_config_before_load_convenience_function(self):
        """Test get_config convenience function before loading config."""
        with pytest.raises(ConfigurationError):
            get_config()


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_complete_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Create initial configuration
        config_dict = {
            "config_version": "1.0",
            "mcts": {"simulations": 800, "threads": 8},
            "neural_network": {"channels": 256, "learning_rate": 0.001},
            "training": {"batch_size": 512},
            "game": {"game_type": "gomoku"},
            "system": {"log_level": "INFO"}
        }

        config_path = os.path.join(self.temp_dir, "workflow_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        # Load configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()

        # Verify loaded correctly
        assert config.mcts.simulations == 800
        assert config.neural_network.channels == 256

        # Modify configuration
        config.mcts.simulations = 1600
        config.neural_network.learning_rate = 0.01

        # Save modified configuration
        output_path = os.path.join(self.temp_dir, "modified_config.yaml")
        manager.save_config(config, output_path)

        # Load saved configuration and verify changes
        manager2 = ConfigManager(output_path)
        config2 = manager2.load_config()

        assert config2.mcts.simulations == 1600
        assert config2.neural_network.learning_rate == 0.01

    def test_configuration_with_all_sections(self):
        """Test configuration loading with all sections populated."""
        config_dict = {
            "config_version": "1.0",
            "created_by": "test",
            "mcts": {
                "simulations": 800,
                "exploration_constant": 1.0,
                "virtual_loss": 1.0,
                "threads": 8,
                "max_tree_size_mb": 1024,
                "batch_size_min": 32,
                "batch_size_max": 64
            },
            "neural_network": {
                "channels": 256,
                "blocks": 20,
                "learning_rate": 0.001,
                "use_mixed_precision": True,
                "device": "cuda"
            },
            "training": {
                "self_play_games_per_iteration": 50,
                "batch_size": 512,
                "experience_buffer_size": 1000000
            },
            "game": {
                "game_type": "gomoku",
                "board_size": 15,
                "win_condition": 5
            },
            "system": {
                "log_level": "INFO",
                "max_memory_gb": 32.0,
                "enable_tensorboard": True
            }
        }

        config_path = os.path.join(self.temp_dir, "complete_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        manager = ConfigManager(config_path)
        config = manager.load_config()

        # Verify all sections loaded correctly
        assert config.config_version == "1.0"
        assert config.created_by == "test"
        assert config.mcts.simulations == 800
        assert config.neural_network.channels == 256
        assert config.training.batch_size == 512
        assert config.game.game_type == "gomoku"
        assert config.system.log_level == "INFO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])