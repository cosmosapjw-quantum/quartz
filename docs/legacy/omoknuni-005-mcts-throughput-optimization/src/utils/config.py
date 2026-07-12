"""
Configuration Management System for AlphaZero Engine

Provides unified configuration loading, validation, and environment override
capabilities for all components of the AlphaZero engine.

Features:
- YAML-based configuration files
- Environment variable overrides
- Type validation and defaults
- Nested configuration support
- Configuration merging and inheritance
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Type
from dataclasses import dataclass, field, fields
import warnings

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class MCTSConfig:
    """MCTS search configuration parameters."""

    # Core search parameters
    simulations: int = 800
    exploration_constant: float = 1.0  # PUCT exploration constant
    virtual_loss: float = 1.0
    threads: int = 8

    # Tree management
    max_tree_size_mb: int = 1024  # Maximum tree memory in MB
    transposition_table_size_mb: int = 256  # TT memory in MB
    enable_transposition_table: bool = True

    # Performance tuning
    batch_size_min: int = 32
    batch_size_max: int = 64
    inference_timeout_ms: float = 3.0

    # Game-specific parameters
    temperature: float = 1.0
    temperature_decay: float = 0.95
    dirichlet_alpha: float = 0.3  # Gomoku default
    dirichlet_weight: float = 0.25


@dataclass
class NeuralNetworkConfig:
    """Neural network architecture and inference configuration."""

    # Model architecture
    channels: int = 256
    blocks: int = 20
    se_ratio: float = 0.25  # Squeeze-Excitation ratio
    dropout: float = 0.0

    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Mixed precision
    use_mixed_precision: bool = True
    gradient_clipping: float = 1.0

    # Inference optimization
    batch_size_preferred: int = 64
    use_tensorrt: bool = False
    use_pinned_memory: bool = True

    # Device configuration
    device: str = "cuda"
    gpu_id: int = 0
    cpu_threads: int = 0  # 0 = auto-detect


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    # Self-play generation
    self_play_games_per_iteration: int = 50
    parallel_self_play_games: int = 4
    max_game_length: int = 450  # Maximum moves per game

    # Training parameters
    training_steps_per_iteration: int = 1000
    batch_size: int = 512
    validation_split: float = 0.1

    # Experience buffer
    experience_buffer_size: int = 1_000_000
    min_experience_size: int = 10_000
    sampling_temperature: float = 1.0

    # Learning schedule
    lr_schedule: str = "cosine"  # cosine, exponential, constant
    lr_warmup_steps: int = 1000
    lr_min: float = 1e-6

    # Evaluation and checkpointing
    evaluation_frequency: int = 10  # iterations
    evaluation_games: int = 100
    save_frequency: int = 5  # iterations
    max_checkpoints: int = 10

    # Early stopping
    patience: int = 50  # iterations without improvement
    min_delta: float = 0.001  # minimum improvement threshold


@dataclass
class GameConfig:
    """Game-specific configuration parameters."""

    # Game selection
    game_type: str = "gomoku"  # gomoku, chess, go

    # Game-specific parameters
    board_size: int = 15  # Gomoku: 15, Go: 9-19, Chess: 8
    win_condition: int = 5  # Gomoku: 5-in-a-row
    rule_variant: str = "standard"  # standard, renju, chinese, etc.

    # Feature extraction
    history_length: int = 8  # Move history planes
    use_enhanced_features: bool = True

    # Game-specific MCTS tuning
    dirichlet_alpha_overrides: Dict[str, float] = field(default_factory=lambda: {
        "gomoku": 0.3,
        "chess": 0.2,
        "go": 0.03
    })


@dataclass
class SystemConfig:
    """System-wide configuration parameters."""

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/alphazero.log"
    enable_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"

    # Performance monitoring
    enable_profiling: bool = False
    profile_output_dir: str = "logs/profiles"
    metrics_update_frequency: int = 10  # seconds

    # Resource limits
    max_memory_gb: float = 32.0
    max_gpu_memory_fraction: float = 0.9
    enable_memory_growth: bool = True

    # Paths
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "training_data"
    results_dir: str = "results"


@dataclass
class AlphaZeroConfig:
    """Complete AlphaZero engine configuration."""

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    neural_network: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    game: GameConfig = field(default_factory=GameConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Meta configuration
    config_version: str = "1.0"
    created_by: str = "alphazero-engine"


class ConfigManager:
    """Manages configuration loading, validation, and environment overrides."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file (default: config/default.yaml)
        """
        self.config_path = config_path or "config/default.yaml"
        self._config: Optional[AlphaZeroConfig] = None

    def load_config(self, config_path: Optional[str] = None) -> AlphaZeroConfig:
        """Load configuration from file with environment overrides.

        Args:
            config_path: Optional path override

        Returns:
            AlphaZeroConfig: Loaded and validated configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if config_path:
            self.config_path = config_path

        # Load base configuration
        config_dict = self._load_yaml_file(self.config_path)

        # Apply environment overrides
        config_dict = self._apply_environment_overrides(config_dict)

        # Validate and create configuration object
        self._config = self._create_config_from_dict(config_dict)

        # Post-load validation
        self._validate_config(self._config)

        logger.info(f"Configuration loaded from {self.config_path}")
        return self._config

    def get_config(self) -> AlphaZeroConfig:
        """Get loaded configuration.

        Returns:
            AlphaZeroConfig: Current configuration

        Raises:
            ConfigurationError: If configuration not loaded
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded. Call load_config() first.")
        return self._config

    def save_config(self, config: AlphaZeroConfig, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration to save
            output_path: Output file path
        """
        config_dict = self._config_to_dict(config)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")

    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Dict[str, Any]: Loaded configuration dictionary

        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {file_path}")

            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)

            if not isinstance(config_dict, dict):
                raise ConfigurationError(f"Invalid configuration format in {file_path}")

            return config_dict

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file {file_path}: {e}")

    def _apply_environment_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables use the format: ALPHAZERO_<SECTION>_<KEY>
        Example: ALPHAZERO_MCTS_SIMULATIONS=1600

        Args:
            config_dict: Base configuration dictionary

        Returns:
            Dict[str, Any]: Configuration with environment overrides applied
        """
        env_prefix = "ALPHAZERO_"

        for env_var, value in os.environ.items():
            if not env_var.startswith(env_prefix):
                continue

            # Parse environment variable
            var_parts = env_var[len(env_prefix):].lower().split('_')
            if len(var_parts) < 2:
                continue

            # Try to match known section names first (handle compound names like neural_network)
            section = None
            key = None
            known_sections = ['mcts', 'neural_network', 'training', 'game', 'system']

            for known_section in known_sections:
                section_parts = known_section.split('_')
                if len(var_parts) >= len(section_parts):
                    if var_parts[:len(section_parts)] == section_parts:
                        section = known_section
                        key = '_'.join(var_parts[len(section_parts):])
                        break

            # Fallback to first part as section if no known section matched
            if section is None:
                section = var_parts[0]
                key = '_'.join(var_parts[1:])

            # Ensure section exists
            if section not in config_dict:
                config_dict[section] = {}

            try:
                # Attempt type conversion based on existing value
                existing_value = config_dict[section].get(key)
                if existing_value is not None:
                    if isinstance(existing_value, bool):
                        converted_value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(existing_value, int):
                        converted_value = int(value)
                    elif isinstance(existing_value, float):
                        converted_value = float(value)
                    else:
                        converted_value = value
                else:
                    converted_value = value

                config_dict[section][key] = converted_value
                logger.info(f"Applied environment override: {env_var}={value}")

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to apply environment override {env_var}: {e}")

        return config_dict

    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> AlphaZeroConfig:
        """Create configuration object from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            AlphaZeroConfig: Configuration object

        Raises:
            ConfigurationError: If configuration creation fails
        """
        try:
            # Create sub-configurations
            mcts_config = MCTSConfig(**config_dict.get('mcts', {}))
            nn_config = NeuralNetworkConfig(**config_dict.get('neural_network', {}))
            training_config = TrainingConfig(**config_dict.get('training', {}))
            game_config = GameConfig(**config_dict.get('game', {}))
            system_config = SystemConfig(**config_dict.get('system', {}))

            # Create main configuration (filter out known sections and any intermediate sections from env vars)
            known_sections = ['mcts', 'neural_network', 'training', 'game', 'system']
            main_config_dict = {k: v for k, v in config_dict.items()
                              if k in ['config_version', 'created_by']}

            config = AlphaZeroConfig(
                mcts=mcts_config,
                neural_network=nn_config,
                training=training_config,
                game=game_config,
                system=system_config,
                **main_config_dict
            )

            return config

        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration parameters: {e}")

    def _validate_config(self, config: AlphaZeroConfig) -> None:
        """Validate configuration values.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If validation fails
        """
        # Validate MCTS parameters
        if config.mcts.simulations < 1:
            raise ConfigurationError("MCTS simulations must be positive")

        if config.mcts.exploration_constant < 0:
            raise ConfigurationError("MCTS exploration constant must be non-negative")

        if config.mcts.threads < 1 or config.mcts.threads > 32:
            raise ConfigurationError("MCTS threads must be between 1 and 32")

        # Validate neural network parameters
        if config.neural_network.channels < 1:
            raise ConfigurationError("Neural network channels must be positive")

        if config.neural_network.blocks < 1:
            raise ConfigurationError("Neural network blocks must be positive")

        if config.neural_network.learning_rate <= 0:
            raise ConfigurationError("Learning rate must be positive")

        # Validate training parameters
        if config.training.batch_size < 1:
            raise ConfigurationError("Training batch size must be positive")

        if config.training.experience_buffer_size < config.training.min_experience_size:
            raise ConfigurationError("Experience buffer size must be >= min_experience_size")

        # Validate game parameters
        if config.game.game_type not in ['gomoku', 'chess', 'go']:
            raise ConfigurationError(f"Unsupported game type: {config.game.game_type}")

        # Validate system parameters
        if config.system.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ConfigurationError(f"Invalid log level: {config.system.log_level}")

        logger.info("Configuration validation passed")

    def _config_to_dict(self, config: AlphaZeroConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary.

        Args:
            config: Configuration object

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_obj in fields(obj):
                    value = getattr(obj, field_obj.name)
                    if hasattr(value, '__dataclass_fields__'):
                        result[field_obj.name] = dataclass_to_dict(value)
                    else:
                        result[field_obj.name] = value
                return result
            else:
                return obj

        return dataclass_to_dict(config)


# Singleton instance for global access
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance.

    Returns:
        ConfigManager: Global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[str] = None) -> AlphaZeroConfig:
    """Load configuration from file (convenience function).

    Args:
        config_path: Path to configuration file

    Returns:
        AlphaZeroConfig: Loaded configuration
    """
    return get_config_manager().load_config(config_path)


def get_config() -> AlphaZeroConfig:
    """Get current configuration (convenience function).

    Returns:
        AlphaZeroConfig: Current configuration
    """
    return get_config_manager().get_config()