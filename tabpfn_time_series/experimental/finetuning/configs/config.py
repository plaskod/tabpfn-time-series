"""
Configuration system for TabPFN Time Series fine-tuning with experiment types.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, List, Union

import pprint
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager for TabPFN Time Series fine-tuning."""

    # Default locations to search for config files
    CONFIG_DIR = Path(__file__).parent
    EXPERIMENTS_DIR = CONFIG_DIR / "experiments"
    BASE_MODEL_DIR = CONFIG_DIR / "base_model"
    METHODS_DIR = CONFIG_DIR / "methods"

    @classmethod
    def load_experiment(cls, experiment_name: str) -> Optional[DictConfig]:
        """Load an experiment configuration by name."""
        experiment_path = cls._find_config(experiment_name, "experiments")
        if experiment_path is None:
            logger.error(f"Experiment configuration not found: {experiment_name}")
            return None

        try:
            # Load experiment config
            experiment_config = OmegaConf.load(experiment_path)

            # If the experiment references a method, load and merge it
            method_name = experiment_config.get("method", "default")
            method_config = cls.load_method(method_name)

            if method_config:
                # Merge method config into experiment config (experiment takes precedence)
                config = OmegaConf.merge(method_config, experiment_config)
                return config

            return experiment_config
        except Exception as e:
            logger.error(
                f"Error loading experiment configuration {experiment_name}: {e}"
            )
            return None

    @classmethod
    def load_method(cls, method_name: str) -> Optional[DictConfig]:
        """Load a method configuration by name."""
        method_path = cls._find_config(method_name, "methods")
        if method_path is None:
            logger.error(f"Method configuration not found: {method_name}")
            return None

        try:
            return OmegaConf.load(method_path)
        except Exception as e:
            logger.error(f"Error loading method configuration {method_name}: {e}")
            return None

    @classmethod
    def load_base_model(cls, base_model_name: str) -> Optional[DictConfig]:
        """Load a base model configuration by name."""
        base_model_path = cls._find_config(base_model_name, "base_model")
        if base_model_path is None:
            logger.error(f"Base model configuration not found: {base_model_name}")
            return None

        try:
            return OmegaConf.load(base_model_path)
        except Exception as e:
            logger.error(
                f"Error loading base model configuration {base_model_name}: {e}"
            )
            return None

    @classmethod
    def list_configs(cls, config_type: str = "experiments") -> List[str]:
        """List all available configurations of a specific type."""
        configs = []

        # Determine the directory to search based on config_type
        config_dir = cls._get_config_dir(config_type)

        if not config_dir.exists():
            return configs

        for config_file in config_dir.glob("**/*.yaml"):
            # Use filename without extension as the config name
            config_name = config_file.stem
            configs.append(config_name)

        return sorted(configs)

    @classmethod
    def update_from_args(cls, config: DictConfig, args: Any) -> DictConfig:
        """Update configuration from command line arguments.

        This method overrides configuration values with values from command line arguments.
        It handles both top-level arguments and nested arguments (like optimization.lr).
        """
        args_dict = {}

        # Map command line arguments to nested config paths
        # For example, --lr will update optimization.lr in the config
        nested_path_mapping = {
            "lr": "optimization.lr",
            "max_epochs": "optimization.max_epochs",
            "accumulate_grad_steps": "optimization.gradient_accumulation_steps",
        }

        # Process each command line argument
        for arg_name, arg_value in vars(args).items():
            # Case 1: Handle arguments that need to be placed in nested config structures
            if arg_name in nested_path_mapping and arg_value is not None:
                path = nested_path_mapping[arg_name].split(".")

                # Build the nested dictionary structure
                current_level = args_dict
                for i, key in enumerate(path):
                    if i == len(path) - 1:
                        # Set the value at the final level
                        current_level[key] = arg_value
                    else:
                        # Create intermediate dictionary if it doesn't exist
                        if key not in current_level:
                            current_level[key] = {}
                        current_level = current_level[key]

            # Case 2: Argument exists in config and has a non-None value - update it
            elif arg_name in config and arg_value is not None:
                args_dict[arg_name] = arg_value

            # Case 3: Argument doesn't exist in config - add it regardless of value
            elif arg_name not in config:
                args_dict[arg_name] = arg_value

            # Case 4: Argument exists in config but has None value - keep original config value
            # (No action needed as we don't add it to args_dict)

        # Convert the dictionary to OmegaConf format and merge with existing config
        args_conf = OmegaConf.create(args_dict)
        return OmegaConf.merge(config, args_conf)

    @classmethod
    def save_config(cls, config: DictConfig, file_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(file_path, "w") as f:
            OmegaConf.save(config, f)

    @classmethod
    def _find_config(
        cls, config_name: str, config_type: str = "experiments"
    ) -> Optional[Path]:
        """Find a configuration file by name and type."""
        # Check if config_name is already a path
        if os.path.exists(config_name):
            return Path(config_name)

        # Determine the directory to search based on config_type
        config_dir = cls._get_config_dir(config_type)

        if not config_dir.exists():
            os.makedirs(config_dir, exist_ok=True)

        # Try with .yaml extension
        if not config_name.endswith(".yaml"):
            config_path = config_dir / f"{config_name}.yaml"
            if config_path.exists():
                return config_path

        # Try as is
        config_path = config_dir / config_name
        if config_path.exists():
            return config_path

        return None

    @classmethod
    def _get_config_dir(cls, config_type: str) -> Path:
        """Get the directory for a specific config type."""
        if config_type == "experiments":
            return cls.EXPERIMENTS_DIR
        elif config_type == "base_model":
            return cls.BASE_MODEL_DIR
        elif config_type == "methods":
            return cls.METHODS_DIR
        else:
            return cls.CONFIG_DIR

    @classmethod
    def pretty_print(cls, config):
        """Pretty print the configuration."""
        config_dict = OmegaConf.to_container(config, resolve=True)
        return pprint.pformat(config_dict, width=100, sort_dicts=False)
