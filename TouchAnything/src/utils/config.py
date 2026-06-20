import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configs, with override_config taking precedence.
    Recursively merges nested dictionaries.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _expand_config_values(value):
    """Recursively expand environment variables and '~' in config strings."""
    if isinstance(value, dict):
        return {k: _expand_config_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config_with_base(config_path: str, base_config_path: str = None) -> Dict[str, Any]:
    """
    Load config and optionally merge with base config.
    
    Example:
        config = load_config_with_base(
            'configs/hamer_pose_training_330_full.yaml',
            'configs/base.yaml',
        )
    """
    if base_config_path is None:
        config_dir = Path(config_path).parent
        default_base = config_dir / 'base.yaml'
        fallback_base = config_dir.parent / 'base.yaml'
        if default_base.exists():
            base_config_path = default_base
        elif fallback_base.exists():
            base_config_path = fallback_base
        else:
            base_config_path = default_base
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Load specific config
    specific_config = load_config(config_path)
    
    # Merge configs
    merged_config = merge_configs(base_config, specific_config)
    merged_config = _expand_config_values(merged_config)

    return merged_config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
