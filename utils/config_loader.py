import yaml 
from pathlib import Path
from typing import Any,Dict,Optional
from dataclasses import dataclass

@dataclass
class Config:

    def __init__(self,config_dict:Dict[str,Any]):
        self._config = config_dict
    
    def get(self,key_path:str,default:Any=None)->Any:
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value,dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self,key:str)->Any:
        return self._config[key]
    
    def __contains__(self,key:str)->bool:
        return key in self._config

# Global configuration instance (lazy-loaded)
_config:Optional[Config] = None

def load_config(config_path:Optional[str]=None)->Config:
    global _config
    if config_path is None:
        # Try to find config relative to this file
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        # If provided path doesn't exist, try relative to project root
        if not config_path.exists():
            utils_dir = Path(__file__).parent
            project_root = utils_dir.parent
            config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Current working directory: {Path.cwd()}"
        )
    
    with open(config_path,"r") as f:
        config_dict = yaml.safe_load(f)
    
    _config = Config(config_dict)
    return _config

def get_config()->Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config

def reload_config(config_path:Optional[str]=None)->Config:
    global _config
    _config = None
    return load_config(config_path)

def get_default_provider() -> str:
    """Get default provider."""
    return get_config().get("providers.default", "openai")

def get_enabled_providers() -> list[str]:
    """Get enabled providers."""
    return get_config().get("providers.enabled", ["openai"])

def get_max_retries() -> int:
    return get_config().get("retry.max_retries", 3)

def get_backoff_base() -> float:
    return get_config().get("retry.backoff.base_seconds", 0.5)

def get_backoff_jitter() -> float:
    return get_config().get("retry.backoff.jitter_factor", 0.25)

def get_default_temperature(task_type: Optional[str] = None) -> float:
    if task_type:
        temp = get_config().get(f"defaults.by_task.{task_type}.temperature")
        if temp is not None:
            return temp
    
    return get_config().get("defaults.temperature", 0.2)

def get_default_max_tokens(task_type: Optional[str] = None) -> int:
    if task_type:
        max_tok = get_config().get(f"defaults.by_task.{task_type}.max_tokens")
        if max_tok is not None:
            return max_tok
    
    return get_config().get("defaults.max_tokens", 1000)

def is_logging_enabled() -> bool:
    return get_config().get("logging.enabled", True)

def get_log_path() -> Path:
    log_dir = get_config().get("logging.output_dir", "logs")
    log_file = get_config().get("logging.output_file", "runs.csv")
    return Path(log_dir) / log_file

def should_auto_route_reasoning() -> bool:
    return get_config().get("models.auto_routing", True)

def get_reasoning_techniques() -> list[str]:
    return get_config().get("models.reasoning_techniques", ["cot", "tot"])
