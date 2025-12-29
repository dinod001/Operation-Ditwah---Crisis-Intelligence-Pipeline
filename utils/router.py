import yaml
from pathlib import Path
from typing import Literal, Optional
from .config_loader import get_config, should_auto_route_reasoning, get_reasoning_techniques


def pick_model(
    provider: Literal["openai", "google", "groq"],
    technique: str,
    tier: Optional[Literal["general", "strong", "reason"]] = None,
    config_path: str = "config/models.yaml",
) -> str:
    """
    Pick an appropriate model for the given provider and reasoning technique.

    Args:
        provider: API provider name ("openai", "google", or "groq")
        technique: Reasoning technique requested (e.g., "cot", "tot", "strong")
        tier: Optional model tier ("general", "strong", "reason"). If None, determined automatically
        config_path: Path to the YAML model configuration file

    Returns:
        Model name as a string
    """

    # Locate config file
    config_file = Path(config_path)
    if not config_file.exists():
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        config_file = project_root / "config" / "models.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Model config not found. Tried:\n"
            f"  - {config_path}\n"
            f"  - {config_file}\n"
            f"Current working directory: {Path.cwd()}"
        )

    # Load YAML configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Check if provider exists in config
    if provider not in config:
        raise KeyError(f"Provider '{provider}' not found in {config_path}")

    # Determine tier if not provided
    if tier is None:
        technique_lower = technique.lower()
        if any(x in technique_lower for x in ["cot", "tot", "reason", "think"]):
            tier = "reason"
        elif any(x in technique_lower for x in ["strong", "complex", "advanced"]):
            tier = "strong"
        else:
            tier = "general"

    # Fallback to general if tier not found for provider
    if tier not in config[provider]:
        tier = "general"

    return config[provider][tier]  # Return the model name


def list_available_models(config_path: str = "config/models.yaml") -> dict[str, dict]:
    """
    List all available models from the YAML configuration.

    Args:
        config_path: Path to the YAML model configuration file

    Returns:
        Dictionary of providers and their available model tiers
    """

    # Locate config file
    config_file = Path(config_path)
    if not config_file.exists():
        utils_dir = Path(__file__).parent
        project_root = utils_dir.parent
        config_file = project_root / "config" / "models.yaml"
    
    if not config_file.exists():
        return {}  # Return empty dict if config file missing

    # Load and return YAML config as a dictionary
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_context_window(model: str) -> int:
    """
    Get approximate context window size for a model in tokens.

    Args:
        model: Model name string

    Returns:
        Context window size (max tokens the model can handle in a single request)
    """

    # OpenAI models
    if "gpt-4o" in model or "o3" in model or "o1" in model:
        return 128_000
    if "gpt-4" in model:
        return 128_000
    if "gpt-3.5" in model:
        return 16_385

    # Google Gemini models
    if "gemini-2.0" in model:
        return 1_000_000
    if "gemini-1.5" in model:
        return 1_000_000

    # Groq models
    if "llama-3.1" in model:
        return 131_072
    if "deepseek-r1" in model:
        return 65_536
    if "llama-3.2" in model:
        return 131_072

    # Default conservative estimate for unknown models
    return 8_000


def should_use_reasoning_model(technique: str) -> bool:
    """
    Determine if the reasoning model should be used based on technique.

    Args:
        technique: The reasoning technique requested (e.g., "cot", "tot", "strong")

    Returns:
        True if the technique matches a reasoning model and auto-routing is enabled, False otherwise
    """

    # Check global setting for auto-routing reasoning
    if not should_auto_route_reasoning():
        return False
    
    technique_lower = technique.lower()
    reasoning_techniques = get_reasoning_techniques()
    
    # Check for exact matches first
    if technique_lower in reasoning_techniques:
        return True
    
    # Check for substring matches (e.g., "cot_reasoning" contains "cot")
    return any(keyword in technique_lower for keyword in reasoning_techniques)
