"""
Live Interaction Mode for Nova Sonic
"""

from .sonic_stream_manager import SonicStreamManager
from .bedrock_model_client import BedrockModelClient, ScriptedUser, EXAMPLE_SCRIPTS
from .config_manager import ConfigManager, TestConfig, InferenceConfig, ToolConfig
from .interaction_logger import InteractionLogger, InteractionLog, Turn
from .model_registry import ModelRegistry, get_model_registry

__version__ = "1.0.0"

__all__ = [
    "SonicStreamManager",
    "BedrockModelClient",
    "ScriptedUser",
    "EXAMPLE_SCRIPTS",
    "ConfigManager",
    "TestConfig",
    "InferenceConfig",
    "ToolConfig",
    "InteractionLogger",
    "InteractionLog",
    "Turn",
    "ModelRegistry",
    "get_model_registry",
]
