import json
from pathlib import Path

from audio_generator import SoundType
from resource_helpers import bundled_resource_path, resource_path


__all__ = ("DEFAULT_WIDTH", "DEFAULT_HEIGHT", "MIN_WIDTH", "MIN_HEIGHT", "FIRE_RATE_MS", \
           "OVERLAP_BUFFER_MS", "DARK_BG", "GRID_COLOR", "CENTER_LINE", "BLACK", "WHITE", "BLUE",\
           "RED", "GREEN", "YELLOW", "GRAY", "ORANGE", "PURPLE", "CYAN", "DARK_BLUE", "DARK_RED",\
           "DARK_CYAN", "DARK_PURPLE", "DEFAULT_CONFIG", "ICON_PATH", "load_config",\
           "VK_LBUTTON", "MAX_BUFFER_SIZE", "MATH_LOG2")


try:
    # Use the PNG for the pygame window icon (better quality)
    ICON_PATH = bundled_resource_path("assets/favicon-512x512.png")
except:
    ICON_PATH = None

# Display Configuration
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 700
MIN_WIDTH = 800
MIN_HEIGHT = 600

# Gameplay Constants
FIRE_RATE_MS = 1000 / 13
OVERLAP_BUFFER_MS = 70

# Color Palette
DARK_BG = (15, 15, 20)
GRID_COLOR = (30, 35, 40)
CENTER_LINE = (60, 65, 70)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (80, 180, 255)
RED = (255, 80, 80)
GREEN = (100, 255, 150)
YELLOW = (255, 220, 80)
GRAY = (130, 130, 130)
ORANGE = (255, 140, 60)
PURPLE = (180, 100, 255)
CYAN = (60, 200, 255)
DARK_BLUE = (40, 80, 120)
DARK_RED = (120, 40, 40)
DARK_CYAN = (40, 100, 120)
DARK_PURPLE = (90, 50, 130)

DEFAULT_CONFIG = {
  "keys": {
    "left": "a",
    "right": "d",
    "walk": "shift",
    "crouch": "ctrl",
    "pause": "tab",
    "practice_mode": "f12"
  },
  "video": {
    "enabled": True,
    "vsync": True,
    "target_fps": None
  },
  "audio": {
    "volume": 1.0,
    "sound_type": 1,
    "loop_duration": 500,
    "accel_sound_type": 3,
    "constant_sound_type": 4,
    "decel_sound_type": 2
  }
}

# Windows API
VK_LBUTTON = 0x01

# Performance constants
MAX_BUFFER_SIZE = 1750
MATH_LOG2 = 0.6931471805599453

def load_config() -> dict:
    def apply_sound_option(config: dict, user_cfg: dict) -> dict:
        sound_options = {
            1: SoundType.MOVING_SHOOTING,
            2: SoundType.FOOTSTEP,
            3: SoundType.SHOOTING,
            4: SoundType.RUNNING_GUNNING,
            5: SoundType.ABILITY,
            6: SoundType.ALERT,
        }
        short_sound_options = {
            1: SoundType.RELOAD,
            2: SoundType.JUMP,
            3: SoundType.SUCCESS,
            4: SoundType.ERROR,
        }
        
        audio_cfg = user_cfg.get("audio", {})
        option = audio_cfg.get("sound_type", user_cfg.get("sound_type", 1))
        
        config["audio"]["sound_type"] = sound_options.get(option, sound_options[1])
        config["audio"]["loop_duration"] = audio_cfg.get("loop_duration", user_cfg.get("loop_duration", 1000)) / 1000
        config["audio"]["volume"] = audio_cfg.get("volume", user_cfg.get("volume", 1.0))
        
        accel_option = audio_cfg.get("accel_sound_type", 3)
        constant_option = audio_cfg.get("constant_sound_type", 4)
        decel_option = audio_cfg.get("decel_sound_type", 2)
        config["audio"]["accel_sound_type"] = sound_options.get(accel_option, sound_options[3])
        config["audio"]["constant_sound_type"] = sound_options.get(constant_option, sound_options[4])
        config["audio"]["decel_sound_type"] = short_sound_options.get(decel_option, short_sound_options[1])
        
        return config
    
    config_path: Path = resource_path("config.json")

    if not config_path.exists():
        return apply_sound_option(DEFAULT_CONFIG.copy(), {})

    try:
        with config_path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    except Exception as e:
        print(f"[Config] Failed to load config.json, using defaults: {e}")
        return apply_sound_option(DEFAULT_CONFIG.copy(), {})

    merged = DEFAULT_CONFIG.copy()
    merged["keys"] = {
        **DEFAULT_CONFIG["keys"],
        **user_cfg.get("keys", {})
    }
    
    video_cfg = user_cfg.get("video", {})
    legacy_cfg = user_cfg.get("low_resources", {})
    
    merged["video"] = {
        **DEFAULT_CONFIG["video"],
        **video_cfg
    }
    
    if "no_graphics" in legacy_cfg:
        merged["video"]["enabled"] = not legacy_cfg["no_graphics"]
    if "target_fps" in legacy_cfg and legacy_cfg["target_fps"] is not None:
        merged["video"]["vsync"] = False
        merged["video"]["target_fps"] = legacy_cfg["target_fps"]
    
    merged["audio"] = {
        **DEFAULT_CONFIG["audio"],
        **user_cfg.get("audio", {})
    }
    
    return apply_sound_option(merged, user_cfg)
