"""
Author: Claude Ai

Input Monitor - Enhanced with Pattern Practice Mode

Real-time visualization of keyboard and mouse inputs with velocity tracking.
Includes pattern recording and practice modes for training strafe and click timing.

FIX: Corrected pattern segment timing to use actual press/release timestamps
ENHANCEMENT: Added tolerance adjustment, inappropriate input detection, walk/crouch tracking
REFACTOR: Extracted pattern evaluation into separate PatternEvaluator class
UPDATE: Pattern recorder now uses timeline visualization matching evaluator
"""

import json
import sys
import time
import ctypes
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from queue import Queue
from enum import Enum

import pygame
import keyboard
import numpy as np

from audio_generator import PygameAudioPlayer, SoundType
from pattern_eval import Pattern, PatternEvaluator, PatternManager, PatternSegment
from resource_helpers import bundled_resource_path, resource_path


# Use the PNG for the pygame window icon (better quality)
try:
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
        "pause": "tab"
    },
    "video": {
        "enabled": True,
        "vsync": True,
        "target_fps": None
    },
    "audio": {
        "volume": 1.0,
        "sound_type": 1,
        "loop_duration": 1000,
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


class AppMode(Enum):
    """Application modes."""
    MONITOR = 0
    PATTERN_CREATE = 1
    PATTERN_SELECT = 2
    PATTERN_PRACTICE = 3


class InaccuracyType(Enum):
    """Types of shooting inaccuracy."""
    NONE = 0
    ACCELERATING = 1
    CONSTANT = 2
    DECELERATING = 3


class Toast:
    """Temporary notification message."""
    
    def __init__(self, message: str, duration_ms: int = 2000, color: Tuple[int, int, int] = GREEN):
        self.message = message
        self.duration_ms = duration_ms
        self.color = color
        self.start_time = time.time()
        self.alpha = 255
    
    def update(self) -> bool:
        elapsed_ms = (time.time() - self.start_time) * 1000
        if elapsed_ms >= self.duration_ms:
            return False
        
        if elapsed_ms > self.duration_ms - 500:
            fade_progress = (self.duration_ms - elapsed_ms) / 500
            self.alpha = int(255 * fade_progress)
        
        return True
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, window_width: int, window_height: int):
        text_surface = font.render(self.message, True, self.color)
        text_rect = text_surface.get_rect()
        
        padding = 20
        bg_width = text_rect.width + padding * 2
        bg_height = text_rect.height + padding
        
        bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
        bg_color = (*DARK_BG, min(200, self.alpha))
        pygame.draw.rect(bg_surface, bg_color, bg_surface.get_rect(), border_radius=10)
        
        border_color = (*self.color, self.alpha)
        pygame.draw.rect(bg_surface, border_color, bg_surface.get_rect(), width=2, border_radius=10)
        
        bg_x = (window_width - bg_width) // 2
        bg_y = window_height - bg_height - 100
        
        screen.blit(bg_surface, (bg_x, bg_y))
        
        text_with_alpha = text_surface.copy()
        text_with_alpha.set_alpha(self.alpha)
        text_x = bg_x + padding
        text_y = bg_y + padding // 2
        screen.blit(text_with_alpha, (text_x, text_y))


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


class VelocitySimulator:
    """Simulates player movement velocity with acceleration and deceleration."""
    
    __slots__ = ('velocity', 'direction', 'accel_progress', 'max_velocity', 
                 'accel_time', 'velocity_threshold', 'decel_half_life',
                 '_log2_decel', '_accel_exp', 'base_max_velocity', 'base_accel_time',
                 'walk_velocity_multiplier', 'walk_accel_multiplier', 'is_walking',
                 'is_accelerating', 'prev_velocity')
    
    def __init__(self):
        self.velocity = 0.0
        self.direction = 0
        self.accel_progress = 0.0
        
        self.base_max_velocity = 1.0
        self.base_accel_time = 0.480
        self.velocity_threshold = 0.0148
        self.decel_half_life = 0.02125
        
        self.walk_velocity_multiplier = 0.52
        self.walk_accel_multiplier = 1.00
        
        self.max_velocity = self.base_max_velocity
        self.accel_time = self.base_accel_time
        self.is_walking = False
        
        self.is_accelerating = False
        self.prev_velocity = 0.0
        
        self._log2_decel = MATH_LOG2 / self.decel_half_life
        self._accel_exp = 1.45
    
    def update(self, dt: float, a_held: bool, d_held: bool, shift_held: bool = False) -> float:
        """Update velocity based on input state and return current velocity with direction."""
        self._update_walk_state(shift_held)
        self.prev_velocity = self.velocity
        
        desired_direction = 0
        if a_held and not d_held:
            desired_direction = -1
        elif d_held and not a_held:
            desired_direction = 1
        
        if desired_direction == 0:
            self._apply_deceleration(dt)
            self.is_accelerating = False
        elif desired_direction == self.direction or self.direction == 0:
            self._apply_acceleration(dt, desired_direction)
            self.is_accelerating = self.velocity > self.prev_velocity
        else:
            self._apply_direction_change(dt, desired_direction)
            self.is_accelerating = False
        
        return self.velocity * self.direction
    
    def _update_walk_state(self, shift_held: bool):
        """Update physics parameters based on walk state."""
        was_walking = self.is_walking
        self.is_walking = shift_held
        
        if self.is_walking:
            self.max_velocity = self.base_max_velocity * self.walk_velocity_multiplier
            self.accel_time = self.base_accel_time * self.walk_accel_multiplier
        else:
            self.max_velocity = self.base_max_velocity
            self.accel_time = self.base_accel_time
        
        if was_walking != self.is_walking and self.accel_progress > 0:
            if self.max_velocity > 0:
                eased_velocity_ratio = self.velocity / self.max_velocity
                self.accel_progress = min(1.0, eased_velocity_ratio ** (1.0 / self._accel_exp))
    
    def is_moving(self) -> bool:
        """Check if velocity exceeds accuracy threshold."""
        return self.velocity > self.velocity_threshold
    
    def is_near_max_velocity(self) -> bool:
        """Check if velocity is approaching maximum (above 75%)."""
        return self.velocity > 0.75 * self.max_velocity
    
    def get_velocity_ratio(self) -> float:
        """Get current velocity as ratio of max velocity (0.0 to 1.0)."""
        return self.velocity / self.max_velocity if self.max_velocity > 0 else 0.0
    
    def is_decelerating(self) -> bool:
        """Check if we're currently decelerating."""
        return self.is_moving() and not self.is_accelerating and self.velocity < self.prev_velocity
    
    def _apply_acceleration(self, dt: float, direction: int):
        """Accelerate in the desired direction with easing curve."""
        self.accel_progress = min(1.0, self.accel_progress + dt / self.accel_time)
        eased_progress = self.accel_progress ** self._accel_exp
        self.velocity = eased_progress * self.max_velocity
        self.direction = direction
    
    def _apply_deceleration(self, dt: float):
        """Exponential decay when no input."""
        self.accel_progress = 0.0
        decay_factor = np.exp(-dt * self._log2_decel)
        self.velocity *= decay_factor
        
        if self.velocity < 0.01:
            self.velocity = 0.0
            self.direction = 0
    
    def _apply_direction_change(self, dt: float, new_direction: int):
        """Handle counter-strafing when changing direction."""
        self.accel_progress = 0.0
        decay_factor = np.exp(-dt * self._log2_decel)
        self.velocity *= decay_factor
        
        if self.velocity < 0.01:
            self.velocity = 0.0
            self.direction = new_direction


class ShootingTracker:
    """Tracks shooting mechanics including fire rate, accuracy, and grace periods."""
    
    __slots__ = ('velocity_sim', 'mouse_held', 'mouse_press_time', 'last_bullet_time',
                 'inaccuracy_type', 'movement_start_time', 'was_moving')
    
    def __init__(self, velocity_sim: VelocitySimulator):
        self.velocity_sim = velocity_sim
        self.mouse_held = False
        self.mouse_press_time = 0.0
        self.last_bullet_time = 0.0
        self.inaccuracy_type = InaccuracyType.NONE
        self.movement_start_time = 0.0
        self.was_moving = False
    
    def on_mouse_press(self, current_time: float):
        """Handle mouse button press."""
        self.mouse_press_time = current_time
        self.last_bullet_time = current_time - (FIRE_RATE_MS / 1000.0)
        self.mouse_held = True
        self.inaccuracy_type = InaccuracyType.NONE
        self.movement_start_time = 0.0
        self.was_moving = False
    
    def on_mouse_release(self):
        """Handle mouse button release."""
        self.mouse_held = False
        self.movement_start_time = 0.0
        self.was_moving = False
    
    def check_bullet_fire(self, current_time: float) -> InaccuracyType:
        """Check if a bullet should fire and return the type of inaccuracy."""
        if not self.mouse_held:
            return InaccuracyType.NONE
        
        is_moving_now = self.velocity_sim.is_moving()
        
        if is_moving_now and not self.was_moving and self.mouse_held:
            time_since_click = (current_time - self.mouse_press_time) * 1000
            if time_since_click > 0:
                self.movement_start_time = current_time
        elif not is_moving_now:
            self.movement_start_time = 0.0
        
        self.was_moving = is_moving_now
        
        self.last_bullet_time = current_time
        
        inaccuracy = self._check_inaccuracy(current_time, is_moving_now)
        if inaccuracy != InaccuracyType.NONE:
            self.inaccuracy_type = inaccuracy
        
        return inaccuracy
    
    def _check_inaccuracy(self, current_time: float, is_moving: bool) -> InaccuracyType:
        """Determine the type of inaccuracy based on movement and grace period."""
        if not is_moving:
            return InaccuracyType.NONE
        
        if self.movement_start_time > 0:
            time_since_movement_start = (current_time - self.movement_start_time) * 1000
            if time_since_movement_start <= OVERLAP_BUFFER_MS:
                return InaccuracyType.NONE
        
        if self.velocity_sim.is_accelerating:
            return InaccuracyType.ACCELERATING
        elif self.velocity_sim.is_decelerating():
            return InaccuracyType.DECELERATING
        else:
            return InaccuracyType.CONSTANT


class RingBuffer:
    """Efficient ring buffer using numpy for O(1) append and fast iteration."""
    
    __slots__ = ('_buffer', '_head', '_size', '_capacity')
    
    def __init__(self, capacity: int, dtype=np.float32):
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._head = 0
        self._size = 0
        self._capacity = capacity
    
    def append(self, value):
        """Add value to buffer."""
        self._buffer[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
    
    def get_recent(self, count: Optional[int] = None) -> np.ndarray:
        """Get most recent values in chronological order."""
        if count is None:
            count = self._size
        else:
            count = min(count, self._size)
        
        if count == 0:
            return np.array([], dtype=self._buffer.dtype)
        
        start_idx = (self._head - count) % self._capacity
        if start_idx < self._head:
            return self._buffer[start_idx:self._head].copy()
        else:
            return np.concatenate([
                self._buffer[start_idx:],
                self._buffer[:self._head]
            ])
    
    def clear(self):
        """Clear the buffer."""
        self._head = 0
        self._size = 0
    
    def __len__(self):
        return self._size


class InputMonitor:
    """Main application for visualizing keyboard and mouse inputs with pattern practice."""
    
    def __init__(self):
        self.config = load_config()
        self.video_enabled = self.config['video']['enabled']
        
        if not self.video_enabled:
            print("[Input Monitor] Video disabled mode not supported for pattern practice")
            sys.exit(1)
        
        pygame.init()
        
        self.window_width = DEFAULT_WIDTH
        self.window_height = DEFAULT_HEIGHT
        
        use_vsync = self.config['video']['vsync']
        target_fps = self.config['video']['target_fps']
        
        if use_vsync:
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.DOUBLEBUF | pygame.SCALED,
                vsync=1
            )
            self.use_vsync = True
            self.target_fps = None
        else:
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.DOUBLEBUF | pygame.SCALED
            )
            self.use_vsync = False
            self.target_fps = target_fps
        
        pygame.display.set_caption("Input Monitor - Pattern Practice")
        
        if ICON_PATH and ICON_PATH.exists():
            try:
                icon_image = pygame.image.load(str(ICON_PATH))
                pygame.display.set_icon(icon_image)
            except:
                pass
        
        self.clock = pygame.time.Clock()
        
        self._font_cache = {}
        self._static_surface = None
        self._need_static_redraw = True
        
        self.update_fonts()
        
        # Core components
        self.velocity_sim = VelocitySimulator()
        self.shooting_tracker = ShootingTracker(self.velocity_sim)
        self.pattern_manager = PatternManager()
        
        # Data buffers
        self.time_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.float64)
        self.a_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.d_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.shift_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.ctrl_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.click_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.inaccuracy_type_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.velocity_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.float32)
        self.bullet_fired_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        
        # Input state
        self.a_key_held = False
        self.d_key_held = False
        self.shift_key_held = False
        self.ctrl_key_held = False
        self.prev_mouse_held = False
        self.volume = float(self.config['audio']['volume'])
        
        # Audio feedback
        self.accel_beeper = PygameAudioPlayer(channel_id=0)
        self.constant_beeper = PygameAudioPlayer(channel_id=1)
        self.decel_beeper = PygameAudioPlayer(channel_id=2)
        self.decel_beeper.set_min_duration(2)
        self.success_player = PygameAudioPlayer(channel_id=3)
        self.error_player = PygameAudioPlayer(channel_id=4)
        self.beeper_queue = Queue()
        self.active_beeper = None
        
        # Application state
        self.mode = AppMode.MONITOR
        self.paused = False
        self.running = True
        self.current_time = 0.0
        self.last_update = time.time()
        
        # Pattern creation state
        self.recording_pattern = False
        self.recorded_segments: List[PatternSegment] = []
        self.current_segment_start = 0.0
        self.current_segment_keys = set()
        self.last_segment_end_time = 0.0
        self.recording_start_time = 0.0  # NEW: Track when recording started
        self.pattern_name = ""
        self.pattern_difficulty = "MEDIUM"
        self.pattern_tolerance = 50
        self.name_input_active = False
        
        # Pattern selection state
        self.available_patterns: List[Pattern] = []
        self.selected_pattern_index = 0
        
        # Pattern practice state - now managed by PatternEvaluator
        self.pattern_evaluator: Optional[PatternEvaluator] = None
        
        # Toast notifications
        self.toasts: List[Toast] = []
        
        self._setup_input_hooks()
        
        # Load available patterns
        self.available_patterns = self.pattern_manager.load_patterns()
    
    def clear_inputs(self):
        self.a_key_held = False
        self.d_key_held = False
        self.shift_key_held = False
        self.ctrl_key_held = False
        self.prev_mouse_held = False        
    
    def update_fonts(self):
        """Scale fonts based on window height."""
        scale = self.window_height / DEFAULT_HEIGHT
        font_size = max(20, int(28 * scale))
        small_font_size = max(14, int(20 * scale))
        large_font_size = max(30, int(42 * scale))
        
        self.font = pygame.font.Font(None, font_size)
        self.small_font = pygame.font.Font(None, small_font_size)
        self.large_font = pygame.font.Font(None, large_font_size)
        self._font_cache.clear()
        self._need_static_redraw = True
    
    def get_scaled_value(self, base_value: int, dimension: str = 'height') -> int:
        """Scale a value based on current window size."""
        if dimension == 'height':
            return int(base_value * (self.window_height / DEFAULT_HEIGHT))
        return int(base_value * (self.window_width / DEFAULT_WIDTH))
    
    def handle_resize(self, new_width: int, new_height: int):
        """Handle window resize events."""
        self.window_width = max(MIN_WIDTH, new_width)
        self.window_height = max(MIN_HEIGHT, new_height)
        self.update_fonts()
        self._need_static_redraw = True
    
    def add_toast(self, message: str, color: Tuple[int, int, int] = GREEN, duration_ms: int = 2000):
        """Add a toast notification."""
        self.toasts.append(Toast(message, duration_ms, color))
    
    def _setup_input_hooks(self):
        """Configure keyboard event handlers."""
        keys = self.config['keys']
        keyboard.on_press_key(keys['left'], lambda _: self._on_key_press(keys['left']), suppress=False)
        keyboard.on_release_key(keys['left'], lambda _: self._on_key_release(keys['left']), suppress=False)
        keyboard.on_press_key(keys['right'], lambda _: self._on_key_press(keys['right']), suppress=False)
        keyboard.on_release_key(keys['right'], lambda _: self._on_key_release(keys['right']), suppress=False)
        keyboard.on_press_key(keys['walk'], lambda _: self._on_key_press(keys['walk']), suppress=False)
        keyboard.on_release_key(keys['walk'], lambda _: self._on_key_release(keys['walk']), suppress=False)
        keyboard.on_press_key(keys['crouch'], lambda _: self._on_key_press(keys['crouch']), suppress=False)
        keyboard.on_release_key(keys['crouch'], lambda _: self._on_key_release(keys['crouch']), suppress=False)
        keyboard.on_press_key(keys['pause'], lambda _: self._toggle_pause(), suppress=False)
        
        # Mode switching
        keyboard.on_press_key('m', lambda _: self._switch_mode(), suppress=False)
        keyboard.on_press_key('space', lambda _: self._handle_space(), suppress=False)
        keyboard.on_press_key('r', lambda _: self._handle_r(), suppress=False)
        keyboard.on_press_key('esc', lambda _: self._handle_esc(), suppress=False)
        
        # Tolerance adjustment (using [ and ] keys)
        keyboard.on_press_key('[', lambda _: self._adjust_tolerance(-5), suppress=False)
        keyboard.on_press_key(']', lambda _: self._adjust_tolerance(5), suppress=False)
    
    def _adjust_tolerance(self, delta: int):
        """Adjust pattern tolerance value."""
        if self.mode == AppMode.PATTERN_CREATE:
            self.pattern_tolerance = max(10, min(200, self.pattern_tolerance + delta))
            self.add_toast(f"Tolerance: ±{self.pattern_tolerance}ms", CYAN, 1000)
    
    def _on_key_press(self, key: str):
        """Handle key press events."""
        keys = self.config['keys']
        current_real_time = time.time()
        
        if key == keys['left']:
            self.a_key_held = True
            if self.recording_pattern and 'a' not in self.current_segment_keys:
                self.current_segment_keys.add('a')
                if self.current_segment_start == 0.0:
                    self.current_segment_start = current_real_time
        elif key == keys['right']:
            self.d_key_held = True
            if self.recording_pattern and 'd' not in self.current_segment_keys:
                self.current_segment_keys.add('d')
                if self.current_segment_start == 0.0:
                    self.current_segment_start = current_real_time
        elif key == keys['walk']:
            self.shift_key_held = True
            if self.recording_pattern and 'walk' not in self.current_segment_keys:
                self.current_segment_keys.add('walk')
                if self.current_segment_start == 0.0:
                    self.current_segment_start = current_real_time
        elif key == keys['crouch']:
            self.ctrl_key_held = True
            if self.recording_pattern and 'crouch' not in self.current_segment_keys:
                self.current_segment_keys.add('crouch')
                if self.current_segment_start == 0.0:
                    self.current_segment_start = current_real_time
    
    def _on_key_release(self, key: str):
        """Handle key release events."""
        keys = self.config['keys']
        current_real_time = time.time()
        
        if key == keys['left']:
            self.a_key_held = False
            if self.recording_pattern and 'a' in self.current_segment_keys:
                self._finalize_segment('a', current_real_time)
        elif key == keys['right']:
            self.d_key_held = False
            if self.recording_pattern and 'd' in self.current_segment_keys:
                self._finalize_segment('d', current_real_time)
        elif key == keys['walk']:
            self.shift_key_held = False
            if self.recording_pattern and 'walk' in self.current_segment_keys:
                self._finalize_segment('walk', current_real_time)
        elif key == keys['crouch']:
            self.ctrl_key_held = False
            if self.recording_pattern and 'crouch' in self.current_segment_keys:
                self._finalize_segment('crouch', current_real_time)
    
    def _finalize_segment(self, key: str, current_real_time: float):
        """Finalize a recorded pattern segment."""
        if key not in self.current_segment_keys:
            return
        
        # Calculate the duration of this segment
        duration_ms = int((current_real_time - self.current_segment_start) * 1000)
        
        # Check if we need to insert a pause BEFORE this segment
        if self.last_segment_end_time > 0 and self.current_segment_start > self.last_segment_end_time:
            pause_duration_ms = int((self.current_segment_start - self.last_segment_end_time) * 1000)
            if pause_duration_ms > 50:  # Minimum pause to record
                pause_segment = PatternSegment(key='pause', duration_ms=pause_duration_ms)
                self.recorded_segments.append(pause_segment)
                # self.add_toast(f"PAUSE: {pause_duration_ms}ms", PURPLE, 1000)
        
        # Now record the actual segment
        if duration_ms > 20:  # Minimum segment duration
            segment = PatternSegment(key=key, duration_ms=duration_ms)
            self.recorded_segments.append(segment)
            
            key_colors = {
                'a': BLUE,
                'd': RED,
                'click': WHITE,
                'walk': CYAN,
                'crouch': PURPLE
            }
            color = key_colors.get(key, GRAY)
            # self.add_toast(f"{key.upper()}: {duration_ms}ms", color, 1500)
        
        # Update state
        self.current_segment_keys.discard(key)
        self.last_segment_end_time = current_real_time
        
        # Reset segment start time only if no other keys are being held
        if not self.current_segment_keys:
            self.current_segment_start = 0.0
    
    def _toggle_pause(self):
        """Toggle pause state."""
        if self.mode == AppMode.MONITOR:
            self.paused = not self.paused
    
    def _switch_mode(self):
        """Cycle through application modes."""
        if self.name_input_active:
            return
        
        if self.mode == AppMode.MONITOR:
            self.mode = AppMode.PATTERN_SELECT
            self.available_patterns = self.pattern_manager.load_patterns()
            self.selected_pattern_index = 0
            self.add_toast("Pattern Selection Mode", CYAN)
        elif self.mode == AppMode.PATTERN_SELECT:
            self.mode = AppMode.MONITOR
            self.add_toast("Monitor Mode", GREEN)
        elif self.mode == AppMode.PATTERN_CREATE:
            if not self.recording_pattern:
                self.mode = AppMode.MONITOR
                self.add_toast("Monitor Mode", GREEN)
        elif self.mode == AppMode.PATTERN_PRACTICE:
            self.mode = AppMode.PATTERN_SELECT
            self.pattern_evaluator = None
            self.add_toast("Pattern Selection Mode", CYAN)
    
    def _handle_space(self):
        """Handle spacebar press based on current mode."""
        if self.mode == AppMode.PATTERN_CREATE:
            if not self.name_input_active:
                self._toggle_recording()
        elif self.mode == AppMode.PATTERN_SELECT:
            if self.available_patterns:
                self._start_practice()
    
    def _handle_r(self):
        """Handle R key - create new pattern from selection mode."""
        if self.mode == AppMode.PATTERN_SELECT:
            self.mode = AppMode.PATTERN_CREATE
            self.recording_pattern = False
            self.recorded_segments = []
            self.pattern_name = ""
            self.pattern_difficulty = "MEDIUM"
            self.pattern_tolerance = 50
            self.name_input_active = False
            self.recording_start_time = 0.0
            self.add_toast("Pattern Creation Mode", PURPLE)
    
    def _handle_esc(self):
        """Handle escape key."""
        if self.mode == AppMode.PATTERN_CREATE:
            if self.recording_pattern:
                self._stop_recording()
            elif self.name_input_active:
                self.name_input_active = False
                self.pattern_name = ""
            else:
                self.mode = AppMode.PATTERN_SELECT
                self.add_toast("Pattern Selection Mode", CYAN)
        elif self.mode == AppMode.PATTERN_PRACTICE:
            self.mode = AppMode.PATTERN_SELECT
            self.pattern_evaluator = None
            self.add_toast("Pattern Selection Mode", CYAN)
    
    def _toggle_recording(self):
        """Toggle pattern recording."""
        if not self.recording_pattern:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start recording a pattern."""
        self.recording_pattern = True
        self.recorded_segments = []
        self.current_segment_start = 0.0
        self.last_segment_end_time = 0.0
        self.current_segment_keys = set()
        self.recording_start_time = time.time()  # Track when we started
        self.add_toast("Recording Started", GREEN)
    
    def _stop_recording(self):
        """Stop recording a pattern."""
        self.recording_pattern = False
        current_real_time = time.time()
        # Finalize any active segments
        for key in list(self.current_segment_keys):
            self._finalize_segment(key, current_real_time)
        
        if len(self.recorded_segments) > 0:
            self.add_toast("Recording Stopped - Enter Name", YELLOW)
            self.name_input_active = True
        else:
            self.add_toast("No segments recorded", RED)
            self.recording_pattern = False
    
    def _start_practice(self):
        """Start practicing the selected pattern."""
        if 0 <= self.selected_pattern_index < len(self.available_patterns):
            pattern = self.available_patterns[self.selected_pattern_index]
            self.pattern_evaluator = PatternEvaluator(pattern)
            self.mode = AppMode.PATTERN_PRACTICE
            self.time_points.clear()
            self.a_points.clear()
            self.d_points.clear()
            self.shift_points.clear()
            self.ctrl_points.clear()
            self.click_points.clear()
            self.add_toast(f"Practice: {pattern.name}", GREEN)
    
    def _poll_mouse_state(self):
        """Check mouse button state using Windows API and handle state changes."""
        state = ctypes.windll.user32.GetAsyncKeyState(VK_LBUTTON)
        current_mouse_held = bool(state & 0x8000)
        current_real_time = time.time()
        
        if current_mouse_held and not self.prev_mouse_held:
            self.shooting_tracker.on_mouse_press(time.time())
            if self.recording_pattern and 'click' not in self.current_segment_keys:
                self.current_segment_keys.add('click')
                if self.current_segment_start == 0.0:
                    self.current_segment_start = current_real_time
        elif not current_mouse_held and self.prev_mouse_held:
            self.shooting_tracker.on_mouse_release()
            if self.recording_pattern and 'click' in self.current_segment_keys:
                self._finalize_segment('click', current_real_time)
        
        if self.active_beeper and (not self.velocity_sim.is_moving() or not current_mouse_held):
            self.beeper_queue.put(('stop', None))
            self.active_beeper = None
        
        self.prev_mouse_held = current_mouse_held
    
    def update_data(self):
        """Update simulation state and append data points."""
        if self.paused and self.mode == AppMode.MONITOR:
            return
        
        self._poll_mouse_state()
        
        current_real_time = time.time()
        dt = current_real_time - self.last_update
        self.last_update = current_real_time
        self.current_time += dt
        
        # Update physics
        prev_velocity = self.velocity_sim.velocity
        velocity = self.velocity_sim.update(dt, self.a_key_held, self.d_key_held, 
                                           self.shift_key_held or self.ctrl_key_held)
        
        # Check for bullet fire
        inaccuracy_type = self.shooting_tracker.check_bullet_fire(current_real_time)
        
        # Audio feedback for monitor mode
        if self.mode == AppMode.MONITOR:
            if inaccuracy_type != InaccuracyType.NONE and self.active_beeper != inaccuracy_type:
                if self.active_beeper is not None:
                    self.beeper_queue.put(('stop', None))
                self.beeper_queue.put(('start', inaccuracy_type))
                self.active_beeper = inaccuracy_type
        
        # Pattern practice checking
        if self.mode == AppMode.PATTERN_PRACTICE and self.pattern_evaluator:
            error_msg = self.pattern_evaluator.check_progress(
                self.current_time, current_real_time,
                self.a_key_held, self.d_key_held, self.shooting_tracker.mouse_held,
                self.shift_key_held, self.ctrl_key_held
            )
            
            if error_msg:
                self.error_player.play(SoundType.ERROR, self.volume)
                self.add_toast(error_msg, RED, 1000)
                self.clear_inputs()
                self.pattern_evaluator.restart()
            elif self.pattern_evaluator.completed:
                result = self.pattern_evaluator.evaluate_attempt()
                if result['success']:
                    self.success_player.play(SoundType.SUCCESS, self.volume)
                    self.add_toast(result['message'], GREEN, 1000)
                else:
                    self.error_player.play(SoundType.ERROR, self.volume)
                    color = YELLOW if result['success_rate'] >= 60 else RED
                    self.add_toast(result['message'], color, 1000)
                self.pattern_evaluator.restart()
        
        self._process_beeper_queue()
        
        # Append data points
        self.time_points.append(self.current_time)
        self.a_points.append(1 if self.a_key_held else 0)
        self.d_points.append(1 if self.d_key_held else 0)
        self.shift_points.append(1 if self.shift_key_held else 0)
        self.ctrl_points.append(1 if self.ctrl_key_held else 0)
        self.click_points.append(1 if self.shooting_tracker.mouse_held else 0)
        self.inaccuracy_type_points.append(inaccuracy_type.value)
        self.velocity_points.append(velocity)
        self.bullet_fired_points.append(1 if inaccuracy_type != InaccuracyType.NONE else 0)
    
    def _process_beeper_queue(self):
        """Process queued beeper commands."""
        self.accel_beeper.update()
        self.constant_beeper.update()
        self.decel_beeper.update()
        self.success_player.update()
        self.error_player.update()
        
        while not self.beeper_queue.empty():
            try:
                cmd, data = self.beeper_queue.get_nowait()
                
                if cmd == 'start':
                    inaccuracy_type = data
                    if inaccuracy_type == InaccuracyType.ACCELERATING:
                        self.accel_beeper.start(
                            self.config["audio"]["accel_sound_type"], 
                            self.volume, 
                            loop_duration=self.config["audio"]["loop_duration"]
                        )
                    elif inaccuracy_type == InaccuracyType.CONSTANT:
                        self.constant_beeper.start(
                            self.config["audio"]["constant_sound_type"], 
                            self.volume, 
                            loop_duration=self.config["audio"]["loop_duration"]
                        )
                    elif inaccuracy_type == InaccuracyType.DECELERATING:
                        self.decel_beeper.play(
                            self.config["audio"]["decel_sound_type"], 
                            self.volume
                        )
                elif cmd == 'stop':
                    self.accel_beeper.stop()
                    self.constant_beeper.stop()
                    self.decel_beeper.stop()
            except:
                break
    
    def _get_cached_text(self, text: str, color: Tuple[int, int, int], font_type: str = 'normal') -> pygame.Surface:
        """Get cached rendered text surface."""
        key = (text, color, font_type)
        if key not in self._font_cache:
            if font_type == 'large':
                font = self.large_font
            elif font_type == 'small':
                font = self.small_font
            else:
                font = self.font
            self._font_cache[key] = font.render(text, True, color)
        return self._font_cache[key]
    
    def draw(self):
        """Main draw function - routes to appropriate screen."""
        self.screen.fill(DARK_BG)
        
        if self.mode == AppMode.MONITOR:
            self._draw_monitor_screen()
        elif self.mode == AppMode.PATTERN_CREATE:
            self._draw_pattern_create_screen()
        elif self.mode == AppMode.PATTERN_SELECT:
            self._draw_pattern_select_screen()
        elif self.mode == AppMode.PATTERN_PRACTICE:
            self._draw_pattern_practice_screen()
        
        # Draw toasts
        self.toasts = [toast for toast in self.toasts if toast.update()]
        for toast in self.toasts:
            toast.draw(self.screen, self.font, self.window_width, self.window_height)
        
        pygame.display.flip()
    
    def _draw_pattern_create_screen(self):
            """Draw the pattern creation screen with timeline visualization."""
            self._draw_grid()
            
            # Header - Title on left
            title = self._get_cached_text("Pattern Creation Mode", WHITE, 'large')
            self.screen.blit(title, (self.get_scaled_value(40), self.get_scaled_value(30)))
            
            # Timer in top right corner (if recording)
            if self.recording_pattern:
                elapsed_time = time.time() - self.recording_start_time
                elapsed_text = self._get_cached_text(f"Time: {elapsed_time:.1f}s", CYAN, 'large')
                timer_x = self.window_width - elapsed_text.get_width() - self.get_scaled_value(40)
                self.screen.blit(elapsed_text, (timer_x, self.get_scaled_value(30)))
            
            # Recording status below title
            y_pos = self.get_scaled_value(90)
            if self.recording_pattern:
                status = self._get_cached_text("RECORDING", GREEN, 'large')
                self.screen.blit(status, (self.get_scaled_value(40), y_pos))
            else:
                status = self._get_cached_text("Ready to Record", YELLOW, 'large')
                self.screen.blit(status, (self.get_scaled_value(40), y_pos))
            
            # Current tolerance on the right side below timer
            y_pos = self.get_scaled_value(90)
            tolerance_text = f"Tolerance: ±{self.pattern_tolerance}ms"
            tolerance_surf = self._get_cached_text(tolerance_text, YELLOW)
            tolerance_x = self.window_width - tolerance_surf.get_width() - self.get_scaled_value(40)
            self.screen.blit(tolerance_surf, (tolerance_x, y_pos))

            
            # Draw pattern timeline visualization (if not entering name, or as background)
            chart_y = self.get_scaled_value(260)
            chart_height = self.get_scaled_value(380)
            
            if self.recorded_segments or self.recording_pattern:
                self._draw_recording_timeline(chart_y, chart_height)
            
            # Name input - takes over the screen when active
            if self.name_input_active:
                # Dim overlay
                overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                pygame.draw.rect(overlay, (0, 0, 0, 180), overlay.get_rect())
                self.screen.blit(overlay, (0, 0))
                
                # Center the input dialog
                dialog_width = self.get_scaled_value(600)
                dialog_height = self.get_scaled_value(200)
                dialog_x = (self.window_width - dialog_width) // 2
                dialog_y = (self.window_height - dialog_height) // 2
                
                # Dialog background
                dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
                pygame.draw.rect(self.screen, (20, 25, 30), dialog_rect, border_radius=10)
                pygame.draw.rect(self.screen, CYAN, dialog_rect, 3, border_radius=10)
                
                # Title
                title_y = dialog_y + self.get_scaled_value(30)
                name_label = self._get_cached_text("Pattern Name:", WHITE, 'large')
                label_x = dialog_x + (dialog_width - name_label.get_width()) // 2
                self.screen.blit(name_label, (label_x, title_y))
                
                # Input box
                input_y = dialog_y + self.get_scaled_value(90)
                input_box = pygame.Rect(dialog_x + self.get_scaled_value(40), input_y, 
                                    dialog_width - self.get_scaled_value(80), self.get_scaled_value(50))
                pygame.draw.rect(self.screen, (30, 35, 40), input_box, 0, border_radius=5)
                pygame.draw.rect(self.screen, WHITE, input_box, 2, border_radius=5)
                
                # Text
                name_text = self._get_cached_text(self.pattern_name + "_", YELLOW)
                text_x = input_box.x + 15
                text_y = input_box.y + (input_box.height - name_text.get_height()) // 2
                self.screen.blit(name_text, (text_x, text_y))
                
                # Hint
                hint_y = dialog_y + dialog_height - self.get_scaled_value(40)
                hint = self._get_cached_text("Enter: Save  |  ESC: Cancel", GRAY, 'small')
                hint_x = dialog_x + (dialog_width - hint.get_width()) // 2
                self.screen.blit(hint, (hint_x, hint_y))
            else:               
                # Instructions at bottom left (only when not entering name)
                y_pos = self.window_height - self.get_scaled_value(150)
                instructions = [
                    ("SPACE: Start/Stop", CYAN),
                    ("Brackets [ ]: Tolerance", YELLOW),
                    ("ESC: Cancel", RED),
                    ("M: Menu", GRAY),
                ]

                # --- Calculate background size ---
                padding = self.get_scaled_value(15)
                line_height = self.get_scaled_value(25)
                text_x = self.get_scaled_value(40)

                box_width = self.get_scaled_value(160)  
                box_height = len(instructions) * line_height + padding

                box_x = text_x - padding
                box_y = y_pos - padding

                # --- Create transparent background ---
                bg_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, 150))  # black with 150 alpha (0-255)

                self.screen.blit(bg_surface, (box_x, box_y))

                # --- Draw text on top ---
                for text, color in instructions:
                    surf = self._get_cached_text(text, color, 'small')
                    self.screen.blit(surf, (text_x, y_pos))
                    y_pos += line_height
                

    
    def _draw_recording_timeline(self, chart_y: int, chart_height: int):
        """Draw the recording timeline with recorded segments and current input state."""
        # Calculate total duration and time scale
        if self.recorded_segments:
            # Create temporary pattern from recorded segments
            temp_pattern = Pattern(
                name="Recording",
                difficulty=self.pattern_difficulty,
                segments=self.recorded_segments,
                tolerance_ms=self.pattern_tolerance
            )
            total_duration = temp_pattern.get_total_duration_ms()
        else:
            total_duration = 1000  # Default 1 second if no segments yet
        
        # Add currently recording segment duration if applicable
        if self.recording_pattern and self.current_segment_start > 0:
            current_segment_duration = (time.time() - self.current_segment_start) * 1000
            total_duration += current_segment_duration
        
        time_scale = self.window_width / max(total_duration, 1000)
        
        y_offset = chart_y + chart_height // 2
        baseline_offset = self.get_scaled_value(100)
        wave_height = self.get_scaled_value(70)
        
        # Define baselines for all tracks
        a_baseline = y_offset - baseline_offset * 1.5
        d_baseline = y_offset + baseline_offset * 1.5
        click_baseline = y_offset
        walk_baseline = y_offset - baseline_offset * 0.5
        crouch_baseline = y_offset + baseline_offset * 0.5
        
        # Draw baseline guides
        pygame.draw.line(self.screen, CENTER_LINE, (0, a_baseline), (self.window_width, a_baseline), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, d_baseline), (self.window_width, d_baseline), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, click_baseline), (self.window_width, click_baseline), 2)
        pygame.draw.line(self.screen, CENTER_LINE, (0, walk_baseline), (self.window_width, walk_baseline), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, crouch_baseline), (self.window_width, crouch_baseline), 1)
        
        # Draw recorded segments
        current_x = 0
        for i, segment in enumerate(self.recorded_segments):
            segment_width = int(segment.duration_ms * time_scale)
            
            if segment.key == 'pause':
                color = (100, 50, 150)
                y_pos = y_offset
                pygame.draw.line(self.screen, color,
                               (current_x, y_pos - wave_height // 2),
                               (current_x + segment_width, y_pos - wave_height // 2), 3)
                pygame.draw.line(self.screen, color,
                               (current_x, y_pos + wave_height // 2),
                               (current_x + segment_width, y_pos + wave_height // 2), 3)
                pygame.draw.line(self.screen, color,
                               (current_x, y_pos - wave_height // 2),
                               (current_x, y_pos + wave_height // 2), 2)
                pygame.draw.line(self.screen, color,
                               (current_x + segment_width, y_pos - wave_height // 2),
                               (current_x + segment_width, y_pos + wave_height // 2), 2)
            elif segment.key == 'a':
                color = DARK_BLUE
                y_start = a_baseline
                y_end = a_baseline - wave_height
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, BLUE, False, points, 2)
            elif segment.key == 'd':
                color = DARK_RED
                y_start = d_baseline
                y_end = d_baseline + wave_height
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, RED, False, points, 2)
            elif segment.key == 'click':
                color = (80, 80, 80)
                y_start = click_baseline
                pygame.draw.line(self.screen, color, 
                               (current_x, y_start), 
                               (current_x + segment_width, y_start), 4)
            elif segment.key == 'walk':
                color = DARK_CYAN
                y_start = walk_baseline
                y_end = walk_baseline - wave_height // 2
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, CYAN, False, points, 2)
            elif segment.key == 'crouch':
                color = DARK_PURPLE
                y_start = crouch_baseline
                y_end = crouch_baseline + wave_height // 2
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, PURPLE, False, points, 2)
            else:
                current_x += segment_width
                continue
            
            # Duration label
            dur_label = self._get_cached_text(f"{segment.duration_ms}ms", GRAY, 'small')
            label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
            if segment.key == 'pause':
                label_y = y_offset - self.get_scaled_value(30)
            elif segment.key == 'a':
                label_y = a_baseline - wave_height - self.get_scaled_value(20)
            elif segment.key == 'd':
                label_y = d_baseline + wave_height + self.get_scaled_value(10)
            elif segment.key == 'walk':
                label_y = walk_baseline - wave_height // 2 - self.get_scaled_value(20)
            elif segment.key == 'crouch':
                label_y = crouch_baseline + wave_height // 2 + self.get_scaled_value(10)
            else:
                label_y = click_baseline + self.get_scaled_value(10)
            self.screen.blit(dur_label, (label_x, label_y))
            
            current_x += segment_width
        
        # Draw currently active inputs (while recording)
        if self.recording_pattern and self.current_segment_start > 0:
            current_duration_ms = (time.time() - self.current_segment_start) * 1000
            segment_width = int(current_duration_ms * time_scale)
            
            # Draw semi-transparent overlays for currently held keys
            if 'a' in self.current_segment_keys:
                y_start = a_baseline
                y_end = a_baseline - wave_height
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                surface = pygame.Surface((segment_width, wave_height), pygame.SRCALPHA)
                pygame.draw.polygon(surface, (*BLUE, 128), [(p[0]-current_x, p[1]-y_end) for p in points])
                self.screen.blit(surface, (current_x, y_end))
                pygame.draw.lines(self.screen, BLUE, False, points, 3)
                
                # Show duration
                dur_label = self._get_cached_text(f"{int(current_duration_ms)}ms", GREEN, 'small')
                label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
                label_y = a_baseline - wave_height - self.get_scaled_value(20)
                self.screen.blit(dur_label, (label_x, label_y))
            
            if 'd' in self.current_segment_keys:
                y_start = d_baseline
                y_end = d_baseline + wave_height
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                surface = pygame.Surface((segment_width, wave_height), pygame.SRCALPHA)
                pygame.draw.polygon(surface, (*RED, 128), [(p[0]-current_x, p[1]-y_start) for p in points])
                self.screen.blit(surface, (current_x, y_start))
                pygame.draw.lines(self.screen, RED, False, points, 3)
                
                # Show duration
                dur_label = self._get_cached_text(f"{int(current_duration_ms)}ms", GREEN, 'small')
                label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
                label_y = d_baseline + wave_height + self.get_scaled_value(10)
                self.screen.blit(dur_label, (label_x, label_y))
            
            if 'click' in self.current_segment_keys:
                pygame.draw.line(self.screen, WHITE, 
                               (current_x, click_baseline), 
                               (current_x + segment_width, click_baseline), 6)
                
                # Show duration
                dur_label = self._get_cached_text(f"{int(current_duration_ms)}ms", GREEN, 'small')
                label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
                label_y = click_baseline + self.get_scaled_value(10)
                self.screen.blit(dur_label, (label_x, label_y))
            
            if 'walk' in self.current_segment_keys:
                y_start = walk_baseline
                y_end = walk_baseline - wave_height // 2
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                surface = pygame.Surface((segment_width, wave_height // 2), pygame.SRCALPHA)
                pygame.draw.polygon(surface, (*CYAN, 128), [(p[0]-current_x, p[1]-y_end) for p in points])
                self.screen.blit(surface, (current_x, y_end))
                pygame.draw.lines(self.screen, CYAN, False, points, 3)
                
                # Show duration
                dur_label = self._get_cached_text(f"{int(current_duration_ms)}ms", GREEN, 'small')
                label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
                label_y = walk_baseline - wave_height // 2 - self.get_scaled_value(20)
                self.screen.blit(dur_label, (label_x, label_y))
            
            if 'crouch' in self.current_segment_keys:
                y_start = crouch_baseline
                y_end = crouch_baseline + wave_height // 2
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                surface = pygame.Surface((segment_width, wave_height // 2), pygame.SRCALPHA)
                pygame.draw.polygon(surface, (*PURPLE, 128), [(p[0]-current_x, p[1]-y_start) for p in points])
                self.screen.blit(surface, (current_x, y_start))
                pygame.draw.lines(self.screen, PURPLE, False, points, 3)
                
                # Show duration
                dur_label = self._get_cached_text(f"{int(current_duration_ms)}ms", GREEN, 'small')
                label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
                label_y = crouch_baseline + wave_height // 2 + self.get_scaled_value(10)
                self.screen.blit(dur_label, (label_x, label_y))
    
    def _draw_pattern_select_screen(self):
        """Draw the pattern selection screen."""
        self._draw_grid()
        
        # Header
        title = self._get_cached_text("Select Pattern", WHITE, 'large')
        self.screen.blit(title, (self.get_scaled_value(40), self.get_scaled_value(30)))
        
        # Instructions
        y_pos = self.get_scaled_value(100)
        instructions = [
            ("UP/DOWN: Navigate patterns", CYAN),
            ("SPACE: Start practice", GREEN),
            ("R: Create new pattern", PURPLE),
            ("M: Return to monitor", GRAY),
        ]
        
        for text, color in instructions:
            surf = self._get_cached_text(text, color, 'small')
            self.screen.blit(surf, (self.get_scaled_value(40), y_pos))
            y_pos += self.get_scaled_value(30)
        
        # List patterns
        y_pos = self.get_scaled_value(240)
        
        if not self.available_patterns:
            no_patterns = self._get_cached_text("No patterns available. Press R to create one.", YELLOW)
            self.screen.blit(no_patterns, (self.get_scaled_value(40), y_pos))
        else:
            box_height = self.get_scaled_value(120)
            spacing = self.get_scaled_value(20)
            total_item_height = box_height + spacing
            visible_area_start = y_pos
            visible_area_height = self.window_height - y_pos - self.get_scaled_value(50)
            max_visible_items = int(visible_area_height / total_item_height)
            
            if self.selected_pattern_index < max_visible_items // 2:
                scroll_offset = 0
            elif self.selected_pattern_index >= len(self.available_patterns) - max_visible_items // 2:
                scroll_offset = max(0, len(self.available_patterns) - max_visible_items)
            else:
                scroll_offset = self.selected_pattern_index - max_visible_items // 2
            
            for i in range(len(self.available_patterns)):
                pattern = self.available_patterns[i]
                is_selected = i == self.selected_pattern_index
                
                box_width = self.window_width - self.get_scaled_value(80)
                box_x = self.get_scaled_value(40)
                box_y = y_pos + (i - scroll_offset) * total_item_height
                
                if box_y < visible_area_start - box_height or box_y > self.window_height + box_height:
                    continue
                
                if is_selected:
                    pygame.draw.rect(self.screen, (40, 60, 80), 
                                   (box_x, box_y, box_width, box_height), 0, border_radius=8)
                    pygame.draw.rect(self.screen, CYAN, 
                                   (box_x, box_y, box_width, box_height), 3, border_radius=8)
                else:
                    pygame.draw.rect(self.screen, GRID_COLOR, 
                                   (box_x, box_y, box_width, box_height), 2, border_radius=8)
                
                name = self._get_cached_text(pattern.name, WHITE)
                self.screen.blit(name, (box_x + 20, box_y + 15))
                
                diff_color = GREEN if pattern.difficulty == "EASY" else (YELLOW if pattern.difficulty == "MEDIUM" else RED)
                difficulty = self._get_cached_text(f"Difficulty: {pattern.difficulty}", diff_color, 'small')
                self.screen.blit(difficulty, (box_x + 20, box_y + 50))
                
                segments_text = f"{len(pattern.segments)} segments, {pattern.get_total_duration_ms()}ms total, ±{pattern.tolerance_ms}ms tolerance"
                segments = self._get_cached_text(segments_text, GRAY, 'small')
                self.screen.blit(segments, (box_x + 20, box_y + 80))
            
            if scroll_offset > 0:
                arrow_y = y_pos - self.get_scaled_value(25)
                arrow_x = self.window_width // 2
                up_text = self._get_cached_text("^ More patterns above", CYAN, 'small')
                self.screen.blit(up_text, (arrow_x - up_text.get_width() // 2, arrow_y))
            
            if scroll_offset + max_visible_items < len(self.available_patterns):
                arrow_y = self.window_height - self.get_scaled_value(35)
                arrow_x = self.window_width // 2
                down_text = self._get_cached_text("v More patterns below", CYAN, 'small')
                self.screen.blit(down_text, (arrow_x - down_text.get_width() // 2, arrow_y))
    
    def _draw_pattern_practice_screen(self):
        """Draw the pattern practice screen."""
        if not self.pattern_evaluator:
            return
        
        self._draw_grid()
        
        pattern = self.pattern_evaluator.pattern
        
        # Header
        title = self._get_cached_text(f"Pattern: {pattern.name}", WHITE, 'large')
        self.screen.blit(title, (self.get_scaled_value(40), self.get_scaled_value(30)))
        
        # Instructions and tolerance
        y_pos = self.get_scaled_value(90)
        instructions = [
            ("Pattern loops automatically", CYAN),
            (f"Tolerance: ±{pattern.tolerance_ms}ms", YELLOW),
            ("ESC: Exit practice", RED),
        ]
        
        for text, color in instructions:
            surf = self._get_cached_text(text, color, 'small')
            self.screen.blit(surf, (self.get_scaled_value(40), y_pos))
            y_pos += self.get_scaled_value(25)
        
        # Statistics
        y_pos += self.get_scaled_value(15)
        stats_text = f"Attempts: {self.pattern_evaluator.attempts}  |  Successes: {self.pattern_evaluator.successes}"
        if self.pattern_evaluator.attempts > 0:
            success_rate = (self.pattern_evaluator.successes / self.pattern_evaluator.attempts * 100)
            stats_text += f"  |  Success Rate: {success_rate:.0f}%"
        
        stats = self._get_cached_text(stats_text, GRAY, 'small')
        self.screen.blit(stats, (self.get_scaled_value(40), y_pos))
        
        # Show waiting status if not started
        if self.pattern_evaluator.waiting_for_start:
            y_pos -= self.get_scaled_value(80)
            first_segment = pattern.segments[0]
            if first_segment.key == 'pause':
                waiting_text = "Pattern will start automatically..."
            else:
                key_name = first_segment.key.upper()
                waiting_text = f"Press {key_name} to start..."
            waiting = self._get_cached_text(waiting_text, CYAN, 'large')
            waiting_x = (self.window_width - waiting.get_width()) * 0.9
            self.screen.blit(waiting, (waiting_x, y_pos))
        
        # Draw pattern timeline
        chart_y = self.get_scaled_value(270)
        chart_height = self.get_scaled_value(380)
        self._draw_practice_timeline(chart_y, chart_height)
    
    def _draw_practice_timeline(self, chart_y: int, chart_height: int):
        """Draw the practice timeline with pattern overlay including walk/crouch tracks."""
        if not self.pattern_evaluator:
            return
        
        pattern = self.pattern_evaluator.pattern
        total_duration = pattern.get_total_duration_ms()
        time_scale = self.window_width / max(total_duration, 1000)
        
        y_offset = chart_y + chart_height // 2
        baseline_offset = self.get_scaled_value(100)
        wave_height = self.get_scaled_value(70)
        
        # Define baselines for all tracks
        a_baseline = y_offset - baseline_offset * 1.5
        d_baseline = y_offset + baseline_offset * 1.5
        click_baseline = y_offset
        walk_baseline = y_offset - baseline_offset * 0.5
        crouch_baseline = y_offset + baseline_offset * 0.5
        
        # Draw baseline guides
        pygame.draw.line(self.screen, CENTER_LINE, (0, a_baseline), (self.window_width, a_baseline), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, d_baseline), (self.window_width, d_baseline), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, click_baseline), (self.window_width, click_baseline), 2)
        pygame.draw.line(self.screen, CENTER_LINE, (0, walk_baseline), (self.window_width, walk_baseline), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, crouch_baseline), (self.window_width, crouch_baseline), 1)
        
        # Draw pattern segments
        current_x = 0
        for i, segment in enumerate(pattern.segments):
            segment_width = int(segment.duration_ms * time_scale)
            
            if segment.key == 'pause':
                color = (100, 50, 150)
                y_pos = y_offset
                pygame.draw.line(self.screen, color,
                               (current_x, y_pos - wave_height // 2),
                               (current_x + segment_width, y_pos - wave_height // 2), 3)
                pygame.draw.line(self.screen, color,
                               (current_x, y_pos + wave_height // 2),
                               (current_x + segment_width, y_pos + wave_height // 2), 3)
                pygame.draw.line(self.screen, color,
                               (current_x, y_pos - wave_height // 2),
                               (current_x, y_pos + wave_height // 2), 2)
                pygame.draw.line(self.screen, color,
                               (current_x + segment_width, y_pos - wave_height // 2),
                               (current_x + segment_width, y_pos + wave_height // 2), 2)
            elif segment.key == 'a':
                color = DARK_BLUE
                y_start = a_baseline
                y_end = a_baseline - wave_height
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, WHITE, False, points, 2)
            elif segment.key == 'd':
                color = DARK_RED
                y_start = d_baseline
                y_end = d_baseline + wave_height
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, WHITE, False, points, 2)
            elif segment.key == 'click':
                color = (80, 80, 80)
                y_start = click_baseline
                pygame.draw.line(self.screen, color, 
                               (current_x, y_start), 
                               (current_x + segment_width, y_start), 4)
            elif segment.key == 'walk':
                color = DARK_CYAN
                y_start = walk_baseline
                y_end = walk_baseline - wave_height // 2
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, CYAN, False, points, 2)
            elif segment.key == 'crouch':
                color = DARK_PURPLE
                y_start = crouch_baseline
                y_end = crouch_baseline + wave_height // 2
                points = [
                    (current_x, y_start),
                    (current_x, y_end),
                    (current_x + segment_width, y_end),
                    (current_x + segment_width, y_start)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.lines(self.screen, PURPLE, False, points, 2)
            else:
                current_x += segment_width
                continue
            
            # Duration label
            dur_label = self._get_cached_text(f"{segment.duration_ms}ms", GRAY, 'small')
            label_x = current_x + segment_width // 2 - dur_label.get_width() // 2
            if segment.key == 'pause':
                label_y = y_offset - self.get_scaled_value(30)
            elif segment.key == 'a':
                label_y = a_baseline - wave_height - self.get_scaled_value(20)
            elif segment.key == 'd':
                label_y = d_baseline + wave_height + self.get_scaled_value(10)
            elif segment.key == 'walk':
                label_y = walk_baseline - wave_height // 2 - self.get_scaled_value(20)
            elif segment.key == 'crouch':
                label_y = crouch_baseline + wave_height // 2 + self.get_scaled_value(10)
            else:
                label_y = click_baseline + self.get_scaled_value(10)
            self.screen.blit(dur_label, (label_x, label_y))
            
            current_x += segment_width
        
        # Draw progress indicator
        if not self.pattern_evaluator.waiting_for_start:
            elapsed_ms = (self.current_time - self.pattern_evaluator.start_time) * 1000
            progress_x = int(min(elapsed_ms, total_duration) * time_scale)
            pygame.draw.line(self.screen, YELLOW, 
                            (progress_x, chart_y), 
                            (progress_x, chart_y + chart_height), 3)
        
        # Draw user inputs overlay
        if not self.pattern_evaluator.waiting_for_start:
            self._draw_user_inputs_overlay(chart_y, chart_height, time_scale, 
                                           a_baseline, d_baseline, click_baseline, 
                                           walk_baseline, crouch_baseline, wave_height)
    
    def _draw_user_inputs_overlay(self, chart_y: int, chart_height: int, time_scale: float,
                                  a_baseline: int, d_baseline: int, click_baseline: int,
                                  walk_baseline: int, crouch_baseline: int, wave_height: int):
        """Draw the user's actual inputs overlaid on the pattern including walk/crouch."""
        if len(self.time_points) == 0 or not self.pattern_evaluator:
            return
        
        times = self.time_points.get_recent()
        a_states = self.a_points.get_recent(len(self.time_points))
        d_states = self.d_points.get_recent(len(self.time_points))
        click_states = self.click_points.get_recent(len(self.time_points))
        walk_states = self.shift_points.get_recent(len(self.time_points))
        crouch_states = self.ctrl_points.get_recent(len(self.time_points))
        
        time_offsets = (times - self.pattern_evaluator.start_time) * 1000
        x_coords = (time_offsets * time_scale).astype(np.int32)
        
        valid_mask = (x_coords >= 0) & (x_coords < self.window_width)
        if not np.any(valid_mask):
            return
        
        x_coords = x_coords[valid_mask]
        a_states = a_states[valid_mask]
        d_states = d_states[valid_mask]
        click_states = click_states[valid_mask]
        walk_states = walk_states[valid_mask]
        crouch_states = crouch_states[valid_mask]
        
        # Draw A key
        a_y = np.where(a_states > 0, a_baseline - wave_height, a_baseline)
        a_points = np.column_stack([x_coords, a_y])
        if len(a_points) > 1:
            pygame.draw.lines(self.screen, BLUE, False, a_points.tolist(), 3)
        
        # Draw D key
        d_y = np.where(d_states > 0, d_baseline + wave_height, d_baseline)
        d_points = np.column_stack([x_coords, d_y])
        if len(d_points) > 1:
            pygame.draw.lines(self.screen, RED, False, d_points.tolist(), 3)
        
        # Draw walk
        walk_y = np.where(walk_states > 0, walk_baseline - wave_height // 2, walk_baseline)
        walk_points = np.column_stack([x_coords, walk_y])
        if len(walk_points) > 1:
            pygame.draw.lines(self.screen, CYAN, False, walk_points.tolist(), 3)
        
        # Draw crouch
        crouch_y = np.where(crouch_states > 0, crouch_baseline + wave_height // 2, crouch_baseline)
        crouch_points = np.column_stack([x_coords, crouch_y])
        if len(crouch_points) > 1:
            pygame.draw.lines(self.screen, PURPLE, False, crouch_points.tolist(), 3)
        
        # Draw clicks
        click_points_list = x_coords[click_states > 0]
        for x in click_points_list:
            pygame.draw.circle(self.screen, WHITE, (int(x), click_baseline), 4)
    
    def _draw_static_elements(self):
        """Pre-render static grid and reference lines."""
        self._static_surface = pygame.Surface((self.window_width, self.window_height))
        self._static_surface.fill(DARK_BG)
        
        grid_size = self.get_scaled_value(40)
        for i in range(0, self.window_height, grid_size):
            pygame.draw.line(self._static_surface, GRID_COLOR, (0, i), (self.window_width, i), 1)
        for i in range(0, self.window_width, grid_size):
            pygame.draw.line(self._static_surface, GRID_COLOR, (i, 0), (i, self.window_height), 1)
        
        chart_height = self.get_scaled_value(500)
        chart_y_offset = self.get_scaled_value(80)
        center_y = chart_y_offset + chart_height // 2
        baseline_offset = self.get_scaled_value(120)
        
        pygame.draw.line(self._static_surface, CENTER_LINE, (0, center_y), (self.window_width, center_y), 2)
        pygame.draw.line(self._static_surface, CENTER_LINE, (0, center_y - baseline_offset), 
                        (self.window_width, center_y - baseline_offset), 1)
        pygame.draw.line(self._static_surface, CENTER_LINE, (0, center_y + baseline_offset), 
                        (self.window_width, center_y + baseline_offset), 1)
    
    def _draw_grid(self):
        """Draw background grid."""
        grid_size = self.get_scaled_value(40)
        for i in range(0, self.window_height, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, i), (self.window_width, i), 1)
        for i in range(0, self.window_width, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (i, 0), (i, self.window_height), 1)
    
    def _draw_timeline_data(self, chart_y_offset: int, chart_height: int):
        """Draw all timeline data with vectorized operations."""
        time_range = 5.0
        center_y = chart_y_offset + chart_height // 2
        baseline_offset = self.get_scaled_value(120)
        wave_height = self.get_scaled_value(80)
        
        a_baseline = center_y - baseline_offset
        d_baseline = center_y + baseline_offset
        
        times = self.time_points.get_recent()
        if len(times) == 0:
            return
        
        time_offsets = self.current_time - times
        valid_mask = time_offsets <= time_range
        
        if not np.any(valid_mask):
            return
        
        times = times[valid_mask]
        time_offsets = time_offsets[valid_mask]
        a_states = self.a_points.get_recent(len(self.time_points))[valid_mask]
        d_states = self.d_points.get_recent(len(self.time_points))[valid_mask]
        shift_states = self.shift_points.get_recent(len(self.time_points))[valid_mask]
        velocities = self.velocity_points.get_recent(len(self.time_points))[valid_mask]
        click_states = self.click_points.get_recent(len(self.time_points))[valid_mask]
        inaccuracy_types = self.inaccuracy_type_points.get_recent(len(self.time_points))[valid_mask]
        bullet_fired = self.bullet_fired_points.get_recent(len(self.time_points))[valid_mask]
        
        x_coords = self.window_width - (time_offsets * self.window_width / time_range).astype(np.int32)
        
        self._draw_shift_background(x_coords, shift_states, center_y, chart_height)
        
        # Draw A key
        a_y = np.where(a_states > 0, a_baseline - wave_height, a_baseline)
        a_points = np.column_stack([x_coords, a_y])
        if len(a_points) > 1:
            pygame.draw.lines(self.screen, DARK_BLUE, False, a_points.tolist(), 1)
        
        # Draw D key
        d_y = np.where(d_states > 0, d_baseline + wave_height, d_baseline)
        d_points = np.column_stack([x_coords, d_y])
        if len(d_points) > 1:
            pygame.draw.lines(self.screen, DARK_RED, False, d_points.tolist(), 1)
        
        self._draw_key_segments_fast(x_coords, times, a_states, a_y, BLUE, a_baseline - self.get_scaled_value(105))
        self._draw_key_segments_fast(x_coords, times, d_states, d_y, RED, d_baseline + self.get_scaled_value(105))
        
        # Velocity
        velocity_scale = self.get_scaled_value(100)
        vel_y = center_y - (velocities * velocity_scale).astype(np.int32)
        vel_points = np.column_stack([x_coords, vel_y])
        if len(vel_points) > 1:
            self._draw_velocity_line_with_walk(x_coords, vel_y, shift_states)
        
        self._draw_click_segments_fast(x_coords, click_states, inaccuracy_types, center_y)
        
        # Bullets
        bullet_mask = bullet_fired > 0
        if np.any(bullet_mask):
            tick_height = self.get_scaled_value(15)
            bullet_x = x_coords[bullet_mask]
            bullet_types = inaccuracy_types[bullet_mask]
            for x, inacc_type in zip(bullet_x, bullet_types):
                if inacc_type == InaccuracyType.ACCELERATING.value:
                    color = ORANGE
                elif inacc_type == InaccuracyType.CONSTANT.value:
                    color = PURPLE
                elif inacc_type == InaccuracyType.DECELERATING.value:
                    color = CYAN
                else:
                    color = RED
                pygame.draw.line(self.screen, color, 
                               (int(x), center_y - tick_height), 
                               (int(x), center_y + tick_height), 3)
    
    def _draw_shift_background(self, x_coords: np.ndarray, shift_states: np.ndarray, 
                               center_y: int, chart_height: int):
        """Draw subtle background highlight when shift is held."""
        shift_changes = np.diff(shift_states, prepend=0)
        starts = np.where(shift_changes == 1)[0]
        ends = np.where(shift_changes == -1)[0]
        
        if len(starts) == 0:
            return
        
        if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1]):
            ends = np.append(ends, len(shift_states) - 1)
        
        highlight_color = (30, 50, 80, 60)
        surface = pygame.Surface((self.window_width, chart_height // 2), pygame.SRCALPHA)
        
        for start_idx, end_idx in zip(starts, ends):
            if end_idx >= start_idx:
                x1 = int(x_coords[end_idx])
                x2 = int(x_coords[start_idx])
                width = x2 - x1
                if width > 0:
                    rect = pygame.Rect(x1, 0, width, chart_height // 2)
                    pygame.draw.rect(surface, highlight_color, rect)
        
        self.screen.blit(surface, (0, center_y - chart_height // 4))
    
    def _draw_velocity_line_with_walk(self, x_coords: np.ndarray, vel_y: np.ndarray, 
                                      shift_states: np.ndarray):
        """Draw velocity line with different colors for walking vs running."""
        points = np.column_stack([x_coords, vel_y])
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, YELLOW, False, points.tolist(), 2)
            
            shift_changes = np.diff(shift_states, prepend=0)
            starts = np.where(shift_changes == 1)[0]
            ends = np.where(shift_changes == -1)[0]
            
            if len(starts) > 0:
                if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1]):
                    ends = np.append(ends, len(shift_states) - 1)
                
                for start_idx, end_idx in zip(starts, ends):
                    if end_idx > start_idx:
                        segment = points[start_idx:end_idx + 1]
                        if len(segment) > 1:
                            pygame.draw.lines(self.screen, (100, 200, 255), False, segment.tolist(), 2)
    
    def _draw_key_segments_fast(self, x_coords: np.ndarray, times: np.ndarray, 
                                states: np.ndarray, y_coords: np.ndarray,
                                color: Tuple[int, int, int], label_y: int):
        """Fast segment drawing using vectorized operations."""
        state_changes = np.diff(states, prepend=0)
        starts = np.where(state_changes == 1)[0]
        ends = np.where(state_changes == -1)[0]
        
        if len(starts) == 0:
            return
        
        if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1]):
            ends = np.append(ends, len(states) - 1)
        
        for start_idx, end_idx in zip(starts, ends):
            if end_idx > start_idx:
                segment_x = x_coords[start_idx:end_idx + 1]
                segment_y = y_coords[start_idx:end_idx + 1]
                points = np.column_stack([segment_x, segment_y])
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, color, False, points.tolist(), 3)
                    
                    duration_ms = int((times[end_idx] - times[start_idx]) * 1000)
                    label = self._get_cached_text(f"{duration_ms} ms", GRAY, 'small')
                    text_x = int(segment_x[0])
                    text_y = label_y - label.get_height() // 2
                    
                    if 0 <= text_x <= self.window_width - label.get_width():
                        self.screen.blit(label, (text_x, text_y))
    
    def _draw_click_segments_fast(self, x_coords: np.ndarray, click_states: np.ndarray,
                                  inaccuracy_types: np.ndarray, center_y: int):
        """Fast click segment drawing with different colors."""
        click_changes = np.diff(click_states, prepend=0)
        starts = np.where(click_changes == 1)[0]
        ends = np.where(click_changes == -1)[0]
        
        if len(starts) == 0:
            return
        
        if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1]):
            ends = np.append(ends, len(click_states) - 1)
        
        dot_radius = max(4, self.get_scaled_value(6))
        
        for start_idx, end_idx in zip(starts, ends):
            if end_idx >= start_idx:
                inaccuracy_type = inaccuracy_types[start_idx]
                segment_x = x_coords[start_idx:end_idx + 1]
                segment_y = np.full(len(segment_x), center_y)
                points = np.column_stack([segment_x, segment_y]).tolist()
                
                if len(points) > 1:
                    if inaccuracy_type == InaccuracyType.ACCELERATING.value:
                        pygame.draw.lines(self.screen, ORANGE, False, points, 4)
                    elif inaccuracy_type == InaccuracyType.CONSTANT.value:
                        pygame.draw.lines(self.screen, PURPLE, False, points, 4)
                    elif inaccuracy_type == InaccuracyType.DECELERATING.value:
                        pygame.draw.lines(self.screen, CYAN, False, points, 4)
                    elif inaccuracy_type != InaccuracyType.NONE.value:
                        pygame.draw.lines(self.screen, GREEN, False, points, 4)
                    
                    pygame.draw.lines(self.screen, WHITE, False, points, 2)
                
                if points:
                    pygame.draw.circle(self.screen, WHITE, (int(points[0][0]), int(points[0][1])), dot_radius, 0)
                    if len(points) > 1:
                        pygame.draw.circle(self.screen, WHITE, (int(points[-1][0]), int(points[-1][1])), dot_radius, 0)
    
    def _draw_header(self):
        """Draw application header with legend."""
        header_padding = self.get_scaled_value(20)
        header_y = self.get_scaled_value(15)
        
        title = self._get_cached_text("INPUT MONITOR", WHITE)
        self.screen.blit(title, (header_padding, header_y))
        
        legend_spacing = self.get_scaled_value(90)
        legend_x = self.window_width - self.get_scaled_value(400)
        
        left = self.config['keys']['left'].upper()
        right = self.config['keys']['right'].upper()
        labels = [
            (f"{left} Key", BLUE),
            (f"{right} Key", RED),
            ("Click", WHITE),
            ("Velocity", YELLOW)
        ]
        
        for i, (text, color) in enumerate(labels):
            label = self._get_cached_text(text, color)
            offset = legend_spacing * i if i < 3 else legend_spacing * (i - 0.2)
            self.screen.blit(label, (legend_x + offset, header_y))
    
    def _draw_status_bar(self):
        """Draw status bar with mode and state information."""
        header_padding = self.get_scaled_value(20)
        bottom_y = self.window_height - self.get_scaled_value(35)
        
        status = "PAUSED" if self.paused else "RECORDING"
        status_color = RED if self.paused else GREEN
        status_text = self._get_cached_text(status, status_color)
        self.screen.blit(status_text, (header_padding, bottom_y))
        
        current_x = header_padding + status_text.get_width() + self.get_scaled_value(30)
        
        if self.velocity_sim.is_walking:
            walk_text = self._get_cached_text("WALKING", BLUE)
            self.screen.blit(walk_text, (current_x, bottom_y))
            current_x += walk_text.get_width() + self.get_scaled_value(20)
        
        if self.active_beeper == InaccuracyType.ACCELERATING:
            inacc_text = self._get_cached_text("ACCEL INACCURATE", ORANGE)
            self.screen.blit(inacc_text, (current_x, bottom_y))
        elif self.active_beeper == InaccuracyType.CONSTANT:
            inacc_text = self._get_cached_text("CONST INACCURATE", PURPLE)
            self.screen.blit(inacc_text, (current_x, bottom_y))
        elif self.active_beeper == InaccuracyType.DECELERATING:
            inacc_text = self._get_cached_text("DECEL INACCURATE", CYAN)
            self.screen.blit(inacc_text, (current_x, bottom_y))
        
        keys = self.config['keys']
        help_x = self.window_width - self.get_scaled_value(400)
        help_text = self._get_cached_text(f"M: Pattern Mode | {str(keys['pause']).upper()}: Pause", CENTER_LINE)
        self.screen.blit(help_text, (help_x, bottom_y))
    
    def _draw_monitor_screen(self):
        """Draw the standard monitor visualization."""
        if self._need_static_redraw:
            self._draw_static_elements()
            self._need_static_redraw = False
        
        if self._static_surface:
            self.screen.blit(self._static_surface, (0, 0))
        
        if len(self.time_points) > 1:
            chart_height = self.get_scaled_value(500)
            chart_y_offset = self.get_scaled_value(80)
            self._draw_timeline_data(chart_y_offset, chart_height)
        
        self._draw_header()
        self._draw_status_bar()
    
    def handle_pygame_events(self):
        """Handle pygame events including text input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.handle_resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if self.mode == AppMode.PATTERN_CREATE and self.name_input_active:
                    if event.key == pygame.K_RETURN:
                        if self.pattern_name and self.recorded_segments:
                            pattern = Pattern(
                                name=self.pattern_name,
                                difficulty=self.pattern_difficulty,
                                segments=self.recorded_segments,
                                tolerance_ms=self.pattern_tolerance
                            )
                            if self.pattern_manager.save_pattern(pattern):
                                self.add_toast(f"Pattern '{self.pattern_name}' saved!", GREEN)
                                self.mode = AppMode.PATTERN_SELECT
                                self.available_patterns = self.pattern_manager.load_patterns()
                            else:
                                self.add_toast("Failed to save pattern", RED)
                            
                            self.recording_pattern = False
                            self.recorded_segments = []
                            self.pattern_name = ""
                            self.name_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.pattern_name = self.pattern_name[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.name_input_active = False
                        self.pattern_name = ""
                    elif event.unicode and event.unicode.isprintable():
                        if len(self.pattern_name) < 30:
                            self.pattern_name += event.unicode
                
                elif self.mode == AppMode.PATTERN_SELECT:
                    if event.key == pygame.K_UP:
                        self.selected_pattern_index = max(0, self.selected_pattern_index - 1)
                    elif event.key == pygame.K_DOWN:
                        self.selected_pattern_index = min(len(self.available_patterns) - 1, 
                                                         self.selected_pattern_index + 1)
                    elif event.key == pygame.K_DELETE:
                        if self.available_patterns:
                            pattern = self.available_patterns[self.selected_pattern_index]
                            if self.pattern_manager.delete_pattern(pattern.name):
                                self.add_toast(f"Deleted '{pattern.name}'", YELLOW)
                                self.available_patterns = self.pattern_manager.load_patterns()
                                self.selected_pattern_index = min(self.selected_pattern_index, 
                                                                 len(self.available_patterns) - 1)
    
    def run(self):
        """Main application loop."""
        try:
            while self.running:
                self.handle_pygame_events()
                self.update_data()
                self.draw()
                
                if self.target_fps is not None:
                    self.clock.tick(self.target_fps)
                else:
                    self.clock.tick()
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources on exit."""
        if self.active_beeper:
            try:
                self.accel_beeper.stop()
                self.constant_beeper.stop()
                self.decel_beeper.stop()
            except:
                pass

        keyboard.unhook_all()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    monitor = InputMonitor()
    monitor.run()