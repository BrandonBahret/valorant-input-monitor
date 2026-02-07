"""
Author: Claude Ai

Input Monitor - Real-time visualization of keyboard and mouse inputs with velocity tracking.

Displays A/D key presses, mouse clicks, and simulated velocity on a scrolling timeline.
Tracks shooting accuracy based on movement velocity and provides audio feedback.
"""

import sys
import time
import ctypes
from typing import List, Tuple, Optional
from queue import Queue

import pygame
import keyboard
import numpy as np

from beeper import ContinuousWavePlayer


# Display Configuration
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 700
MIN_WIDTH = 800
MIN_HEIGHT = 600
FPS = 165

# Gameplay Constants
FIRE_RATE_MS = 1000 / 64
OVERLAP_BUFFER_MS = 70

# Color Palette
DARK_BG = (15, 15, 20)
GRID_COLOR = (30, 35, 40)
CENTER_LINE = (60, 65, 70)
WHITE = (255, 255, 255)
BLUE = (80, 180, 255)
RED = (255, 80, 80)
GREEN = (100, 255, 150)
YELLOW = (255, 220, 80)
GRAY = (130, 130, 130)

# Windows API
VK_LBUTTON = 0x01

# Performance constants
MAX_BUFFER_SIZE = 2000  # Fixed buffer size
MATH_LOG2 = 0.6931471805599453  # Pre-computed ln(2)


class VelocitySimulator:
    """Simulates player movement velocity with acceleration and deceleration."""
    
    __slots__ = ('velocity', 'direction', 'accel_progress', 'max_velocity', 
                 'accel_time', 'velocity_threshold', 'decel_half_life',
                 '_log2_decel', '_accel_exp')
    
    def __init__(self):
        self.velocity = 0.0
        self.direction = 0
        self.accel_progress = 0.0
        
        # Physics parameters
        self.max_velocity = 1.0
        self.accel_time = 0.480
        self.velocity_threshold = 0.0145
        self.decel_half_life = 0.02125
        
        # Pre-compute constants
        self._log2_decel = MATH_LOG2 / self.decel_half_life
        self._accel_exp = 1.45
    
    def update(self, dt: float, a_held: bool, d_held: bool) -> float:
        """Update velocity based on input state and return current velocity with direction."""
        desired_direction = 0
        if a_held and not d_held:
            desired_direction = -1
        elif d_held and not a_held:
            desired_direction = 1
        
        if desired_direction == 0:
            self._apply_deceleration(dt)
        elif desired_direction == self.direction or self.direction == 0:
            self._apply_acceleration(dt, desired_direction)
        else:
            self._apply_direction_change(dt, desired_direction)
        
        return self.velocity * self.direction
    
    def is_moving(self) -> bool:
        """Check if velocity exceeds accuracy threshold."""
        return self.velocity > self.velocity_threshold
    
    def _apply_acceleration(self, dt: float, direction: int):
        """Accelerate in the desired direction with easing curve."""
        self.accel_progress = min(1.0, self.accel_progress + dt / self.accel_time)
        eased_progress = self.accel_progress ** self._accel_exp
        self.velocity = eased_progress * self.max_velocity
        self.direction = direction
    
    def _apply_deceleration(self, dt: float):
        """Exponential decay when no input."""
        self.accel_progress = 0.0
        # Use pre-computed constant
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
                 'has_inaccurate_bullet', 'movement_start_time', 'was_moving')
    
    def __init__(self, velocity_sim: VelocitySimulator):
        self.velocity_sim = velocity_sim
        self.mouse_held = False
        self.mouse_press_time = 0.0
        self.last_bullet_time = 0.0
        self.has_inaccurate_bullet = False
        self.movement_start_time = 0.0
        self.was_moving = False
    
    def on_mouse_press(self, current_time: float):
        """Handle mouse button press."""
        self.mouse_press_time = current_time
        self.last_bullet_time = current_time - (FIRE_RATE_MS / 1000.0)
        self.mouse_held = True
        self.has_inaccurate_bullet = False
        self.movement_start_time = 0.0
        self.was_moving = False
    
    def on_mouse_release(self):
        """Handle mouse button release."""
        self.mouse_held = False
        self.movement_start_time = 0.0
        self.was_moving = False
    
    def check_bullet_fire(self, current_time: float) -> bool:
        """Check if a bullet should fire and return if it's inaccurate."""
        if not self.mouse_held:
            return False
        
        is_moving_now = self.velocity_sim.is_moving()
        
        # Track when movement starts during active shooting
        if is_moving_now and not self.was_moving and self.mouse_held:
            time_since_click = (current_time - self.mouse_press_time) * 1000
            if time_since_click > 0:
                self.movement_start_time = current_time
        elif not is_moving_now:
            self.movement_start_time = 0.0
        
        self.was_moving = is_moving_now
        
        self.last_bullet_time = current_time
        
        # Determine if bullet is inaccurate
        is_inaccurate = self._check_inaccuracy(current_time, is_moving_now)
        if is_inaccurate:
            self.has_inaccurate_bullet = True
        
        return is_inaccurate
    
    def _check_inaccuracy(self, current_time: float, is_moving: bool) -> bool:
        """Determine if current bullet is inaccurate based on movement and grace period."""
        if not is_moving:
            return False
        
        if self.movement_start_time > 0:
            time_since_movement_start = (current_time - self.movement_start_time) * 1000
            return time_since_movement_start > OVERLAP_BUFFER_MS
        
        return True


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
    
    def __len__(self):
        return self._size


class InputMonitor:
    """Main application for visualizing keyboard and mouse inputs."""
    
    def __init__(self):
        pygame.init()
        
        self.window_width = DEFAULT_WIDTH
        self.window_height = DEFAULT_HEIGHT
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.DOUBLEBUF, vsync=1
        )
        pygame.display.set_caption("Input Monitor")
        self.clock = pygame.time.Clock()
        
        # Core components
        self.velocity_sim = VelocitySimulator()
        self.shooting_tracker = ShootingTracker(self.velocity_sim)
        
        # Audio feedback
        self.beeper = ContinuousWavePlayer()
        self.beeper_queue = Queue()
        self.beeper_active = False
        
        # Optimized data storage using numpy ring buffers
        self.time_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.float64)
        self.a_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.d_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.click_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.click_inaccurate = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        self.velocity_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.float32)
        self.bullet_fired_points = RingBuffer(MAX_BUFFER_SIZE, dtype=np.uint8)
        
        # Input state
        self.a_key_held = False
        self.d_key_held = False
        self.prev_mouse_held = False
        
        # Application state
        self.paused = False
        self.running = True
        self.current_time = 0.0
        self.last_update = time.time()
        
        # Cached rendering resources
        self._font_cache = {}
        self._static_surface = None
        self._need_static_redraw = True
        
        self.update_fonts()
        self._setup_input_hooks()
    
    def update_fonts(self):
        """Scale fonts based on window height."""
        scale = self.window_height / DEFAULT_HEIGHT
        font_size = max(20, int(28 * scale))
        small_font_size = max(14, int(20 * scale))
        
        self.font = pygame.font.Font(None, font_size)
        self.small_font = pygame.font.Font(None, small_font_size)
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
    
    def _setup_input_hooks(self):
        """Configure keyboard event handlers."""
        keyboard.on_press_key('a', lambda _: self._on_key_press('a'), suppress=False)
        keyboard.on_release_key('a', lambda _: self._on_key_release('a'), suppress=False)
        keyboard.on_press_key('d', lambda _: self._on_key_press('d'), suppress=False)
        keyboard.on_release_key('d', lambda _: self._on_key_release('d'), suppress=False)
        keyboard.on_press_key('tab', lambda _: self._toggle_pause(), suppress=False)
    
    def _on_key_press(self, key: str):
        """Handle key press events."""
        if key == 'a':
            self.a_key_held = True
        elif key == 'd':
            self.d_key_held = True
    
    def _on_key_release(self, key: str):
        """Handle key release events."""
        if key == 'a':
            self.a_key_held = False
        elif key == 'd':
            self.d_key_held = False
    
    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
    
    def _poll_mouse_state(self):
        """Check mouse button state using Windows API and handle state changes."""
        state = ctypes.windll.user32.GetAsyncKeyState(VK_LBUTTON)
        current_mouse_held = bool(state & 0x8000)
        
        # Detect state transitions
        if current_mouse_held and not self.prev_mouse_held:
            self.shooting_tracker.on_mouse_press(time.time())
        elif not current_mouse_held and self.prev_mouse_held:
            self.shooting_tracker.on_mouse_release()
        
        # Manage audio feedback
        if self.beeper_active and (not self.velocity_sim.is_moving() or not current_mouse_held):
            self.beeper_queue.put('stop')
            self.beeper_active = False
        
        self.prev_mouse_held = current_mouse_held
    
    def update_data(self):
        """Update simulation state and append data points."""
        if self.paused:
            return
        
        self._poll_mouse_state()
        
        current_real_time = time.time()
        dt = current_real_time - self.last_update
        self.last_update = current_real_time
        self.current_time += dt
        
        # Update physics
        velocity = self.velocity_sim.update(dt, self.a_key_held, self.d_key_held)
        
        # Check for bullet fire
        bullet_fired_inaccurate = self.shooting_tracker.check_bullet_fire(current_real_time)
        
        # Start beeper for inaccurate shots
        if bullet_fired_inaccurate and not self.beeper_active:
            self.beeper_queue.put('start')
            self.beeper_active = True
        
        # Process audio queue
        self._process_beeper_queue()
        
        # Append data points - single write operation per buffer
        self.time_points.append(self.current_time)
        self.a_points.append(1 if self.a_key_held else 0)
        self.d_points.append(1 if self.d_key_held else 0)
        self.click_points.append(1 if self.shooting_tracker.mouse_held else 0)
        self.click_inaccurate.append(1 if self.shooting_tracker.has_inaccurate_bullet else 0)
        self.velocity_points.append(velocity)
        self.bullet_fired_points.append(1 if bullet_fired_inaccurate else 0)
    
    def _process_beeper_queue(self):
        """Process queued beeper commands."""
        self.beeper.update()
        while not self.beeper_queue.empty():
            try:
                cmd = self.beeper_queue.get_nowait()
                if cmd == 'start':
                    self.beeper.start()
                elif cmd == 'stop':
                    self.beeper.stop()
            except:
                break
    
    def _get_cached_text(self, text: str, color: Tuple[int, int, int], font_type: str = 'normal') -> pygame.Surface:
        """Get cached rendered text surface."""
        key = (text, color, font_type)
        if key not in self._font_cache:
            font = self.font if font_type == 'normal' else self.small_font
            self._font_cache[key] = font.render(text, True, color)
        return self._font_cache[key]
    
    def draw_chart(self):
        """Render the main visualization with optimizations."""
        # Draw static elements if needed
        if self._need_static_redraw:
            self._draw_static_elements()
            self._need_static_redraw = False
        
        # Clear screen
        self.screen.fill(DARK_BG)
        
        # Draw static background
        if self._static_surface:
            self.screen.blit(self._static_surface, (0, 0))
        
        # Draw dynamic timeline data
        if len(self.time_points) > 1:
            chart_height = self.get_scaled_value(500)
            chart_y_offset = self.get_scaled_value(80)
            self._draw_timeline_data(chart_y_offset, chart_height)
        
        # Draw dynamic overlays
        self._draw_header()
        self._draw_status_bar()
    
    def _draw_static_elements(self):
        """Pre-render static grid and reference lines."""
        self._static_surface = pygame.Surface((self.window_width, self.window_height))
        self._static_surface.fill(DARK_BG)
        
        # Draw grid
        grid_size = self.get_scaled_value(40)
        for i in range(0, self.window_height, grid_size):
            pygame.draw.line(self._static_surface, GRID_COLOR, (0, i), (self.window_width, i), 1)
        for i in range(0, self.window_width, grid_size):
            pygame.draw.line(self._static_surface, GRID_COLOR, (i, 0), (i, self.window_height), 1)
        
        # Draw center lines
        chart_height = self.get_scaled_value(500)
        chart_y_offset = self.get_scaled_value(80)
        center_y = chart_y_offset + chart_height // 2
        baseline_offset = self.get_scaled_value(120)
        
        pygame.draw.line(self._static_surface, CENTER_LINE, (0, center_y), (self.window_width, center_y), 2)
        pygame.draw.line(self._static_surface, CENTER_LINE, (0, center_y - baseline_offset), 
                        (self.window_width, center_y - baseline_offset), 1)
        pygame.draw.line(self._static_surface, CENTER_LINE, (0, center_y + baseline_offset), 
                        (self.window_width, center_y + baseline_offset), 1)
    
    def _draw_timeline_data(self, chart_y_offset: int, chart_height: int):
        """Draw all timeline data with vectorized operations."""
        time_range = 5.0
        center_y = chart_y_offset + chart_height // 2
        baseline_offset = self.get_scaled_value(120)
        wave_height = self.get_scaled_value(80)
        
        a_baseline = center_y - baseline_offset
        d_baseline = center_y + baseline_offset
        
        # Get recent data as numpy arrays
        times = self.time_points.get_recent()
        if len(times) == 0:
            return
        
        # Vectorized time offset calculation
        time_offsets = self.current_time - times
        valid_mask = time_offsets <= time_range
        
        if not np.any(valid_mask):
            return
        
        # Filter arrays
        times = times[valid_mask]
        time_offsets = time_offsets[valid_mask]
        a_states = self.a_points.get_recent(len(self.time_points))[valid_mask]
        d_states = self.d_points.get_recent(len(self.time_points))[valid_mask]
        velocities = self.velocity_points.get_recent(len(self.time_points))[valid_mask]
        click_states = self.click_points.get_recent(len(self.time_points))[valid_mask]
        click_bad = self.click_inaccurate.get_recent(len(self.time_points))[valid_mask]
        bullet_fired = self.bullet_fired_points.get_recent(len(self.time_points))[valid_mask]
        
        # Vectorized x coordinate calculation
        x_coords = self.window_width - (time_offsets * self.window_width / time_range).astype(np.int32)
        
        # Draw A key line (vectorized)
        a_y = np.where(a_states > 0, a_baseline - wave_height, a_baseline)
        a_points = np.column_stack([x_coords, a_y])
        if len(a_points) > 1:
            pygame.draw.lines(self.screen, (40, 80, 120), False, a_points.tolist(), 1)
        
        # Draw D key line (vectorized)
        d_y = np.where(d_states > 0, d_baseline + wave_height, d_baseline)
        d_points = np.column_stack([x_coords, d_y])
        if len(d_points) > 1:
            pygame.draw.lines(self.screen, (120, 40, 40), False, d_points.tolist(), 1)
        
        # Draw active segments for A and D
        self._draw_key_segments_fast(x_coords, times, a_states, a_y, BLUE, a_baseline - self.get_scaled_value(105))
        self._draw_key_segments_fast(x_coords, times, d_states, d_y, RED, d_baseline + self.get_scaled_value(105))
        
        # Draw velocity line (vectorized)
        velocity_scale = self.get_scaled_value(100)
        vel_y = center_y - (velocities * velocity_scale).astype(np.int32)
        vel_points = np.column_stack([x_coords, vel_y])
        if len(vel_points) > 1:
            pygame.draw.lines(self.screen, YELLOW, False, vel_points.tolist(), 2)
        
        # Draw click segments
        self._draw_click_segments_fast(x_coords, click_states, click_bad, center_y)
        
        # Draw bullet markers
        bullet_mask = bullet_fired > 0
        if np.any(bullet_mask):
            tick_height = self.get_scaled_value(15)
            bullet_x = x_coords[bullet_mask]
            for x in bullet_x:
                pygame.draw.line(self.screen, RED, 
                               (int(x), center_y - tick_height), 
                               (int(x), center_y + tick_height), 3)
    
    def _draw_key_segments_fast(self, x_coords: np.ndarray, times: np.ndarray, 
                                states: np.ndarray, y_coords: np.ndarray,
                                color: Tuple[int, int, int], label_y: int):
        """Fast segment drawing using vectorized operations."""
        # Find state transitions
        state_changes = np.diff(states, prepend=0)
        starts = np.where(state_changes == 1)[0]
        ends = np.where(state_changes == -1)[0]
        
        # Handle edge cases
        if len(starts) == 0:
            return
        
        if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1]):
            ends = np.append(ends, len(states) - 1)
        
        # Draw segments
        for start_idx, end_idx in zip(starts, ends):
            if end_idx > start_idx:
                segment_x = x_coords[start_idx:end_idx + 1]
                segment_y = y_coords[start_idx:end_idx + 1]
                points = np.column_stack([segment_x, segment_y])
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, color, False, points.tolist(), 3)
                    
                    # Draw duration label
                    duration_ms = int((times[end_idx] - times[start_idx]) * 1000)
                    label = self._get_cached_text(f"{duration_ms} ms", GRAY, 'small')
                    text_x = int(segment_x[0])
                    text_y = label_y - label.get_height() // 2
                    
                    if 0 <= text_x <= self.window_width - label.get_width():
                        self.screen.blit(label, (text_x, text_y))
    
    def _draw_click_segments_fast(self, x_coords: np.ndarray, click_states: np.ndarray,
                                  click_bad: np.ndarray, center_y: int):
        """Fast click segment drawing."""
        # Find click transitions
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
                is_bad = click_bad[start_idx] > 0
                segment_x = x_coords[start_idx:end_idx + 1]
                segment_y = np.full(len(segment_x), center_y)
                points = np.column_stack([segment_x, segment_y]).tolist()
                
                if len(points) > 1:
                    if is_bad:
                        pygame.draw.lines(self.screen, GREEN, False, points, 4)
                    pygame.draw.lines(self.screen, WHITE, False, points, 2)
                
                # Draw dots
                if points:
                    pygame.draw.circle(self.screen, WHITE, (int(points[0][0]), int(points[0][1])), dot_radius, 0)
                    if len(points) > 1:
                        pygame.draw.circle(self.screen, WHITE, (int(points[-1][0]), int(points[-1][1])), dot_radius, 0)
    
    def _draw_header(self):
        """Draw application header with legend using cached text."""
        header_padding = self.get_scaled_value(20)
        header_y = self.get_scaled_value(15)
        
        title = self._get_cached_text("INPUT MONITOR", WHITE)
        self.screen.blit(title, (header_padding, header_y))
        
        legend_spacing = self.get_scaled_value(100)
        legend_x = self.window_width - self.get_scaled_value(480)
        
        labels = [
            ("A Key", BLUE),
            ("D Key", RED),
            ("Click", WHITE),
            ("Velocity", YELLOW)
        ]
        
        for i, (text, color) in enumerate(labels):
            label = self._get_cached_text(text, color)
            offset = legend_spacing * (i if i < 2 else i + 0.9 if i == 3 else i)
            self.screen.blit(label, (legend_x + offset, header_y))
    
    def _draw_status_bar(self):
        """Draw status bar with pause state."""
        header_padding = self.get_scaled_value(20)
        bottom_y = self.window_height - self.get_scaled_value(35)
        
        status = "PAUSED" if self.paused else "RECORDING"
        status_color = RED if self.paused else GREEN
        status_text = self._get_cached_text(status, status_color)
        self.screen.blit(status_text, (header_padding, bottom_y))
        
        help_text = self._get_cached_text("TAB: Pause/Resume", CENTER_LINE)
        help_x = self.window_width - self.get_scaled_value(220)
        self.screen.blit(help_text, (help_x, bottom_y))
    
    def run(self):
        """Main application loop."""
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.handle_resize(event.w, event.h)
                
                self.update_data()
                self.draw_chart()
                pygame.display.flip()
                self.clock.tick(FPS)
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources on exit."""
        if self.beeper_active:
            try:
                self.beeper.stop()
            except:
                pass
        keyboard.unhook_all()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    monitor = InputMonitor()
    monitor.run()