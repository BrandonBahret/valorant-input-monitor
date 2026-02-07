"""
Author: Claude Ai

Input Monitor - Real-time visualization of keyboard and mouse inputs with velocity tracking.

Displays A/D key presses, mouse clicks, and simulated velocity on a scrolling timeline.
Tracks shooting accuracy based on movement velocity and provides audio feedback.
"""

import sys
import time
import ctypes
from typing import List, Tuple, Deque
from collections import deque
from queue import Queue

import pygame
import keyboard

from beeper import ContinuousWavePlayer


# Display Configuration
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 700
MIN_WIDTH = 800
MIN_HEIGHT = 600
FPS = 165

# Gameplay Constants
FIRE_RATE_MS = 1000 / 16
OVERLAP_BUFFER_MS = 100  # Grace period for movement after accurate shooting starts

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


class Segment(list):
    """Extended list that stores duration metadata for visualization segments."""
    
    def __init__(self, *args):
        super().__init__(*args)
        self.duration_ms = 0
    
    def xy_points(self) -> List[Tuple[int, int]]:
        """Extract only x,y coordinates for drawing, ignoring timestamp."""
        return [(p[0], p[1]) for p in self]


class VelocitySimulator:
    """Simulates player movement velocity with acceleration and deceleration."""
    
    def __init__(self):
        self.velocity = 0.0
        self.direction = 0
        self.accel_progress = 0.0
        
        # Physics parameters
        self.max_velocity = 1.0
        self.accel_time = 0.475
        self.velocity_threshold = 0.0151
        self.decel_half_life = 0.021
    
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
        return abs(self.velocity) > self.velocity_threshold
    
    def _apply_acceleration(self, dt: float, direction: int):
        """Accelerate in the desired direction with easing curve."""
        self.accel_progress = min(1.0, self.accel_progress + dt / self.accel_time)
        eased_progress = self.accel_progress ** 1.5
        self.velocity = eased_progress * self.max_velocity
        self.direction = direction
    
    def _apply_deceleration(self, dt: float):
        """Exponential decay when no input."""
        import math
        self.accel_progress = 0.0
        decay_factor = math.exp(-dt * math.log(2) / self.decel_half_life)
        self.velocity *= decay_factor
        
        if self.velocity < 0.01:
            self.velocity = 0.0
            self.direction = 0
    
    def _apply_direction_change(self, dt: float, new_direction: int):
        """Handle counter-strafing when changing direction."""
        import math
        self.accel_progress = 0.0
        counter_strafe_half_life = self.decel_half_life
        decay_factor = math.exp(-dt * math.log(2) / counter_strafe_half_life)
        self.velocity *= decay_factor
        
        if self.velocity < 0.01:
            self.velocity = 0.0
            self.direction = new_direction


class ShootingTracker:
    """Tracks shooting mechanics including fire rate, accuracy, and grace periods."""
    
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
        
        # Check fire rate
        time_since_last_bullet = (current_time - self.last_bullet_time) * 1000
        if time_since_last_bullet < FIRE_RATE_MS:
            return False
        
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


class InputMonitor:
    """Main application for visualizing keyboard and mouse inputs."""
    
    def __init__(self):
        pygame.init()
        
        self.window_width = DEFAULT_WIDTH
        self.window_height = DEFAULT_HEIGHT
        # self.screen = pygame.display.set_mode(
        #     (self.window_width, self.window_height),
        #     pygame.RESIZABLE | pygame.DOUBLEBUF, vsync=1
        # )
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
        
        # Data storage
        self.max_points = self.window_width
        self.time_points: Deque[float] = deque(maxlen=self.max_points)
        self.a_points: Deque[int] = deque(maxlen=self.max_points)
        self.d_points: Deque[int] = deque(maxlen=self.max_points)
        self.click_points: Deque[Tuple[int, bool]] = deque(maxlen=self.max_points)
        self.velocity_points: Deque[float] = deque(maxlen=self.max_points)
        self.bullet_fired_points: Deque[bool] = deque(maxlen=self.max_points)
        
        # Input state
        self.a_key_held = False
        self.d_key_held = False
        self.prev_mouse_held = False
        
        # Application state
        self.paused = False
        self.running = True
        self.current_time = 0.0
        self.last_update = time.time()
        
        self.update_fonts()
        self._setup_input_hooks()
    
    def update_fonts(self):
        """Scale fonts based on window height."""
        scale = self.window_height / DEFAULT_HEIGHT
        font_size = max(20, int(28 * scale))
        small_font_size = max(14, int(20 * scale))
        
        self.font = pygame.font.Font(None, font_size)
        self.small_font = pygame.font.Font(None, small_font_size)
    
    def get_scaled_value(self, base_value: int, dimension: str = 'height') -> int:
        """Scale a value based on current window size."""
        if dimension == 'height':
            return int(base_value * (self.window_height / DEFAULT_HEIGHT))
        return int(base_value * (self.window_width / DEFAULT_WIDTH))
    
    def handle_resize(self, new_width: int, new_height: int):
        """Handle window resize events and update data structures."""
        self.window_width = max(MIN_WIDTH, new_width)
        self.window_height = max(MIN_HEIGHT, new_height)
        
        new_max_points = self.window_width
        if new_max_points != self.max_points:
            self.max_points = new_max_points
            self.time_points = deque(self.time_points, maxlen=self.max_points)
            self.a_points = deque(self.a_points, maxlen=self.max_points)
            self.d_points = deque(self.d_points, maxlen=self.max_points)
            self.click_points = deque(self.click_points, maxlen=self.max_points)
            self.velocity_points = deque(self.velocity_points, maxlen=self.max_points)
            self.bullet_fired_points = deque(self.bullet_fired_points, maxlen=self.max_points)
        
        self.update_fonts()
    
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
        
        # Append data points
        self.time_points.append(self.current_time)
        self.a_points.append(1 if self.a_key_held else 0)
        self.d_points.append(1 if self.d_key_held else 0)
        self.click_points.append((
            1 if self.shooting_tracker.mouse_held else 0,
            self.shooting_tracker.has_inaccurate_bullet
        ))
        self.velocity_points.append(velocity)
        self.bullet_fired_points.append(bullet_fired_inaccurate)
    
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
    
    def _finalize_segment(self, segment: List) -> Segment:
        """Convert list to Segment and calculate duration from timestamps."""
        if not isinstance(segment, Segment):
            segment = Segment(segment)
        
        if len(segment) >= 2:
            t_start = segment[0][2]
            t_end = segment[-1][2]
            segment.duration_ms = int((t_end - t_start) * 1000)
        else:
            segment.duration_ms = 0
        
        return segment
    
    def draw_chart(self):
        """Render the main visualization."""
        self.screen.fill(DARK_BG)
        
        chart_height = self.get_scaled_value(500)
        chart_y_offset = self.get_scaled_value(80)
        grid_size = self.get_scaled_value(40)
        
        self._draw_grid(grid_size)
        self._draw_center_line(chart_y_offset, chart_height)
        
        if len(self.time_points) > 1:
            self._draw_timeline_data(chart_y_offset, chart_height)
        
        self._draw_header()
        self._draw_status_bar()
    
    def _draw_grid(self, grid_size: int):
        """Draw subtle background grid."""
        for i in range(0, self.window_height, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, i), (self.window_width, i), 1)
        for i in range(0, self.window_width, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (i, 0), (i, self.window_height), 1)
    
    def _draw_center_line(self, chart_y_offset: int, chart_height: int):
        """Draw horizontal reference lines."""
        center_y = chart_y_offset + chart_height // 2
        baseline_offset = self.get_scaled_value(120)
        
        pygame.draw.line(self.screen, CENTER_LINE, (0, center_y), (self.window_width, center_y), 2)
        pygame.draw.line(self.screen, CENTER_LINE, (0, center_y - baseline_offset), 
                        (self.window_width, center_y - baseline_offset), 1)
        pygame.draw.line(self.screen, CENTER_LINE, (0, center_y + baseline_offset), 
                        (self.window_width, center_y + baseline_offset), 1)
    
    def _draw_timeline_data(self, chart_y_offset: int, chart_height: int):
        """Draw all timeline data including keys, velocity, and clicks."""
        time_range = 5.0
        center_y = chart_y_offset + chart_height // 2
        baseline_offset = self.get_scaled_value(120)
        wave_height = self.get_scaled_value(80)
        
        a_baseline = center_y - baseline_offset
        d_baseline = center_y + baseline_offset
        
        # Collect visualization data
        a_line_points, d_line_points = [], []
        a_segments, d_segments = self._collect_key_segments(
            time_range, a_baseline, d_baseline, wave_height,
            a_line_points, d_line_points
        )
        
        velocity_line_points = self._collect_velocity_points(time_range, center_y)
        click_segments_normal, click_segments_timed = self._collect_click_segments(
            time_range, center_y
        )
        
        # Render layers
        self._draw_baseline_waves(a_line_points, d_line_points)
        self._draw_active_segments(a_segments, d_segments, a_baseline, d_baseline)
        self._draw_velocity_line(velocity_line_points)
        self._draw_click_segments(click_segments_normal, click_segments_timed)
        self._draw_bullet_markers(time_range, center_y)
    
    def _collect_key_segments(self, time_range: float, a_baseline: int, d_baseline: int,
                              wave_height: int, a_line_points: List, d_line_points: List
                              ) -> Tuple[List[Segment], List[Segment]]:
        """Collect key press segments with timestamps."""
        a_segments, d_segments = [], []
        current_a_segment, current_d_segment = [], []
        prev_a_active, prev_d_active = False, False
        
        for i in range(len(self.time_points)):
            time_offset = self.current_time - self.time_points[i]
            if time_offset > time_range:
                continue
            
            x = self.window_width - int(time_offset * self.window_width / time_range)
            if x < 0:
                continue
            
            # A key processing
            a_active = self.a_points[i] > 0
            if a_active:
                y = a_baseline - wave_height
                if not prev_a_active:
                    current_a_segment.append((x, a_baseline, self.time_points[i]))
                current_a_segment.append((x, y, self.time_points[i]))
            else:
                y = a_baseline
                if prev_a_active:
                    current_a_segment.append((x, a_baseline, self.time_points[i]))
                    a_segments.append(self._finalize_segment(current_a_segment[:]))
                    current_a_segment = []
            a_line_points.append((x, y))
            prev_a_active = a_active
            
            # D key processing
            d_active = self.d_points[i] > 0
            if d_active:
                y = d_baseline + wave_height
                if not prev_d_active:
                    current_d_segment.append((x, d_baseline, self.time_points[i]))
                current_d_segment.append((x, y, self.time_points[i]))
            else:
                y = d_baseline
                if prev_d_active:
                    current_d_segment.append((x, d_baseline, self.time_points[i]))
                    d_segments.append(self._finalize_segment(current_d_segment[:]))
                    current_d_segment = []
            d_line_points.append((x, y))
            prev_d_active = d_active
        
        if current_a_segment:
            a_segments.append(self._finalize_segment(current_a_segment[:]))
        if current_d_segment:
            d_segments.append(self._finalize_segment(current_d_segment[:]))
        
        return a_segments, d_segments
    
    def _collect_velocity_points(self, time_range: float, center_y: int) -> List[Tuple[int, int]]:
        """Collect velocity visualization points."""
        velocity_line_points = []
        velocity_scale = self.get_scaled_value(100)
        
        for i in range(len(self.time_points)):
            time_offset = self.current_time - self.time_points[i]
            if time_offset > time_range:
                continue
            
            x = self.window_width - int(time_offset * self.window_width / time_range)
            if x < 0:
                continue
            
            vel = self.velocity_points[i]
            y = center_y - int(vel * velocity_scale)
            velocity_line_points.append((x, y))
        
        return velocity_line_points
    
    def _collect_click_segments(self, time_range: float, center_y: int
                                ) -> Tuple[List[List], List[List]]:
        """Collect click segments, separating normal and inaccurate clicks."""
        click_segments_normal, click_segments_timed = [], []
        current_segment = []
        current_is_timed = False
        
        for i in range(len(self.time_points)):
            time_offset = self.current_time - self.time_points[i]
            if time_offset > time_range:
                continue
            
            x = self.window_width - int(time_offset * self.window_width / time_range)
            if x < 0:
                continue
            
            click_value, click_has_bad_bullet = self.click_points[i]
            click_active = click_value > 0
            
            if click_active:
                if not current_segment or click_has_bad_bullet != current_is_timed:
                    if current_segment:
                        target = click_segments_timed if current_is_timed else click_segments_normal
                        target.append(current_segment[:])
                    current_segment = [(x, center_y)]
                    current_is_timed = click_has_bad_bullet
                else:
                    current_segment.append((x, center_y))
            else:
                if current_segment:
                    target = click_segments_timed if current_is_timed else click_segments_normal
                    target.append(current_segment[:])
                    current_segment = []
        
        if current_segment:
            target = click_segments_timed if current_is_timed else click_segments_normal
            target.append(current_segment)
        
        return click_segments_normal, click_segments_timed
    
    def _draw_baseline_waves(self, a_line_points: List, d_line_points: List):
        """Draw dimmed baseline waves for key states."""
        if len(a_line_points) > 1:
            pygame.draw.lines(self.screen, (40, 80, 120), False, a_line_points, 1)
        if len(d_line_points) > 1:
            pygame.draw.lines(self.screen, (120, 40, 40), False, d_line_points, 1)
    
    def _draw_active_segments(self, a_segments: List[Segment], d_segments: List[Segment],
                             a_baseline: int, d_baseline: int):
        """Draw active key press segments with duration labels."""
        label_offset = self.get_scaled_value(105)
        a_label_y = a_baseline - label_offset
        d_label_y = d_baseline + label_offset
        
        for segment in a_segments:
            if len(segment) > 1:
                pygame.draw.lines(self.screen, BLUE, False, segment.xy_points(), 3)
                self._draw_segment_duration(segment, a_label_y, GRAY)
        
        for segment in d_segments:
            if len(segment) > 1:
                pygame.draw.lines(self.screen, RED, False, segment.xy_points(), 3)
                self._draw_segment_duration(segment, d_label_y, GRAY)
    
    def _draw_segment_duration(self, segment: Segment, label_y: int, color: Tuple[int, int, int]):
        """Draw duration label for a segment."""
        if len(segment) < 2 or not hasattr(segment, "duration_ms"):
            return
        
        x_start = segment[0][0]
        label = self.small_font.render(f"{segment.duration_ms} ms", True, color)
        text_x = x_start
        text_y = label_y - label.get_height() // 2
        
        if 0 <= text_x <= self.window_width - label.get_width():
            self.screen.blit(label, (text_x, text_y))
    
    def _draw_velocity_line(self, velocity_line_points: List[Tuple[int, int]]):
        """Draw velocity visualization line."""
        if len(velocity_line_points) > 1:
            pygame.draw.lines(self.screen, YELLOW, False, velocity_line_points, 2)
    
    def _draw_click_segments(self, click_segments_normal: List, click_segments_timed: List):
        """Draw click segments with appropriate styling."""
        dot_radius = max(4, self.get_scaled_value(6))
        
        # Normal clicks
        for segment in click_segments_normal:
            if len(segment) > 1:
                pygame.draw.lines(self.screen, WHITE, False, segment, 2)
            if segment:
                pygame.draw.circle(self.screen, WHITE, segment[0], dot_radius, 0)
                if len(segment) > 1:
                    pygame.draw.circle(self.screen, WHITE, segment[-1], dot_radius, 0)
        
        # Inaccurate clicks with green overlay
        for segment in click_segments_timed:
            if len(segment) > 1:
                pygame.draw.lines(self.screen, GREEN, False, segment, 4)
                pygame.draw.lines(self.screen, WHITE, False, segment, 2)
            if segment:
                pygame.draw.circle(self.screen, WHITE, segment[0], dot_radius, 0)
                if len(segment) > 1:
                    pygame.draw.circle(self.screen, WHITE, segment[-1], dot_radius, 0)
    
    def _draw_bullet_markers(self, time_range: float, center_y: int):
        """Draw markers for inaccurate bullet fires with velocity values."""
        last_text_x = None
        text_x_threshold = self.get_scaled_value(100)
        tick_height = self.get_scaled_value(15)
        
        for i in range(len(self.time_points)):
            time_offset = self.current_time - self.time_points[i]
            if time_offset > time_range:
                continue
            
            x = self.window_width - int(time_offset * self.window_width / time_range)
            if x < 0:
                continue
            
            if self.bullet_fired_points[i]:
                pygame.draw.line(self.screen, RED, 
                               (x, center_y - tick_height), 
                               (x, center_y + tick_height), 3)
                
                # vel = abs(self.velocity_points[i])
                # vel_text = f"{vel:.3f}"
                # vel_label = self.small_font.render(vel_text, True, GRAYf)
                
                # text_x = x - vel_label.get_width() // 2
                # text_y = center_y + self.get_scaled_value(25)
                
                # if last_text_x is None or abs(text_x - last_text_x) >= text_x_threshold:
                #     self.screen.blit(vel_label, (text_x, text_y))
                #     last_text_x = text_x
    
    def _draw_header(self):
        """Draw application header with legend."""
        header_padding = self.get_scaled_value(20)
        header_y = self.get_scaled_value(15)
        
        title = self.font.render("INPUT MONITOR", True, WHITE)
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
            label = self.font.render(text, True, color)
            offset = legend_spacing * (i if i < 2 else i + 0.9 if i == 3 else i)
            self.screen.blit(label, (legend_x + offset, header_y))
    
    def _draw_status_bar(self):
        """Draw status bar with pause state and controls."""
        header_padding = self.get_scaled_value(20)
        bottom_y = self.window_height - self.get_scaled_value(35)
        
        status = "PAUSED" if self.paused else "RECORDING"
        status_color = RED if self.paused else GREEN
        status_text = self.font.render(status, True, status_color)
        self.screen.blit(status_text, (header_padding, bottom_y))
        
        help_text = self.font.render("TAB: Pause/Resume", True, CENTER_LINE)
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