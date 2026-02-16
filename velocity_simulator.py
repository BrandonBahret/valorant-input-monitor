from enum import Enum

import numpy as np

from app_config import *


class InaccuracyType(Enum):
    """Types of shooting inaccuracy."""
    NONE = 0
    ACCELERATING = 1
    CONSTANT = 2
    DECELERATING = 3

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

