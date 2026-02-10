"""
Pygame Audio Player - Audio feedback system for game input monitoring.

Provides a pygame-based audio player that generates various sound effects
for different game events like footsteps, shooting, ability usage, etc.
Designed for Valorant input monitoring but adaptable to other games.
"""

import time
import threading
from typing import Literal
from enum import Enum

import pygame
import numpy as np


class SoundType(Enum):
    """Available sound effect types for game events."""
    FOOTSTEP = "footstep"
    SHOOTING = "shooting"
    MOVING_SHOOTING = "moving_shooting"  # Alert sound (more urgent)
    RUNNING_GUNNING = "running_gunning"  # Subtle alert for run-and-gun
    ABILITY = "ability"
    RELOAD = "reload"
    JUMP = "jump"
    CROUCH = "crouch"
    PLANT = "plant"
    DEFUSE = "defuse"
    ALERT = "alert"  # Generic alert
    SUCCESS = "success"
    ERROR = "error"


class PygameAudioPlayer:
    """
    Pygame-based audio player for game event feedback.
    
    Features:
    - Multiple sound types for different game events
    - Smooth waveform generation using pygame.sndarray
    - Minimum duration enforcement to prevent audio clicks
    - Thread-safe operation
    - No external audio file dependencies
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio player.
        
        Args:
            sample_rate: Audio sample rate in Hz (22050 is efficient for pygame)
        """
        # Initialize pygame mixer
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
                
        self.sample_rate = sample_rate
        
        # Timing and state
        self.start_time = 0.0
        self.min_duration = 0.150  # Minimum play duration in seconds
        self.last_sound_time = time.time()
        
        # Playback state
        self.is_playing = False
        self.pending_stop = False
        self.current_sound = None
        self.continuous_sound = None
        self.is_continuous = False
        
        # Thread safety
        self._lock = threading.Lock()
    
    @property
    def last_sound_delta(self) -> float:
        """Time elapsed since last sound started."""
        return time.time() - self.last_sound_time
    
    def _generate_footstep(self, duration: float = 0.15) -> pygame.mixer.Sound:
        """
        Generate footstep sound - short, low-frequency thud.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Low frequency pulse with noise
        frequency = 120
        wave = np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, 0.3, samples)
        
        # Envelope - quick attack, medium decay
        envelope = np.exp(-t * 12)
        
        # Combine and apply envelope
        sound_data = (wave * 0.7 + noise * 0.3) * envelope
        sound_data = (sound_data * 32767 * 0.3).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_shooting(self, duration: float = 0.08) -> pygame.mixer.Sound:
        """
        Generate shooting sound - sharp, high-frequency crack.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # White noise with band-pass characteristics
        noise = np.random.normal(0, 1, samples)
        
        # Add sharp attack
        frequency = 800
        click = np.sin(2 * np.pi * frequency * t)
        
        # Very fast decay envelope
        envelope = np.exp(-t * 40)
        
        # Combine
        sound_data = (noise * 0.6 + click * 0.4) * envelope
        sound_data = (sound_data * 32767 * 0.4).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_moving_shooting_alert(self, duration: float = 0.3) -> pygame.mixer.Sound:
        """
        Generate alert sound for shooting while moving - distinctive warning tone.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Alternating frequency (beep-beep pattern)
        freq1 = 880  # A5
        freq2 = 1100  # C#6
        
        # Create alternating pattern
        switch_point = samples // 2
        wave = np.concatenate([
            np.sin(2 * np.pi * freq1 * t[:switch_point]),
            np.sin(2 * np.pi * freq2 * t[switch_point:])
        ])
        
        # Sharp envelope for each beep
        env1 = np.exp(-t[:switch_point] * 15)
        env2 = np.exp(-t[switch_point:] * 15)
        envelope = np.concatenate([env1, env2])
        
        sound_data = wave * envelope
        sound_data = (sound_data * 32767 * 0.35).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_running_gunning(self, duration: float = 0.4) -> pygame.mixer.Sound:
        """
        Generate subtle run-and-gun alert - gentle warble/wobble tone.
        
        Designed to be noticeable but not annoying for frequent gameplay feedback.
        The wobble effect mimics the inaccuracy of shooting while moving.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Mid-range base frequency - not shrill, but audible
        base_freq = 750  # Hz
        
        # Create subtle detuned pair for richness
        freq1 = base_freq
        freq2 = base_freq * 1.01  # Slight detuning creates beating/wobble
        
        # Generate two slightly detuned sine waves
        wave1 = np.sin(2 * np.pi * freq1 * t)
        wave2 = np.sin(2 * np.pi * freq2 * t)
        
        # Combine for natural beating effect
        combined = (wave1 + wave2) / 2
        
        # Add gentle frequency modulation (warble)
        mod_rate = 2.5  # Hz - slow wobble
        mod_depth = 30  # Hz
        phase_mod = mod_depth * np.sin(2 * np.pi * mod_rate * t)
        warble = np.sin(2 * np.pi * (base_freq + phase_mod) * t)
        
        # Mix beating and warble
        sound_wave = combined * 0.6 + warble * 0.4
        
        # Gentle envelope - soft attack and release
        attack_time = 0.05
        release_time = 0.15
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        envelope = np.ones(samples)
        # Smooth attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2
        # Smooth release
        envelope[-release_samples:] = np.linspace(1, 0, release_samples) ** 2
        
        sound_data = sound_wave * envelope
        sound_data = (sound_data * 32767 * 0.25).astype(np.int16)  # Subtle volume
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_ability(self, duration: float = 0.25) -> pygame.mixer.Sound:
        """
        Generate ability usage sound - mystical swoosh.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Frequency sweep (high to low)
        start_freq = 1200
        end_freq = 400
        freq_sweep = start_freq - (start_freq - end_freq) * (t / duration)
        
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.sample_rate
        wave = np.sin(phase)
        
        # Smooth envelope
        envelope = np.exp(-t * 8) * (1 - np.exp(-t * 50))
        
        sound_data = wave * envelope
        sound_data = (sound_data * 32767 * 0.3).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_reload(self, duration: float = 0.2) -> pygame.mixer.Sound:
        """
        Generate reload sound - mechanical click sequence.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Two clicks
        click1_pos = int(samples * 0.2)
        click2_pos = int(samples * 0.6)
        
        # Create impulses
        sound_data = np.zeros(samples)
        
        # First click
        click_samples = 500
        click_t = np.linspace(0, 0.02, click_samples, False)
        click = np.sin(2 * np.pi * 300 * click_t) * np.exp(-click_t * 100)
        sound_data[click1_pos:click1_pos + click_samples] = click
        
        # Second click (slightly different pitch)
        click = np.sin(2 * np.pi * 400 * click_t) * np.exp(-click_t * 100)
        sound_data[click2_pos:click2_pos + click_samples] += click
        
        sound_data = (sound_data * 32767 * 0.35).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_jump(self, duration: float = 0.12) -> pygame.mixer.Sound:
        """
        Generate jump sound - quick upward swoosh.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Upward frequency sweep
        start_freq = 200
        end_freq = 600
        freq_sweep = start_freq + (end_freq - start_freq) * (t / duration)
        
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.sample_rate
        wave = np.sin(phase)
        
        # Quick envelope
        envelope = np.exp(-t * 15) * (1 - np.exp(-t * 80))
        
        sound_data = wave * envelope
        sound_data = (sound_data * 32767 * 0.25).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_alert(self, duration: float = 0.25) -> pygame.mixer.Sound:
        """
        Generate generic alert sound - attention-grabbing beep.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Pure sine wave at attention-grabbing frequency
        frequency = 1000
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Add tremolo for urgency
        tremolo = 1 + 0.3 * np.sin(2 * np.pi * 8 * t)
        
        # Envelope
        envelope = np.exp(-t * 10)
        
        sound_data = wave * tremolo * envelope
        sound_data = (sound_data * 32767 * 0.35).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_success(self, duration: float = 0.3) -> pygame.mixer.Sound:
        """
        Generate success sound - pleasant ascending chime.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Major chord progression
        freq1 = 523  # C5
        freq2 = 659  # E5
        freq3 = 784  # G5
        
        wave = (np.sin(2 * np.pi * freq1 * t) +
                0.6 * np.sin(2 * np.pi * freq2 * t) +
                0.4 * np.sin(2 * np.pi * freq3 * t))
        
        # Smooth envelope
        envelope = np.exp(-t * 6)
        
        sound_data = wave * envelope
        sound_data = (sound_data * 32767 * 0.25).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_error(self, duration: float = 0.2) -> pygame.mixer.Sound:
        """
        Generate error sound - harsh descending tone.
        
        Args:
            duration: Sound duration in seconds
            
        Returns:
            pygame.mixer.Sound object
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Dissonant frequencies
        freq1 = 400
        freq2 = 415  # Slightly off for dissonance
        
        wave = (np.sin(2 * np.pi * freq1 * t) +
                np.sin(2 * np.pi * freq2 * t))
        
        # Quick decay
        envelope = np.exp(-t * 12)
        
        sound_data = wave * envelope
        sound_data = (sound_data * 32767 * 0.3).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def _generate_continuous_wave(self, sound_type: SoundType, duration: float = 1.0) -> pygame.mixer.Sound:
        """
        Generate a continuous, loopable waveform for a given sound type.
        
        Creates seamless loops for continuous playback. Some sounds (like footsteps)
        will be repeated patterns, while others (like alerts) will be steady tones.
        
        Args:
            sound_type: Type of sound to generate continuously
            duration: Duration of one loop cycle in seconds
            
        Returns:
            pygame.mixer.Sound object that loops seamlessly
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        if sound_type == SoundType.FOOTSTEP:
            # Repeating footstep pattern (2 steps per cycle)
            step_duration = duration / 2
            step_samples = samples // 2
            
            # Generate two footsteps
            sound_data = np.zeros(samples)
            for i in range(2):
                start_idx = i * step_samples
                step_t = np.linspace(0, step_duration, step_samples, False)
                
                frequency = 120
                wave = np.sin(2 * np.pi * frequency * step_t)
                noise = np.random.normal(0, 0.3, step_samples)
                envelope = np.exp(-step_t * 12)
                
                sound_data[start_idx:start_idx + step_samples] = (wave * 0.7 + noise * 0.3) * envelope
            
        elif sound_type == SoundType.SHOOTING:
            # Rapid fire pattern
            shots_per_cycle = 8
            shot_spacing = samples // shots_per_cycle
            
            sound_data = np.zeros(samples)
            for i in range(shots_per_cycle):
                start_idx = i * shot_spacing
                shot_samples = min(int(0.05 * self.sample_rate), shot_spacing)
                shot_t = np.linspace(0, 0.05, shot_samples, False)
                
                noise = np.random.normal(0, 1, shot_samples)
                click = np.sin(2 * np.pi * 800 * shot_t)
                envelope = np.exp(-shot_t * 40)
                
                shot = (noise * 0.6 + click * 0.4) * envelope
                sound_data[start_idx:start_idx + shot_samples] = shot
        
        elif sound_type == SoundType.MOVING_SHOOTING:
            # Continuous alert tone with pulsing
            frequency = 1000
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Pulsing amplitude
            pulse_freq = 4  # Hz
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * pulse_freq * t)
            
            sound_data = wave * pulse
        
        elif sound_type == SoundType.RUNNING_GUNNING:
            # Continuous subtle warble/wobble - gentle rolling feeling
            base_freq = 750  # Mid-range frequency
            
            # Create detuned pair for natural beating
            freq1 = base_freq
            freq2 = base_freq * 1.01
            wave1 = np.sin(2 * np.pi * freq1 * t)
            wave2 = np.sin(2 * np.pi * freq2 * t)
            beating = (wave1 + wave2) / 2
            
            # Add slow warble (mimics movement inaccuracy)
            warble_rate = 2.5  # Hz - gentle rolling pulse
            warble_depth = 40  # Hz
            freq_mod = base_freq + warble_depth * np.sin(2 * np.pi * warble_rate * t)
            phase = 2 * np.pi * np.cumsum(freq_mod) / self.sample_rate
            warble = np.sin(phase)
            
            # Mix beating and warble for rich, subtle texture
            sound_data = beating * 0.5 + warble * 0.5
            
            # Apply gentle amplitude modulation (rolling effect)
            amp_mod_rate = 2.0  # Hz
            amp_mod = 0.7 + 0.3 * np.sin(2 * np.pi * amp_mod_rate * t)
            sound_data = sound_data * amp_mod
        
        elif sound_type == SoundType.ABILITY:
            # Continuous mystical hum with slow frequency modulation
            base_freq = 400
            mod_freq = 0.5  # Slow modulation
            freq_modulation = 50 * np.sin(2 * np.pi * mod_freq * t)
            
            phase = 2 * np.pi * (base_freq + freq_modulation) * t
            wave = np.sin(np.cumsum(phase) / self.sample_rate)
            
            # Add harmonic
            harmonic = 0.3 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            
            sound_data = wave + harmonic
        
        elif sound_type == SoundType.ALERT:
            # Steady alert tone with tremolo
            frequency = 1000
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Tremolo for urgency
            tremolo_freq = 6
            tremolo = 1 + 0.4 * np.sin(2 * np.pi * tremolo_freq * t)
            
            sound_data = wave * tremolo
        
        else:
            # Default to steady sine wave
            frequency = 440
            sound_data = np.sin(2 * np.pi * frequency * t)
        
        # Normalize and convert
        sound_data = sound_data / np.max(np.abs(sound_data))  # Normalize
        sound_data = (sound_data * 32767 * 0.3).astype(np.int16)
        
        # Convert to stereo
        stereo_data = np.column_stack((sound_data, sound_data))
        
        return pygame.sndarray.make_sound(stereo_data)
    
    def play(self, sound_type: SoundType, volume: float = 1.0):
        """
        Play a sound effect.
        
        Args:
            sound_type: Type of sound to play (from SoundType enum)
            volume: Volume level (0.0 to 1.0)
        """
        with self._lock:
            # Generate appropriate sound
            if sound_type == SoundType.FOOTSTEP:
                sound = self._generate_footstep()
            elif sound_type == SoundType.SHOOTING:
                sound = self._generate_shooting()
            elif sound_type == SoundType.MOVING_SHOOTING:
                sound = self._generate_moving_shooting_alert()
            elif sound_type == SoundType.RUNNING_GUNNING:
                sound = self._generate_running_gunning()
            elif sound_type == SoundType.ABILITY:
                sound = self._generate_ability()
            elif sound_type == SoundType.RELOAD:
                sound = self._generate_reload()
            elif sound_type == SoundType.JUMP:
                sound = self._generate_jump()
            elif sound_type == SoundType.ALERT:
                sound = self._generate_alert()
            elif sound_type == SoundType.SUCCESS:
                sound = self._generate_success()
            elif sound_type == SoundType.ERROR:
                sound = self._generate_error()
            else:
                # Default to alert for unimplemented types
                sound = self._generate_alert()
            
            # Set volume and play
            sound.set_volume(min(max(volume, 0.0), 1.0))
            sound.play()
            
            self.is_playing = True
            self.start_time = time.time()
            self.last_sound_time = time.time()
            self.current_sound = sound
    
    def start(self, sound_type: SoundType, volume: float = 1.0, loop_duration: float = 1.0):
        """
        Start continuous playback of a sound.
        
        The sound will loop continuously until stop() is called. Different sound types
        will create different continuous effects:
        - FOOTSTEP: Repeating footstep pattern
        - SHOOTING: Rapid fire bursts
        - MOVING_SHOOTING: Continuous pulsing alert (urgent)
        - RUNNING_GUNNING: Subtle rolling warble (gentle feedback)
        - ABILITY: Mystical humming
        - ALERT: Steady warning tone with tremolo
        - Others: Continuous wave based on sound characteristics
        
        Args:
            sound_type: Type of sound to play continuously
            volume: Volume level (0.0 to 1.0)
            loop_duration: Duration of one loop cycle in seconds (default: 1.0)
        """
        # Stop any existing continuous playback
        if self.is_continuous:
            self.stop()
        
        with self._lock:
            # Generate the loopable sound
            sound = self._generate_continuous_wave(sound_type, loop_duration)
            
            # Set volume and start looping
            sound.set_volume(min(max(volume, 0.0), 1.0))
            sound.play(loops=-1)  # -1 means loop indefinitely
            
            self.is_continuous = True
            self.is_playing = True
            self.continuous_sound = sound
            self.start_time = time.time()
            self.last_sound_time = time.time()
    
    def stop(self):
        """
        Stop all currently playing sounds, including continuous loops.
        
        Respects minimum duration to prevent audio artifacts. If minimum duration
        hasn't elapsed, marks for pending stop which will execute when update() 
        is called after min_duration.
        """
        if self.is_continuous:
            elapsed = time.time() - self.start_time
            
            if elapsed < self.min_duration:
                self.pending_stop = True
            else:
                self._execute_stop()
        else:
            # For one-shot sounds, stop immediately
            self._execute_stop()
    
    def _execute_stop(self):
        """Immediately stop playback and reset state."""
        pygame.mixer.stop()
        self.is_playing = False
        self.is_continuous = False
        self.pending_stop = False
        self.current_sound = None
        self.continuous_sound = None
    
    def update(self):
        """
        Process pending operations. Call this regularly from main thread.
        
        Handles:
        - Updating playing state based on pygame mixer
        - Deferred stops that were requested before minimum duration elapsed
        """
        # Handle pending stop
        if self.pending_stop:
            elapsed = time.time() - self.start_time
            if elapsed >= self.min_duration:
                self._execute_stop()
        
        # Update playing state
        if self.is_playing and not self.is_continuous and not pygame.mixer.get_busy():
            self.is_playing = False
    
    def set_min_duration(self, duration: float):
        """
        Set the minimum playback duration for continuous sounds.
        
        This prevents audio artifacts from rapid start/stop cycles.
        
        Args:
            duration: Minimum duration in seconds (default: 0.150)
        """
        self.min_duration = max(0.0, duration)
    
    def cleanup(self):
        """Clean up pygame mixer resources."""
        pygame.mixer.quit()


def demo():
    """Demonstrate all available sound types."""
    player = PygameAudioPlayer()
    
    print("Pygame Audio Player Demo")
    print("=" * 50)
    
    print("\n--- ONE-SHOT SOUNDS ---")
    demo_sounds = [
        (SoundType.FOOTSTEP, "Footstep"),
        (SoundType.SHOOTING, "Shooting"),
        (SoundType.RUNNING_GUNNING, "Run-and-Gun Alert (Subtle)"),
        (SoundType.ABILITY, "Ability Usage"),
        (SoundType.RELOAD, "Reload"),
        (SoundType.JUMP, "Jump"),
        (SoundType.SUCCESS, "Success"),
        (SoundType.ERROR, "Error"),
    ]
    
    for sound_type, description in demo_sounds:
        print(f"Playing: {description}")
        player.play(sound_type, volume=0.5)
        time.sleep(0.5)  # Wait between sounds
    
    print("\n--- CONTINUOUS SOUNDS ---")
    continuous_sounds = [
        (SoundType.FOOTSTEP, "Continuous Footsteps", 0.8),
        (SoundType.SHOOTING, "Continuous Shooting", 1.0),
        (SoundType.RUNNING_GUNNING, "Run-and-Gun (Subtle Warble)", 2.5),
        (SoundType.MOVING_SHOOTING, "Moving + Shooting (Urgent Alert)", 1.5),
        (SoundType.ABILITY, "Continuous Ability Hum", 2.0),
        (SoundType.ALERT, "Continuous Alert", 1.5),
    ]
    
    for sound_type, description, duration in continuous_sounds:
        print(f"\nPlaying (continuous): {description}")
        player.start(sound_type, volume=0.4)
        time.sleep(duration)
        player.stop()
        time.sleep(0.3)  # Pause between demos
    
    print("\n--- MINIMUM DURATION TEST ---")
    print("Testing rapid start/stop with minimum duration protection...")
    print("Starting run-and-gun alert, stopping immediately (min duration enforced)")
    player.start(SoundType.RUNNING_GUNNING, volume=0.4)
    time.sleep(0.05)  # Try to stop after only 50ms
    player.stop()
    print("Stop requested after 50ms, but will play for minimum 150ms")
    
    # Keep calling update until the pending stop completes
    while player.is_playing:
        player.update()
        time.sleep(0.01)
    
    print("Sound stopped cleanly after minimum duration")
    time.sleep(0.5)
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    
    # Cleanup
    time.sleep(0.5)
    player.cleanup()


if __name__ == "__main__":
    demo()