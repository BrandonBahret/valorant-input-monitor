"""
Continuous Wave Player - Audio feedback system with smooth waveform generation.

Provides a threaded audio player that generates pleasant bell-like tones with
harmonics and tremolo effects. Supports minimum duration enforcement to prevent
audio artifacts from rapid start/stop cycles.
"""

import time
import threading
from typing import Generator

import pyaudio
import numpy as np


class ContinuousWavePlayer:
    """
    Threaded audio player that generates continuous waveforms with harmonics.
    
    Features:
    - Smooth sine wave generation with harmonics
    - Tremolo effect for pleasant audio feedback
    - Minimum duration enforcement to prevent audio clicks
    - Thread-safe operation with pause/resume support
    """
    
    def __init__(self, frequency: float = 440, sample_rate: int = 44100, 
                 amplitude: float = 0.15):
        """
        Initialize the wave player.
        
        Args:
            frequency: Base frequency in Hz (default: 440 Hz / A4)
            sample_rate: Audio sample rate in Hz
            amplitude: Wave amplitude (0.0 to 1.0)
        """
        self.frequency = frequency
        self.original_freq = frequency
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        
        # Timing and state
        self.start_time = 0.0
        self.min_duration = 0.400  # Minimum play duration in seconds
        self.last_beep_time = time.time()
        
        # Playback state
        self.is_playing = False
        self.is_paused = False
        self.pending_stop = False
        
        # Thread safety
        self._lock = threading.Lock()
        self._thread = None
    
    @property
    def last_beep_delta(self) -> float:
        """Time elapsed since last beep started."""
        return time.time() - self.last_beep_time
    
    def start(self):
        """Start audio playback in a separate thread."""
        if self.is_playing:
            return
        
        self.start_time = time.time()
        self.last_beep_time = time.time()
        self.pending_stop = False
        self.is_playing = True
        
        self._thread = threading.Thread(target=self._play_audio, daemon=True)
        self._thread.start()
    
    def stop(self):
        """
        Stop audio playback, respecting minimum duration.
        
        If minimum duration hasn't elapsed, marks for pending stop
        which will execute when update() is called after min_duration.
        """
        elapsed = time.time() - self.start_time
        
        if elapsed < self.min_duration:
            self.pending_stop = True
        else:
            self._execute_stop()
    
    def pause(self):
        """Pause audio generation without stopping the thread."""
        self.is_paused = True
    
    def resume(self):
        """Resume audio generation."""
        self.is_paused = False
    
    def update(self):
        """
        Process pending operations. Call this regularly from main thread.
        
        Handles deferred stops that were requested before minimum duration elapsed.
        """
        if self.pending_stop:
            elapsed = time.time() - self.start_time
            if elapsed >= self.min_duration:
                self._execute_stop()
    
    def _execute_stop(self):
        """Immediately stop playback and reset state."""
        self.is_playing = False
        self.is_paused = False
        self.pending_stop = False
        self.frequency = self.original_freq
    
    def _generate_waveform(self) -> Generator[bytes, None, None]:
        """
        Generate audio samples as a continuous waveform.
        
        Creates a bell-like tone by combining:
        - Base sine wave
        - Two harmonic overtones
        - Subtle tremolo effect
        
        Yields:
            Audio sample chunks as bytes
        """
        t = 0.0
        chunk_size = 1024
        increment = 1.0 / self.sample_rate
        
        while self.is_playing:
            if self.is_paused:
                time.sleep(0.01)
                continue
            
            with self._lock:
                time_array = t + np.arange(chunk_size) * increment
                
                # Generate base tone and harmonics
                base_wave = np.sin(2 * np.pi * self.frequency * time_array)
                harmonic1 = 0.3 * np.sin(2 * np.pi * self.frequency * 2 * time_array)
                harmonic2 = 0.15 * np.sin(2 * np.pi * self.frequency * 3 * time_array)
                
                combined = base_wave + harmonic1 + harmonic2
                
                # Apply tremolo for gentle volume modulation
                tremolo_rate = 4.5  # Hz
                tremolo_depth = 0.15
                tremolo = 1 - tremolo_depth * (
                    np.sin(2 * np.pi * tremolo_rate * time_array) * 0.5 + 0.5
                )
                
                samples = (self.amplitude * combined * tremolo).astype(np.float32)
                t += chunk_size * increment
            
            yield samples.tobytes()
    
    def _play_audio(self):
        """Audio playback thread target. Manages PyAudio stream lifecycle."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=2,  # Changed to stereo
            rate=self.sample_rate,
            output=True
        )
        
        waveform_generator = self._generate_waveform()
        
        try:
            while self.is_playing:
                try:
                    mono_data = next(waveform_generator)
                    # Convert mono to stereo by duplicating samples
                    mono_samples = np.frombuffer(mono_data, dtype=np.float32)
                    stereo_samples = np.repeat(mono_samples, 2)
                    stream.write(stereo_samples.tobytes())
                except StopIteration:
                    break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    # Demonstration
    beeper = ContinuousWavePlayer(frequency=523)  # C5 note
    
    print("Playing for 2 seconds...")
    beeper.start()
    time.sleep(2)
    
    print("Stopping...")
    beeper.stop()
    
    # Allow pending stop to complete
    while beeper.is_playing:
        beeper.update()
        time.sleep(0.01)
    
    print("Done!")