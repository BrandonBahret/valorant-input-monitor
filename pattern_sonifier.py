"""
pattern_sonifier.py
-------------------
Converts Pattern data into audible rhythm sequences.

Key → Timbre mapping
--------------------
  d       → high pitched sine tone  (880 Hz)
  a       → low pitched sine tone   (220 Hz)
  click   → sharp percussive burst  (shaped white noise)
  walk    → mid-range sine tone     (440 Hz)
  crouch  → deep thud               (110 Hz)
  pause   → silence

Usage
-----
    from pattern_sonifier import PatternSonifier

    # Create once at app startup — initialises pygame.mixer if available
    sonifier = PatternSonifier()

    # Play the target pattern rhythm (e.g. bound to F5 while practising)
    sonifier.play_pattern(pattern)

    # Stop any currently-playing audio immediately
    sonifier.stop()

    # Register a global F5 hotkey that works even when the game is fullscreened.
    # Provide a callable that returns the current pattern at call time.
    sonifier.register_hotkey(
        get_pattern=lambda: current_pattern,
    )

    # Tear down the hotkey and mixer when shutting down
    sonifier.shutdown()
"""

from __future__ import annotations

import io
import math
import random
import struct
import threading
import time
import wave
from typing import Callable, List, Optional

from dataclasses import dataclass

from pattern_eval_structs import Pattern


# ---------------------------------------------------------------------------
# Optional dependency guards — nothing here is a hard requirement
# ---------------------------------------------------------------------------
try:
    import pygame
    import pygame.mixer
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import keyboard as _keyboard_lib
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _KEYBOARD_AVAILABLE = False

# winsound: Windows-only, used only as a last-resort fallback
try:
    import winsound as _winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE:  int = 44_100   # Hz
_CHANNELS:     int = 1        # mono
_SAMPLE_WIDTH: int = 2        # bytes — 16-bit signed

# Frequencies per key type
_FREQ: dict[str, float] = {
    'd':      880.0,
    'a':      220.0,
    'walk':   440.0,
    'crouch': 110.0,
    'click':  0.0,    # handled separately (percussive noise)
    'pause':  0.0,    # silence
}

# Human-readable labels used in console output
_LABEL: dict[str, str] = {
    'd':      '▲ d      (880 Hz  — high)',
    'a':      '▼ a      (220 Hz  — low)',
    'walk':   '— walk   (440 Hz  — mid)',
    'crouch': '● crouch (110 Hz  — deep)',
    'click':  '✕ click  (percussive noise)',
    'pause':  '  pause  (silence)',
}

# ANSI colour codes for console feedback (degrade gracefully on dumb terminals)
_ANSI: dict[str, str] = {
    'd':      '\033[96m',   # bright cyan
    'a':      '\033[94m',   # bright blue
    'walk':   '\033[92m',   # bright green
    'crouch': '\033[95m',   # bright magenta
    'click':  '\033[93m',   # bright yellow
    'pause':  '\033[90m',   # dark grey
    'reset':  '\033[0m',
}

_RNG = random.Random(0xDEADBEEF)   # deterministic seed → same click sound every run


# ---------------------------------------------------------------------------
# Low-level PCM helpers
# ---------------------------------------------------------------------------

def _n_samples(duration_ms: int) -> int:
    """Number of 16-bit samples for a given duration in milliseconds."""
    return int(_SAMPLE_RATE * duration_ms / 1000)


def _pack(samples: List[int]) -> bytes:
    """Pack a list of int16 values into little-endian bytes."""
    return struct.pack(f'<{len(samples)}h', *samples)


def _sine_pcm(freq: float, duration_ms: int,
              amplitude: float = 0.65,
              attack_ms: int = 15,
              release_ms: int = 35) -> bytes:
    """
    Warm, woodwind-like tone with harmonics and smooth envelope.
    
    Creates a richer, more organic sound by:
    - Adding harmonics (2nd and 3rd partials) for warmth
    - Using exponential attack/release curves for natural breath quality
    - Slightly reduced fundamental for a softer timbre
    """
    n         = _n_samples(duration_ms)
    n_attack  = _n_samples(attack_ms)
    n_release = _n_samples(release_ms)
    samples   = []

    for i in range(n):
        # Exponential envelope curves (more natural than linear)
        if i < n_attack:
            # Soft attack - exponential ease-in
            t = i / n_attack
            env = 1.0 - math.exp(-5.0 * t)
        elif i >= n - n_release:
            # Gentle release - exponential decay
            t = (n - i) / n_release
            env = math.exp(-3.0 * (1.0 - t))
        else:
            env = 1.0

        # Fundamental frequency (reduced to 70% for softer tone)
        fundamental = 0.70 * math.sin(2.0 * math.pi * freq * i / _SAMPLE_RATE)
        
        # 2nd harmonic (octave) - adds brightness but stays soft
        harmonic_2 = 0.15 * math.sin(2.0 * math.pi * freq * 2.0 * i / _SAMPLE_RATE)
        
        # 3rd harmonic (perfect fifth above octave) - adds warmth
        harmonic_3 = 0.10 * math.sin(2.0 * math.pi * freq * 3.0 * i / _SAMPLE_RATE)
        
        # Combine harmonics with envelope
        signal = fundamental + harmonic_2 + harmonic_3
        s = amplitude * env * signal
        samples.append(max(-32768, min(32767, int(s * 32767))))

    return _pack(samples)


def _click_pcm(duration_ms: int, amplitude: float = 0.55) -> bytes:
    """
    Soft articulation transient: gentle noise burst with smooth decay.
    
    More like a soft tongue click or breath articulation rather than
    a harsh percussive sound. Creates subtle distinction from pitched tones
    while maintaining the organic, woodwind-like quality.
    """
    burst_ms = min(duration_ms, 45)  # Shorter, softer burst
    n_burst  = _n_samples(burst_ms)
    samples  = []

    for i in range(n_burst):
        # Gentler exponential decay (less aggressive than -8.0)
        env   = math.exp(-5.0 * i / max(n_burst, 1))
        
        # Filtered noise - cut high frequencies for warmth
        noise = _RNG.uniform(-1.0, 1.0)
        
        # Apply additional smoothing to reduce harshness
        if i < 5:  # Very short attack
            smooth = i / 5.0
            noise *= smooth
        
        samples.append(max(-32768, min(32767, int(amplitude * env * noise * 32767))))

    # Silence for the remainder of the segment duration
    samples.extend([0] * _n_samples(duration_ms - burst_ms))
    return _pack(samples)


def _silence_pcm(duration_ms: int) -> bytes:
    """Return completely silent PCM of exactly duration_ms length."""
    return _pack([0] * _n_samples(duration_ms))


def _pcm_for_key(key: str, duration_ms: int, amplitude: float = 0.58) -> bytes:
    """Dispatch to the correct PCM generator for a given key identifier."""
    if duration_ms <= 0:
        return b''
    if key == 'pause':
        return _silence_pcm(duration_ms)
    if key == 'click':
        return _click_pcm(duration_ms, amplitude)
    freq = _FREQ.get(key, 440.0)
    return _sine_pcm(freq, duration_ms, amplitude)


def _mix_pcm(buf_a: bytes, buf_b: bytes) -> bytes:
    """
    Mix two mono 16-bit PCM buffers by averaging paired samples.
    The shorter buffer is zero-padded to match the longer one.
    Output is clamped to the int16 range — no clipping distortion.
    """
    n_a = len(buf_a) // 2
    n_b = len(buf_b) // 2
    n   = max(n_a, n_b)

    sa = list(struct.unpack(f'<{n_a}h', buf_a)) + [0] * (n - n_a)
    sb = list(struct.unpack(f'<{n_b}h', buf_b)) + [0] * (n - n_b)

    mixed = [max(-32768, min(32767, (x + y) // 2)) for x, y in zip(sa, sb)]
    return _pack(mixed)


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM bytes in a valid WAV container (in-memory)."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pattern → PCM builders  (public, importable for testing)
# ---------------------------------------------------------------------------

def build_pattern_pcm(pattern: Pattern, amplitude: float = 0.58) -> bytes:
    """
    Build a mono PCM buffer from the pattern's segments.
    
    Each segment produces a warm, woodwind-like tone whose pitch identifies 
    the key type and whose duration matches the segment's duration_ms exactly.
    """
    buffer = bytearray()
    
    for segment in pattern.segments:
        seg_pcm = _pcm_for_key(segment.key, segment.duration_ms, amplitude)
        buffer.extend(seg_pcm)
    
    return bytes(buffer)


# ---------------------------------------------------------------------------
# Console visualisation helpers
# ---------------------------------------------------------------------------

def _print_pattern_legend(pattern: Pattern) -> None:
    """Compact visual of the pattern printed to stdout."""
    print(f"\n[Sonifier] ♪ Playing: {pattern.name!r}")
    for seg in pattern.segments:
        colour = _ANSI.get(seg.key, '')
        reset  = _ANSI['reset']
        blocks = max(1, seg.duration_ms // 50)
        bar    = '█' * blocks
        label  = _LABEL.get(seg.key, seg.key)
        print(f"  {colour}{bar:<22}  {seg.duration_ms:>5} ms   {label}{reset}")
    print()


# ---------------------------------------------------------------------------
# Playback engine (pygame primary / winsound fallback)
# ---------------------------------------------------------------------------

class _PlaybackEngine:
    """
    Thread-safe audio playback wrapper.

    play() immediately stops any prior sound and starts the new one in a
    daemon thread so the caller is never blocked.
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._pygame_ok   = False

        if _PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.pre_init(_SAMPLE_RATE, -16, _CHANNELS, 512)
                    pygame.mixer.init()
                self._pygame_ok = True
            except Exception as exc:
                print(f"[Sonifier] pygame.mixer init failed — {exc}")

    def play(self, wav_bytes: bytes) -> None:
        """Stop current audio and play wav_bytes in a new background thread."""
        self.stop()
        self._stop_event.clear()
        t = threading.Thread(target=self._run, args=(wav_bytes,), daemon=True)
        with self._lock:
            self._thread = t
        t.start()

    def stop(self) -> None:
        """Signal the playback thread to stop and wait up to 0.5 s for it."""
        self._stop_event.set()
        with self._lock:
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=0.5)

    # -- private -------------------------------------------------------------

    def _run(self, wav_bytes: bytes) -> None:
        if self._pygame_ok:
            self._play_pygame(wav_bytes)
        elif _WINSOUND_AVAILABLE:
            self._play_winsound(wav_bytes)
        else:
            print("[Sonifier] No audio backend available. "
                  "Install pygame:  pip install pygame")

    def _play_pygame(self, wav_bytes: bytes) -> None:
        try:
            buf   = io.BytesIO(wav_bytes)
            sound = pygame.mixer.Sound(buf)
            ch    = sound.play()
            if ch is None:
                return
            while ch.get_busy() and not self._stop_event.is_set():
                time.sleep(0.01)
            if self._stop_event.is_set():
                ch.stop()
        except Exception as exc:
            print(f"[Sonifier] pygame playback error — {exc}")

    def _play_winsound(self, wav_bytes: bytes) -> None:
        """Blocking fallback via winsound (Windows only, no stop support)."""
        try:
            _winsound.PlaySound(wav_bytes,
                                _winsound.SND_MEMORY | _winsound.SND_NODEFAULT)
        except Exception as exc:
            print(f"[Sonifier] winsound error — {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PatternSonifier:
    """
    Converts patterns into audio and manages the F5 hotkey.

    Typical lifecycle
    -----------------
    1.  ``sonifier = PatternSonifier()``           — create once at startup
    2.  ``sonifier.register_hotkey(get_p)``        — optional, for fullscreen use
    3.  ``sonifier.play_pattern(p)``               — hear the target rhythm
    4.  ``sonifier.shutdown()``                    — clean up on exit
    """

    def __init__(self) -> None:
        self._engine             = _PlaybackEngine()
        self._hotkey_registered  = False
        self._hotkey_key: str    = 'f5'

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_pattern(self, pattern: Pattern) -> None:
        """
        Render and play the target pattern rhythm.

        Each segment produces a tone whose **pitch identifies the key type**
        and whose **duration matches the segment's duration_ms** exactly, so
        internalising the audio is equivalent to internalising the timing.
        """
        _print_pattern_legend(pattern)
        pcm = build_pattern_pcm(pattern)
        self._engine.play(_pcm_to_wav(pcm))

    def stop(self) -> None:
        """Interrupt any currently-playing audio immediately."""
        self._engine.stop()

    # ------------------------------------------------------------------
    # Global hotkey
    # ------------------------------------------------------------------

    def register_hotkey(self,
                        get_pattern: Callable[[], Optional[Pattern]],
                        hotkey: str = 'f5') -> bool:
        """
        Register a global hotkey (default **F5**) for in-game audio preview.

        The ``keyboard`` library hooks at the OS level via a low-level thread,
        so this works even when the game window has exclusive focus (fullscreen).

        Parameters
        ----------
        get_pattern:
            Callable → current Pattern (or None if none loaded).
        hotkey:
            Key string accepted by the ``keyboard`` library (e.g. ``'f5'``).

        Returns
        -------
        bool
            True if the hotkey was successfully registered.

        Notes
        -----
        * On Linux, the ``keyboard`` library typically requires root or the
          ``input`` group membership to access /dev/input.
        * Calling this method more than once replaces the previous callback.
        """
        if not _KEYBOARD_AVAILABLE:
            print(
                "[Sonifier] 'keyboard' library not installed — global hotkey unavailable.\n"
                "           Install with:  pip install keyboard"
            )
            return False

        # Remove any existing hotkey first so we don't stack callbacks
        if self._hotkey_registered:
            self.unregister_hotkey(self._hotkey_key)

        def _callback() -> None:
            pattern = get_pattern()
            if pattern is None:
                print("[Sonifier] F5 pressed — no pattern currently loaded.")
                return            
            self.play_pattern(pattern)

        try:
            _keyboard_lib.add_hotkey(hotkey, _callback, suppress=False)
            self._hotkey_registered = True
            self._hotkey_key        = hotkey
            print(f"[Sonifier] Global hotkey '{hotkey.upper()}' registered — "
                  f"press to preview the current pattern.")
            return True
        except Exception as exc:
            print(f"[Sonifier] Failed to register hotkey '{hotkey}': {exc}")
            return False

    def unregister_hotkey(self, hotkey: Optional[str] = None) -> None:
        """Remove the previously registered global hotkey."""
        if not _KEYBOARD_AVAILABLE or not self._hotkey_registered:
            return
        key = hotkey or self._hotkey_key
        try:
            _keyboard_lib.remove_hotkey(key)
        except Exception:
            pass
        self._hotkey_registered = False

    def shutdown(self) -> None:
        """
        Stop audio playback and release all resources.
        Call once on application exit.
        """
        self.stop()
        self.unregister_hotkey()
        if _PYGAME_AVAILABLE and self._engine._pygame_ok:
            try:
                pygame.mixer.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Minimal self-test   (python pattern_sonifier.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from dataclasses import dataclass, field as _field

    @dataclass
    class _Seg:
        key: str
        duration_ms: int

    @dataclass
    class _Pat:
        name: str         = 'Self-test'
        difficulty: str   = 'EASY'
        segments: list    = _field(default_factory=list)
        tolerance_ms: int = 50

        def get_total_duration_ms(self):
            return sum(s.duration_ms for s in self.segments)

    pat = _Pat(segments=[
        _Seg('d',      200),
        _Seg('pause',  100),
        _Seg('a',      300),
        _Seg('click',  150),
        _Seg('d',      200),
    ])

    s = PatternSonifier()
    total_ms = pat.get_total_duration_ms()

    print("=== PLAYING PATTERN ===")
    s.play_pattern(pat)
    time.sleep(total_ms / 1000 + 0.4)

    s.shutdown()
    print("Self-test complete.")