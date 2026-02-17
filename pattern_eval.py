import json
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass

from pattern_eval_structs import Pattern
from resource_helpers import resource_path


class PatternManager:
    """Manages pattern storage and retrieval."""
    
    def __init__(self):
        self.patterns_dir = resource_path("patterns")
        self.patterns_dir.mkdir(exist_ok=True)
    
    def save_pattern(self, pattern: Pattern) -> bool:
        try:
            filename = f"{pattern.name.lower().replace(' ', '_')}.json"
            filepath = self.patterns_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pattern.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"[PatternManager] Failed to save pattern: {e}")
            return False
    
    def load_patterns(self) -> List[Pattern]:
        patterns = []
        try:
            for filepath in self.patterns_dir.glob("*.json"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    patterns.append(Pattern.from_dict(data))
        except Exception as e:
            print(f"[PatternManager] Failed to load patterns: {e}")
        return patterns
    
    def delete_pattern(self, pattern_name: str) -> bool:
        try:
            filename = f"{pattern_name.lower().replace(' ', '_')}.json"
            filepath = self.patterns_dir / filename
            if filepath.exists():
                filepath.unlink()
                return True
        except Exception as e:
            print(f"[PatternManager] Failed to delete pattern: {e}")
        return False


@dataclass
class SegmentTiming:
    """Tracks detailed timing information for a pattern segment."""
    segment_index: int
    expected_key: str
    expected_duration_ms: int
    actual_duration_ms: int
    key_matched: bool
    timing_error_ms: int
    
    def is_success(self, tolerance_ms: int) -> bool:
        """Check if this segment was executed successfully."""
        return self.key_matched and abs(self.timing_error_ms) <= tolerance_ms

class PatternEvaluator:
    """Manages pattern practice evaluation and progress tracking."""
    
    def __init__(self, pattern: Pattern, input_tolerance_ms: int = 150):
        self.pattern = pattern
        self.input_tolerance_ms = input_tolerance_ms
        
        # Progress tracking
        self.start_time = 0.0
        self.current_segment = 0
        self.segment_start_time = 0.0
        self.segment_key_pressed_time = 0.0
        self.segment_key_released_time = 0.0
        self.last_key_state = False
        self.completed_once = False
        
        # State
        self.waiting_for_start = True
        self.waiting_for_release = False  # Cooldown: all keys must be released before restarting
        self.completed = False
        self.disqualified = False
        self.showing_result = False  # New: Indicates we are reviewing the results
        
        # Statistics
        self.attempts = 0
        self.successes = 0
        self.segment_timings: List[SegmentTiming] = []

    def _all_keys_released(self, a_held: bool, d_held: bool, click_held: bool,
                           shift_held: bool, ctrl_held: bool) -> bool:
        """Returns True when every tracked input is no longer held."""
        return not any([a_held, d_held, click_held, shift_held, ctrl_held])

    def restart(self):
        """Reset for a new attempt."""
        self.completed = False
        self.disqualified = False
        self.showing_result = False
        self.start_time = 0.0
        self.current_segment = 0
        self.segment_timings = []
        self.segment_start_time = 0.0
        self.segment_key_pressed_time = 0.0
        self.segment_key_released_time = 0.0
        self.last_key_state = False

        self.waiting_for_release = True
        self.waiting_for_start = False
    
    def get_segment_time_bounds(self, index: int) -> Tuple[float, float]:
        """
        Get the absolute time bounds for a specific segment from pattern start.
        Returns time in milliseconds relative to pattern start.
        """
        if index < 0 or index >= len(self.pattern.segments):
            return (0.0, 0.0)
        
        # Calculate absolute start time from pattern beginning
        start_ms = 0.0
        for i in range(index):
            start_ms += self.pattern.segments[i].duration_ms
        
        end_ms = start_ms + self.pattern.segments[index].duration_ms
        return start_ms, end_ms
    
    def is_input_valid_with_tolerance(self, input_key: str, current_time_seconds: float) -> bool:
        """
        Check if an input is valid considering tolerance window.
        """
        if self.waiting_for_start or self.start_time == 0:
            return False
        
        # Calculate elapsed time since pattern start in milliseconds
        elapsed_ms = (current_time_seconds - self.start_time) * 1000
        
        segments = self.pattern.segments
        
        # Check all segments to see if input is valid within tolerance
        # We check a range around the current segment to handle transitions
        start_idx = max(0, self.current_segment - 1)
        end_idx = min(len(segments), self.current_segment + 2)
        
        for test_idx in range(start_idx, end_idx):
            seg = segments[test_idx]
            
            # Skip if this segment doesn't use this key
            if seg.key != input_key:
                continue
            
            # Get the time bounds for this segment
            seg_start_ms, seg_end_ms = self.get_segment_time_bounds(test_idx)
            
            # Apply tolerance window - allow early entry and late exit
            tolerant_start = seg_start_ms - self.input_tolerance_ms
            tolerant_end = seg_end_ms + self.input_tolerance_ms
            
            # Check if current time falls within the tolerance window
            if tolerant_start <= elapsed_ms <= tolerant_end:
                return True
        
        return False
    
    def check_inappropriate_inputs(self, current_time_seconds: float, 
                                  a_held: bool, d_held: bool, click_held: bool,
                                  shift_held: bool, ctrl_held: bool) -> bool:
        """
        Check if any inappropriate inputs are being pressed.
        """
        if self.waiting_for_start or self.disqualified or self.showing_result:
            return False
        
        # Map of input keys to their held states
        inputs_to_check = [
            ('a', a_held),
            ('d', d_held),
            ('click', click_held),
            ('walk', shift_held),
            ('crouch', ctrl_held)
        ]
        
        # Check each held input
        for input_key, is_held in inputs_to_check:
            if is_held:
                # If this input is held but not valid for current time, it's inappropriate
                if not self.is_input_valid_with_tolerance(input_key, current_time_seconds):
                    print(f"[DEBUG] Inappropriate input detected: {input_key} at {(current_time_seconds - self.start_time) * 1000:.1f}ms")
                    return True
        
        return False
    
    def is_start_key_pressed(self,
                      a_held: bool, d_held: bool, click_held: bool,
                      shift_held: bool, ctrl_held: bool) -> bool:
        
        first_segment = self.pattern.segments[0]
        started = False
        key_is_held = False
        key_just_pressed = False
        
        if first_segment.key == 'a':
            key_is_held = a_held
            key_just_pressed = key_is_held and not self.last_key_state
            started = key_just_pressed
        elif first_segment.key == 'd':
            key_is_held = d_held
            key_just_pressed = key_is_held and not self.last_key_state
            started = key_just_pressed
        elif first_segment.key == 'click':
            key_is_held = click_held
            key_just_pressed = key_is_held and not self.last_key_state
            started = key_just_pressed
        elif first_segment.key == 'walk':
            key_is_held = shift_held
            key_just_pressed = key_is_held and not self.last_key_state
            started = key_just_pressed
        elif first_segment.key == 'crouch':
            key_is_held = ctrl_held
            key_just_pressed = key_is_held and not self.last_key_state
            started = key_just_pressed
        elif first_segment.key == 'pause':
            started = True
            key_is_held = False
        
        return started
    
    def check_progress(self, current_time: float, current_real_time: float,
                      a_held: bool, d_held: bool, click_held: bool,
                      shift_held: bool, ctrl_held: bool) -> Optional[str]:
        """
        Check if user is following pattern correctly.
        Returns error message if disqualified, None otherwise.
        """
        # --- Cooldown gate ---------------------------------------------------
        if self.waiting_for_release:
            if self._all_keys_released(a_held, d_held, click_held, shift_held, ctrl_held):
                print("[DEBUG] All keys released after disqualification — ready for next attempt.")
                self.waiting_for_release = False
                self.waiting_for_start = True
                self.last_key_state = False 
            else:
                first_key = self.pattern.segments[0].key
                if first_key == 'a': self.last_key_state = a_held
                elif first_key == 'd': self.last_key_state = d_held
                elif first_key == 'click': self.last_key_state = click_held
                elif first_key == 'walk': self.last_key_state = shift_held
                elif first_key == 'crouch': self.last_key_state = ctrl_held
                else: self.last_key_state = False
            return None
        # ---------------------------------------------------------------------
        
        # Wait for the user to start with the correct first input
        if self.waiting_for_start:
            first_segment = self.pattern.segments[0]
            started = False
            key_is_held = False
            key_just_pressed = False
            
            if first_segment.key == 'a':
                key_is_held = a_held
                key_just_pressed = key_is_held and not self.last_key_state
                started = key_just_pressed
            elif first_segment.key == 'd':
                key_is_held = d_held
                key_just_pressed = key_is_held and not self.last_key_state
                started = key_just_pressed
            elif first_segment.key == 'click':
                key_is_held = click_held
                key_just_pressed = key_is_held and not self.last_key_state
                started = key_just_pressed
            elif first_segment.key == 'walk':
                key_is_held = shift_held
                key_just_pressed = key_is_held and not self.last_key_state
                started = key_just_pressed
            elif first_segment.key == 'crouch':
                key_is_held = ctrl_held
                key_just_pressed = key_is_held and not self.last_key_state
                started = key_just_pressed
            elif first_segment.key == 'pause':
                started = True
                key_is_held = False
            
            if self.disqualified or self.showing_result and not started:
                return None
            
            # Update last key state even while waiting
            if not started:
                if first_segment.key == 'a': self.last_key_state = a_held
                elif first_segment.key == 'd': self.last_key_state = d_held
                elif first_segment.key == 'click': self.last_key_state = click_held
                elif first_segment.key == 'walk': self.last_key_state = shift_held
                elif first_segment.key == 'crouch': self.last_key_state = ctrl_held
            
            if started:
                self.waiting_for_start = False
                self.completed = False
                self.start_time = current_time
                self.segment_start_time = current_time
                if key_is_held:
                    self.segment_key_pressed_time = current_real_time
                else:
                    self.segment_key_pressed_time = 0.0
                self.segment_key_released_time = 0.0
                self.last_key_state = key_is_held
                
            return None
        
        # Check if pattern is completed
        if self.current_segment >= len(self.pattern.segments):
            if not self.completed:
                print(f"[DEBUG] Pattern complete! Recorded {len(self.segment_timings)} segments, expected {len(self.pattern.segments)}")
                self.completed = True
            return None
        
        segment = self.pattern.segments[self.current_segment]
        segment_elapsed_ms = (current_time - self.segment_start_time) * 1000
        
        # Check for inappropriate inputs - pass current_time in seconds
        if self.check_inappropriate_inputs(current_time, a_held, d_held, 
                                          click_held, shift_held, ctrl_held):
            self.disqualified = True
            self.attempts += 1
            return "Wrong input!"
        
        # Special handling for pause segments
        if segment.key == 'pause':
            if segment_elapsed_ms >= segment.duration_ms:
                timing = SegmentTiming(
                    segment_index=self.current_segment,
                    expected_key='pause',
                    expected_duration_ms=segment.duration_ms,
                    actual_duration_ms=int(segment_elapsed_ms),
                    key_matched=True,
                    timing_error_ms=0
                )
                self.segment_timings.append(timing)
                
                self.current_segment += 1
                self.segment_start_time = current_time
                self.segment_key_pressed_time = 0.0
                self.segment_key_released_time = 0.0
                self.last_key_state = False
            return None
        
        # Check if the correct key is currently being held
        key_is_held = False
        if segment.key == 'a': key_is_held = a_held
        elif segment.key == 'd': key_is_held = d_held
        elif segment.key == 'click': key_is_held = click_held
        elif segment.key == 'walk': key_is_held = shift_held
        elif segment.key == 'crouch': key_is_held = ctrl_held
        
        # Track key press timestamp
        key_just_pressed = not self.last_key_state and key_is_held
        key_just_released = self.last_key_state and not key_is_held
        
        if key_just_pressed:
            self.segment_key_pressed_time = current_real_time
        
        if key_just_released:
            self.segment_key_released_time = current_real_time
        
        self.last_key_state = key_is_held
        
        # End segment when key is released
        min_time_ms = 20
        tolerance = self.pattern.tolerance_ms
        max_time_ms = segment.duration_ms + tolerance + 50
        
        should_end_segment = False
        
        if key_just_released and self.segment_key_pressed_time > 0:
            should_end_segment = True
        elif segment_elapsed_ms > max_time_ms:
            should_end_segment = True
        
        if should_end_segment:
            if self.segment_key_pressed_time > 0 and self.segment_key_released_time > 0:
                actual_duration_ms = int((self.segment_key_released_time - self.segment_key_pressed_time) * 1000)
            elif self.segment_key_pressed_time > 0:
                actual_duration_ms = int((current_real_time - self.segment_key_pressed_time) * 1000)
            else:
                actual_duration_ms = 0
            
            timing_error = actual_duration_ms - segment.duration_ms
            
            print(f"[DEBUG] Segment {self.current_segment} ended: key={segment.key}, expected={segment.duration_ms}ms, actual={actual_duration_ms}ms, error={timing_error}ms")
            
            timing = SegmentTiming(
                segment_index=self.current_segment,
                expected_key=segment.key,
                expected_duration_ms=segment.duration_ms,
                actual_duration_ms=actual_duration_ms,
                key_matched=actual_duration_ms >= min_time_ms,
                timing_error_ms=timing_error
            )
            
            self.segment_timings.append(timing)
            
            self.current_segment += 1
            self.segment_start_time = current_time
            self.segment_key_pressed_time = 0.0
            self.segment_key_released_time = 0.0
            self.last_key_state = False
        
        return None
    
    def evaluate_attempt(self) -> Dict:
        """
        Evaluate the completed pattern attempt.
        Returns dict with evaluation results.
        """
        if not self.segment_timings:
            return {
                'success': False,
                'message': "No segments recorded",
                'success_rate': 0,
                'avg_error': 0,
                'successful_count': 0,
                'total_count': 0
            }
        
        tolerance = self.pattern.tolerance_ms
        
        print(f"\n=== Pattern Evaluation ===")
        for i, timing in enumerate(self.segment_timings):
            print(f"Segment {i}: expected={timing.expected_duration_ms}ms, actual={timing.actual_duration_ms}ms, error={timing.timing_error_ms}ms, matched={timing.key_matched}")
        
        successful_segments = [t for t in self.segment_timings if t.is_success(tolerance)]
        total_segments = len(self.segment_timings)
        success_count = len(successful_segments)
        success_rate = (success_count / total_segments) * 100 if total_segments > 0 else 0
        
        valid_timings = [t for t in self.segment_timings if t.key_matched]
        timing_errors = [abs(t.timing_error_ms) for t in valid_timings]
        avg_error = sum(timing_errors) / len(timing_errors) if timing_errors else 0
        
        print(f"Success: {success_count}/{total_segments} = {success_rate:.0f}%, Avg Error: {avg_error:.0f}ms")
        
        self.attempts += 1
        success = success_rate >= 80 and avg_error <= tolerance
        
        if success:
            self.successes += 1
        
        message = f"{success_rate:.0f}% ({success_count}/{total_segments}) | Avg: {avg_error:.0f}ms"
        if success:
            message = f"Success! {message}"
        
        self.completed_once = True
        return {
            'success': success,
            'message': message,
            'success_rate': success_rate,
            'avg_error': avg_error,
            'successful_count': success_count,
            'total_count': total_segments
        }