from dataclasses import asdict, dataclass
from typing import List


@dataclass
class PatternSegment:
    """Represents a single segment in a pattern."""
    key: str  # 'a', 'd', 'click', 'walk', 'crouch', 'pause'
    duration_ms: int
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class Pattern:
    """Represents a complete pattern for practice."""
    name: str
    difficulty: str  # "EASY", "MEDIUM", "HARD"
    segments: List[PatternSegment]
    tolerance_ms: int = 50
    
    def to_dict(self):
        return {
            'name': self.name,
            'difficulty': self.difficulty,
            'segments': [s.to_dict() for s in self.segments],
            'tolerance_ms': self.tolerance_ms
        }
    
    @classmethod
    def from_dict(cls, data):
        segments = [PatternSegment.from_dict(s) for s in data['segments']]
        return cls(
            name=data['name'],
            difficulty=data['difficulty'],
            segments=segments,
            tolerance_ms=data.get('tolerance_ms', 50)
        )
    
    def get_total_duration_ms(self) -> int:
        return sum(s.duration_ms for s in self.segments)