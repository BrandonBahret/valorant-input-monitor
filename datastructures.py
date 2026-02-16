from typing import Optional

import numpy as np


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