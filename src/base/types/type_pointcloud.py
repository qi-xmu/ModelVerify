# import numpy as np
from dataclasses import dataclass, field

from numpy.typing import NDArray


@dataclass
class Point:
    id: int = 0
    position: NDArray = field(default_factory=lambda: __import__("numpy").zeros(3))
    confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 0


@dataclass
class PointCloudFrame:
    t_us: int = 0
    points: list[Point] = field(default_factory=list)

    @property
    def size(self):
        return len(self.points)

    @property
    def confidence(self):
        return sum(p.confidence for p in self.points)
