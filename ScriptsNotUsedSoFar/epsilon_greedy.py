from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class EpsSchedule:
    eps0: float = 0.2
    eps_min: float = 0.05
    power: float = 0.5
    def at(self, t: int) -> float:
        return max(self.eps_min, self.eps0 / ((t + 1) ** self.power))

class OnlineRidgePerArm:
    def __init__(self, d: int, lam: float = 1e-2):
        self.d = d
        self.S = [lam * np.eye(d), lam * np.eye(d)]
        self.b = [np.zeros(d), np.zeros(d)]
    def update(self, x: np.ndarray, a: int, y: float) -> None:
        self.S[a] += np.outer(x, x); self.b[a] += x * y
    def theta(self, a: int) -> np.ndarray:
        try: return np.linalg.solve(self.S[a], self.b[a])
        except np.linalg.LinAlgError: return np.linalg.lstsq(self.S[a], self.b[a], rcond=None)[0]
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        th0, th1 = self.theta(0), self.theta(1)
        return float(x @ th0), float(x @ th1)

class EpsGreedyBinary:
    """ε-greedy logger for A∈{0,1}; no clipping."""
    def __init__(self, d: int, schedule: EpsSchedule, lam: float = 1e-2,
                 rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(0)
        self.sched = schedule
        self.model = OnlineRidgePerArm(d, lam=lam)
    def step(self, t: int, x: np.ndarray) -> Tuple[int, float]:
        q0, q1 = self.model.predict(x)
        eps = self.sched.at(t)
        if q1 > q0:   pi1 = 1.0 - 0.5 * eps
        elif q1 < q0: pi1 = 0.5 * eps
        else:         pi1 = 0.5
        a = 1 if (self.rng.rand() < pi1) else 0
        return a, pi1
    def update(self, x: np.ndarray, a: int, y: float) -> None:
        self.model.update(x, a, y)
    def current_thetas(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.theta(0), self.model.theta(1)

