from math import pi

import numpy as np
from scipy.linalg import solve_continuous_are

g = 9.81
cvx = (g * pi) / 40.0
comega = (0.6 * g) / pi

A = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, cvx, -60, 0, 0],
        [0, 0, 0, 0, -60, 0],
        [0, 0, 0, 0, 0, -2 * pi],
    ]
)

B = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1.0, 0],
        [0, comega],
    ]
)

hover_thrust = (g - 7.5) / 7.5


class LQR:
    def __init__(self, Q, R) -> None:
        self.Q = Q
        self.R = R
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ B.T @ P

    def get_action(self, obs) -> np.ndarray:
        action = -self.K @ obs
        action[0] += hover_thrust
        return np.clip(action, -1.0, 1.0)
