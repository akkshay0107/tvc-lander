from math import pi

import numpy as np
from scipy.linalg import solve_continuous_are

g = 9.81
cvx = (g * pi) / 40.0
comega = (0.6 * g) / pi

A = np.array(
    [
        [0, 0, 0, 1, 0, 0],  # dx = vx
        [0, 0, 0, 0, 1, 0],  # dy = vy
        [0, 0, 0, 0, 0, 1],  # dtheta = omega
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

Q = np.diag([10, 10, 100, 1, 1, 10])
R = np.diag([0.1, 0.1])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

hover_thrust = (g - 7.5) / 7.5


def get_action(obs):
    action = -K @ obs
    action[0] += hover_thrust
    return np.clip(action, -1.0, 1.0)
