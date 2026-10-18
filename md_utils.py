"""
Общие функции для сценариев задачи N тел.

Содержит:
- verlet_step: один шаг интегрирования Верле
- compute_forces_pp: прямой расчёт сил O(N²)
"""

import numpy as np

# Параметры
G = 1.0
SOFTENING = 0.01
DT = 0.005


def verlet_step(pos, vel, acc, dt):
    """Один шаг Верле (velocity Verlet)."""
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2
    new_vel = vel + 0.5 * acc * dt
    return new_pos, new_vel


def compute_forces_pp(pos, masses, softening=SOFTENING):
    """Прямой расчёт сил O(N²)."""
    n = len(pos)
    forces = np.zeros_like(pos)
    for i in range(n):
        dr = pos - pos[i]
        r2 = np.sum(dr ** 2, axis=1) + softening ** 2
        r = np.sqrt(r2)
        inv_r3 = 1.0 / (r2 * r + 1e-12)
        inv_r3[i] = 0.0
        forces[i] = G * np.sum((masses * inv_r3)[:, None] * dr, axis=0)
    return forces