"""
Сценарий 3: Много тел (N = 500).
Гравитационный коллапс с деревом Барнса–Хата.
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import trange

from md_utils import G, SOFTENING, DT


# ==========================================================
# Дерево Барнса-Хата
# ==========================================================

class QuadNode:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.mass = 0.0
        self.com = np.zeros(2)
        self.children = None
        self.index = None

    def size(self):
        return max(self.x_max - self.x_min, self.y_max - self.y_min)

    def insert(self, pos, mass, idx):
        if self.mass == 0.0 and self.children is None:
            self.mass = mass
            self.com = pos.copy()
            self.index = idx
            return
        if self.children is None:
            self._subdivide()
            old_pos = self.com.copy()
            old_mass = self.mass
            old_idx = self.index
            self.mass = 0.0
            self.com = np.zeros(2)
            self._insert_into_child(old_pos, old_mass, old_idx)
            self.index = None
        self._insert_into_child(pos, mass, idx)
        self.mass += mass
        self.com = (self.com * (self.mass - mass) + pos * mass) / self.mass

    def _subdivide(self):
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        self.children = [
            QuadNode(self.x_min, xm, self.y_min, ym),
            QuadNode(xm, self.x_max, self.y_min, ym),
            QuadNode(self.x_min, xm, ym, self.y_max),
            QuadNode(xm, self.x_max, ym, self.y_max),
        ]

    def _insert_into_child(self, pos, mass, idx):
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        child_idx = 0
        if pos[0] > xm:
            child_idx += 1
        if pos[1] > ym:
            child_idx += 2
        self.children[child_idx].insert(pos, mass, idx)


def build_tree(pos, masses):
    x_min, y_min = pos.min(axis=0)
    x_max, y_max = pos.max(axis=0)
    pad = 0.01 * max(x_max - x_min, y_max - y_min)
    root = QuadNode(x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    for i, (p, m) in enumerate(zip(pos, masses)):
        root.insert(p, m, i)
    return root


def compute_force_bh(root, pos, theta=0.5, softening=SOFTENING):
    if root.mass == 0.0 or (root.children is None and root.index is None):
        return np.zeros(2)
    if root.children is None and root.index is not None:
        return np.zeros(2)
    dr = root.com - pos
    r2 = dr @ dr + softening ** 2
    r = np.sqrt(r2)
    if root.size() / r < theta:
        return G * root.mass * dr / (r2 * r + 1e-12)
    force = np.zeros(2)
    if root.children is not None:
        for child in root.children:
            force += compute_force_bh(child, pos, theta, softening)
    return force


# ==========================================================
# Сценарий 3
# ==========================================================

def scenario_many(n=500, n_steps=2000, save_every=20, seed=42):
    print(f"[Сценарий 3] Много тел (N={n})")
    os.makedirs("results", exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    rng = np.random.default_rng(seed)
    radius = 10.0
    r = radius * np.sqrt(rng.random(n))
    theta = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    vel = np.zeros_like(pos)
    for i in range(n):
        r_i = np.linalg.norm(pos[i])
        if r_i > 1e-6:
            v_circ = np.sqrt(G * 1.0 / (r_i + SOFTENING))
            tangent = np.array([-pos[i, 1], pos[i, 0]]) / r_i
            vel[i] = v_circ * tangent * 0.8

    masses = np.ones(n) / n

    frames = []
    energies = []
    tqdm_bar = trange(n_steps, desc="N-body")

    for step in tqdm_bar:
        tree = build_tree(pos, masses)
        forces = np.zeros_like(pos)
        for i in range(n):
            forces[i] = compute_force_bh(tree, pos[i])
        acc = forces / masses[:, None]

        new_pos = pos + vel * DT + 0.5 * acc * DT ** 2
        new_vel = vel + 0.5 * acc * DT
        pos = np.mod(new_pos, 2 * radius) - radius
        vel = new_vel

        if step % 20 == 0:
            ke = 0.5 * np.sum(masses[:, None] * vel ** 2)
            energies.append(ke)

        if step % save_every == 0:
            frames.append(pos.copy())

    print("  Создание GIF...")
    images = []
    for i, p in enumerate(frames):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(p[:, 0], p[:, 1], s=4, c="white", edgecolors="none")
        ax.set_facecolor("black")
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_title(f"Много тел (N={n}), шаг {i * save_every}",
                     color="white", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight",
                    facecolor="black")
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    gif_path = "results/scenario_many.gif"
    imageio.mimsave(gif_path, images, fps=15, loop=0)
    print(f"  Сохранён: {gif_path}")

    # Финальный кадр
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pos[:, 0], pos[:, 1], s=6, c="white", edgecolors="none")
    ax.set_facecolor("black")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_title(f"Много тел (N={n}): гравитационный коллапс",
                 color="white", fontsize=13)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig("results/scenario_many_final.png", dpi=150,
                bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print("  Сохранён: results/scenario_many_final.png")

    # График энергии
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(energies)) * 20, energies, "b-", linewidth=2)
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Кинетическая энергия")
    ax.set_title("Кинетическая энергия системы от времени")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/scenario_many_energy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Сохранён: results/scenario_many_energy.png")


if __name__ == "__main__":
    scenario_many()