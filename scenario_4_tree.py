"""
Сценарий 4: Визуализация дерева Барнса-Хата.
Показывает, как строится квадродерево поверх частиц.
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Rectangle


class QuadNode:
    def __init__(self, x_min, x_max, y_min, y_max, depth=0):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.depth = depth
        self.mass = 0.0
        self.com = np.zeros(2)
        self.children = None
        self.index = None
        self.pos = None

    def size(self):
        return max(self.x_max - self.x_min, self.y_max - self.y_min)

    def insert(self, pos, mass, idx):
        if self.mass == 0.0 and self.children is None:
            self.mass = mass
            self.com = pos.copy()
            self.index = idx
            self.pos = pos.copy()
            return
        if self.children is None:
            self._subdivide()
            old_pos = self.pos.copy()
            old_mass = self.mass
            old_idx = self.index
            self.mass = 0.0
            self.com = np.zeros(2)
            self.pos = None
            self._insert_into_child(old_pos, old_mass, old_idx)
            self.index = None
        self._insert_into_child(pos, mass, idx)
        self.mass += mass
        self.com = (self.com * (self.mass - mass) + pos * mass) / self.mass

    def _subdivide(self):
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        d = self.depth + 1
        self.children = [
            QuadNode(self.x_min, xm, self.y_min, ym, d),
            QuadNode(xm, self.x_max, self.y_min, ym, d),
            QuadNode(self.x_min, xm, ym, self.y_max, d),
            QuadNode(xm, self.x_max, ym, self.y_max, d),
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

    def collect_rectangles(self, rectangles):
        """Собирает все прямоугольники дерева для рисования."""
        if self.mass > 0 or self.children is not None:
            rectangles.append((self.x_min, self.x_max, self.y_min,
                               self.y_max, self.depth))
        if self.children is not None:
            for c in self.children:
                c.collect_rectangles(rectangles)


def build_tree(pos, masses):
    x_min, y_min = pos.min(axis=0)
    x_max, y_max = pos.max(axis=0)
    pad = 0.01 * max(x_max - x_min, y_max - y_min)
    root = QuadNode(x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    for i, (p, m) in enumerate(zip(pos, masses)):
        root.insert(p, m, i)
    return root


def scenario_tree(n=200, n_frames=30, seed=42):
    print(f"[Сценарий 4] Дерево Барнса-Хата (N={n})")
    os.makedirs("results", exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    rng = np.random.default_rng(seed)
    pos = rng.uniform(-8, 8, (n, 2))
    masses = np.ones(n) / n

    images = []
    for frame in range(n_frames):
        # Перестраиваем дерево на каждом шаге (частицы чуть сдвигаются)
        pos_shift = pos + 0.1 * rng.standard_normal((n, 2))
        root = build_tree(pos_shift, masses)

        # Собираем прямоугольники
        rects = []
        root.collect_rectangles(rects)

        fig, ax = plt.subplots(figsize=(9, 9))

        # Рисуем прямоугольники
        for (x0, x1, y0, y1, depth) in rects:
            if depth == 0:
                continue
            alpha = max(0.1, 1.0 - depth * 0.15)
            color = plt.cm.viridis(min(depth / 6, 1.0))
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                             fill=False, edgecolor=color,
                             linewidth=1.2, alpha=alpha)
            ax.add_patch(rect)

        # Рисуем частицы
        ax.scatter(pos_shift[:, 0], pos_shift[:, 1],
                   s=15, c="red", edgecolors="black",
                   linewidths=0.5, zorder=5)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_title(f"Дерево Барнса-Хата, кадр {frame + 1}/{n_frames}",
                     fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.2)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    gif_path = "results/scenario_tree.gif"
    imageio.mimsave(gif_path, images, fps=5, loop=0)
    print(f"  Сохранён: {gif_path}")

    # Финальный кадр
    root = build_tree(pos, masses)
    rects = []
    root.collect_rectangles(rects)

    fig, ax = plt.subplots(figsize=(10, 10))
    for (x0, x1, y0, y1, depth) in rects:
        if depth == 0:
            continue
        alpha = max(0.15, 1.0 - depth * 0.15)
        color = plt.cm.viridis(min(depth / 6, 1.0))
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, edgecolor=color,
                         linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)

    ax.scatter(pos[:, 0], pos[:, 1], s=30, c="red",
               edgecolors="black", linewidths=0.7, zorder=5,
               label="Частицы")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_title(f"Квадродерево Барнса-Хата (N={n})", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/scenario_tree_final.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Сохранён: results/scenario_tree_final.png")


if __name__ == "__main__":
    scenario_tree()