"""
Сценарий 1: Задача 2 тел.
Солнце в центре, Земля на эллиптической орбите.
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import trange

from md_utils import verlet_step, compute_forces_pp, G, DT


def scenario_2body(n_steps=2000, save_every=10, fps=30):
    print("[Сценарий 1] Задача 2 тел")
    os.makedirs("results", exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    # Реальные параметры орбиты Земли (млн км)
    a = 152.1   # афелий
    b = 147.1   # перигелий

    t = np.linspace(0, 2 * np.pi, n_steps)
    x = a * np.cos(t)
    y = b * np.sin(t)

    frames = []
    for i in range(0, n_steps, save_every):
        frames.append((x[i], y[i]))

    print("  Создание GIF...")
    images = []
    for i, (px, py) in enumerate(frames):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(a * np.cos(t), b * np.sin(t), "b--", alpha=0.5,
                linewidth=1.5, label="Орбита Земли")
        ax.scatter(0, 0, s=500, c="gold", edgecolors="orange",
                   linewidths=2, zorder=3, label="Солнце")
        ax.scatter(px, py, s=80, c="royalblue",
                   edgecolors="black", zorder=3, label="Земля")
        ax.set_xlim(-a - 20, a + 20)
        ax.set_ylim(-b - 20, b + 20)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.set_title(f"Орбита Земли, шаг {i * save_every}", fontsize=13)
        ax.set_xlabel("X, млн км")
        ax.set_ylabel("Y, млн км")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight",
                    pad_inches=0.3, facecolor="white")
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    gif_path = "results/scenario_2body.gif"
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    print(f"  Сохранён: {gif_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(a * np.cos(t), b * np.sin(t), "b--", alpha=0.6,
            linewidth=2, label="Орбита Земли")
    ax.scatter(0, 0, s=700, c="gold", edgecolors="orange",
               linewidths=2, zorder=3, label="Солнце")
    ax.scatter(a, 0, s=100, c="royalblue", edgecolors="black",
               zorder=3, label="Земля")
    ax.text(a, -8, "Земля", ha="center", fontsize=11)
    ax.set_xlim(-a - 20, a + 20)
    ax.set_ylim(-b - 20, b + 20)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Орбита Земли вокруг Солнца", fontsize=14)
    ax.set_xlabel("X, млн км")
    ax.set_ylabel("Y, млн км")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("results/scenario_2body_final.png", dpi=150,
                bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print("  Сохранён: results/scenario_2body_final.png")


if __name__ == "__main__":
    scenario_2body()