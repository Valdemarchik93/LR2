"""
Сценарий 2: Солнечная система.
Солнце + 4 планеты (Меркурий, Венера, Земля, Марс).
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def scenario_solar(n_steps=4000, save_every=20, fps=30):
    print("[Сценарий 2] Солнечная система")
    os.makedirs("results", exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    # Реальные параметры (млн км, годы)
    planets = [
        ("Меркурий", 57.9,  56.7,  0.241, "gray"),
        ("Венера",   108.2, 108.1, 0.615, "orange"),
        ("Земля",    149.6, 149.5, 1.000, "royalblue"),
        ("Марс",     227.9, 226.9, 1.881, "red"),
    ]

    t_max = 2.0 * 365.25  # 2 земных года
    t = np.linspace(0, t_max, n_steps)

    frames = []
    for i in range(0, n_steps, save_every):
        frames.append(t[i])

    print("  Создание GIF...")
    images = []
    for frame_idx, t_now in enumerate(frames):
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, a, b, period, color in planets:
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(a * np.cos(theta), b * np.sin(theta), "--",
                    color=color, alpha=0.4, linewidth=1)

        ax.scatter(0, 0, s=600, c="gold", edgecolors="orange",
                   linewidths=2, zorder=5, label="Солнце")

        for name, a, b, period, color in planets:
            angle = 2 * np.pi * (t_now / (period * 365.25))
            px = a * np.cos(angle)
            py = b * np.sin(angle)
            ax.scatter(px, py, s=100, c=color, edgecolors="black",
                       zorder=4, label=name)

        ax.set_xlim(-260, 260)
        ax.set_ylim(-260, 260)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.set_title(f"Солнечная система, t = {t_now:.0f} дней", fontsize=13)
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

    gif_path = "results/scenario_solar.gif"
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    print(f"  Сохранён: {gif_path}")

    # Финальный кадр
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, a, b, period, color in planets:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(a * np.cos(theta), b * np.sin(theta), "--",
                color=color, alpha=0.5, linewidth=1.2)
    ax.scatter(0, 0, s=700, c="gold", edgecolors="orange",
               linewidths=2, zorder=5, label="Солнце")
    for name, a, b, period, color in planets:
        angle = 2 * np.pi * (t[-1] / (period * 365.25))
        px = a * np.cos(angle)
        py = b * np.sin(angle)
        ax.scatter(px, py, s=120, c=color, edgecolors="black",
                   zorder=4, label=name)
    ax.set_xlim(-260, 260)
    ax.set_ylim(-260, 260)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Солнечная система: 4 планеты", fontsize=14)
    ax.set_xlabel("X, млн км")
    ax.set_ylabel("Y, млн км")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("results/scenario_solar_final.png", dpi=150,
                bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print("  Сохранён: results/scenario_solar_final.png")


if __name__ == "__main__":
    scenario_solar()