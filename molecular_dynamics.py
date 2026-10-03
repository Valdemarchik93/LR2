"""
Моделирование двумерного газа частиц методом молекулярной динамики.

Реализовано:
- потенциал Леннарда-Джонса
- интегрирование методом Верле
- периодические граничные условия
- метод сеток для поиска соседей
- расчёт энергии, температуры, распределения скоростей
- визуализация (GIF, графики)
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import trange


# ==========================================================
# Параметры модели
# ==========================================================

N = 200                # число частиц
L = 20.0               # размер ячейки (квадрат L × L)
DT = 0.005             # шаг по времени
N_STEPS = 3000         # число шагов
SAVE_EVERY = 30        # сохранять кадр каждые N шагов
TARGET_T = 1.0         # целевая температура

# Параметры потенциала Леннарда-Джонса
EPSILON = 1.0          # глубина потенциальной ямы
SIGMA = 1.0            # характерное расстояние
R_CUT = 2.5 * SIGMA    # радиус обрезания потенциала

# Масса частицы
MASS = 1.0


# ==========================================================
# Потенциал Леннарда-Джонса
# ==========================================================

def lj_force(r):
    """Сила Леннарда-Джонса как функция расстояния r.

    F(r) = 24 * eps * (2*(sigma/r)^13 - (sigma/r)^7) / r
    Возвращает скаляр (сила вдоль оси r).
    """
    sr6 = (SIGMA / r) ** 6
    sr12 = sr6 * sr6
    return 24.0 * EPSILON * (2.0 * sr12 - sr6) / r


def lj_potential(r):
    """Потенциал Леннарда-Джонса."""
    sr6 = (SIGMA / r) ** 6
    sr12 = sr6 * sr6
    return 4.0 * EPSILON * (sr12 - sr6)


# ==========================================================
# Инициализация
# ==========================================================

def init_positions(n, l):
    """Частицы на квадратной решётке + небольшой шум."""
    side = int(np.ceil(np.sqrt(n)))
    dx = l / side
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= n:
                break
            x = (i + 0.5) * dx
            y = (j + 0.5) * dx
            pos.append([x, y])
    pos = np.array(pos[:n])
    pos += 0.05 * dx * (np.random.rand(*pos.shape) - 0.5)
    return pos


def init_velocities(n, target_temperature):
    """Случайные скорости из распределения Максвелла,
    затем корректировка на нулевой суммарный импульс."""
    vel = np.random.randn(n, 2) * np.sqrt(target_temperature)
    vel -= vel.mean(axis=0)              # нулевой суммарный импульс
    return vel


# ==========================================================
# Периодические границы и сетка
# ==========================================================

def minimum_image(dr, l):
    """Учёт периодических границ: возвращает минимальный вектор."""
    return dr - l * np.round(dr / l)


def build_cell_grid(pos, l, r_cut):
    """Разбивает область на ячейки размера >= r_cut."""
    n_cells = max(1, int(l / r_cut))
    cell_size = l / n_cells
    grid = {}
    for idx, (x, y) in enumerate(pos):
        cx = int(x / cell_size) % n_cells
        cy = int(y / cell_size) % n_cells
        grid.setdefault((cx, cy), []).append(idx)
    return grid, n_cells, cell_size


def neighbor_pairs(pos, l, r_cut):
    """Возвращает пары (i, j) частиц на расстоянии < r_cut."""
    grid, n_cells, cell_size = build_cell_grid(pos, l, r_cut)
    pairs = []
    for (cx, cy), ids in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = (cx + dx) % n_cells
                ny = (cy + dy) % n_cells
                other = grid.get((nx, ny), [])
                for i in ids:
                    for j in other:
                        if i < j:
                            pairs.append((i, j))
    return pairs


# ==========================================================
# Силы
# ==========================================================

def compute_forces(pos, l, r_cut):
    """Считает силы и потенциальную энергию."""
    forces = np.zeros_like(pos)
    potential = 0.0

    pairs = neighbor_pairs(pos, l, r_cut)
    for i, j in pairs:
        dr = minimum_image(pos[j] - pos[i], l)
        r = np.linalg.norm(dr)
        if r < 1e-6 or r >= r_cut:
            continue
        f = lj_force(r)
        forces[i] -= f * dr / r
        forces[j] += f * dr / r
        potential += lj_potential(r)
    return forces, potential


# ==========================================================
# Термостат (простое масштабирование скоростей)
# ==========================================================

def temperature(vel):
    """Температура в 2D: T = <m*v^2> / (2*kB) с kB=1."""
    return MASS * np.sum(vel ** 2) / (2 * len(vel))


def rescale_velocities(vel, target_t):
    """Масштабирует скорости к целевой температуре."""
    t_now = temperature(vel)
    if t_now > 1e-12:
        vel *= np.sqrt(target_t / t_now)
    return vel


# ==========================================================
# Основной цикл (метод Верле)
# ==========================================================

def simulate():
    np.random.seed(42)

    pos = init_positions(N, L)
    vel = init_velocities(N, TARGET_T)
    vel = rescale_velocities(vel, TARGET_T)

    forces, potential = compute_forces(pos, L, R_CUT)

    # История
    energies = {"kinetic": [], "potential": [], "total": []}
    frames = []
    velocity_snapshots = []
    step_saved_positions = []

    print("Запуск молекулярной динамики...")
    for step in trange(N_STEPS, desc="MD"):
        # 1. Обновление скоростей (половина шага)
        vel += 0.5 * DT * forces / MASS

        # 2. Обновление координат (полный шаг)
        pos += DT * vel

        # 3. Периодические границы
        pos = np.mod(pos, L)

        # 4. Пересчёт сил
        forces, potential = compute_forces(pos, L, R_CUT)

        # 5. Обновление скоростей (вторая половина шага)
        vel += 0.5 * DT * forces / MASS

        # 6. Термостат (каждые 20 шагов)
        if step % 20 == 0:
            vel = rescale_velocities(vel, TARGET_T)

        # 7. Сбор статистики
        kinetic = 0.5 * MASS * np.sum(vel ** 2)
        energies["kinetic"].append(kinetic)
        energies["potential"].append(potential)
        energies["total"].append(kinetic + potential)

        if step % SAVE_EVERY == 0:
            frames.append(pos.copy())
            velocity_snapshots.append(vel.copy())
            step_saved_positions.append(step)

    return pos, vel, energies, frames, velocity_snapshots, step_saved_positions


# ==========================================================
# Визуализация
# ==========================================================

def create_gif(frames, filename="results/md_animation.gif", fps=15):
    os.makedirs("results", exist_ok=True)
    images = []
    print(f"Создание GIF: {len(frames)} кадров...")
    for pos in frames:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pos[:, 0], pos[:, 1], s=30, c="royalblue",
                   edgecolors="black", linewidths=0.5)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title(f"Молекулярная динамика, N={N}")
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    imageio.mimsave(filename, images, fps=fps, loop=0)
    print(f"GIF сохранён: {filename}")


def plot_energy(energies, filename="results/md_energy.png"):
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(len(energies["total"])) * DT
    ax.plot(t, energies["kinetic"], label="Кинетическая", alpha=0.7)
    ax.plot(t, energies["potential"], label="Потенциальная", alpha=0.7)
    ax.plot(t, energies["total"], label="Полная", linewidth=2, color="black")
    ax.set_xlabel("Время")
    ax.set_ylabel("Энергия")
    ax.set_title("Энергия системы от времени")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_velocity_distribution(velocity_snapshots,
                                filename="results/md_velocity.png"):
    os.makedirs("results", exist_ok=True)
    vel = velocity_snapshots[-1]
    speed = np.linalg.norm(vel, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(speed, bins=30, density=True, alpha=0.7,
            color="steelblue", edgecolor="black")

    # Теоретическое распределение Максвелла в 2D
    t_now = temperature(vel)
    v = np.linspace(0, speed.max() * 1.2, 200)
    f_maxwell = (v / t_now) * np.exp(-v ** 2 / (2 * t_now))
    ax.plot(v, f_maxwell, "r-", linewidth=2, label="Максвелл (теория)")

    ax.set_xlabel("Скорость |v|")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title(f"Распределение скоростей (T ≈ {t_now:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {filename}")


def radial_distribution(pos, l, n_bins=60, r_max=None):
    """Радиальная функция распределения g(r)."""
    if r_max is None:
        r_max = l / 2
    hist = np.zeros(n_bins)
    r_edges = np.linspace(0, r_max, n_bins + 1)

    pairs = neighbor_pairs(pos, l, r_max)
    for i, j in pairs:
        dr = minimum_image(pos[j] - pos[i], l)
        r = np.linalg.norm(dr)
        if r < r_max:
            idx = int(r / r_max * n_bins)
            if 0 <= idx < n_bins:
                hist[idx] += 2  # обе частицы

    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    shell_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    rho0 = N / (l * l)
    g_r = hist / (N * rho0 * shell_area + 1e-12)
    return r_centers, g_r


def plot_rdf(pos, filename="results/md_rdf.png"):
    os.makedirs("results", exist_ok=True)
    r, g = radial_distribution(pos, L)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, g, "b-", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Расстояние r")
    ax.set_ylabel("g(r)")
    ax.set_title("Радиальная функция распределения")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {filename}")


# ==========================================================
# Точка входа
# ==========================================================

if __name__ == "__main__":
    pos, vel, energies, frames, velocity_snapshots, _ = simulate()

    print("\nСоздание визуализаций...")
    create_gif(frames)
    plot_energy(energies)
    plot_velocity_distribution(velocity_snapshots)
    plot_rdf(pos)

    print("\nГотово! Результаты в папке 'results/'")