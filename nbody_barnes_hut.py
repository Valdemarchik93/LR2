"""
Моделирование гравитационной задачи N тел методом молекулярной динамики.

Реализовано:
- гравитационное взаимодействие (закон Ньютона);
- два метода расчёта сил:
    * прямой (PP, O(N^2));
    * дерево Барнса-Хата (BH, O(N log N));
- интегрирование методом Верле;
- мягкость (softening) для устранения сингулярности;
- сравнение времени PP и BH;
- анимация GIF и графики.

Запуск:
    python nbody_barnes_hut.py --n 500 --steps 2000 --method bh
"""

import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import click
from tqdm import trange


# ==========================================================
# Параметры модели
# ==========================================================

G = 1.0                # гравитационная постоянная
SOFTENING = 0.1        # параметр мягкости (чтобы не было деления на ноль)
THETA = 0.5            # критерий раскрытия узла дерева
DT = 0.005             # шаг по времени


# ==========================================================
# Инициализация
# ==========================================================

def init_particles(n, radius=10.0, seed=42):
    """Случайное распределение в диске с небольшим вращением."""
    rng = np.random.default_rng(seed)

    # Позиции: равномерно в диске
    r = radius * np.sqrt(rng.random(n))
    theta = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Скорости: круговое движение (стабильный диск)
    vel = np.zeros_like(pos)
    for i in range(n):
        r_i = np.linalg.norm(pos[i])
        if r_i > 1e-6:
            v_circ = np.sqrt(G * 1.0 / (r_i + SOFTENING))
            # тангенциальное направление
            tangent = np.array([-pos[i, 1], pos[i, 0]]) / r_i
            vel[i] = v_circ * tangent * 0.8  # чуть меньше круговой — будет коллапс

    masses = np.ones(n) / n  # одинаковая масса
    return pos, vel, masses


# ==========================================================
# Дерево Барнса-Хата
# ==========================================================

class QuadNode:
    """Узел квадрадерева (2D)."""

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.mass = 0.0
        self.com = np.zeros(2)     # центр масс
        self.children = None       # 4 ребёнка или None
        self.index = None          # если лист — индекс частицы

    def size(self):
        return max(self.x_max - self.x_min, self.y_max - self.y_min)

    def insert(self, pos, mass, idx):
        """Вставка частицы в дерево."""
        # Если узел пустой и лист — сохраняем частицу
        if self.mass == 0.0 and self.children is None:
            self.mass = mass
            self.com = pos.copy()
            self.index = idx
            return

        # Если это лист с частицей — делим его
        if self.children is None:
            self._subdivide()
            old_pos = self.com.copy()
            old_mass = self.mass
            old_idx = self.index
            # Перекладываем старую частицу
            self.mass = 0.0
            self.com = np.zeros(2)
            self._insert_into_child(old_pos, old_mass, old_idx)
            self.index = None

        # Вставляем новую частицу в нужного ребёнка
        self._insert_into_child(pos, mass, idx)

        # Обновляем суммарную массу и центр масс
        self.mass += mass
        self.com = (self.com * (self.mass - mass) + pos * mass) / self.mass

    def _subdivide(self):
        """Создаёт 4 ребёнка."""
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        self.children = [
            QuadNode(self.x_min, xm, self.y_min, ym),  # SW
            QuadNode(xm, self.x_max, self.y_min, ym),  # SE
            QuadNode(self.x_min, xm, ym, self.y_max),  # NW
            QuadNode(xm, self.x_max, ym, self.y_max),  # NE
        ]

    def _insert_into_child(self, pos, mass, idx):
        """Определяет, в какого ребёнка вставлять."""
        xm = 0.5 * (self.x_min + self.x_max)
        ym = 0.5 * (self.y_min + self.y_max)
        child_idx = 0
        if pos[0] > xm:
            child_idx += 1
        if pos[1] > ym:
            child_idx += 2
        self.children[child_idx].insert(pos, mass, idx)


def build_tree(pos, masses):
    """Строит квадрадерево по всем частицам."""
    x_min, y_min = pos.min(axis=0)
    x_max, y_max = pos.max(axis=0)
    # Немного расширим границы
    pad = 0.01 * max(x_max - x_min, y_max - y_min)
    root = QuadNode(x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    for i, (p, m) in enumerate(zip(pos, masses)):
        root.insert(p, m, i)
    return root


def compute_force_bh(root, pos, mass_target, theta=THETA, softening=SOFTENING):
    """Считает силу, действующую на частицу pos, через дерево."""
    if root.mass == 0.0 or (root.children is None and root.index is None):
        return np.zeros(2)
    # Если это лист и это та же частица — пропускаем
    if root.children is None and root.index is not None:
        return np.zeros(2)

    dr = root.com - pos
    r2 = dr @ dr + softening ** 2
    r = np.sqrt(r2)

    # Критерий Барнса-Хата: если узел достаточно далеко — считаем как одну частицу
    if root.size() / r < theta:
        return G * root.mass * dr / (r2 * r + 1e-12)

    # Иначе — раскрываем узел
    force = np.zeros(2)
    if root.children is not None:
        for child in root.children:
            force += compute_force_bh(child, pos, mass_target, theta, softening)
    return force


# ==========================================================
# Прямой метод (PP)
# ==========================================================

def compute_forces_pp(pos, masses, softening=SOFTENING):
    """Прямой расчёт сил O(N^2) — честный Python-цикл.

    ВАЖНО: эта версия сознательно написана без векторизации NumPy,
    чтобы корректно сравнить алгоритмическую сложность PP и BH.
    """
    n = len(pos)
    forces = np.zeros_like(pos)

    for i in range(n):
        fx, fy = 0.0, 0.0
        xi, yi = pos[i]
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            r2 = dx * dx + dy * dy + softening ** 2
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            f = G * masses[j] * inv_r3
            fx += f * dx
            fy += f * dy
        forces[i, 0] = fx
        forces[i, 1] = fy

    return forces


# ==========================================================
# Метод Верле (leap-frog)
# ==========================================================

def verlet_step(pos, vel, acc, dt):
    """Один шаг Верле: r(t+dt) = 2r(t) - r(t-dt) + a*dt^2."""
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2
    new_vel = vel + 0.5 * acc * dt
    return new_pos, new_vel


# ==========================================================
# Энергии
# ==========================================================

def kinetic_energy(vel, masses):
    return 0.5 * np.sum(masses[:, None] * vel ** 2)


def potential_energy_pp(pos, masses, softening=SOFTENING):
    """Потенциальная энергия (для проверки сохранения)."""
    n = len(pos)
    u = 0.0
    for i in range(n):
        dr = pos[i + 1:] - pos[i]
        r2 = np.sum(dr ** 2, axis=1) + softening ** 2
        r = np.sqrt(r2)
        u -= G * masses[i] * np.sum(masses[i + 1:] / r)
    return u


# ==========================================================
# Основной цикл симуляции
# ==========================================================

def simulate(n=500, steps=2000, method="bh", save_every=20, seed=42):
    """Запускает симуляцию N тел.

    method = "bh" или "pp".
    """
    print(f"Инициализация {n} частиц...")
    pos, vel, masses = init_particles(n, seed=seed)

    frames = []
    times = []
    energies = {"kinetic": [], "potential": [], "total": []}
    step_times = []

    print(f"Метод: {method.upper()}, шагов: {steps}")

    for step in trange(steps, desc="N-body"):
        t0 = time.time()

        # Расчёт сил
        if method == "pp":
            forces = compute_forces_pp(pos, masses)
        else:
            tree = build_tree(pos, masses)
            forces = np.zeros_like(pos)
            for i in range(n):
                forces[i] = compute_force_bh(tree, pos[i], masses[i])

        # Ускорение
        acc = forces / masses[:, None]

        # Шаг Верле
        pos, vel = verlet_step(pos, vel, acc, DT)

        # Проверка на разлёт (стабильность)
        if not np.all(np.isfinite(pos)):
            print(f"⚠️  Численная неустойчивость на шаге {step}")
            break

        step_times.append(time.time() - t0)

        # Энергии (считаем через PP — точно)
        if step % 20 == 0:
            ke = kinetic_energy(vel, masses)
            pe = potential_energy_pp(pos, masses)
            energies["kinetic"].append(ke)
            energies["potential"].append(pe)
            energies["total"].append(ke + pe)
            times.append(step * DT)

        # Сохранение кадра
        if step % save_every == 0:
            frames.append(pos.copy())

    return {
        "pos": pos,
        "vel": vel,
        "masses": masses,
        "frames": frames,
        "times": times,
        "energies": energies,
        "step_times": step_times,
    }


# ==========================================================
# Визуализация
# ==========================================================

def create_gif(frames, filename="results/nbody.gif", fps=15):
    os.makedirs("results", exist_ok=True)
    images = []
    print(f"Создание GIF: {len(frames)} кадров...")
    for i, pos in enumerate(frames):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pos[:, 0], pos[:, 1], s=3, c="white",
                   edgecolors="none")
        ax.set_facecolor("black")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title(f"N-body, кадр {i+1}/{len(frames)}", color="white")
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=60,
                    bbox_inches="tight", facecolor="black")
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    imageio.mimsave(filename, images, fps=fps, loop=0)
    print(f"GIF сохранён: {filename}")


def plot_energy(times, energies, filename="results/nbody_energy.png"):
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, energies["kinetic"], label="Кинетическая", alpha=0.7)
    ax.plot(times, energies["potential"], label="Потенциальная", alpha=0.7)
    ax.plot(times, energies["total"], label="Полная",
            linewidth=2, color="black")
    ax.set_xlabel("Время")
    ax.set_ylabel("Энергия")
    ax.set_title("Энергия системы от времени")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_time_vs_n():
    """Сравнение времени PP и BH при разных N."""
    ns = [50, 100, 200, 500, 1000, 2000, 5000]
    times_pp = []
    times_bh = []

    print("Замер времени PP vs BH...")
    for n in ns:
        pos, _, masses = init_particles(n, seed=42)

        # PP
        t0 = time.time()
        compute_forces_pp(pos, masses)
        dt_pp = time.time() - t0
        times_pp.append(dt_pp)

        # BH
        t0 = time.time()
        tree = build_tree(pos, masses)
        forces = np.zeros_like(pos)
        for i in range(n):
            forces[i] = compute_force_bh(tree, pos[i], masses[i])
        dt_bh = time.time() - t0
        times_bh.append(dt_bh)

        ratio = dt_bh / dt_pp if dt_pp > 0 else float('inf')
        print(f"  N={n:5d}: PP={dt_pp:.4f}s, BH={dt_bh:.4f}s, "
              f"BH/PP={ratio:.2f}")

    # Теоретические кривые, отмасштабированные под данные
    ns_arr = np.array(ns, dtype=float)
    # Подгоняем коэффициенты по последней точке PP
    coef_pp = times_pp[-1] / (ns_arr[-1] ** 2)
    coef_bh = times_bh[-1] / (ns_arr[-1] * np.log(ns_arr[-1]))
    pp_theory = coef_pp * ns_arr ** 2
    bh_theory = coef_bh * ns_arr * np.log(ns_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, times_pp, "o-", label="Прямой (PP), эксперимент",
            linewidth=2, markersize=8, color="crimson")
    ax.plot(ns, times_bh, "s-", label="Барнс-Хат (BH), эксперимент",
            linewidth=2, markersize=8, color="steelblue")
    ax.plot(ns, pp_theory, "--", color="crimson", alpha=0.5,
            label="теория O(N²)")
    ax.plot(ns, bh_theory, "--", color="steelblue", alpha=0.5,
            label="теория O(N log N)")

    ax.set_xlabel("Число частиц N")
    ax.set_ylabel("Время одного расчёта сил, с")
    ax.set_title("Сравнение времени PP и BH")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("results/nbody_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранено: results/nbody_time.png")