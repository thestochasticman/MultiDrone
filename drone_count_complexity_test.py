from MultiPlanner.multi_plan import multi_plan
from MultiPlanner.multi_drone import MultiDrone
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import yaml
import time

from pathlib import Path
from typing import Sequence, Tuple, Dict, Union
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_metrics(
    times_taken: Sequence[float],
    drone_counts: Sequence[Union[int, float]],
    outdir: str = "outputs/drone_complexity_tests",
    fname: str = "num_drones_vs_time_taken",
    ylabel: str = "time (s)",
    add_trend: bool = False,
    yscale: str = "linear",
) -> Dict[str, Path]:
    """
    Plot time taken vs number of drones and save figure(s).

    Args:
        times_taken: y-values; must be same length as drone_counts.
        drone_counts: x-values; typically integers like [1,2,3,...].
        outdir: directory to save figures.
        fname: base filename (without extension); sanitized automatically.
        ylabel: y-axis label.
        add_trend: if True and len>=2, overlay a linear trendline.
        yscale: 'linear' or 'log'.

    Returns:
        {'png': Path, 'pdf': Path}
    """
    # --- validate & coerce ---
    x = np.asarray(drone_counts, dtype=float)
    y = np.asarray(times_taken, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("times_taken and drone_counts must be 1D sequences.")
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: len(x)={len(x)} vs len(y)={len(y)}.")
    if len(x) < 2:
        raise ValueError("Need at least 2 points to plot a line.")

    # sort by x for a tidy plot
    order = np.argsort(x)
    x, y = x[order], y[order]

    # --- prepare output paths ---
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    safe_fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", fname)
    png_path = out / f"{safe_fname}.png"
    pdf_path = out / f"{safe_fname}.pdf"

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel("num drones")
    ax.set_ylabel(ylabel)
    ax.set_title("num drones vs time taken")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_yscale(yscale)

    # integer ticks for drone counts (when reasonable)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # optional linear trend
    if add_trend and len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        yfit = np.polyval(coeffs, x)
        # R^2
        ss_res = float(np.sum((y - yfit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.plot(x, yfit, linestyle="--", linewidth=1.6, label=f"trend (R²={r2:.3f})")
        ax.legend()

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}

# Example:
# plot_metrics([0.8, 1.9, 3.7], [1, 2, 3])


def main():
    ap = argparse.ArgumentParser(description="Multi-drone RRT-Connect")

    ap.add_argument("--eta", type=float, default=1.2, help="Step size per extend")
    ap.add_argument("--iters", type=int, default=25000 * 2, help="Max Iter")
    ap.add_argument("--goal-bias", type=float, default=0.10, help="Probability of sampling the goal")
    ap.add_argument("--p-bridge", type=float, default=0.40, help="Probability of bridge sampling")
    ap.add_argument("--p-obstacle", type=float, default=0.40, help="Probability of obstacle sampling")
    ap.add_argument("--ball_radius", type=float, default=1.2, help="radius of the ball around to sample around q1 for sampling near obstacle")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = ap.parse_args()

    

    envs = [
        'drone_complexity_test_envs/env1.yaml',
        'drone_complexity_test_envs/env2.yaml',
        'drone_complexity_test_envs/env3.yaml',
        'drone_complexity_test_envs/env4.yaml',
        'drone_complexity_test_envs/env5.yaml',
        # 'drone_complexity_test_envs/env6.yaml',
        # 'drone_complexity_test_envs/env7.yaml',
        # 'drone_complexity_test_envs/env8.yaml',
        # 'drone_complexity_test_envs/env9.yaml'
    ]

    
    drone_count = 0
    drone_counts = []
    times_taken = []
    for env in envs:
        with open(env, 'r') as f:
            config = yaml.safe_load(f)
            n_drones = len(config.get('initial_configuration'))

        sim = MultiDrone(n_drones, environment_file=env)
        start_time = time.perf_counter()
        path, _ = multi_plan(
            sim,
            max_iterations=args.iters,
            eta=args.eta,
            goal_bias=args.goal_bias,
            p_bridge=args.p_bridge,
            p_obstacle=args.p_obstacle,
            ball_radius=args.ball_radius,
            seed=args.seed,
            verbose=False
        )
        times_taken += [time.perf_counter() - start_time]
        drone_counts += [drone_count]
        drone_count += 1

        
        sim.visualize_paths(path)

    print(times_taken)
    plot_metrics(times_taken, drone_counts)



        
        

if __name__ == '__main__':
    main()

