from MultiPlanner.multi_plan import multi_plan
from MultiPlanner.multi_drone import MultiDrone
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import yaml

def plot_metrics(data: dict, outdir: str = "outputs/space_complexity_test"):
    """
    Plots each metric in `data` (except 'num_corridors') vs 'num_corridors'
    and saves figures as PNG (and PDF) in `outdir`.
    """
    if "num_corridors" not in data:
        raise KeyError("data must contain a 'num_corridors' key")

    x = np.asarray(data["num_corridors"])
    n = len(x)

    # Basic validation
    for k, v in data.items():
        if len(v) != n:
            raise ValueError(f"All arrays must have the same length. '{k}' has length {len(v)}, expected {n}.")

    # Sort by x so lines are tidy even if input order varies
    order = np.argsort(x)
    x_sorted = x[order]

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Plot each metric against num_corridors
    for metric, y in data.items():
        if metric == "num_corridors":
            continue
        y_sorted = np.asarray(y)[order]

        plt.figure()
        plt.plot(x_sorted, y_sorted, marker="o")
        plt.xlabel("num_corridors")
        plt.ylabel(metric)
        plt.title(f"{metric} vs num_corridors")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        png_path = out / f"{metric}_vs_num_corridors.png"
        pdf_path = out / f"{metric}_vs_num_corridors.pdf"
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()

    print(f"Saved figures to: {out.resolve()}")

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
        'complexity_test_envs/env1.yaml',
        'complexity_test_envs/env2.yaml',
        'complexity_test_envs/env3.yaml',
        'complexity_test_envs/env4.yaml',
        'complexity_test_envs/env5.yaml',
        'complexity_test_envs/env6.yaml',
        'complexity_test_envs/env7.yaml',
        'complexity_test_envs/env8.yaml',
        'complexity_test_envs/env9.yaml'
    ]

    obstacle_runs_info = {
        'iterations': [],
        'time_taken': [],
        'path_size': [],
        'num_corridors': [],
    }
    num_corridors = 0
    for env in envs:
        with open(env, 'r') as f:
            config = yaml.safe_load(f)
            n_drones = len(config.get('initial_configuration'))

        sim = MultiDrone(1, environment_file=env)
        path, runs_info = multi_plan(
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
        obstacle_runs_info['iterations'] += [list(runs_info)[0].iterations]
        obstacle_runs_info['path_size'] += [list(runs_info)[0].path_size]
        obstacle_runs_info['time_taken'] += [list(runs_info)[0].time_taken]
        obstacle_runs_info['num_corridors'] += [num_corridors]
        num_corridors += 1
        
        
        print(runs_info)
        sim.visualize_paths(path)

    print(obstacle_runs_info)
    plot_metrics(obstacle_runs_info)



        
        

if __name__ == '__main__':
    main()

