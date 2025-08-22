from Planner.plan import plan
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Single-drone RRT-Connect with narrow-passage sampling")
    ap.add_argument("--env", type=str, default="single_drone_obs/env2.yaml" ,help="Environment YAML (MultiDrone format)")
    ap.add_argument("--eta", type=float, default=1.2, help="Step size per extend")
    ap.add_argument("--iters", type=int, default=25000 * 2, help="Iteration budget")
    ap.add_argument("--goal-bias", type=float, default=0.10, help="Probability of sampling the goal")
    ap.add_argument("--p-bridge", type=float, default=0.40, help="Probability of bridge sampling")
    ap.add_argument("--p-obstacle", type=float, default=0.40, help="Probability of Gaussian boundary sampling")
    ap.add_argument("--sigma", type=float, default=1.2, help="Gaussian sigma for boundary sampling")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--verbose", action="store_true", help="Print status")
    args = ap.parse_args()

    from Planner.multi_drone import MultiDrone

    sim = MultiDrone(num_drones=1, environment_file=args.env)

    path = plan(sim,
        max_iterations=args.iters,
        eta=args.eta,
        goal_bias=args.goal_bias,
        p_bridge=args.p_bridge,
        p_obstacle=args.p_obstacle,
        ball_radius=1,
        seed=args.seed,
        verbose=args.verbose)
    
    sim.visualize_paths([np.asarray(p).reshape(1,3) for p in path])
    # print(path)

if __name__ == '__main__':
    main()