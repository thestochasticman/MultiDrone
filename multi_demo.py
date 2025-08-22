from MultiPlanner.multi_drone import MultiDrone
from MultiPlanner.multi_plan import multi_plan
import argparse
import numpy as np
import yaml

def main():
    ap = argparse.ArgumentParser(description="Multi-drone RRT-Connect")
    ap.add_argument("--env", type=str, default="multi_drone_obs/env2.yaml" ,help="Environment YAML (MultiDrone format)")
    ap.add_argument("--eta", type=float, default=1.2, help="Step size per extend")
    ap.add_argument("--iters", type=int, default=25000 * 2, help="Max Iter")
    ap.add_argument("--goal-bias", type=float, default=0.10, help="Probability of sampling the goal")
    ap.add_argument("--p-bridge", type=float, default=0.40, help="Probability of bridge sampling")
    ap.add_argument("--p-obstacle", type=float, default=0.40, help="Probability of Gaussian boundary sampling")
    ap.add_argument("--ball_radius", type=float, default=1.2, help="radius of the ball around to sample around q1 for sampling near obstacle")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--verbose", action="store_true", help="Print status")
    args = ap.parse_args()



    with open(args.env, 'r') as f:
        config = yaml.safe_load(f)
        n_drones = len(config.get('initial_configuration'))

    sim = MultiDrone(num_drones=n_drones, environment_file=args.env)

    path = multi_plan(
        sim,
        max_iterations=args.iters,
        eta=args.eta,
        goal_bias=args.goal_bias,
        p_bridge=args.p_bridge,
        p_obstacle=args.p_obstacle,
        ball_radius=args.ball_radius,
        seed=args.seed,
        verbose=args.verbose)
    
    sim.visualize_paths(path)

if __name__ == '__main__':
    main()