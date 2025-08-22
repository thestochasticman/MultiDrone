import numpy as np
from multi_drone import MultiDrone

def get_path(sim: MultiDrone)->list[np.array]:
  i_c = sim.initial_configuration
  g_p = sim.goal_positions
  if sim.motion_valid(i_c, g_p):
    return [i_c, g_p]


if __name__ == '__main__':
  sim = MultiDrone(num_drones=1, environment_file='single_drone_no_obs/env.yaml')
  path = get_path(sim)

  sim.visualize_paths(path)
  goal_reached = sim.is_goal(path[-1])
  print(f"goal reached: {goal_reached}")