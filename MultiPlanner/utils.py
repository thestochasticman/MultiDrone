from dataclasses import asdict, is_dataclass
from Planner.multi_drone import MultiDrone
import numpy as np
import yaml

def to_serializable(x):
    if is_dataclass(x):
        return to_serializable(asdict(x))
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x  # assume already YAML-friendly


def get_drone_sims(starts: np.ndarray, goals: np.ndarray, config: dict):
    drone_sims = []
    for i in range(len(starts)):
        config_copy = config.copy()
        # one start (shape [1,3]) is correct for a 1-drone sim
        config_copy['initial_configuration'] = [np.asarray(starts[i], float).tolist()]
        # IMPORTANT: no extra [] around goals[i]
        config_copy['goals'] = [{
            'position': np.asarray(goals[i], float).tolist(),          # <- fixed
            'radius': float(config['goals'][i]['radius']),
            # keep color if you have it:
            **({'color': config['goals'][i].get('color')} if 'color' in config['goals'][i] else {})
        }]

        # (optional) sanity checks
        assert np.array(config_copy['goals'][0]['position']).shape == (3,), \
            f"goal position must be [x,y,z], got {config_copy['goals'][0]['position']}"

        sim_path = f'tmp/{i}.yaml'
        config_copy = to_serializable(config_copy)
        with open(sim_path, 'w') as f:
            yaml.safe_dump(config_copy, f, sort_keys=False)

        drone_sims += [MultiDrone(1, sim_path)]
        
    return drone_sims