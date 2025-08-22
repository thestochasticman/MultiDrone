All commands are run from the root of the directory.

## Conda Environment

Please run

```
conda env create -f conda_env.yml
```

to activate conda, please run

```
conda activate comp4620-a1-yasaradeel
```
### To do a simple run, please run 

```
python multi_demo.py --env path_to_your_yaml
```

#### Examples

```
python multi_demo.py --env multi_drone_obs/env2.yaml

```
![multi_drone_example](outputs/multi_drone_example.png)
Use the arguments as provided in the argspace or use default ones.

### Code

All of it exists in the MultiPlanner directory. I have tried to give it a modular package structure.


### Environments are in
  * complexity_test_envs
  * drone_complexity_envs
  * multi_drone_obs
  * single_drone_obs

### Space Complexity Test

```
python space_complexity_test.py
```
A single drone goes over different corridors.

![6 corridors](outputs/space_complexity_test/6_corridor.png)

![iterations_vs_num_corridors](outputs/space_complexity_test/iterations_vs_num_corridors.png)
![time_taken_vs_num_corridors](outputs/space_complexity_test/time_taken_vs_num_corridors.png)


### Drone Complexity Test

```
python drone_count_complexity_test.py
```

Multiple Drones try to navigate through 2 corridors.

![5 Drones](outputs/drone_complexity_tests/5-drones.png)

![5 Drones times](outputs/drone_complexity_tests/num_drones_vs_time_taken.png)


