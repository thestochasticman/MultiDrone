from dataclasses import dataclass

@dataclass
class RunInfo:
    drone_idx   : int
    iterations  : int
    path_size   : int
    time_taken  : float

    def __hash__(self):
        '-'.join(
            [
                str(self.drone_idx),
                str(self.iterations),
                str(self.path_size),
                str(self.time_taken)
            ]
        )
        return int(self.drone_idx * self.iterations * self.path_size * self.time_taken)

    # def __str__(self):
    #     id_string = f"drone_id: {self.drone_idx}"
    #     iter_string = f"RRT iterations: {self.iterations}"
    #     path_size=f"RRT path size: {self.path_size}"
    #     time_taken = f"RRT_time_taken: {self.time_taken}"

    #     print(time_taken)