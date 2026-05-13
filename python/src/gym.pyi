from typing import Dict, List, Tuple

class PyEnvironment:
    def __init__(self, max_steps: int, group_size: int) -> None:
        """
        Initializes the environment.

        Args:
            max_steps: The maximum number of steps per episode.
            group_size: Number of parallel agents.
        """
        ...

    @property
    def steps(self) -> int: ...
    @property
    def max_steps(self) -> int: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def act_dim(self) -> int: ...
    @property
    def group_size(self) -> int: ...
    def set_spawn_x_range(self, min: float, max: float) -> None:
        """
        Set the x-coordinate spawn range for all the rockets.

        Args:
            min: lower x bound
            max: upper x bound
        """
        ...

    def set_spawn_y_range(self, min: float, max: float) -> None:
        """
        Set the y-coordinate spawn range for all the rockets.

        Args:
            min: lower y bound
            max: upper y bound
        """
        ...

    def set_spawn_angle_range(self, min: float, max: float) -> None:
        """
        Set the theta-coordinate spawn range for all the rockets.

        Args:
            min: lower theta bound
            max: upper theta bound
        """
        ...

    def reset(self) -> List[List[float]]:
        """
        Resets the environment for all agents.

        Returns:
            A list of initial observations for all agents.
        """
        ...

    def step(
        self, actions: List[List[float]]
    ) -> Tuple[List[List[float]], List[float], List[bool], List[str]]:
        """
        Takes a step for all agents.

        Args:
            actions: A list of actions, one for each agent. Each action is [thrust, gimbal_angle].

        Returns:
            A tuple containing (next_observations, rewards, dones, reasons).
        """
        ...

    def sample(self) -> List[float]:
        """
        Samples a random valid state for a single rocket.

        Returns:
            A random observation for a single rocket.
        """
        ...

    def render_info(self) -> Dict[str, float]:
        """
        Returns a dictionary with information for rendering the first rocket.

        Returns:
            A dictionary with rocket state information.
        """
        ...
