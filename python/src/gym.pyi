from typing import Dict, List, Tuple

class PyEnvironment:
    tot_steps: int

    @property
    def steps(self) -> int: ...
    @property
    def max_steps(self) -> int: ...
    @property
    def prev_potential(self) -> float: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def act_dim(self) -> int: ...
    def __init__(self, max_steps: int) -> None:
        """
        Initializes the environment.

        Args:
            max_steps: The maximum number of steps per episode.
        """
        ...

    def reset(self) -> List[float]:
        """
        Resets the environment to a new random state.

        Returns:
            The initial observation of the new episode.
        """
        ...

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, str]:
        """
        Takes a step in the environment.

        Args:
            action: A list containing [thrust, gimbal_angle].

        Returns:
            A tuple containing (next_observation, reward, done, reason).
        """
        ...

    def sample(self) -> List[float]:
        """
        Samples a random valid state.

        Returns:
            A random observation.
        """
        ...

    def render_info(self) -> Dict[str, float]:
        """
        Returns a dictionary with information for rendering.

        Returns:
            A dictionary with rocket state information.
        """
        ...
