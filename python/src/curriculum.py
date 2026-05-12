import collections
from dataclasses import dataclass
from typing import Deque, List


@dataclass(frozen=True, slots=True)
class BoxBound:
    """
    Defines the spawn boundaries for a training task.
    All ranges are inclusive [min, max].
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    angle_min: float
    angle_max: float

    def __str__(self) -> str:
        return (
            f"X=[{self.x_min:.1f}, {self.x_max:.1f}], "
            f"Y=[{self.y_min:.1f}, {self.y_max:.1f}], "
            f"Angle=[{self.angle_min:.2f}, {self.angle_max:.2f}]"
        )


class CurriculumManager:
    """
    Manages the progression through a series of BoxBound tasks based on performance.
    """

    def __init__(
        self,
        env,
        tasks: List[BoxBound],
        window_size: int = 50,
        up_threshold: float = 0.7,
        down_threshold: float = 0.2,
    ) -> None:
        """
        Initializes the CurriculumManager.

        Args:
            env: The environment instance (must have set_spawn_* methods).
            tasks: A list of BoxBound objects defining the curriculum levels.
            window_size: Number of rollouts to average for success rate calculation.
            up_threshold: Success rate above which difficulty increases.
            down_threshold: Success rate below which difficulty decreases.
        """
        self.env = env
        self.tasks = tasks
        self.task_idx = 0
        self.success_history: Deque[float] = collections.deque(maxlen=window_size)
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

        if not self.tasks:
            raise ValueError("CurriculumManager requires at least one task.")

        self._sync_env()

    def update_metrics(self, success_count: int, group_size: int) -> None:
        """
        Records the results of a rollout and adjusts difficulty if thresholds are met.

        Args:
            success_count: Number of successful agents in the group.
            group_size: Total number of agents in the group.
        """
        rate = success_count / group_size
        self.success_history.append(rate)

        # Need enough data points before making a decision
        if len(self.success_history) < 10:
            return

        avg_rate = sum(self.success_history) / len(self.success_history)

        if avg_rate > self.up_threshold and self.task_idx < len(self.tasks) - 1:
            self.task_idx += 1
            print(
                f"[Curriculum] Level UP to {self.task_idx}. Avg Success: {avg_rate:.2f}"
            )
            self._sync_env()
            self.success_history.clear()
        elif avg_rate < self.down_threshold and self.task_idx > 0:
            self.task_idx -= 1
            print(
                f"[Curriculum] Level DOWN to {self.task_idx}. Avg Success: {avg_rate:.2f}"
            )
            self._sync_env()
            self.success_history.clear()

    def _sync_env(self) -> None:
        """Applies the current task's boundaries to the environment."""
        task = self.tasks[self.task_idx]
        self.env.set_spawn_x_range(task.x_min, task.x_max)
        self.env.set_spawn_y_range(task.y_min, task.y_max)
        self.env.set_spawn_angle_range(task.angle_min, task.angle_max)

    def __str__(self) -> str:
        avg_rate = (
            sum(self.success_history) / len(self.success_history)
            if self.success_history
            else 0.0
        )
        return (
            f"Curriculum(level={self.task_idx}/{len(self.tasks) - 1}, "
            f"avg_success={avg_rate:.2f}, "
            f"current_task={self.tasks[self.task_idx]})"
        )
