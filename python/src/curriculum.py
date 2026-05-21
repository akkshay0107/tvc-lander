from dataclasses import dataclass
from typing import List


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

    def __init__(self, env, tasks: List[BoxBound]) -> None:
        self.env = env
        self.tasks = tasks
        self._task_idx = 0

        if not self.tasks:
            raise ValueError("CurriculumManager requires at least one task.")

        self.apply()

    @property
    def task_idx(self) -> int:
        return self._task_idx

    @property
    def current_task(self) -> BoxBound:
        return self.tasks[self._task_idx]

    def step_up(self) -> bool:
        if self._task_idx < len(self.tasks) - 1:
            self._task_idx += 1
            self.apply()
            return True
        return False

    def step_down(self) -> bool:
        if self._task_idx > 0:
            self._task_idx -= 1
            self.apply()
            return True
        return False

    def apply(self) -> None:
        """Applies the current task's boundaries to the environment."""
        task = self.current_task
        self.env.set_spawn_x_range(task.x_min, task.x_max)
        self.env.set_spawn_y_range(task.y_min, task.y_max)
        self.env.set_spawn_angle_range(task.angle_min, task.angle_max)

    def __str__(self) -> str:
        return (
            f"Curriculum(level={self._task_idx}/{len(self.tasks) - 1}, "
            f"current_task={self.current_task})"
        )
