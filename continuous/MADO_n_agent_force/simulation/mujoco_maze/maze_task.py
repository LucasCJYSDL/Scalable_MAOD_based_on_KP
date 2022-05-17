"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np

from simulation.mujoco_maze.maze_env_utils import MazeCell


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 2.0,
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    AGENT_NUM: Optional[int] = None
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=8.0, point=4.0, swimmer=4.0)
    INNER_REWARD_SCALING: float = 0.01
    POSITION_ONLY = False
    # For Fall/Push/BlockMaze
    OBSERVE_BLOCKS: bool = False
    # For Billiard
    OBSERVE_BALLS: bool = False
    OBJECT_BALL_SIZE: float = 1.0
    # Unused now
    PUT_SPIN_NEAR_AGENT: bool = False
    TOP_DOWN_VIEW: bool = False

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale

    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: List[np.ndarray]) -> bool:
        for o in obs:
            is_goal = False
            for goal in self.goals:
                if goal.neighbor(o):
                    is_goal = True
                    break
            if not is_goal:
                return False

        return True

    def set_goal_area(self, goal: np.ndarray, threshold: float) -> None:
        self.goals = [MazeGoal(pos=goal, threshold=threshold)]

    def set_goal_threshold(self, threshold: float):
        for goal in self.goals:
            goal.threshold = threshold

    def is_goal_area(self, loc: np.ndarray) -> bool:
        return self.goals[0].euc_dist(loc) <= self.scale

    @abstractmethod
    def reward(self, obs: List[np.ndarray], is_multiple=False) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass

    @abstractmethod
    def init_positions(self) -> List[np.ndarray]:
        pass


# class DistRewardMixIn:
#     REWARD_THRESHOLD: float = -1000.0
#     goals: List[MazeGoal]
#     scale: float
#
#     def reward(self, obs: List[np.ndarray]) -> float:
#         return -self.goals[0].euc_dist(obs) / self.scale


class GoalRewardUMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: List[np.ndarray], is_multiple=False) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    AGENT_NUM: int = 2
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.scale = scale
        self.goals = [MazeGoal(np.array([0.0 * scale, -4.0 * scale]))] # relative location of the first agent

    def reward(self, obs: List[np.ndarray], is_multiple=False) -> float:
        if is_multiple:
            return 100.0 if self.termination(obs) else self.PENALTY * 100.0
        return 1.0 if self.termination(obs) else self.PENALTY

    def init_positions(self) -> List[np.ndarray]:
        return [np.array([1.0 * self.scale, 0.0 * self.scale]), np.array([5.0 * self.scale, 0.0 * self.scale]),
                np.array([1.0 * self.scale, -4.0 * self.scale]), np.array([5.0 * self.scale, -4.0 * self.scale])]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, E, B, B, B, E, B, B],
            [B, E, E, E, B, E, E, R, B],
            [B, E, E, E, E, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B]]

class DistReward4rooms(GoalReward4Rooms):
    POSITION_ONLY = True
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def reward(self, obs: List[np.ndarray], is_multiple=False) -> float:
        ori_rwd = super(DistReward4rooms, self).reward(obs)
        return -self.goals[0].euc_dist(obs[0]) / self.scale / 10.0 + ori_rwd * 100.0




class GoalRewardLongCorridor(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    AGENT_NUM: int = 2
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=4.0, swimmer=2.0)

    def __init__(self, scale: float, goal: Tuple[float, float] = (0.0, -4.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: List[np.ndarray], is_multiple=False) -> float:
        if is_multiple:
            return 100.0 if self.termination(obs) else self.PENALTY * 100.0
        return 1.0 if self.termination(obs) else self.PENALTY

    def init_positions(self) -> List[np.ndarray]:
        return [np.array([3.0 * self.scale, 0.0 * self.scale]), np.array([3.0 * self.scale, 2.0 * self.scale]),
                np.array([6.0 * self.scale, 4.0 * self.scale]), np.array([0.0 * self.scale, 5.0 * self.scale])]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, B, B, E, B, B, E, B],
            [B, E, E, B, E, B, E, E, B],
            [B, B, B, B, E, B, B, B, B],
            [B, E, B, E, E, E, B, R, B],
            [B, E, B, B, E, B, B, E, B],
            [B, R, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B]
        ]


class DistRewardLongCorridor(GoalRewardLongCorridor):
    POSITION_ONLY = True
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def reward(self, obs: List[np.ndarray], is_multiple=False) -> float:
        ori_rwd = super(DistRewardLongCorridor, self).reward(obs)
        return -self.goals[0].euc_dist(obs[0]) / self.scale / 10.0 + ori_rwd * 100.0




class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "UMaze": [GoalRewardUMaze],
        "4Rooms": [GoalReward4Rooms, DistReward4rooms],
        "LongCorridor": [GoalRewardLongCorridor, DistRewardLongCorridor]
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
