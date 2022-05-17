"""
Mujoco Maze environment.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple, Type, Dict

import gym
import numpy as np

from simulation.mujoco_maze import maze_env_utils, maze_task
from simulation.mujoco_maze.agent_model import AgentModel

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"


class MazeEnv(gym.Env):
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
        include_position: bool = True,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        inner_reward_scaling: float = 1.0,
        restitution_coef: float = 0.8,
        task_kwargs: dict = {},
        camera_move_x: Optional[float] = None,
        camera_move_y: Optional[float] = None,
        camera_zoom: Optional[float] = None,
        image_shape: Tuple[int, int] = (600, 480),
        **kwargs,
    ) -> None:
        self.t = 0  # time steps
        self._task = maze_task(maze_size_scaling, **task_kwargs)
        self._maze_height = height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self._inner_reward_scaling = inner_reward_scaling
        self._observe_blocks = self._task.OBSERVE_BLOCKS
        self._put_spin_near_agent = self._task.PUT_SPIN_NEAR_AGENT
        # Observe other objectives
        self._observe_balls = self._task.OBSERVE_BALLS
        self._top_down_view = self._task.TOP_DOWN_VIEW
        self._restitution_coef = restitution_coef

        self._maze_structure = structure = self._task.create_maze()
        # Elevate the maze to allow for falling.
        self.elevated = any(maze_env_utils.MazeCell.CHASM in row for row in structure)
        # Are there any movable blocks?
        self.blocks = any(any(r.can_move() for r in row) for row in structure)

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self._init_positions = [
            (x - torso_x, y - torso_y) for x, y in self._find_all_robots()
        ]
        self.agent_num = len(self._init_positions)

        if model_cls.MANUAL_COLLISION:
            if model_cls.RADIUS is None:
                raise ValueError("Manual collision needs radius of the model")
            self._collision = maze_env_utils.CollisionDetector(
                structure,
                size_scaling,
                torso_x,
                torso_y,
                model_cls.RADIUS,
            )
        else:
            self._collision = None

        # Let's create MuJoCo XML
        self.wrapped_env = []

        for idx in range(self.agent_num):
            xml_path = os.path.join(MODEL_DIR, model_cls.FILE+'_{}.xml'.format(idx))
            # read the info about the agent
            tree = ET.parse(xml_path)
            worldbody = tree.find(".//worldbody")

            height_offset = 0.0
            if self.elevated:
                # Increase initial z-pos of ant.
                height_offset = height * size_scaling
                torso = tree.find(".//body[@name='torso']")
                torso.set("pos", f"0 0 {0.75 + height_offset:.2f}")
            if self.blocks:
                # If there are movable blocks, change simulation settings to perform
                # better contact detection.
                default = tree.find(".//default")
                default.find(".//geom").set("solimp", ".995 .995 .01")


            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    struct = structure[i][j]
                    if struct.is_robot() and self._put_spin_near_agent:
                        struct = maze_env_utils.MazeCell.SPIN
                    x, y = j * size_scaling - torso_x, i * size_scaling - torso_y
                    h = height / 2 * size_scaling
                    size = size_scaling * 0.5
                    if self.elevated and not struct.is_chasm():
                        # Create elevated platform.
                        ET.SubElement(
                            worldbody,
                            "geom",
                            name=f"elevated_{i}_{j}",
                            pos=f"{x} {y} {h}",
                            size=f"{size} {size} {h}",
                            type="box",
                            material="",
                            contype="1",
                            conaffinity="1",
                            rgba="0.9 0.9 0.9 1",
                        )
                    # create walls
                    if struct.is_block():
                        # Unmovable block.
                        # Offset all coordinates so that robot starts at the origin.
                        ET.SubElement(
                            worldbody,
                            "geom",
                            name=f"block_{i}_{j}",
                            pos=f"{x} {y} {h + height_offset}",
                            size=f"{size} {size} {h}",
                            type="box",
                            material="",
                            contype="1",
                            conaffinity="1",
                            rgba="0.4 0.4 0.4 1",
                        )
                    assert not struct.can_move() and not struct.is_object_ball()


            torso = tree.find(".//body[@name='torso']")
            geoms = torso.findall(".//geom")
            for geom in geoms:
                if "name" not in geom.attrib:
                    raise Exception("Every geom of the torso must have a name")

            # Set goals
            self._task.set_goal_threshold(maze_size_scaling)
            for i, goal in enumerate(self._task.goals):
                z = goal.pos[2] if goal.dim >= 3 else 0.0
                if goal.custom_size is None:
                    size = f"{maze_size_scaling * 0.1}"
                else:
                    size = f"{goal.custom_size}"
                ET.SubElement(
                    worldbody,
                    "site",
                    name=f"goal_site{i}",
                    pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                    size=size,
                    rgba=goal.rgb.rgba_str(),
                )

            _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
            tree.write(file_path)
            self.wrapped_env.append(model_cls(file_path=file_path, **kwargs))

        self.observation_space = self._get_obs_space()
        self._camera_move_x = camera_move_x
        self._camera_move_y = camera_move_y
        self._camera_zoom = camera_zoom
        self._image_shape = image_shape

    @property
    def has_extended_obs(self) -> bool:
        return self._top_down_view or self._observe_blocks or self._observe_balls

    def _get_obs_space(self) -> gym.spaces.Box: # danger: you should add the agent loop out of the simulation when sampling the obs
        shape = self.get_obs()[0].shape
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        # Set velocity limits
        wrapped_obs_space = self.wrapped_env[0].observation_space
        high[: wrapped_obs_space.shape[0]] = wrapped_obs_space.high
        low[: wrapped_obs_space.shape[0]] = wrapped_obs_space.low
        # Set coordinate limits
        low[0], high[0], low[1], high[1] = self._xy_limits()
        # Set orientation limits
        return gym.spaces.Box(low, high)

    def _xy_limits(self) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = 100, 100, -100, -100
        structure = self._maze_structure
        # enumerate the empty blocks
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_block():
                continue
            xmin, xmax = min(xmin, j), max(xmax, j)
            ymin, ymax = min(ymin, i), max(ymax, i)
        x0, y0 = self._init_torso_x, self._init_torso_y
        scaling = self._maze_size_scaling
        xmin, xmax = (xmin - 0.5) * scaling - x0, (xmax + 0.5) * scaling - x0
        ymin, ymax = (ymin - 0.5) * scaling - y0, (ymax + 0.5) * scaling - y0
        return xmin, xmax, ymin, ymax

    def get_obs(self) -> List[np.ndarray]:
        obs_list = []
        for idx in range(self.agent_num):
            wrapped_obs = self.wrapped_env[idx]._get_obs()
            assert not self._top_down_view
            view = []

            additional_obs = []
            assert not self._observe_balls and not self._observe_blocks

            obs = np.concatenate([wrapped_obs[:3]] + additional_obs + [wrapped_obs[3:]])
            obs_list.append(np.concatenate([obs, *view, np.array([self.t * 0.001])])) # TODO: Fine-tuning

        return obs_list

    def get_state(self) -> np.ndarray:
        obs_list = self.get_obs()
        return np.array(obs_list).flatten()

    def get_env_info(self) -> Dict:
        env_info = {"state_shape": self.agent_num * int(np.prod(self.observation_space.shape)),
                    "obs_shape": int(np.prod(self.observation_space.shape)),
                    "action_shape": int(np.prod(self.action_space.shape)),
                    "n_agents": self.agent_num,
                    "episode_limit": 500}
        return env_info

    def reset(self) -> List[np.ndarray]:
        self.t = 0
        # back to MujocoEnv to see this
        for idx in range(self.agent_num):
            self.wrapped_env[idx].reset()
            xy = self.wrapped_env[idx].get_xy()
            xy += np.array(self._init_positions[idx])
            self.wrapped_env[idx].set_xy(xy) # danger
        # Samples a new goal
        assert not self._task.sample_goals()

        return self.get_obs()

    def _maybe_move_camera(self, viewer: Any) -> None:
        from mujoco_py import const

        if self._camera_move_x is not None:
            viewer.move_camera(const.MOUSE_ROTATE_V, self._camera_move_x, 0.0)
        if self._camera_move_y is not None:
            viewer.move_camera(const.MOUSE_ROTATE_H, 0.0, self._camera_move_y)
        if self._camera_zoom is not None:
            viewer.move_camera(const.MOUSE_ZOOM, 0, self._camera_zoom)

    def render_agent(self, agent_id, mode="human", **kwargs) -> Optional[np.ndarray]: # danger

        if self.wrapped_env[agent_id].viewer is None:
            self.wrapped_env[agent_id].render(mode, **kwargs)
            self._maybe_move_camera(self.wrapped_env[agent_id].viewer)
        return self.wrapped_env[agent_id].render(mode, **kwargs)

    @property
    def action_space(self): # # danger: you should add the agent loop out of the simulation when sampling the actions
        return self.wrapped_env[0].action_space

    def _find_robot(self) -> Tuple[float, float]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def _find_all_robots(self) -> List[Tuple[float, float]]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        coords = []
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                coords.append((j * size_scaling, i * size_scaling))
        return coords

    def step(self, action: np.ndarray) -> Tuple[List[np.ndarray], float, bool, dict]:
        self.t += 1
        info = {'position': []}
        for idx in range(self.agent_num): # currently, we don't consider the collision among the agents
            if self.wrapped_env[idx].MANUAL_COLLISION:
                old_pos = self.wrapped_env[idx].get_xy()
                inner_next_obs, inner_reward, _, _ = self.wrapped_env[idx].step(action[idx])
                new_pos = self.wrapped_env[idx].get_xy()

                # Checks that the new_position is in the wall
                collision = self._collision.detect(old_pos, new_pos)
                if collision is not None:
                    pos = collision.point + self._restitution_coef * collision.rest() # there is energy cost in the collision process
                    if self._collision.detect(old_pos, pos) is not None:
                        # If pos is also not in the wall, we give up computing the position
                        self.wrapped_env[idx].set_xy(old_pos)
                    else:
                        self.wrapped_env[idx].set_xy(pos)

            else:
                inner_next_obs, inner_reward, _, _ = self.wrapped_env[idx].step(action[idx])

            info['position'].append(self.wrapped_env[idx].get_xy())

        next_obs = self.get_obs()
        outer_reward = self._task.reward(next_obs, is_multiple=True) # no inner reward
        done = self._task.termination(next_obs)

        return next_obs, outer_reward, done, info

    def close(self) -> None:
        for idx in range(self.agent_num):
            self.wrapped_env[idx].close()



