from functools import partial
from agents.multiple.components.episode_buffer import EpisodeBatch
import numpy as np

import gym
from simulation import mujoco_maze

class EpisodeRunner:

    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = gym.make(self.args.env)
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0
        self.t_env = 0

    def setup(self, scheme, groups, preprocess, mac, low_level_groups=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, low_level_mac, option_list, test_mode=False):
        assert low_level_mac is None and option_list is None
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while True:
            # danger
            ori_obs = self.env.get_obs()
            new_obs = []
            for idx in range(self.args.n_agents):
                temp_obs = []
                temp_obs.append(ori_obs[idx].copy())
                ori_len = len(ori_obs)
                assert ori_len == self.args.n_agents
                for ori_idx in range(ori_len):
                    if ori_idx == idx:
                        continue
                    temp_obs.append(ori_obs[ori_idx].copy())
                new_obs.append(np.array(temp_obs).flatten())

            pre_transition_data = {
                "state": [self.env.get_state()],
                "obs": [new_obs]
            }

            if self.args.is_discrete:
                pre_transition_data["avail_actions"] = [self.env.get_avail_actions()],

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            _, reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            if terminated or (self.t >= self.episode_limit): # danger
                break

        # danger
        ori_obs = self.env.get_obs()
        new_obs = []
        for idx in range(self.args.n_agents):
            temp_obs = []
            temp_obs.append(ori_obs[idx].copy())
            ori_len = len(ori_obs)
            assert ori_len == self.args.n_agents
            for ori_idx in range(ori_len):
                if ori_idx == idx:
                    continue
                temp_obs.append(ori_obs[ori_idx].copy())
            new_obs.append(np.array(temp_obs).flatten())

        last_data = {
            "state": [self.env.get_state()],
            "obs": [new_obs]}
        if self.args.is_discrete:
            last_data["avail_actions"] = [self.env.get_avail_actions()]
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        return self.batch, {'return': episode_return, 'length': self.t}

