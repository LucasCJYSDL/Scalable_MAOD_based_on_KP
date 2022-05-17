import gym
import numpy as np
from typing import List, Tuple

from functools import partial
from agents.multiple.components.episode_buffer import EpisodeBatch
from agents.multiple.controllers.basic_controller import BasicMAC
from agents.spectral.learners.option_wrapper import Option
from simulation import mujoco_maze

class HierarchicalEpisodeRunner:

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
        if low_level_groups is not None:
            self.new_low_batch = partial(EpisodeBatch, scheme, low_level_groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        else:
            self.new_low_batch = None

        self.mac = mac

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        if self.new_low_batch is not None:
            self.low_level_batch = self.new_low_batch()
        self.env.reset()
        self.t = 0
        self.option_durations = [-1 for _ in range(self.args.n_agents)]
        self.last_high_level_actions = [0 for _ in range(self.args.n_agents)]

    def run(self, low_level_mac: BasicMAC, option_list, test_mode=False):
        assert low_level_mac is not None and option_list is not None
        assert self.args.is_discrete and self.args.is_hierarchical
        if isinstance(option_list[0], Option):
            mode = 'multiple'
            assert self.args.is_force
        else:
            mode = 'single'

        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        low_level_mac.init_hidden(batch_size=self.batch_size)

        while True:
            # danger
            ori_obs = self.env.get_obs()
            new_obs = []
            for idx in range(self.args.n_agents):
                temp_obs = []
                temp_obs.append(ori_obs[idx].copy())
                ori_len = len(ori_obs)
                for ori_idx in range(ori_len):
                    if ori_idx == idx:
                        continue
                    temp_obs.append(ori_obs[ori_idx].copy())
                new_obs.append(np.array(temp_obs).flatten())

            pre_transition_data = {
                "state": [self.env.get_state()],
                "obs": [new_obs]}

            # update the option_duration list
            for idx in range(self.args.n_agents):
                if self.option_durations[idx] >= self.args.option_duration_limit - 1:
                    self.option_durations[idx] = -1
                    if mode == 'multiple':
                        is_term = np.array([1 for _ in range(self.args.real_n_agents)])
                else:
                    last_high_level_action = int(self.last_high_level_actions[idx])
                    if last_high_level_action == 0:
                        if self.t > 0:
                            assert self.option_durations[idx] == 0
                        self.option_durations[idx] = -1
                        if mode == 'multiple':
                            is_term = np.array([1 for _ in range(self.args.real_n_agents)])
                    else:
                        if mode == 'single':
                            temp_option = option_list[idx][last_high_level_action-1]
                            if temp_option.is_term_true(state=ori_obs, agent_id=idx):
                                self.option_durations[idx] = -1
                        else:
                            temp_option = option_list[last_high_level_action-1]
                            is_term = np.array([0 for _ in range(self.args.real_n_agents)])
                            for r_idx in range(self.args.real_n_agents):
                                if temp_option.is_term_true(state=ori_obs, agent_id=r_idx):
                                    is_term[r_idx] = 1

                            if is_term.any():
                                is_term = np.array([1 for _ in range(self.args.real_n_agents)])
                                self.option_durations[idx] = -1

            if mode == 'multiple':
                if is_term.all():
                    is_stay = np.array([0 for _ in range(self.args.real_n_agents)])
                else:
                    is_stay = is_term.copy()

            # get the available actions
            avai_actions = np.zeros(shape=(self.args.n_agents, self.args.n_actions)).astype(np.int)
            for i in range(self.args.n_agents):
                for j in range(self.args.n_actions):
                    if j == 0:
                        avai_actions[i][j] = 1 # the low level policy is always available
                    else:
                        if mode == 'single':
                            temp_option = option_list[i][j-1]
                        else:
                            temp_option = option_list[j-1]
                        if temp_option.is_init_true(state=ori_obs, agent_id=i):
                            avai_actions[i][j] = 1
            pre_transition_data["avail_actions"] = [avai_actions]

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            if mode == 'single':
                low_level_actions = low_level_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else: # danger
                low_obs = []
                for idx in range(self.args.real_n_agents):
                    temp_obs = []
                    temp_obs.append(ori_obs[idx].copy())
                    ori_len = len(ori_obs)
                    for ori_idx in range(ori_len):
                        if ori_idx == idx:
                            continue
                        temp_obs.append(ori_obs[ori_idx].copy())
                    low_obs.append(np.array(temp_obs).flatten())

                self.low_level_batch.update({"obs": [low_obs]}, ts=self.t)
                low_level_actions = low_level_mac.select_actions(self.low_level_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            cpu_actions = actions.cpu().detach().clone().numpy()
            cpu_low_level_actions = low_level_actions.cpu().detach().clone().numpy()
            assert len(cpu_actions[0]) == self.args.n_agents

            # reorganize the high-level actions and get the init_mask
            init_mask = []
            new_actions = []
            for idx in range(self.args.n_agents):
                if self.option_durations[idx] < 0:
                    init_mask.append([True]) # danger
                    new_actions.append(cpu_actions[0][idx])
                    self.last_high_level_actions[idx] = cpu_actions[0][idx]
                else:
                    init_mask.append([False]) # danger
                    new_actions.append(self.last_high_level_actions[idx])

                self.option_durations[idx] += 1

            # get the low level actions
            env_actions = []
            for idx in range(self.args.n_agents):
                high_level_action = int(new_actions[idx])
                if high_level_action == 0: # use the low level policy
                    if mode == 'single':
                        env_actions.append(cpu_low_level_actions[0][idx])
                    else:
                        for r_idx in range(self.args.real_n_agents):
                            env_actions.append(cpu_low_level_actions[0][r_idx])
                else:
                    if mode == 'single':
                        temp_option = option_list[idx][high_level_action-1]
                        env_actions.append(temp_option.act(state=ori_obs, agent_id=idx))
                    else:
                        temp_option = option_list[high_level_action-1]
                        for r_idx in range(self.args.real_n_agents):
                            if is_stay[r_idx]:
                                env_actions.append(np.array([0.0, 0.0])) # danger, but there is not anything that may interrupt the normal training process
                            else:
                                env_actions.append(temp_option.act(state=ori_obs, agent_id=idx))


            _, reward, terminated, env_info = self.env.step(env_actions)
            episode_return += reward

            post_transition_data = {
                "init": [init_mask], # danger
                "actions": [new_actions], # danger
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
            for ori_idx in range(ori_len):
                if ori_idx == idx:
                    continue
                temp_obs.append(ori_obs[ori_idx].copy())
            new_obs.append(np.array(temp_obs).flatten())

        last_data = {
            "state": [self.env.get_state()],
            "obs": [new_obs]}

        # update the option_duration list
        for idx in range(self.args.n_agents):
            if self.option_durations[idx] >= self.args.option_duration_limit - 1:
                self.option_durations[idx] = -1
            else:
                last_high_level_action = int(self.last_high_level_actions[idx])
                if last_high_level_action == 0:
                    if self.t > 0:
                        assert self.option_durations[idx] == 0
                    self.option_durations[idx] = -1
                else:
                    if mode == 'single':
                        temp_option = option_list[idx][last_high_level_action - 1]
                        if temp_option.is_term_true(state=ori_obs, agent_id=idx):
                            self.option_durations[idx] = -1
                    else:
                        temp_option = option_list[last_high_level_action - 1]
                        is_term = np.array([0 for _ in range(self.args.real_n_agents)])
                        for r_idx in range(self.args.real_n_agents):
                            if temp_option.is_term_true(state=ori_obs, agent_id=r_idx):
                                is_term[r_idx] = 1

                        if is_term.all():
                            self.option_durations[idx] = -1

        # get the available actions
        avai_actions = np.zeros(shape=(self.args.n_agents, self.args.n_actions)).astype(np.int)
        for i in range(self.args.n_agents):
            for j in range(self.args.n_actions):
                if j == 0:
                    avai_actions[i][j] = 1  # the low level policy is always available
                else:
                    if mode == 'single':
                        temp_option = option_list[i][j - 1]
                    else:
                        temp_option = option_list[j - 1]
                    if temp_option.is_init_true(state=ori_obs, agent_id=i):
                        avai_actions[i][j] = 1
        last_data["avail_actions"] = [avai_actions]

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state !!! useless for hppo !!!
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        cpu_actions = actions.cpu().detach().clone().numpy()
        assert len(cpu_actions[0]) == self.args.n_agents

        # reorganize the high-level actions and get the init_mask
        init_mask = []
        new_actions = []
        for idx in range(self.args.n_agents):
            if self.option_durations[idx] < 0:
                init_mask.append([True])  # danger
                new_actions.append(cpu_actions[0][idx])
                self.last_high_level_actions[idx] = cpu_actions[0][idx]
            else:
                init_mask.append([False])  # danger
                new_actions.append(self.last_high_level_actions[idx])

            self.option_durations[idx] += 1

        self.batch.update({"init": [init_mask], "actions": [new_actions]}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        return self.batch, {'return': episode_return, 'length': self.t}

