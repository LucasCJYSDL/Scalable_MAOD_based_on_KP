# -*- coding: utf-8 -*-

import numpy as np
from common.replay_buffer import ReplayBuffer
from agents.agents import Opponent_agents, Search_agents
from agents.target_agents import Target_agents
from learners import REGISTRY as learner_REGISTRY
from network.bandits.uniform import Uniform
from network.bandits.hierarchial import EZ_agent

class Runner:
    def __init__(self, env, args):
        self.args = args
        self.env = env

        self.target_agents = Target_agents(args)
        self.opponent_agents = Opponent_agents(args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.learner = learner_REGISTRY[self.args.learner](self.target_agents, self.args)

        self.noise_generator = None
        if args.alg == 'maven':
            assert self.target_agents.get_actor_name() == 'noise_rnn'
            if args.noise_bandit:
                self.noise_generator = EZ_agent(args)
                self.oppo_noise_generator = EZ_agent(args)
            else:
                self.noise_generator = Uniform(args)
                self.oppo_noise_generator = Uniform(args)

        self.start_itr = 0

        self.start_train_steps = self.start_itr * args.train_steps

    def generate_episode(self, episode_num, evaluate=False):

        epsilon = 0 if evaluate else self.args.epsilon
        if self.args.epsilon_anneal_scale == 'episode' or (self.args.epsilon_anneal_scale == 'itr' and episode_num == 0):
            epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon
        if not evaluate:
            self.args.epsilon = epsilon

        episode_buffer = None
        if not evaluate:
            episode_buffer = {'o':            np.zeros([self.args.episode_limit + 1, self.args.n_agents, self.args.obs_shape]),
                              's':            np.zeros([self.args.episode_limit + 1, self.args.state_shape]),
                              'a':            np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a':     np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a':      np.zeros([self.args.episode_limit + 1, self.args.n_agents, self.args.n_actions]),
                              'r':            np.zeros([self.args.episode_limit, 1]),
                              'done':         np.ones([self.args.episode_limit, 1]),
                              'padded':       np.ones([self.args.episode_limit, 1]),
                              'gamma':        np.zeros([self.args.episode_limit, 1]),
                              'next_idx':     np.zeros([self.args.episode_limit, 1])}
        # roll out
        self.target_agents.init_hidden(1)
        self.opponent_agents.init_hidden(1)
        target_last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        temp_list = []
        self.env.reset()
        obs_list, temp_state = self.env.get_obs_state()
        target_noise = None
        if self.args.alg == 'maven':
            target_noise = self.noise_generator.sample(temp_state, test_mode=False)

        for episode in range(self.args.episode_limit):
            obs_list, temp_state = self.env.get_obs_state()

            avai_action = np.zeros(self.args.n_actions, dtype=np.float)
            action_list = []
            # action_onehot_list = []
            for agent_id in range(self.args.n_agents):
                temp_obs = obs_list[agent_id]

                action = self.target_agents.choose_action(temp_obs, target_last_action[agent_id], agent_id,
                                               avai_action, epsilon, self.replay_buffer, evaluate, noise=target_noise)
                action_list.append(action)
                onehot_action = np.zeros(self.args.n_actions)
                onehot_action[action] = 1
                target_last_action[agent_id] = onehot_action

                temp_dict = {'agent_id': agent_id, 'o': temp_obs, 's':temp_state, 'a': action, 'onehot_a': onehot_action, 'avail_a': avai_action}
                temp_list.append(temp_dict)

            rewards, _, done= self.env.execute_agent_action(action_list)

            episode_buffer['o'][episode] = obs_list
            episode_buffer['s'][episode] = temp_state
            episode_buffer['a'][episode] = np.reshape(action_list, [self.args.n_agents, 1])
            episode_buffer['onehot_a'][episode] = target_last_action
            episode_buffer['avail_a'][episode] = np.zeros([self.args.n_agents, self.args.n_actions], dtype=np.float)
            episode_buffer['r'][episode] = [rewards]
            episode_buffer['done'][episode] = [done]
            episode_buffer['padded'][episode] = [0.]

            if done:
                break

        round_num = episode + 1
        print("The total round number is {}.".format(round_num))
        episode_buffer['o'][round_num] = episode_buffer['o'][round_num-1].copy()
        episode_buffer['s'][round_num] = episode_buffer['s'][round_num-1].copy()
        episode_buffer['avail_a'][round_num] = episode_buffer['avail_a'][round_num-1].copy()

        print("The board score for the target group is {}!".format(episode_buffer['r'][round_num - 1][0]))

        if self.args.alg == 'maven':
            # self.noise_generator.update_returns(target_state_hands, target_noise, episode_buffer['r'][round_num - 1][0])
            episode_buffer['noise'] = np.array(target_noise)

        episode_buffer = self.multi_step_TD(episode_buffer, round_num)

        return episode_buffer, episode_buffer['r'][round_num - 1], round_num

    def multi_step_TD(self, episode_buffer, round_num):

        n = self.args.step_num
        gamma = self.args.gamma
        for e in range(round_num):
            if (e + n) < round_num:
                episode_buffer['gamma'][e] = [gamma ** n]
                temp_rwd = 0.
                for idx in range(e, e + n):
                    factor = gamma ** (idx - e)
                    temp_rwd += factor * episode_buffer['r'][idx][0]
                episode_buffer['r'][e] = [temp_rwd]
                episode_buffer['next_idx'][e] = [n]
            else:
                episode_buffer['done'][e] = [True]
                episode_buffer['gamma'][e] = [gamma ** (round_num - e)]
                temp_rwd = 0.
                for idx in range(e, round_num):
                    factor = gamma ** (idx - e)
                    temp_rwd += factor * episode_buffer['r'][idx][0]
                episode_buffer['r'][e] = [temp_rwd]
                episode_buffer['next_idx'][e] = [round_num - 1 - e]  ## check
            if episode_buffer['next_idx'][e][0] + e - 1 < 0:
                print("Bad index!!!")
                episode_buffer['next_idx'][e][0] = 1 - e
        return episode_buffer

    def run(self):
        train_steps = self.start_train_steps
        for itr in range(self.start_itr, self.args.n_itr):
            print("##########################{}##########################".format(itr))

            scores, steps = [], []
            episode_batch, score, step_num = self.generate_episode(0)
            # print("3: ", episode_batch)
            scores.append(score[0])
            steps.append(step_num)

            for key in episode_batch.keys():
                episode_batch[key] = np.array([episode_batch[key]])
            if self.args.alg == 'coma':
                assert (self.args.n_episodes > 1) and (self.args.n_episodes == self.args.batch_size == self.args.buffer_size) \
                and self.args.train_steps == 1, "COMA should be online learning!!!"
            for e in range(1, self.args.n_episodes):
                episode, score, step_num = self.generate_episode(e)
                scores.append(score[0])
                steps.append(step_num)
                for key in episode.keys():
                    episode[key] = np.array([episode[key]])
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.replay_buffer.store(episode_batch)
            if not self.replay_buffer.can_sample(self.args.batch_size):
                print("No enough episodes!!!")
                continue

            log_dict = self.learner.get_log_dict()
            for _ in range(self.args.train_steps):
                batch = self.replay_buffer.sample(self.args.batch_size)
                max_episode_len = self.target_agents.get_max_episode_len(batch)
                for key in batch.keys():
                    if key == 'noise':
                        continue
                    if key in ['o', 's', 'avail_a']:
                        batch[key] = batch[key][:, :max_episode_len+1]
                    else:
                        batch[key] = batch[key][:, :max_episode_len]
                log_info = self.learner.train(batch, max_episode_len, train_steps)
                for key in log_dict.keys():
                    assert key in log_info, key
                    log_dict[key].append(log_info[key])
                if train_steps > 0 and train_steps % self.args.save_model_period == 0:
                    print("Saving the models!")
                    self.learner.save_models(train_steps)
                    if self.args.alg == 'maven':
                        save_dir = self.learner.get_save_dir()
                        self.noise_generator.save_models(save_dir, train_steps)
                train_steps += 1

            print("Log to the tensorboard!")
            self.learner.log_info(np.mean(steps), scores, log_dict, itr)
