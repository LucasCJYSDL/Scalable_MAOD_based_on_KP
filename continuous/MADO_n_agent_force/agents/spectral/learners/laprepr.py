import os
import collections
import random
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List

from agents.spectral.configs import get_laprepr_args
from agents.spectral.utils import torch_tools, timer_tools, summary_tools
from agents.spectral.modules.episodic_replay_buffer import EpisodicReplayBuffer
from agents.spectral.modules.networks import ReprNetMLP
# from agents.spectral.learners.option_wrapper import Option


def l2_dist(x1, x2, generalized):
    if not generalized:
        return (x1 - x2).pow(2).sum(-1)
    d = x1.shape[1]
    weight = np.arange(d, 0, -1).astype(np.float32)
    weight = torch_tools.to_tensor(weight, x1.device)
    return ((x1 - x2).pow(2)) @ weight.T

def pos_loss(x1, x2, generalized=False):
    return l2_dist(x1, x2, generalized).mean()

# used in the original code
# def _rep_loss(inprods, n, k, c, reg):
#
#     norms = inprods[torch.arange(n), torch.arange(n)]
#     part1 = inprods.pow(2).sum() - norms.pow(2).sum()
#     part1 = part1 / ((n - 1) * n)
#     part2 = - 2 * c * norms.mean() / k
#     part3 = c * c / k
#     # regularization
#     # if reg > 0.0:
#     #     reg_part1 = norms.pow(2).mean()
#     #     reg_part2 = - 2 * c * norms.mean()
#     #     reg_part3 = c * c
#     #     reg_part = (reg_part1 + reg_part2 + reg_part3) / n
#     # else:
#     #     reg_part = 0.0
#     # return part1 + part2 + part3 + reg * reg_part
#     return part1 + part2 + part3

def _rep_loss(inprods, n, k, c, reg):

    norms = inprods[torch.arange(n), torch.arange(n)]
    part1 = (inprods.pow(2).sum() - norms.pow(2).sum()) / ((n - 1) * n)
    part2 = - 2 * c * norms.mean()
    part3 = c * c * k

    return part1 + part2 + part3

def neg_loss(x, c=1.0, reg=0.0, generalized=False): # derivation and modification
    """
    x: n * d.
    The formula shown in the paper
    """
    n = x.shape[0]
    d = x.shape[1]
    if not generalized:
        inprods = x @ x.T
        return _rep_loss(inprods, n, d, c, reg)

    tot_loss = 0.0
    # tot_loss = torch.tensor(0.0, device=x.device, requires_grad=True) # danger
    for k in range(1, d+1):
        inprods = x[:, :k] @ x[:, :k].T
        tot_loss += _rep_loss(inprods, n, k, c, reg)
    return tot_loss


class LapReprLearner:

    def __init__(self, common_args, env, agent_id, seed):
        random.seed(0)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.args = get_laprepr_args(common_args)
        self.env = env
        self.agent_id = agent_id
        # NN
        if self.args.use_position_only:
            self._repr_fn = ReprNetMLP(self.args.obs_pos_dim, n_layers=self.args.lap_n_layers, n_units=self.args.lap_n_units, d=self.args.d)
        else:
            self._repr_fn = ReprNetMLP(self.args.obs_dim, n_layers=self.args.lap_n_layers, n_units=self.args.lap_n_units, d=self.args.d)
        self._repr_fn.to(device=self.args.device)
        # optimizer
        opt = getattr(optim, self.args.lap_opt_args_name)
        self._optimizer = opt(self._repr_fn.parameters(), lr=self.args.lap_opt_args_lr)
        # replay_buffer
        self._replay_buffer = EpisodicReplayBuffer(max_size=self.args.lap_replay_buffer_size)

        self._global_step = 0
        self._train_info = collections.OrderedDict()

        # create ckpt save dir and log dir
        self.saver_dir = os.path.join(self.args.model_dir, "agent_{}".format(self.agent_id))
        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        self.log_dir = os.path.join(self.args.log_dir, "agent_{}".format(self.agent_id))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def _collect_samples(self):
        # start actors, collect trajectories from random actions
        print('Start collecting samples for Agent {}.'.format(self.agent_id))
        timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 1000
        while total_n_steps < self.args.n_samples:
            # cur_obs = self.env.reset(random_init=True, is_sample=True) # random start points for the offline setting
            cur_obs = self.env.reset(random_init=True, is_sample=False)
            # print(cur_obs[:2])
            if self.args.use_position_only:
                cur_obs = cur_obs[:2]
            epi_len = 0
            episode = []
            while True:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)
                if self.args.use_position_only:
                    next_obs = next_obs[:2]
                # redundant info
                transition = {'s': cur_obs, 'a': action, 'r': reward, 'next_s': next_obs, 'done': done}
                cur_obs = next_obs
                epi_len += 1
                episode.append(transition)
                # log
                total_n_steps += 1
                if (total_n_steps + 1) % collect_batch == 0:
                    print('({}/{}) steps collected.'.format(total_n_steps + 1, self.args.n_samples))
                if epi_len >= self.episode_limit:
                    break
            final_transition = {'s': cur_obs, 'a': self.env.action_space.sample(), 'r': 0.0, 'next_s': cur_obs, 'done': True}
            episode.append(final_transition) # to make sure the last state in the episodes can be sampled in the future process
            self._replay_buffer.add_steps(episode)
        time_cost = timer.time_cost()
        print('Data collection for Agent {} finished, time cost: {}s'.format(self.agent_id, time_cost))

    def _collect_samples_with_options(self, option_list, agent_num: int):
        # start actors, collect trajectories from random actions
        print('Start collecting hierarchical samples for Agent {}.'.format(self.agent_id))
        timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 1000
        action_space = len(option_list) + 1
        while total_n_steps < self.args.n_samples:
            cur_obs = self.env.reset(random_init=True, is_sample=False) # random start points for the offline setting
            if self.args.use_position_only:
                cur_obs = cur_obs[:2]
            epi_len = 0
            episode = []
            last_high_act = 0
            option_duration = -1
            while True:
                if (option_duration >= self.args.option_duration_limit - 1) or (last_high_act == 0) \
                        or (option_list[last_high_act-1].is_term_true(state=[cur_obs.copy() for _ in range(agent_num)], agent_id=self.agent_id)):
                    option_duration = -1

                if option_duration == -1:
                    avail_actions = []
                    for i in range(action_space): # a little wrong in logic but does not hurt much
                        # if i == 0:
                        avail_actions.append(i)
                        # else:
                        #     if option_list[i-1].is_init_true(state=[cur_obs.copy() for _ in range(agent_num)], agent_id=self.agent_id):
                        #         avail_actions.append(i)

                    high_level_act = random.choice(avail_actions)
                    last_high_act = high_level_act
                else:
                    high_level_act = last_high_act
                option_duration += 1

                if high_level_act == 0:
                    action = self.env.action_space.sample()
                else:
                    action = option_list[high_level_act-1].act(state=[cur_obs.copy() for _ in range(agent_num)], agent_id=self.agent_id)

                next_obs, reward, done, _ = self.env.step(action)
                if self.args.use_position_only:
                    next_obs = next_obs[:2]
                # redundant info
                transition = {'s': cur_obs, 'a': action, 'r': reward, 'next_s': next_obs, 'done': done}
                cur_obs = next_obs
                epi_len += 1
                episode.append(transition)
                # log
                total_n_steps += 1
                if (total_n_steps + 1) % collect_batch == 0:
                    print('({}/{}) steps collected.'.format(total_n_steps + 1, self.args.n_samples))
                if epi_len >= self.episode_limit:
                    break
            final_transition = {'s': cur_obs, 'a': self.env.action_space.sample(), 'r': 0.0, 'next_s': cur_obs, 'done': True}
            episode.append(final_transition) # to make sure the last state in the episodes can be sampled in the future process
            self._replay_buffer.add_steps(episode)
        time_cost = timer.time_cost()
        print('Hierarchical data collection for Agent {} finished, time cost: {}s'.format(self.agent_id, time_cost))

    def train(self, option_list, agent_num):
        self.episode_limit = self.env.get_env_info()["episode_limit"] # 1000
        if option_list is None:
            self._collect_samples()
        else:
            self._collect_samples_with_options(option_list, agent_num)
        # learning begins
        timer = timer_tools.Timer()
        timer.set_step(0)
        for step in range(self.args.lap_train_steps):
            assert step == self._global_step
            self._train_step()
            # save
            if (step + 1) % self.args.lap_save_freq == 0:
                saver_path = os.path.join(self.saver_dir, 'model_{}.ckpt'.format(step+1))
                torch.save(self._repr_fn.state_dict(), saver_path)
            # print info
            if step == 0 or (step + 1) % self.args.lap_print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                print('Training steps per second: {:.4g}.'.format(steps_per_sec))
                summary_str = summary_tools.get_summary_str(step=self._global_step, info=self._train_info)
                print(summary_str)
                if self.args.visualize:
                    self.visualize_embeddings(sample_num=self.args.ev_n_samples, interval=self.args.ev_interval, step=step)

        # save the final laprepr model
        saver_path = os.path.join(self.saver_dir, 'final_model.ckpt')
        torch.save(self._repr_fn.state_dict(), saver_path)
        # log the time cost
        time_cost = timer.time_cost()
        print('Training finished, time cost {:.4g}s.'.format(time_cost))

    def _train_step(self):
        train_batch = self._get_train_batch()
        loss = self._build_loss(train_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._global_step += 1

    def _get_train_batch(self): # how will the discount influence the performance?
        s1, s2 = self._replay_buffer.sample_steps(self.args.lap_batch_size, mode='pair', discount=self.args.lap_discount)
        s_neg, _ = self._replay_buffer.sample_steps(self.args.lap_batch_size, mode='single')
        s1, s2, s_neg = map(self._get_obs_batch, [s1, s2, s_neg])
        batch = {}
        batch['s1'] = self._tensor(s1)
        batch['s2'] = self._tensor(s2)
        batch['s_neg'] = self._tensor(s_neg)
        return batch

    def _build_loss(self, batch): # modification
        s1 = batch['s1']
        s2 = batch['s2']
        s_neg = batch['s_neg']
        s1_repr = self._repr_fn(s1)
        s2_repr = self._repr_fn(s2)
        s_neg_repr = self._repr_fn(s_neg)
        loss_positive = pos_loss(s1_repr, s2_repr, generalized=self.args.generalized)
        loss_negative = neg_loss(s_neg_repr, c=self.args.c_neg, reg=self.args.reg_neg, generalized=self.args.generalized)
        assert loss_positive.requires_grad and loss_negative.requires_grad # danger
        loss = loss_positive + self.args.w_neg * loss_negative
        info = self._train_info
        info['loss_pos'] = loss_positive.item()
        info['loss_neg'] = loss_negative.item()
        info['loss_total'] = loss.item()
        summary_tools.write_summary(self.writer, info=info, step=self._global_step)
        return loss

    def _get_obs_batch(self, steps): # which way is better for spectral clustering?
        if self.args.use_position_only:
            obs_batch = [s[:2] for s in steps]
        else:
            obs_batch = steps
        return np.stack(obs_batch, axis=0)

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self.args.device)

    def _get_pair_embeddings(self, sample_num, interval):
        pair_embeddings = [[], []]
        with torch.no_grad(): # danger
            cur_idx = 0
            while cur_idx < sample_num:
                next_idx = min(cur_idx + interval, sample_num)
                s1, s2 = self._replay_buffer.sample_steps(next_idx-cur_idx, mode='pair', discount=self.args.lap_discount)
                s1, s2 = map(self._get_obs_batch, [s1, s2])
                s1, s2 = map(self._tensor, [s1, s2]) # danger
                s1_repr = self._repr_fn(s1)
                s2_repr = self._repr_fn(s2)
                pair_embeddings[0] += s1_repr.cpu().tolist()
                pair_embeddings[1] += s2_repr.cpu().tolist()
                cur_idx = next_idx
        pair_embeddings = np.array(pair_embeddings)
        assert pair_embeddings.shape[1] == sample_num

        return pair_embeddings

    def get_eigenvalues(self): # important and dangerous; time complexity: d|S| # check!
        print('Start estimating eigenvalues for Agent {}.'.format(self.agent_id))
        timer = timer_tools.Timer()

        self.eigenvalue_list = []
        d_max = self.args.d
        pair_embeddings = self._get_pair_embeddings(sample_num=self.args.ev_n_samples, interval=self.args.ev_interval) # np.ndarray: [2, |S|, d]
        assert pair_embeddings.shape[2] == d_max
        assert self.args.generalized
        for k in range(d_max):
            # danger
            k_value = 0.5 * (np.square(pair_embeddings[0][:, k] - pair_embeddings[1][:, k])).mean()
            self.eigenvalue_list.append(k_value)

        time_cost = timer.time_cost()
        print('Eigenvalues estimating finished, time cost: {}s, generalized: {}.'.format(time_cost, self.args.generalized))
        print("The eigenvalue list for Agent {} is {}!!!".format(self.agent_id, self.eigenvalue_list))
        return self.eigenvalue_list

    def get_embedding_optimum(self, sample_num, interval, dim, with_degree):
        data_input = self._replay_buffer.get_all_steps(max_num=sample_num)
        obs_input = self._get_obs_batch(data_input)
        obs_input = self._tensor(obs_input)  # maybe too much for the gpu?
        data_size = int(obs_input.shape[0])

        embeddings = []
        with torch.no_grad():  # danger
            cur_idx = 0
            while cur_idx < data_size:
                next_idx = min(cur_idx + interval, data_size)
                data_segment = obs_input[cur_idx:next_idx, :]
                raw_embedding_segment = self._repr_fn(data_segment)
                if with_degree:
                    embedding_segment = raw_embedding_segment[:, 0] * raw_embedding_segment[:, dim]
                else:
                    embedding_segment = raw_embedding_segment[:, dim]
                embeddings = embeddings + embedding_segment.cpu().detach().clone().tolist()
                cur_idx = next_idx
        embeddings = np.array(embeddings)
        assert embeddings.shape[0] == data_size

        embeddings = np.around(embeddings, 6)
        min_idx = np.argmin(embeddings) # TODO: there may be a few points related to the optimum, which can be further filtered based on the goal location
        max_idx = np.argmax(embeddings)

        return [(data_input[min_idx], embeddings[min_idx]), (data_input[max_idx], embeddings[max_idx])]


    def get_embedding(self, data_input, dim, with_degree):
        obs_input = self._get_obs_batch([data_input])
        obs_input = self._tensor(obs_input)  # maybe too much for the gpu?

        with torch.no_grad():
            raw_embedding_segment = self._repr_fn(obs_input)
            if with_degree:
                embedding_segment = raw_embedding_segment[:, 0] * raw_embedding_segment[:, dim]
            else:
                embedding_segment = raw_embedding_segment[:, dim]

        embedding = np.around(embedding_segment.cpu().detach().clone().numpy()[0], 6)

        return embedding

    def visualize_embeddings(self, sample_num, interval, step, dir='./agents/spectral/visualization'):
        import matplotlib.pyplot as plt

        if not os.path.exists(dir):
            os.makedirs(dir)

        data_input = self._replay_buffer.get_all_steps(max_num=sample_num)
        obs_input = self._get_obs_batch(data_input)
        obs_input = self._tensor(obs_input)  # maybe too much for the gpu?
        data_size = int(obs_input.shape[0])

        embeddings = []
        with torch.no_grad():  # danger
            cur_idx = 0
            while cur_idx < data_size:
                next_idx = min(cur_idx + interval, data_size)
                data_segment = obs_input[cur_idx:next_idx, :]
                raw_embedding_segment = self._repr_fn(data_segment)
                embeddings = embeddings + raw_embedding_segment.cpu().detach().clone().tolist()
                cur_idx = next_idx
        embeddings = np.array(embeddings)
        assert embeddings.shape[0] == data_size
        embeddings = np.around(embeddings, 6)

        for dim in range(2):
            axis_x = np.array(data_input)[:, 0]
            axis_y = np.array(data_input)[:, 1]
            value = embeddings[:, dim]
            plt.figure()
            plt.scatter(x=axis_x, y=axis_y, c=value, cmap="viridis", alpha=0.3)
            plt.colorbar()
            plt.savefig(dir + '/'+ 'step_{}_agent_{}_embedding_{}.png'.format(step, self.agent_id, dim))



