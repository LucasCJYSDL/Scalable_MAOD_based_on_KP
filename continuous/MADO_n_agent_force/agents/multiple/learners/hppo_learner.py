import os
import copy
import torch as th
from torch.optim import Adam
import numpy as np
from torch.distributions import Normal
from agents.multiple.modules.critics import REGISTRY as critic_resigtry
from agents.multiple.components.episode_buffer import EpisodeBatch
from torch.utils.tensorboard import SummaryWriter


class HPPOLearner:
    def __init__(self, mac, scheme, args):
        self.args = args
        self.n_agents = args.n_agents

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        if not os.path.exists(self.args.tb_path):
            os.makedirs(self.args.tb_path)
        self.writer = SummaryWriter(self.args.tb_path)

    def _get_option_duration(self, init_mask):
        device = init_mask.device
        init_mask_array = init_mask.cpu().detach().clone().numpy() # (bs, max_step, agent_num)
        option_duration_array = np.zeros_like(init_mask_array).astype(np.int) # (bs, max_step, agent_num)
        batch_size = init_mask_array.shape[0]
        max_length = init_mask_array.shape[1]
        agent_num = init_mask_array.shape[2]

        for bs_idx in range(batch_size):
            temp_array = init_mask_array[bs_idx] # (max_step, agent_num)
            for ag_idx in range(agent_num):
                cur_duration = 0
                for t in range(max_length-1, -1, -1):
                    cur_duration += 1
                    option_duration_array[bs_idx][t][ag_idx] = cur_duration
                    if temp_array[t][ag_idx] > 0.0:
                        cur_duration = 0

        return th.tensor(option_duration_array, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :] # in fact, it's option
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        init_mask = batch["init"][:, :-1].squeeze(3).float() # (bs, max_step-1, agent_num)
        option_duration = self._get_option_duration(init_mask)

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            print("The sum of the mask is zero!")
            return

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()
        self.old_mac.init_hidden(batch.batch_size)
        assert self.args.is_discrete
        old_mac_out = []
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0
        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        for k in range(self.args.epochs):

            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards, critic_mask, option_duration, init_mask)
            advantages = advantages.detach()
            # Calculate policy grad with mask
            self.mac.init_hidden(batch.batch_size)

            mac_out = []
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            pi = mac_out
            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)
            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            if int(self.args.hierarchy_type) == 1:
                pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()
            else:
                pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask * init_mask).sum() / (mask * init_mask).sum()

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            print("Hard update for the target network!!!!!!")
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        log_info = {'critic_loss': np.mean(critic_train_stats['critic_loss']), 'target_mean': np.mean(critic_train_stats['target_mean']),
                    'pg_loss': pg_loss.item()}
        self.write_summary(log_info, episode_num)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            print(log_info)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, option_duration, init_mask):
        # Optimise critic
        target_vals = target_critic(batch)[:, :-1]
        target_vals = target_vals.squeeze(3)

        if int(self.args.hierarchy_type) < 3:
            target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        else:
            target_returns = self.hierarchy_nstep_returns(rewards, mask, target_vals, option_duration)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v) # (bs, max_step-1, agent_num)
        if int(self.args.hierarchy_type) < 4:
            masked_td_error = td_error * mask
            loss = (masked_td_error ** 2).sum() / mask.sum()
        else:
            masked_td_error = td_error * mask * init_mask
            loss = (masked_td_error ** 2).sum() / (mask * init_mask).sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values) # (bs, max_step-1, agent_num)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0]) # (bs, agent_num)
            if int(self.args.hierarchy_type) < 3:
                temp_step = nsteps
            else:
                temp_step = int(nsteps[0, t_start, 0])
            for step in range(temp_step + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == temp_step:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def hierarchy_nstep_returns(self, rewards, mask, target_vals, option_duration):
        nstep_values = th.zeros_like(target_vals) # (bs, max_step-1, agent_num)
        batch_size = rewards.shape[0]
        for bs_idx in range(batch_size):
            for idx in range(self.n_agents):
                temp_rewards = rewards[bs_idx:(bs_idx+1), :, :]
                temp_masks = mask[bs_idx:(bs_idx+1), :, idx:(idx+1)]
                temp_target_vals = target_vals[bs_idx:(bs_idx+1), :, idx:(idx+1)]
                temp_option_duration = option_duration[bs_idx:(bs_idx+1), :, idx:(idx+1)]
                temp_values = self.nstep_returns(temp_rewards, temp_masks, temp_target_vals, temp_option_duration) # (1, max_step-1, 1)
                nstep_values[bs_idx:(bs_idx+1), :, idx:(idx+1)] = temp_values
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.old_mac.load_state_dict(self.mac.state_dict())
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

    def write_summary(self, info, step):
        for key, val in info.items():
            if isinstance(val, (int, float, np.int32, np.int64, np.float32, np.float64)):
                self.writer.add_scalar(key, val, step)
