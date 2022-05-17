import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy
from agents.multiple.components.episode_buffer import EpisodeBatch
from torch.distributions import Normal
import torch as th
from torch.optim import Adam
from agents.multiple.modules.critics import REGISTRY as critic_resigtry


class ActorCriticLearner:
    def __init__(self, mac, scheme, args):
        self.args = args
        self.n_agents = args.n_agents

        self.mac = mac
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

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            print("The sum of the mask is zero!")
            return

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()
        advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards, critic_mask)
        advantages = advantages.detach()

        self.mac.init_hidden(batch.batch_size)
        if self.args.is_discrete:
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
        else:
            mac_mu, mac_sigma = [], []
            for t in range(batch.max_seq_length - 1):
                agent_mus, agent_sigmas = self.mac.forward(batch, t=t)
                mac_mu.append(agent_mus)
                mac_sigma.append(agent_sigmas)
            mac_mu = th.stack(mac_mu, dim=1)
            mac_sigma = th.stack(mac_sigma, dim=1)
            m = Normal(mac_mu, mac_sigma)
            log_pi_taken = m.log_prob(actions).sum(-1, keepdim=True)
            log_pi_taken[mask == 0] = 0.0
            log_pi_taken = log_pi_taken.squeeze(3)
            entropy = m.entropy().sum(-1)

        pg_loss = -((advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
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

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        target_vals = target_critic(batch)[:, :-1]
        target_vals = target_vals.squeeze(3)

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

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
        nstep_values = th.zeros_like(values)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def cuda(self):
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
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

    def write_summary(self, info, step):
        for key, val in info.items():
            if isinstance(val, (int, float, np.int32, np.int64, np.float32, np.float64)):
                self.writer.add_scalar(key, val, step)
