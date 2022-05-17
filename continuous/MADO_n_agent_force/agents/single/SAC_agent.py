import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter
from simulation.mujoco_maze import maze_env_single
from agents.single.customized.collector import Collector
from tianshou.data import ReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic, RecurrentActorProb, RecurrentCritic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0) # required
    parser.add_argument('--recurrent', type=bool, default=False)
    parser.add_argument('--stack-num', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=3)
    parser.add_argument('--logdir', type=str, default='./agents/single/log')
    parser.add_argument('--render', type=float, default=0.01) # render time
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


class SACAgent(object):
    def __init__(self, env: maze_env_single.MazeEnv, new_goal_area: Optional[Dict]=None):
        self.env = env
        if new_goal_area is not None:
            self.env.set_goal_area(new_goal_area['goal'], new_goal_area['threshold'])
        self.args = get_args()

    def learn(self):

        self.args.state_shape = self.env.observation_space.shape
        self.args.action_shape = self.env.action_space.shape
        self.args.max_action = np.max(self.env.action_space.high)
        print("Observations shape:", self.args.state_shape) # danger
        print("Actions shape:", self.args.action_shape)
        print("Action range:", np.min(self.env.action_space.low), np.max(self.env.action_space.high))
        train_envs = self.env
        test_envs = self.env

        # seed # required
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        train_envs.seed(self.args.seed)
        test_envs.seed(self.args.seed)
        # model
        if not self.args.recurrent:
            net_a = Net(self.args.state_shape, hidden_sizes=self.args.hidden_sizes, device=self.args.device)
            # fine-tuning, max_action is not used if unbounded==True
            actor = ActorProb(net_a, self.args.action_shape, max_action=self.args.max_action, device=self.args.device, unbounded=False, conditioned_sigma=True).to(self.args.device)
        else:
            actor = RecurrentActorProb(layer_num=len(self.args.hidden_sizes), state_shape=self.args.state_shape, action_shape=self.args.action_shape, max_action=self.args.max_action,
                                       device=self.args.device, unbounded=True, conditioned_sigma=True, hidden_layer_size=self.args.hidden_sizes[0]).to(self.args.device)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.args.actor_lr)

        if not self.args.recurrent:
            net_c1 = Net(self.args.state_shape, self.args.action_shape, hidden_sizes=self.args.hidden_sizes, concat=True, device=self.args.device)
            critic1 = Critic(net_c1, device=self.args.device).to(self.args.device)
            net_c2 = Net(self.args.state_shape, self.args.action_shape, hidden_sizes=self.args.hidden_sizes, concat=True, device=self.args.device)
            critic2 = Critic(net_c2, device=self.args.device).to(self.args.device)
        else:
            critic1 = RecurrentCritic(layer_num=len(self.args.hidden_sizes), state_shape=self.args.state_shape, action_shape=self.args.action_shape,
                                       device=self.args.device, hidden_layer_size=self.args.hidden_sizes[0]).to(self.args.device)
            critic2 = RecurrentCritic(layer_num=len(self.args.hidden_sizes), state_shape=self.args.state_shape, action_shape=self.args.action_shape,
                                       device=self.args.device, hidden_layer_size=self.args.hidden_sizes[0]).to(self.args.device)

        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.args.critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.args.critic_lr)

        if self.args.auto_alpha: # fine_tune: True or False
            target_entropy = -np.prod(self.env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.args.alpha_lr)
            self.args.alpha = (target_entropy, log_alpha, alpha_optim)

        self.policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=self.args.tau,
            gamma=self.args.gamma,
            alpha=self.args.alpha, # auto or not
            estimation_step=self.args.n_step,
            action_space=self.env.action_space
        )

        # load a previous policy
        if self.args.resume_path:
            self.policy.load_state_dict(torch.load(self.args.resume_path, map_location=self.args.device))
            print("Loaded agent from: ", self.args.resume_path)

        # collector
        if not self.args.recurrent:
            buffer = ReplayBuffer(self.args.buffer_size)
        else:
            buffer = ReplayBuffer(self.args.buffer_size, stack_num=self.args.stack_num)

        train_collector = Collector(self.policy, train_envs, buffer, exploration_noise=True, random_init=True, is_sample=False)
        test_collector = Collector(self.policy, test_envs, random_init=True, is_sample=False)
        train_collector.collect(n_step=self.args.start_timesteps, random=True)
        # log
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'seed_{self.args.seed}_{t0}_sac'
        if self.args.recurrent:
            log_file += '_rnn'
        log_path = os.path.join(self.args.logdir, 'sac', log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(self.args)) # required
        logger = TensorboardLogger(writer) # required

        def save_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth')) # required

        if not self.args.watch:
            # trainer
            result = offpolicy_trainer(
                self.policy,
                train_collector,
                test_collector,
                self.args.epoch,
                self.args.step_per_epoch,
                self.args.step_per_collect,
                self.args.test_num,
                self.args.batch_size,
                save_fn=save_fn,
                logger=logger,
                update_per_step=self.args.update_per_step,
                test_in_train=False # to save time
            )
            pprint.pprint(result)

        # # Let's watch its performance!
        # self.policy.eval() # required
        # test_envs.seed(self.args.seed) # required
        # test_collector.reset()
        # result = test_collector.collect(n_episode=self.args.test_num, render=self.args.render)
        # print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = Batch(obs=np.array([obs[:3]]), info={})
        act_raw = self.policy.forward(batch=obs_batch).act
        # print("1: ", act_raw)
        act = self.policy.map_action(act_raw)
        # print("2: ", act)
        return act.cpu().detach().clone().numpy()[0]

if __name__ == '__main__':
    from simulation import mujoco_maze
    import time

    test_env = gym.make("Point4Rooms-a0-v1")
    test_agent = SACAgent(test_env, new_goal_area={'goal': np.array([8.0, 0.0]), 'threshold': 2.0})
    test_agent.learn()

    s = test_env.reset()
    for _ in range(1000):
        action = test_agent.get_action(s)
        print("a: ", action)
        s, r, done, _ = test_env.step(action)
        print("s: ", s)
        print("r: ", r)
        print("done: ", done)
        test_env.render()
        time.sleep(0.01)
        if done:
            break



