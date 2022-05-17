import os
import torch
import datetime
import gym
import numpy as np

from agents.spectral.configs import get_common_args
from agents.spectral.learners import laprepr, option_generator, option_agent
from agents.spectral.utils import timer_tools
from agents.spectral.learners.option_wrapper import Option
from simulation import mujoco_maze

class SpectralAgent(object):
    def __init__(self, env_id: str, seed: int, agent_num: int):
        self.env_id = env_id
        self.agent_num = agent_num
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # load the arguments
        self.args = get_common_args()
        unique_token = f"{env_id}_seed{seed}_{datetime.datetime.now()}"
        self.args.model_dir = './agents/spectral/' + self.args.model_dir + '/' + unique_token
        self.args.log_dir = './agents/spectral/' + self.args.log_dir + '/' + unique_token

        if torch.cuda.is_available() and self.args.cuda:
            self.args.device = torch.device('cuda')
            # if self.args.gpu is not None:
            #     os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        else:
            self.args.device = torch.device('cpu')
        print('device: {}.'.format(self.args.device))
        # environments
        pre = self.env_id.split('-')[0]
        suf = self.env_id.split('-')[-1]
        self.env_list = []
        self.env_id_list = []
        for idx in range(self.agent_num):
            temp_env_id = pre + '-' + "a{}".format(idx) + '-' + suf
            print("Making environment: {}!!!!!!".format(temp_env_id))
            self.env_id_list.append(temp_env_id)
            temp_env = gym.make(temp_env_id)
            temp_env.seed(self.seed)
            self.env_list.append(temp_env)

        env_info = self.env_list[0].get_env_info()
        self.args.obs_dim = env_info["obs_shape"]
        self.args.obs_pos_dim = env_info["obs_pos_shape"]
        self.args.act_dim = env_info["action_shape"]

    def get_option_list(self, mode, option_list=None):
        timer = timer_tools.Timer()
        self._collect_laplacian_spectrum(option_list)
        # get the termination and initiation set
        self.option_generator = option_generator.OptionGenerator(self.args, self.eigenvalue_list, self.laprepr_list)
        self.option_agent = option_agent.OptionAgent(self.args, self.env_id_list, self.option_generator)

        if mode == 'multiple':
            multi_subgoal_list, multi_threshold = self.option_generator.get_multi_options() # [(min_agent_0: np.ndarray, min_agent_1, ...), (max_agent_0, max_agent_1, ...)], threshold: int
            self.multi_option_agents = self.option_agent.get_multi_option_agents(multi_subgoal_list, multi_threshold)
            print('Total time cost: {:.4g}s.'.format(timer.time_cost()))
            return self.multi_option_agents
        else:
            assert mode == 'single'
            single_subgoal_list, single_threshold = self.option_generator.get_single_options() # [(agent_0_min: np.ndarray, agent_0_max), (agent_1_min, agent_1_max), ...], threshold_list: List[int]
            self.single_option_agents = self.option_agent.get_single_option_agents(single_subgoal_list, single_threshold)
            print('Total time cost: {:.4g}s.'.format(timer.time_cost()))
            return self.single_option_agents


    def _collect_laplacian_spectrum(self, option_list):
        self.laprepr_list = []
        self.eigenvalue_list = []

        for idx in range(self.agent_num):
            learner = laprepr.LapReprLearner(self.args, self.env_list[idx], agent_id=idx, seed=self.seed)
            if option_list is None:
                learner.train(option_list, self.agent_num)
            else:
                if isinstance(option_list[0], Option):
                    learner.train(option_list, self.agent_num)
                else:
                    learner.train(list(option_list[idx]), self.agent_num)
            self.laprepr_list.append(learner)
            self.eigenvalue_list.append(learner.get_eigenvalues())
            # if self.args.visualize:
            #     learner.visualize_embeddings(sample_num=self.args.ev_n_samples, interval=self.args.ev_interval)


if __name__ == '__main__':
    test_agent = SpectralAgent(env_id="Point4Rooms-v0", seed=0, agent_num=2)
    test_agent.get_option_list(mode='single')




