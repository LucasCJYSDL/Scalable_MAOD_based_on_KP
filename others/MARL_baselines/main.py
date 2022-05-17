# -*- coding: utf-8 -*-
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from common.arguments import get_common_args, get_q_decom_args
from runner import Runner
import os
import random
import numpy as np
import torch


def main(env, arg):
    runner = Runner(env, arg)
    runner.run()


if __name__ == '__main__':

    arguments = get_q_decom_args(get_common_args())

    if arguments.gpu is not None:
        arguments.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)

    file_name = os.path.dirname(os.path.realpath(__file__)) + '/tasks/' + arguments.task + '.txt'
    print("Grid World file path: ", file_name)
    environment = make_grid_world_from_file(file_name=file_name)

    env_info = environment.get_env_info()
    arguments.n_actions = env_info['n_actions']
    arguments.n_agents = env_info['n_agents']
    arguments.state_shape = env_info['state_shape']
    arguments.obs_shape = env_info['obs_shape']
    arguments.episode_limit = env_info['episode_limit']

    main(environment, arguments)

