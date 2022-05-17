# -*- coding: utf-8 -*-

import argparse


def get_common_args():

    parser = argparse.ArgumentParser()
    # parameter for the game
    parser.add_argument('--debug', type=bool, default=False, help='whether to show the simulation information')
    # the algorithmm to use
    parser.add_argument('--alg', type=str, default='maven', help='training algorithm to use, you can choose from qmix, cwqmix, owqmix, coma, msac, maven')
    parser.add_argument('--task', type=str, default='maze4')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    # others
    parser.add_argument('--result_dir', type=str, default='./log', help='where to save the log files')
    parser.add_argument('--model_dir', type=str, default='./ckpt', help='where to save the ckpt files')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pre-trained model')
    parser.add_argument('--load_model_path', type=str, default='', help='where to upload the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')

    args = parser.parse_args()
    return args


def get_q_decom_args(args):
    # basic info part
    # whether to reuse the agent network among the agents; if true, the agent id will be part of the input
    args.reuse_network = True
    # whether to use last action as part of the input
    args.last_action = True
    # network
    args.agent = 'basic_agent'
    args.actor = 'rnn'
    args.pre_hidden_dim = 64
    args.rnn_hidden_dim = 64

    # loss function and optimizer part
    # discount factor
    args.gamma = 1.0
    # n-step sarsa
    args.step_num = 2
    # learning rate
    args.lr = 1e-3
    # RMSProp alpha
    args.optim_alpha = 0.99
    # RMSProp epsilon
    args.optim_eps = 0.00001
    # gradient clip
    args.clip_norm = 10 ## check

    # exploration part
    # epsilon greedy
    args.epsilon = 0.5
    args.min_epsilon = 0.001
    args.epsilon_decay = (args.epsilon - args.min_epsilon) / 1000
    args.epsilon_anneal_scale = 'itr'
    # ucb1
    args.ucb_coe = 0.1
    # boltzmann
    args.boltzmann_coe = 500
    # multi-nominal

    # others
    # total iteration number
    args.n_itr = 5000
    # how many episodes in an iteration
    args.n_episodes = 1
    # how many training times in an iteration
    args.train_steps = 10
    args.batch_size = 32
    args.buffer_size = int(5e3)
    # interval for saving the ckpt
    args.save_model_period = 5000 ## training steps
    # interval for saving the replay buffer
    args.save_buffer_period = 200 ## iteration number
    # interval for updating the target network
    args.target_update_period = 200 ## training steps
    # interval for updating the opponent agent
    args.opponent_update_period = 5000 ## training steps
    # interval for online evaluation
    args.evaluation_period = 10000 ## training steps
    # board number for online evaluation
    args.board_num = 4

    # specific algo part
    if args.alg == 'qmix':
        args = _qmix_args(args)
    elif (args.alg == 'cwqmix') or (args.alg == 'owqmix'):
        args = _wqmix_args(args)
    elif args.alg == 'coma':
        args = _coma(args)
    elif args.alg == 'msac':
        args = _msac(args)
    elif args.alg == 'maven':
        args = _maven(args)
    else:
        raise NotImplementedError

    return args


def _qmix_args(args):

    args.exploration = "ucb1"
    args.agent_output_type = "q_value"
    args.learner = "qmix_learner"
    args.double_q = True

    args.mixer = "qmix"
    args.mixing_embed_dim = 64
    args.hypernet_layers = 1
    args.hypernet_embed = 64

    return args

def _wqmix_args(args):

    args.exploration = "ucb1"
    args.agent_output_type = "q_value"
    args.learner = "wqmix_learner"
    args.double_q = True

    args.mixer = "qmix"
    args.mixing_embed_dim = 64
    args.hypernet_layers = 1
    args.hypernet_embed = 64

    args.central_loss = 1
    args.qmix_loss = 1
    args.w = 0.1  # $\alpha$ in the paper

    if args.alg == "cwqmix":
        args.hysteretic_qmix = False  # False -> CW-QMIX, True -> OW-QMIX
    else:
        assert args.alg == "owqmix"
        args.hysteretic_qmix = True

    args.central_mixer = "ff" ## 'ff' or 'atten'
    args.central_mixing_embed_dim = 64 ## fine tune, recommend: 256 for 'ff', 128 for 'atten'
    args.central_action_embed = 1
    args.central_agent = "basic_central_agent"
    args.central_actor = "central_rnn"
    args.central_pre_hidden_dim = 64
    args.central_rnn_hidden_dim = 64

    return args

def _coma(args):

    args.exploration = "multinomial"
    args.agent_output_type = "pi_logits"
    args.batch_size = 8
    args.buffer_size = 8 ## fine-tune
    args.n_episodes = 8 ## online rl, so the above three should be the same

    args.learner = "coma_learner"
    args.critic_embed_dim = 64 ## 128 in its initial setting
    args.critic_lr = 0.0005
    args.td_lambda = 0.8
    # args.recurrent_critic = False

    args.train_steps = 1 ## online learning
    args.save_buffer_period = 10000
    args.step_num = 1
    args.save_model_period = 1000 ## training steps
    args.target_update_period = 200 ## training steps
    args.opponent_update_period = 1000 ## training steps
    args.evaluation_period = 2000 ## training steps

    return args

def _msac(args):

    args.exploration = "multinomial"
    args.agent_output_type = "pi_logits"
    args.learner = "msac_learner"
    args.double_q = False

    args.mixer = None
    args.central_loss = 1
    args.actor_loss = 1
    args.central_mixing_embed_dim = 64
    args.central_mixer = "ff"

    args.entropy_temp = 0.01

    args.central_action_embed = 1
    args.central_agent = "basic_central_agent"
    args.central_actor = "central_rnn"
    args.central_pre_hidden_dim = 64
    args.central_rnn_hidden_dim = 64

    return args

def _maven(args):

    args.exploration = "ucb1"
    args.agent_output_type = "q_value"
    args.learner = "maven_learner"
    args.double_q = True

    args.mixer = 'noise_qmix'
    args.mixing_embed_dim = 64
    args.hypernet_layers = 1
    args.hypernet_embed = 64
    args.skip_connections = False
    args.hyper_initialization_nonzeros = 0

    args.agent = 'noise_agent'
    args.actor = 'noise_rnn'
    args.noise_dim = 4 ## default: 2
    args.noise_embedding_dim = 64 ## default: 32
    args.hyper = False ## default: True

    args.rnn_discrim = False
    args.rnn_agg_size = 32

    args.discrim_size = 64 ## default: 64
    args.discrim_layers = 3
    args.noise_bandit = False ## default: False
    args.entropy_scaling = 0.001
    args.bandit_buffer = 512
    args.bandit_iters = 8
    args.bandit_batch = 64
    args.mi_loss = 1
    args.hard_qs = False

    return args

if __name__ == '__main__':
    args = get_common_args()
    args = get_q_decom_args(args)
    print(args.hypernet_layers)
    print(getattr(args, "hypernet_layers", 1))

