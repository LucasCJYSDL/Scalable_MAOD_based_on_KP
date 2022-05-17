# -*- coding: utf-8 -*-

import argparse


def get_common_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./log', help='where to save the log files')
    parser.add_argument('--model_dir', type=str, default='./ckpt', help='where to save the ckpt files')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--gpu', type=str, default='cuda', help='which gpu to use')
    parser.add_argument('--render', type=bool, default=False, help='whether to render the env')
    parser.add_argument('--visualize', type=bool, default=False, help='whether to visualize the learned embeddings')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode or not')
    parser.add_argument('--range_threshold', type=float, default=2.0, help='range threshold for termination conditions')

    args = parser.parse_args()
    return args


def get_laprepr_args(args):

    args.d = 5 # the smallest d eigenvectors
    args.n_samples = 70000 # the total number of samples for training
    args.w_neg = 1.0
    args.c_neg = 1.0
    args.reg_neg = 0.0
    args.generalized = True # generalized spectral drawing or not

    args.use_position_only = True # important hyperparameters
    args.lap_n_layers = 3
    args.lap_n_units = 256

    args.lap_batch_size = 128
    args.lap_discount = 0.9 # important hyperparameters # 0.9
    args.lap_replay_buffer_size = 10000 # in fact 10000 * epi_length; original parameter: 100000,
    args.lap_opt_args_name = 'Adam'
    args.lap_opt_args_lr = 0.001
    args.lap_train_steps = 25000
    args.lap_print_freq = 5000
    args.lap_save_freq = 10000

    args.ev_n_samples = 70000
    args.ev_interval = 10000
    args.option_duration_limit = 100

    return args

def get_generator_args(args):
    args.with_degree = False # There are two versions of spectral partitioning algorithms.
    args.og_n_samples = 70000
    args.og_interval = 10000

    args.threshold = 0.5 # threshold for partitioning the whole state space to initiation and termination set, very important!!!

    return args


