import datetime
import os
import numpy as np
import threading
import torch as th
from types import SimpleNamespace as SN
from typing import Optional, List, Tuple
from os.path import dirname, abspath

from agents.multiple.controllers.basic_controller import BasicMAC
from agents.multiple.learners import REGISTRY as le_REGISTRY
from agents.multiple.runners import REGISTRY as r_REGISTRY
from agents.multiple.controllers import REGISTRY as mac_REGISTRY
from agents.multiple.components.episode_buffer import ReplayBuffer
from agents.multiple.components.transforms import OneHot
from agents.spectral.learners.option_wrapper import Option


class RunAgent(object):
    def __init__(self, _config):
        # check args sanity
        self._config = self._args_sanity_check(_config)

        self.args = SN(**(self._config))
        self.args.device = "cuda" if self.args.use_cuda else "cpu"

        unique_token = f"{_config['env']}_{_config['name']}_seed{_config['seed']}_{datetime.datetime.now()}"
        self.args.tb_path = os.path.join(dirname(dirname(abspath(__file__))), "multiple", "results", "tb_logs", "{}").format(unique_token)
        self.args.ckpt_path = os.path.join(dirname(dirname(abspath(__file__))), "multiple", "results", "ckpt", "{}").format(unique_token)

        if self.args.env_args['type'] == 'discrete':
            self.args.is_discrete = True
            self.args.is_hierarchical = False
            self.args.is_force = False
        elif self.args.env_args['type'] == 'continuous':
            self.args.is_discrete = False
            self.args.is_hierarchical = False
            self.args.is_force = False
        elif self.args.env_args['type'] == 'hierarchical':
            self.args.is_discrete = True
            self.args.is_hierarchical = True
            self.args.is_force = False
        else: # change
            assert self.args.env_args['type'] == 'hierarchical_force'
            self.args.is_discrete = True
            self.args.is_hierarchical = True
            self.args.is_force = True

        # Init runner so we can get env info
        self.runner = r_REGISTRY[self.args.runner](args=self.args)
        print("The RUNNER selected is {}!!!".format(self.args.runner))

        # Set up schemes and groups here
        env_info = self.runner.get_env_info()
        if not self.args.is_force: # change
            self.args.n_agents = env_info["n_agents"]
        else:
            self.args.n_agents = 1
            self.args.real_n_agents = env_info["n_agents"]

        if self.args.is_discrete:
            if not self.args.is_hierarchical:
                self.args.n_actions = env_info["n_actions"]
            else:
                assert 'n_actions' not in env_info.keys()

            scheme = {"state": {"vshape": env_info["state_shape"]}, "obs": {"vshape": env_info["obs_shape"] * env_info["n_agents"], "group": "agents"}, # change
                      "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                      "avail_actions": {"vshape": (self.args.n_actions,), "group": "agents", "dtype": th.int},
                      "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": th.uint8}}
            if self.args.is_hierarchical:
                scheme["init"] = {"vshape": (1,), "group": "agents", "dtype": th.uint8}

            preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])}
        else:
            self.args.action_shape = env_info["action_shape"]
            scheme = {"state": {"vshape": env_info["state_shape"]}, "obs": {"vshape": env_info["obs_shape"] * env_info["n_agents"], "group": "agents"}, # change, danger
                      "actions": {"vshape": env_info["action_shape"], "group": "agents"},
                      "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": th.uint8}}
            preprocess = {}

        self.args.state_shape = env_info["state_shape"]
        groups = {"agents": self.args.n_agents}

        self.buffer = ReplayBuffer(
            scheme,
            groups,
            self.args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if self.args.buffer_cpu_only else self.args.device)

        # Setup multiagent controller here
        self.mac = mac_REGISTRY[self.args.mac](self.buffer.scheme, self.args)

        # Give runner the scheme
        if not self.args.is_force:
            self.runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=self.mac)
        else:
            self.runner.setup(scheme=scheme, groups=groups, low_level_groups={"agents": self.args.real_n_agents}, preprocess=preprocess, mac=self.mac)

        # Learner
        self.learner = le_REGISTRY[self.args.learner](self.mac, self.buffer.scheme, self.args)
        if self.args.use_cuda:
            self.learner.cuda()

        # Load the pretrained model to the Learner
        if self.args.pretrained_path != "":
            assert os.path.isdir(self.args.pretrained_path), "Checkpoint directiory {} doesn't exist!".format(self.args.pretrained_path)
            timesteps = []
            # Go through all files
            for name in os.listdir(self.args.pretrained_path):
                full_name = os.path.join(self.args.pretrained_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if self.args.load_step == -1:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - self.args.load_step))

            model_path = os.path.join(self.args.pretrained_path, str(timestep_to_load))

            print("Loading model from {}".format(model_path))
            self.learner.load_models(model_path)

        self.episode = 0
        self.last_test_T = -self.args.test_interval - 1
        self.model_save_time = 0


    def run(self, training_steps=None, low_level_mac=None, option_list=None):
        # Run and train
        self.run_sequential(training_steps, low_level_mac, option_list)

        # Clean up after finishing
        # print("Exiting Main")
        #
        # print("Stopping all threads")
        # for t in threading.enumerate():
        #     if t.name != "MainThread":
        #         print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
        #         t.join(timeout=1)
        #         print("Thread joined")
        #
        # print("Exiting script")


    def run_sequential(self, training_steps, low_level_mac, option_list):

        if self.args.pretrained_path != "" and self.args.evaluate:
            self.runner.log_train_stats_t = self.runner.t_env
            self.evaluate_sequential(low_level_mac, option_list)
            return

        # start training
        if training_steps is None:
            training_steps = self.args.t_max
            assert self.runner.t_env == 0

        print("Beginning training for {} timesteps".format(training_steps))
        step_limit = training_steps + self.runner.t_env

        if self.runner.t_env >= self.args.t_max:
            print("Already reach the training step limit!!!")
            return

        while self.runner.t_env <= step_limit:
            # Run for a whole episode at a time
            print("###############################Episode {} begins!###############################".format(self.episode))

            if self.args.mac == "maddpg_mac":
                self.mac.scale_noise(scale=(self.args.final_noise +
                                    (self.args.init_noise - self.args.final_noise) * (self.args.t_max - self.runner.t_env) / float(self.args.t_max)))
                self.mac.reset_noise()
                run_time = 1
            else:
                run_time = self.args.buffer_size

            returns, lengths = [], []
            for _ in range(run_time):
                episode_batch, info = self.runner.run(low_level_mac=low_level_mac, test_mode=False, option_list=option_list)
                self.buffer.insert_episode_batch(episode_batch)
                returns.append(info['return'])
                lengths.append(info['length'])

            print("The average return is {} and episodic length is {}!".format(np.mean(returns), np.mean(lengths)))
            self.learner.write_summary({'return': np.mean(returns), 'length': np.mean(lengths)}, self.episode)

            if self.buffer.can_sample(self.args.batch_size):
                episode_sample = self.buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t] # the dtype of max_ep_t is th.long, so it can work

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                self.learner.train(episode_sample, self.runner.t_env, self.episode)

            # Execute test runs once in a while
            n_test_runs = self.args.test_nepisode
            if (self.runner.t_env - self.last_test_T) / self.args.test_interval >= 1.0:

                print("t_env: {} / {}".format(self.runner.t_env, self.args.t_max))

                self.last_test_T = self.runner.t_env
                eva_returns, eva_lengths = [], []
                for _ in range(n_test_runs):
                    _, info = self.runner.run(test_mode=True, low_level_mac=low_level_mac, option_list=option_list)
                    eva_returns.append(info['return'])
                    eva_lengths.append(info['length'])
                print("The average return is {} and episodic length is {}!".format(np.mean(eva_returns), np.mean(eva_lengths)))
                self.learner.write_summary({'eva_return': np.mean(eva_returns), 'eva_length': np.mean(eva_lengths)}, self.episode)

            if self.args.save_model and (self.runner.t_env - self.model_save_time >= self.args.save_model_interval or self.model_save_time == 0):
                self.model_save_time = self.runner.t_env
                save_path = os.path.join(self.args.ckpt_path, str(self.episode))
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                print("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                self.learner.save_models(save_path)

            self.episode += self.args.batch_size_run * run_time

        print("Finished Training in this stage!!!")
        if self.runner.t_env >= self.args.t_max:
            self.runner.close_env()
            print("Finished Training!!!")

    def evaluate_sequential(self, low_level_mac, option_list):
        print("Evaluation begins!!!!!!")
        returns, lengths = [], []
        for _ in range(self.args.test_nepisode):
            _, info = self.runner.run(test_mode=True, low_level_mac=low_level_mac, option_list=option_list)
            returns.append(info['return'])
            lengths.append(info['length'])
        print("The average return is {} and episodic length is {}!".format(np.mean(returns), np.mean(lengths)))
        self.runner.close_env()
        print("Finished Evaluating")

    def _args_sanity_check(self, config):

        # set CUDA flags
        # config["use_cuda"] = True # Use cuda whenever possible!
        if config["use_cuda"] and not th.cuda.is_available():
            config["use_cuda"] = False
            print("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

        if config["test_nepisode"] < config["batch_size_run"]:
            config["test_nepisode"] = config["batch_size_run"]
        else:
            config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

        return config
