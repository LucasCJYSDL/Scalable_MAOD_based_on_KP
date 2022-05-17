import numpy as np
import os
import collections
from copy import deepcopy
from typing import Dict, Optional, List, Tuple
import torch as th
import yaml

from agents.multiple.run import RunAgent
from agents.multiple.controllers.basic_controller import BasicMAC
from agents.spectral.learners.option_wrapper import Option


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


class MARLAgent(object):
    def __init__(self, env_dict: Dict, alg: str, seed: int):
        self.env_dict = env_dict
        self.alg = alg
        self.seed = seed

        # Get the defaults from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        # Load algorithm and env base configs
        with open(os.path.join(os.path.dirname(__file__), "config", "algs", "{}.yaml".format(self.alg)), "r") as f:
            alg_config = yaml.load(f, Loader=yaml.Loader)
        config_dict = recursive_dict_update(config_dict, alg_config)
        # print(config_dict)
        config_dict['env'] = self.env_dict['task']
        config_dict['env_args']['type'] = self.env_dict['type']
        config_dict['seed'] = self.seed
        if 'n_actions' in self.env_dict.keys():
            config_dict['n_actions'] = self.env_dict['n_actions']
        print(config_dict)

        config = config_copy(config_dict)
        np.random.seed(config["seed"])
        th.manual_seed(config["seed"])

        self.config = config

        # run the framework
        self.run_agent = RunAgent(config)

    def learn(self, training_steps=None, low_level_mac: Optional[BasicMAC]=None, option_list=None):

        self.run_agent.run(training_steps=training_steps, low_level_mac=low_level_mac, option_list=option_list)

    def get_max_training_steps(self):

        return self.config['t_max']



if __name__ == '__main__':
    test_cls = MARLAgent(env_dict={"task": "Point4Rooms-v0", "type": "continuous"}, alg="mappo", seed=0)
    test_cls.learn()

