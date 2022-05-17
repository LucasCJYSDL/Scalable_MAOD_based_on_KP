import argparse
import os
from agents.multiple.MARL_agent import MARLAgent
from agents.spectral.spectral_agent import SpectralAgent
from agents.hierarchy.single_hierarchical_agent import SingleHierarchicalAgent
from agents.hierarchy.multiple_hierarchical_agent import MultipleHierarchicalAgent

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0) # required
    parser.add_argument('--env_id', type=str, default='PointLongCorridor-v0')
    parser.add_argument('--agent_num', type=int, default=2)
    parser.add_argument('--option_num', type=int, default=2)
    parser.add_argument('--high_level_alg', type=str, default='hmappo') # {hmappo}
    parser.add_argument('--low_level_alg', type=str, default='mappo') # {mappo, maddpg, maa2c}
    parser.add_argument('--single_alg', type=str, default='sac') # {sac}
    parser.add_argument('--mode', type=str, default='multiple') # {none, single, multiple}
    parser.add_argument('--gpu', type=str, default='2')

    return parser.parse_args()

def main(args=get_args()):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == 'none': # complete the tasks without options
        low_level_agent = MARLAgent(env_dict={"task": args.env_id, "type": "continuous"}, alg=args.low_level_alg, seed=args.seed)
        low_level_agent.learn()
        return
    
    if args.mode == 'single': # complete the tasks with single-agent options
        spectral_agent = SpectralAgent(env_id=args.env_id, seed=args.seed, agent_num=args.agent_num)
        # collect the initial single-agent option list: two single-agent options for each agent
        option_list = spectral_agent.get_option_list(mode=args.mode) # [(agent_0_min: np.ndarray, agent_0_max), (agent_1_min, agent_1_max), ...]
        # build the hierarchical agent with the option list
        hierarchical_agent = SingleHierarchicalAgent(args, option_list)
        # extend the option list
        hierarchical_agent.update_option_list()
        # set up the agents and start training
        hierarchical_agent.setup_agents()
        hierarchical_agent.learn()
        return

    if args.mode == 'multiple': # complete the tasks with multiple-agent options
        spectral_agent = SpectralAgent(env_id=args.env_id, seed=args.seed, agent_num=args.agent_num)
        # collect the initial multiple-agent option list with two elements
        option_list = spectral_agent.get_option_list(mode=args.mode) # [(min_agent_0: np.ndarray, min_agent_1, ...), (max_agent_0, max_agent_1, ...)]
        # build the hierarchical agent with the option list
        hierarchical_agent = MultipleHierarchicalAgent(args, option_list)
        # extend the option list
        hierarchical_agent.update_option_list()
        # set up the agents and start training
        hierarchical_agent.setup_agents()
        hierarchical_agent.learn()
        return


if __name__ == '__main__':
    main()
