import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import pandas as pd


def plot():
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(9, 6))
    plt.xlabel('Training Episode', fontsize=15)
    plt.ylabel('Step', fontsize=15)
    plt.tick_params(labelsize=15)

    agent_list = ['COMA', 'CWQMIX', 'OWQMIX', 'MAVEN']
    color_list = ['red', 'green', 'blue', 'orange']
    seed_list = [0, 10, 100]
    mean_list = []
    std_list = []

    for agent_name in agent_list:
        agent_step_list = []
        for seed_num in seed_list:
            temp_dir = './result/' + agent_name + '/seed_{}/'.format(str(seed_num)) + 'step.csv'
            print("Read from ", temp_dir)
            temp_data = pd.read_csv(temp_dir)
            seed_step_list = np.array(temp_data['Value'])
            agent_step_list.append(seed_step_list)
        mean_list.append(np.mean(agent_step_list, axis=0))
        std_list.append(np.std(agent_step_list, axis=0))

    x_list = np.array(temp_data['Step']) # should be the same for each plot

    for agent_idx in range(len(agent_list)):
        agent_color = color_list[agent_idx]
        agent_mean = mean_list[agent_idx]
        agent_std = std_list[agent_idx]
        plt.plot(x_list, agent_mean, ls='-', lw=2.0, color=agent_color, label=agent_list[agent_idx])

        lower = list(map(lambda x: x[0] - x[1], zip(agent_mean, agent_std)))
        upper = list(map(lambda x: x[0] + x[1], zip(agent_mean, agent_std)))
        plt.fill_between(x_list, lower, upper, color=agent_color, alpha=0.2)

    plt.grid(True)
    plt.legend(fontsize=15)

    plt.savefig('MARL_baselines.png')


if __name__ == '__main__':
    plot()
