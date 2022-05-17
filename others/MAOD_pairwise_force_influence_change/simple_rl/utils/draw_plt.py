import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os

def plot(data_dict, name, agent_color_list, task_name, option_type):

    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(9, 6))
    plt.xlabel('Training Episode', fontsize=15)
    plt.ylabel(name, fontsize=16)
    plt.tick_params(labelsize=14)

    agent_idx = 0
    for agent_name in data_dict.keys():
        agent_color = agent_color_list[agent_idx]
        agent_idx += 1
        agent_data = np.array(data_dict[agent_name])
        agent_mean = []
        agent_std = []
        epi_list = []
        instance_num = agent_data.shape[0]
        episode_num = agent_data.shape[1]
        for epi in range(episode_num):
            temp_mean = agent_data[:, epi].mean()
            temp_std = agent_data[:, epi].std()
            agent_mean.append(temp_mean)
            agent_std.append(temp_std)
            epi_list.append(epi+1)
        # print("1: ", agent_data)
        # print("2: ", agent_mean)
        # print("3: ", agent_std)
        plt.plot(epi_list, agent_mean, ls='-', lw=2.0, color=agent_color, label=agent_name)

        lower = list(map(lambda x: x[0] - x[1], zip(agent_mean, agent_std)))
        upper = list(map(lambda x: x[0] + x[1], zip(agent_mean, agent_std)))
        plt.fill_between(epi_list, lower, upper, color=agent_color, alpha=0.2)

    plt.grid(True)
    plt.legend(fontsize=15)
    # plt.subplots_adjust(top=0.98, bottom=0.13, right=0.98, left=0.117, hspace=0.2, wspace=0.2)
    # plt.show()
    file_dir = '../plots'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    plt.savefig(file_dir + '/' + task_name + '_' + option_type + '_' + name)


def plot_collision_change(collision_ratio, step_list, collision_list):
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(9, 6))
    plt.tick_params(labelsize=14)

    ratio_array = np.array(collision_list)/np.array(step_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(collision_ratio, step_list, ls='-', marker='o', lw=2.0, color='blue', label='# of steps')
    lns2 = ax.plot(collision_ratio, collision_list, ls='-', marker='o', lw=2.0, color='orange', label='# of collisions')

    ax2 = ax.twinx()
    ax2.bar(collision_ratio, ratio_array, width=0.05, fc='g', alpha=0.5)
    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=15)

    ax.grid()
    ax.set_xlabel("Probability of Collisions", fontsize=15)
    ax.set_ylabel("Number", fontsize=15)
    ax2.set_ylabel("Ratio", fontsize=15)
    ax.set_ylim(0, 120)
    ax2.set_ylim(0, 0.8)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    for a, b in zip(collision_ratio, ratio_array):
        ax2.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=15)

    plt.savefig('collision_change_rlt.png')
    
    
if __name__ == '__main__':

    # data_dict = {
    #     'agent_1': [[0,1,2,3], [4,5,6,7], [2,3,4,5]],
    #     'agent_2': [[10,11,12,13], [14,15,16,17], [12,13,14,15]]
    # }
    #
    # plot(data_dict, name='test_value', agent_color_list=['blue', 'orange'], task_name='test_task')

    cr = [0.2, 0.4, 0.6]
    sn = [40, 50, 60]
    cn = [5, 6, 20]
    plot_collision_change(cr, sn, cn)