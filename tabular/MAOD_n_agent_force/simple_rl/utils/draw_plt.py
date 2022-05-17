import matplotlib.pyplot as plt
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

if __name__ == '__main__':

    data_dict = {
        'agent_1': [[0,1,2,3], [4,5,6,7], [2,3,4,5]],
        'agent_2': [[10,11,12,13], [14,15,16,17], [12,13,14,15]]
    }

    plot(data_dict, name='test_value', agent_color_list=['blue', 'orange'], task_name='test_task')