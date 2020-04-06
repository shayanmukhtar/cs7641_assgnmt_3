import numpy as np
import gym_maze
import gym
import hiive.mdptoolbox.mdp as mdp
import matplotlib.pyplot as plt
import seaborn
import math


def run_value_iteration(params):
    transition_model = params['T']
    reward_matrix = params['R']
    gammas = params['gamma'] # assume a list, as this is a hyperparameter
    epsilons = params['epsilon']
    num_states = transition_model.shape[1]

    plt.figure()

    runs = []
    for gamma in gammas:
        for epsilon in epsilons:
            value_iterator = mdp.ValueIteration(transition_model, reward_matrix, gamma=gamma, epsilon=epsilon)
            value_iterator.run()
            errors = []
            run_dict = {
                'gamma': gamma,
                'epsilon': epsilon,
                'mdp': value_iterator
            }
            runs.append(run_dict)
            for run_stat in value_iterator.run_stats:
                errors.append(run_stat['Error'])

            label = "Gamma: " + str(gamma) + " Epsilon: " + str(epsilon)
            plt.plot(errors, label=label)

    error_plot_title = params['error_plot_title_value']
    path = params['path_value']

    plt.title(error_plot_title)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.xlim(params['error_plot_x_axes_value'])
    plt.savefig(path + error_plot_title)

    # plot the value map of environment if possible for all the hyperparameters
    for run in runs:
        if params['make_heat_map']:
            plt.figure()
            num_states_1d = int(np.sqrt(num_states))
            state_values = np.reshape(run['mdp'].V, (num_states_1d, num_states_1d))
            figsize = params['figsize']
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            h_map = seaborn.heatmap(state_values, fmt='0.2g', annot=False, ax=ax)
            title = params['heat_map_name'] + " Gamma: " + str(run['gamma']) + " Epsilon: " + str(run['epsilon'])
            h_map.title.set_text(title)
            plt.savefig(path + title + ".png")
            plt.close()

    # evaluate the learners
    plt.figure()
    env = params['env']
    for run in runs:
        cum_reward = 0
        rewards = []
        for episode in range(1000):
            observation = env.reset()
            action = run['mdp'].policy[observation]
            while True:
                observation, reward, done, info = env.step(action)
                cum_reward += reward
                if done:
                    break
                action = run['mdp'].policy[observation]
            rewards.append(cum_reward)
        label = "Gamma: " + str(run['gamma']) + " Epsilon: " + str(run['epsilon'])
        plt.plot(rewards, label=label)

    error_plot_title = params['value_iteration_policy_eval']
    path = params['path_value']

    plt.title(error_plot_title)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig(path + error_plot_title)
