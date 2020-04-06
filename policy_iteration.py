import numpy as np
import matplotlib.pyplot as plt
import hiive.mdptoolbox.mdp as mdp
import seaborn


def run_policy_iteration(params):
    transition_model = params['T']
    reward_matrix = params['R']
    gammas = params['gamma']  # assume a list, as this is a hyperparameter
    epsilons = params['epsilon']
    num_states = transition_model.shape[1]

    plt.figure()

    runs = []
    for gamma in gammas:
        policy_iterator = mdp.PolicyIteration(transition_model, reward_matrix, gamma=gamma)
        policy_iterator.run()
        errors = []
        run_dict = {
            'gamma': gamma,
            'mdp': policy_iterator
        }
        runs.append(run_dict)
        for run_stat in policy_iterator.run_stats:
            errors.append(run_stat['Error'])

        label = "Gamma: " + str(gamma)
        plt.plot(errors, label=label)

    error_plot_title = params['error_plot_title_policy']
    path = params['path_policy']

    plt.title(error_plot_title)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.xlim(params['error_plot_x_axes_policy'])
    plt.savefig(path + error_plot_title)

    for run in runs:
        if params['make_policy_heat_map']:
            plt.figure()
            num_states_1d = int(np.sqrt(num_states))
            state_action_values = np.reshape(run['mdp'].policy, (num_states_1d, num_states_1d))
            state_action_map = params['state_action_map']
            annotations_func = np.vectorize(state_action_map)
            annotations = annotations_func(state_action_values)
            figsize = params['figsize']
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            h_map = seaborn.heatmap(state_action_values, annot=annotations, fmt='', cbar=False, ax=ax)
            title = params['policy_heat_map_name'] + " Gamma: " + str(run['gamma'])
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
        label = "Gamma: " + str(run['gamma'])
        plt.plot(rewards, label=label)

    error_plot_title = params['policy_iteration_policy_eval']
    path = params['path_policy']

    plt.title(error_plot_title)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig(path + error_plot_title)


def main():
    print('')


if __name__ == '__main__':
    main()
