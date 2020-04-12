import hiive.mdptoolbox.example as mdp_examples
import value_iteration
import policy_iteration
import q_learning
import numpy as np
import time


class obs_space(object):
    def __init__(self, n):
        self.n = n


class action_space(object):
    def __init__(self, n):
        self.n = n


class env_class(object):
    def __init__(self, T, R, num_steps=100):
        self.T = T
        self.R = R
        self.state = 0
        self.num_states = T.shape[1]
        self.all_states = [s for s in range(0, self.num_states)]
        self.num_steps = num_steps
        self.current_step = 0
        self.observation_space = obs_space(self.num_states)
        self.action_space = action_space(T.shape[0])

    def reset(self):
        self.state = 0
        self.current_step = 0
        return self.state

    def step(self, action):
        # T is of form [A, S, S_Prime]
        transition_matrix = self.T[action, self.state, :]
        next_state = np.random.choice(a=self.all_states, p=transition_matrix)
        reward = self.R[self.state, action]

        # keep track of state
        self.state = next_state

        # return the standard gym variables
        observation = self.state
        info = {"None", 0.0}
        done = False

        self.current_step += 1
        if self.current_step > self.num_steps:
            done = True

        return observation, reward, done, info


def run_forest_management():
    forest_management_map = mdp_examples.forest(S=16, p=0.1)
    forest_management_env = env_class(T=forest_management_map[0], R=forest_management_map[1])
    print("")
    params = {
        'T': forest_management_map[0],
        'R': forest_management_map[1],
        'gamma': [0.8, 0.9, 0.95, 0.975, 0.99],
        'epsilon': [0.1, 0.01],
        'figsize': 10,
        'env': forest_management_env,
        'path_value': "./Value_Iteration/",
        'value_iteration_policy_eval': 'Forest Management - Policy Evaluation over 1000 Episodes for Value Iteration',
        'error_plot_title_value': 'Forest Management - Iterations vs Error in State Values for Value Iteration',
        'error_plot_x_axes_value': (0, 30),
        'make_heat_map': True,
        'heat_map_name': 'Forest Management - State Values',
        'path_policy': "./Policy_Iteration/",
        'error_plot_title_policy': 'Forest Management - Iterations vs Error in State Values for Policy Iteration',
        'error_plot_x_axes_policy': (-1, 30),
        'make_policy_heat_map': True,
        'state_action_map': lambda s: ['W', 'C'][s],
        'policy_heat_map_name': 'Forest Management - Policy to State Mapping from Policy Iteration',
        'policy_iteration_policy_eval': 'Forest Management - Policy Evaluation over 1000 Episodes for Policy Iteration',
        'q_gamma': [0.975], #[0.95, 0.975, 0.99],
        'q_epsilon': [1e-5],
        'q_alphas': [0.1], #[0.01, 0.001],
        'q_radrs': [0.9999], #[0.999, 0.99999],
        'q_learning_error_plot_title_value': 'Forest Management - Episode vs Error in Q Values for Q Learning',
        'q_path_value': './Q_Learning/',
        'q_episodes': 10000,
        'q_policy_eval': 'Forest Management - Policy Evaluation over 1000 Episodes for Q Learning',
        'reward_shape': None,
        'q_value_heatmap': 'Forest Management - Q-Values Heatmap',
        'use_r_max': False
    }

    seconds_now = time.time()
    value_iteration.run_value_iteration(params)
    print("Value iteration took: " + str(time.time() - seconds_now))
    seconds_now = time.time()
    policy_iteration.run_policy_iteration(params)
    print("Policy iteration took: " + str(time.time() - seconds_now))
    q_learning.run_q_learning(params)


def main():
    run_forest_management()


if __name__ == '__main__':
    main()