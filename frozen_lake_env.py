import gym
import numpy as np
import value_iteration
import policy_iteration
import hiive.mdptoolbox.mdp as mdp
from gym.envs.toy_text.frozen_lake import generate_random_map
import q_learning
import math

def frozen_lake():
    np.random.seed(786)

    frozen_lake_map = generate_random_map(32, 0.90)
    frozen_lake_env = gym.make("FrozenLake-v0", desc=frozen_lake_map)
    frozen_lake_env._max_episode_steps = 1000000
    NUM_ACTIONS = frozen_lake_env.action_space.n
    NUM_STATES = frozen_lake_env.observation_space.n

    # Make a transition model matrix
    transition_model = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))
    reward_matrix = np.zeros((NUM_STATES, NUM_ACTIONS))
    for state in range(NUM_STATES):
        for action in range(NUM_ACTIONS):
            for next_state_tuple in frozen_lake_env.env.P[state][action]:
                next_state = next_state_tuple[1]
                probability = next_state_tuple[0]
                reward = next_state_tuple[2]
                transition_model[action, state, next_state] += probability
                reward_matrix[state, action] = reward

    params = {
        'T': transition_model,
        'R': reward_matrix,
        'gamma': [0.9, 0.95, 0.975, 0.99],
        'epsilon': [0.1, 0.01, 0.001],
        'figsize': 15,
        'env': frozen_lake_env,
        'path_value': "./Value_Iteration/",
        'value_iteration_policy_eval': 'Frozen Lake - Policy Evaluation over 1000 Episodes for Value Iteration',
        'error_plot_title_value': 'Frozen Lake - Iterations vs Error in State Values for Value Iteration',
        'error_plot_x_axes_value': (0, 60),
        'make_heat_map': True,
        'heat_map_name': 'Frozen Lake - State Values',
        'path_policy': "./Policy_Iteration/",
        'error_plot_title_policy': 'Frozen Lake - Iterations vs Error in State Values for Policy Iteration',
        'error_plot_x_axes_policy': (-1, 10),
        'make_policy_heat_map': True,
        'state_action_map': lambda s: ['L', 'D', 'R', 'U'][s],
        'policy_heat_map_name': 'Frozen Lake - Policy to State Mapping from Policy Iteration',
        'policy_iteration_policy_eval': 'Frozen Lake - Policy Evaluation over 1000 Episodes for Policy Iteration',
        'q_gamma': [0.99], #[0.95, 0.975, 0.99],
        'q_epsilon': [1e-5],
        'q_alphas': [0.1], #[0.01, 0.001],
        'q_radrs': [0.999999], #[0.999, 0.99999],
        'q_learning_error_plot_title_value': 'Frozen Lake - Episode vs Error in Q Values for Q Learning - Reward Shaping Euclidean Distance',
        'q_path_value': './Q_Learning/',
        'q_episodes': 300000,
        'q_policy_eval': 'Frozen Lake - Policy Evaluation over 1000 Episodes for Q Learning - Reward Shaping Euclidean Distance',
        'reward_shape': shape_reward_pour_la_lac,
        'q_value_heatmap': 'Q-Values Heatmap - Reward Shaping Euclidean Distance',
        'use_r_max': False
    }
    frozen_lake_env.render()
    # value_iteration.run_value_iteration(params)
    # policy_iteration.run_policy_iteration(params)
    q_learning.run_q_learning(params)


def shape_reward_ng_style(observation, reward, done, q_learner, info=None):
    x_pos_new = observation % 32
    y_pos_new = observation // 32
    manhattan_dist = (32.0 - x_pos_new) + (32.0 - y_pos_new)
    reward_new = -1.0 * (manhattan_dist / 0.7)
    if done is True and reward < 0.5:
        if 'TimeLimit.truncated' in info.keys():
            if not info['TimeLimit.truncated']:
                # this is a hole, assign the worst case
                reward_new = -50000
        else:
            # this is a hole, assign the worst case
            reward_new = -50000
    return reward_new


def shape_reward_pour_la_lac(observation, reward, done, q_learner, info=None):
    if not done:
        x_pos_new = observation % 32
        y_pos_new = observation // 32
        distance_new = math.sqrt((31 - x_pos_new)**2 + (31 - y_pos_new)**2) / 46 # ~ sqrt(32**2 + 32 **2)

        x_pos_old = q_learner.state % 32
        y_pos_old = q_learner.state // 32
        distance_old = math.sqrt((31 - x_pos_old) ** 2 + (31 - y_pos_old) ** 2) / 46  # ~ sqrt(32**2 + 32 **2)

        # if distance_new > distance_old:
        #     # we've gone further away from the goal - thats negative reward
        #     reward_new = reward + (distance_old - distance_new)
        # else:
        #     # we've come closer - thats positive reward
        reward_new = reward + ((distance_old - distance_new) / 2.0)

        return reward_new
    else:
        return reward


def main():
    frozen_lake()


if __name__ == '__main__':
    main()