import numpy as np
import gym_maze
import gym
import hiive.mdptoolbox.mdp as mdp
import matplotlib.pyplot as plt
import seaborn
import math

class Q_Learner(object):
    def __init__(self, num_states, num_actions, alpha, gamma, rar, radr, epsilon, dyna=0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.q_values = np.ndarray((num_states, num_actions))
        self.c_values = np.ndarray((num_states, num_actions))
        self.q_values[:] = 0
        self.state = 0
        self.action = 0
        self.epsilon = epsilon
        self.dyna = dyna
        self.exp_tuples = np.zeros((1, 4)) # s, a, r, s' tuples for dyna

    # called after observation is taken, gives you s_prime and r
    # internally we keep track of s and a
    def determine_action(self, s_prime, reward):
        # the below lines implement epsilon greedy, which won't converge to the optimal q values, only
        # to the optimal policy - but we need q values b
        random_action_determine = np.random.random()
        if random_action_determine < self.rar:
            a_prime = np.random.randint(low=0, high=self.num_actions)
        else:
            a_prime = np.argmax(a=self.q_values[s_prime, :])

        # self.q_values[self.state, self.action] = self.q_values[self.state, self.action] + \
        #     self.alpha*(reward + self.gamma*self.q_values[s_prime, a_prime] - self.q_values[self.state, self.action])
        s = self.state
        a = self.action
        one_minus_alpha = 1.0 - self.alpha
        arg_max = np.argmax(a=self.q_values[s_prime, :])
        self.q_values[s, a] = (one_minus_alpha * self.q_values[s, a]) + (self.alpha * (reward + (self.gamma * self.q_values[s_prime, arg_max])))

        # include the rewards that mean something
        if reward >= 0.5:
            self.exp_tuples = np.vstack((self.exp_tuples, [self.state, self.action, s_prime, reward]))

        if self.dyna > 0 and self.exp_tuples.shape[0] > 1:
            # And draw from them randomly to hallucinate - do all the random number generation
            # one outside the for loop to save some time
            random_experiences = np.random.randint(low=1, high=self.exp_tuples.shape[0], size=self.dyna)
            random_hallucinations = self.exp_tuples[random_experiences, :]
            random_states = random_hallucinations[:, 0].astype(int)
            random_actions = random_hallucinations[:, 1].astype(int)
            random_s_primes = random_hallucinations[:, 2].astype(int)
            random_rewards = random_hallucinations[:, 3]

            # And update the Q tables based on the hallucinations
            for hallucination in range(0, self.dyna):
                # And now update the Q table with the hallucinated experience Tuple
                random_state = random_states[hallucination]
                random_action = random_actions[hallucination]
                inferred_s_prime = random_s_primes[hallucination]
                inferred_reward = random_rewards[hallucination]
                inferred_arg_max = np.argmax(a=self.q_values[inferred_s_prime, :])
                self.q_values[random_state, random_action] = (one_minus_alpha * self.q_values[random_state, random_action]) + (
                            self.alpha * (inferred_reward + (self.gamma * self.q_values[inferred_s_prime, inferred_arg_max])))

        self.state = s_prime
        self.action = a_prime
        self.rar = self.rar * self.radr
        return a_prime

    def set_state_and_query(self, state):
        self.state = state
        random_action_determine = np.random.random()
        if random_action_determine < self.rar:
            a_prime = np.random.randint(low=0, high=self.num_actions)
        else:
            a_prime = np.argmax(a=self.q_values[state, :])

        self.action = a_prime
        return a_prime

    def set_state(self, state):
        self.state = state


def run_q_learning(params):
    env = params['env']
    gammas = params['q_gamma']
    epsilons = params['q_epsilon']
    alphas = params['q_alphas']
    radrs = params['q_radrs']
    max_episodes = params['q_episodes']

    # lets go model free for q-learning to make it more fun
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Make us a list of Q-Learners, all with their respective hyperparameters
    q_learners = []
    for gamma in gammas:
        for alpha in alphas:
            for radr in radrs:
                for epsilon in epsilons:
                    q_learner = Q_Learner(num_states, num_actions, alpha, gamma, 1.0, radr, epsilon=epsilon, dyna=0)
                    q_learners.append(q_learner)

    # run through all the q learners, but also plot their progress in convergence
    plt.figure()
    for q_learner in q_learners:
        if params['use_r_max']:
            q_learner.q_values[:, :] = 1
        errors = []
        for episode in range(max_episodes):
            observation = env.reset()
            action = q_learner.set_state_and_query(observation)
            old_q_values = q_learner.q_values.copy()
            while True:  # loop for each step in the episode
                # env.render()
                observation, reward, done, info = env.step(action)
                # reward_perceived = params['reward_shape'](observation, reward, done, q_learner, info)
                action = q_learner.determine_action(observation, reward)
                if done:
                    print(str(episode))
                    break
            errors.append(abs(old_q_values - q_learner.q_values).max())
        label = "G: " + str(q_learner.gamma) + " A: " + str(q_learner.alpha) + " R: " + str(q_learner.radr)
        plt.plot(errors, label=label)
        print("Final Rar: " + str(q_learner.rar))

    error_plot_title = params['q_learning_error_plot_title_value']
    path = params['q_path_value']

    plt.title(error_plot_title)
    plt.xlabel('Episodes')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(path + error_plot_title)
    plt.close()

    # make a heatmap of the states q values
    if params['make_heat_map']:
        plt.figure()
        num_states_1d = int(np.sqrt(num_states))
        state_value_flat = np.apply_along_axis(max, axis=1, arr=q_learners[-1].q_values)
        state_values = np.reshape(state_value_flat, (num_states_1d, num_states_1d))
        figsize = params['figsize']
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        h_map = seaborn.heatmap(state_values, fmt='0.2g', annot=False, ax=ax)
        title = params['q_value_heatmap']
        h_map.title.set_text(title)
        plt.savefig(path + title + ".png")
        plt.close()

    # evaluate the learners
    plt.figure()
    for q_learner in q_learners:
        rewards = []
        cum_reward = 0

        # turn off learning - just do the policy evaluation
        q_learner.alpha = 0
        q_learner.rar = 0
        for episode in range(1000):
            observation = env.reset()
            action = q_learner.set_state_and_query(observation)
            while True:  # loop for each step in the episode
                # env.render()
                observation, reward, done, info = env.step(action)
                action = q_learner.determine_action(observation, reward)
                cum_reward += reward
                if done:
                    break
            rewards.append(cum_reward)
        label = "G: " + str(q_learner.gamma) + " A: " + str(q_learner.alpha) + " R: " + str(q_learner.radr)
        plt.plot(rewards, label=label)

    error_plot_title = params['q_policy_eval']
    path = params['q_path_value']

    plt.title(error_plot_title)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig(path + error_plot_title)
