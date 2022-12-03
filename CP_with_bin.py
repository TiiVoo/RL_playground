import gym
import numpy as np
from scipy import interpolate
import random


class CartpoleBin:
    def __init__(self, mode, interpolation_table_y, interpolation_table_x, epsilon, alpha):
        match mode:
            case 'rendermode0':
                self.env = gym.make('CartPole-v0')
            case 'rendermode1':
                self.env = gym.make('CartPole-v0', render_mode="rgb_array")
            case 'rendermode2':
                self.env = gym.make('CartPole-v0', render_mode="human")
            case _:
                print('specify render, default selected')
                self.env = gym.make('CartPole-v0')

        # initialize all variables
        self.env.reset()
        self.done = False
        self.epsilon = epsilon
        self.obs_boxed = []
        self.state_new = 0
        self.alpha = alpha
        self.action = self.env.action_space.sample()

        assert (self.action == 1 or self.action == 0)
        #
        assert (len(interpolation_table_y[0]) == 10), "at the moment class works only with discretization of 10."

        # initialize Q matrix
        num_actions = self.env.action_space.n
        num_states = len(interpolation_table_y[0]) ** len(interpolation_table_y)
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

        # initialize interpolation functions for boxed observations
        self.interp_fct = []
        for count, y in enumerate(interpolation_table_y):
            self.interp_fct.append(interpolate.interp1d(interpolation_table_x[count], y,
                                                        kind='nearest', fill_value="extrapolate"))

    def _single_step(self, action):
        self.state = self.state_new
        [self.observation, reward, self.done, _, _] = self.env.step(action)

        if self.done:
            self.reward = -100
        else:
            self.reward = 0

        self._box_observation(self.observation)  # get obs boxed
        self._find_new_state(self.obs_boxed)

    def _box_observation(self, observation):
        self.obs_boxed = []
        for count, obs in enumerate(observation):
            self.obs_boxed.append(self.interp_fct[count](obs))

    def _find_new_state(self, obs_boxed):
        self.state_new = int("".join(map(lambda x: str(int(x)), obs_boxed)))

    def _choose_action_epsilon_greedy(self, state):

        if random.random() < self.epsilon:
            self.action = self.env.action_space.sample()
        else:
            self.action = np.argmax(self.Q[state])

    def _update_Q(self):

        self.Q[self.state][self.action] += self.alpha * (
                self.reward + self.epsilon * np.max(self.Q[self.state_new])
                - self.Q[self.state][self.action])

    def q_learning_episode(self):
        self.env.reset()
        max_iter = 1000
        count = 0
        while not self.done and count < max_iter:
            self._single_step(self.action)  # get new state
            self._update_Q()
            self._choose_action_epsilon_greedy(self.state)  # get action
            count += 1


if __name__ == '__main__':
    n_intp = 10
    interpolation_table_y = [np.linspace(0, 9, n_intp),
                                  np.linspace(0, 9, n_intp),
                                  np.linspace(0, 9, n_intp),
                                  np.linspace(0, 9, n_intp)]
    # position, speed, angle, angular speed
    interpolation_table_x = [np.linspace(-2.4, 2.4, n_intp),
                                  np.linspace(-3, 3, n_intp),
                                  np.linspace(-.2095, .2095, n_intp),
                                  np.linspace(-1 , 1, n_intp)]
    epsilon = 0.1
    alpha = 0.3
    cp = CartpoleBin('rendermode0',
                          interpolation_table_y,
                          interpolation_table_x,
                          epsilon, alpha)

    for i in range(5000):
        cp.q_learning_episode()
        cp.env.reset()
        cp.done = False
        print(i)

    cp2 = CartpoleBin('rendermode2',
                     interpolation_table_y,
                     interpolation_table_x,
                     epsilon, alpha)
    cp2.Q = cp.Q
    for i in range(10):
        cp2.q_learning_episode()
        cp2.env.reset()
        cp2.done = False
        print(i)
    print("")