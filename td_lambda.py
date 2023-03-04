
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import RBF_MountainCar
from RBF_MountainCar import plot_cost_to_go, FeatureExtract, Model, plot_running_avg

class SGDRegressor_Eligibilities:
    def __init__(self,D , learning_rate=0.1):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = learning_rate

    def partial_fit(self, x, y,eligibility):
        self.w += self.lr * (y - x.dot(self.w))*eligibility

    def predict(self, x):
        return x.dot(self.w)

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))
        for i in range(env.action_space.n):
            model = SGDRegressor_Eligibilities(D, learning_rate=learning_rate)
            self.models.append(model)

    def predict(self, s):
        x = self.feature_transformer.transform([s])
        result = np.stack([m.predict(x) for m in self.models]).T
        assert (len(result.shape) == 2)
        return result

    def update(self, s, a, g, gamma, lambda_):
        x = self.feature_transformer.transform([s])
        assert (len(x.shape) == 2)
        self.eligibilities *= gamma * lambda_
        self.eligibilities[a] += x[0] # or x[0] or 1 ???
        self.models[a].partial_fit(x[0], g, self.eligibilities[a])

    def sample_action(self, s, eps):
        # eps = 0
        # Technically, we don't need to do epsilon-greedy
        # because SGDRegressor predicts 0 for all states
        # until they are updated. This works as the
        # "Optimistic Initial Values" method, since all
        # the rewards for Mountain Car are -1.
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))



def run_episode(env, model, eps, gamma):
    observation = env.reset()[0]
    max_iter = 10000
    count = 1
    totalreward = 0
    lambd = 0.5  # choose between 0 and 1

    done = False
    while not done and count < max_iter:
        action = model.sample_action(observation, eps)  # get action
        [next_observation, reward, done, _, _] = env.step(action)  # get new state



        prediction = model.predict(next_observation)

        g = reward + gamma * np.max(prediction[0])

        model.update(observation, action, g, gamma, lambd) # update q matrix

        count += 1
        observation = next_observation
        totalreward += reward
        # print("step:", count, "reward:", reward, "total reward:", totalreward)
        # plot_cost_to_go(env, model)

    return totalreward

def main():
    #env = gym.make('MountainCar-v0')
    env = gym.make('MountainCar-v0', render_mode="human")
    feature_extract = RBF_MountainCar.FeatureExtract(env)
    model = Model(env, feature_extract, learning_rate=0.1)

    n_episodes = 300
    totalrewards = np.empty(n_episodes)
    for i in range(n_episodes):
        eps = 0.1 * (0.97 ** i)
        totalreward = run_episode(env, model, eps, gamma=0.99)
        totalrewards[i] = totalreward
        print("episode:", i, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
    # plot the optimal state-value function
    plot_cost_to_go(env, model)

    #env = gym.make('MountainCar-v0',render_mode="human")
    #model = Model(env, feature_extract, learning_rate="constant")


if __name__ == "__main__":
    main()
