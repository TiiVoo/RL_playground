import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion


class SGDRegressor:
    def __init__(self,D , learning_rate=0.1):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = learning_rate

    def partial_fit(self, x, y):
        self.w += self.lr * (y - x.dot(self.w)).dot(x)

    def predict(self, x):
        return x.dot(self.w)


class FeatureExtract:
    def __init__(self, env, n_components=100):
        observation_examples = np.random.random((20000, 4))*2-2
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=0.1, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.01, n_components=n_components))
        ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # print "observations:", observations
        scaled = self.scaler.transform(observations)
        assert (len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions, learning_rate=learning_rate)
            self.models.append(model)

    def predict(self, s):
        x = self.feature_transformer.transform([s])
        result = np.stack([m.predict(x) for m in self.models]).T
        assert (len(result.shape) == 2)
        return result

    def update(self, s, a, g):
        x = self.feature_transformer.transform([s])
        assert (len(x.shape) == 2)
        self.models[a].partial_fit(x, g)

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
    max_iter = 2000
    count = 0
    totalreward = 0
    done = False
    while not done and count < max_iter:
        action = model.sample_action(observation, eps)  # get action
        prev_observation = observation
        [observation, reward, done, _, _] = env.step(action)  # get new state
        prediction = model.predict(observation)

        if done:
            reward = -200

        g = reward + gamma * np.max(prediction)
        model.update(prev_observation, action, g)  # update q-matrix
        count += 1
        totalreward += reward
        # print("step:", count, "reward:", reward, "total reward:", totalreward)
        # plot_cost_to_go(env, model)

    return totalreward


def main():
    env = gym.make('CartPole-v1', render_mode="human")
    feature_extract = FeatureExtract(env)
    model = Model(env, feature_extract, learning_rate=0.1)

    n_episodes = 500
    total_rewards = np.empty(n_episodes)
    for i in range(n_episodes):
        eps = 1.0/np.sqrt(i+1)
        total_count = run_episode(env, model, eps, gamma=0.99)
        total_rewards[i] = total_count
        print("episode:", i, "total reward:", total_count)
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())

    plt.plot(total_rewards)
    plt.title("Rewards")

    from RBF_MountainCar import plot_running_avg
    plot_running_avg(total_rewards)
    plt.show()
    # plot the optimal state-value function

    env = gym.make('CartPole-v1',render_mode="human")
    #model = Model(env, feature_extract, learning_rate="constant")
    for i in range(20):
        eps = 0.001
        total_count = run_episode(env, model, eps, gamma=0.99)
        print("episode:", i, "total reward:", total_count)

if __name__ == "__main__":
    main()
