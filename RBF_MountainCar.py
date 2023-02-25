import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDRegressor


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


class FeatureExtract:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
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
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()[0]]), [0])
            self.models.append(model)

    def predict(self, s):
        x = self.feature_transformer.transform([s])
        result = np.stack([m.predict(x) for m in self.models]).T
        assert (len(result.shape) == 2)
        return result

    def update(self, s, a, g):
        x = self.feature_transformer.transform([s])
        assert (len(x.shape) == 2)
        self.models[a].partial_fit(x, [g])

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
    count = 0
    totalreward = 0
    done = False
    while not done and count < max_iter:
        action = model.sample_action(observation, eps)  # get action
        prev_observation = observation
        [observation, reward, done, _, _] = env.step(action)  # get new state
        prediction = model.predict(observation)
        g = reward + gamma * np.max(prediction[0])
        model.update(prev_observation, action, g)  # update q-matrix
        count += 1
        totalreward += reward
        # print("step:", count, "reward:", reward, "total reward:", totalreward)
        # plot_cost_to_go(env, model)

    return totalreward


def main():
    env = gym.make('MountainCar-v0')
    #env = gym.make('MountainCar-v0', render_mode="human")
    feature_extract = FeatureExtract(env)
    model = Model(env, feature_extract, learning_rate="constant")

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
