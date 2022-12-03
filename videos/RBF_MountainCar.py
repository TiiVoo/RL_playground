import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDRegressor


# replace Q-function with RBF function?

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
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
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


class SpinMe:
    def __init__(self, env, model, n_episodes=5000):
        self.env = env
        self.model = model
        self.n_episodes = n_episodes

    def run_multi_episode(self):
        for i in range(self.n_episodes):
            self.env.reset()
            max_iter = 1000
            count = 0
            done = False
            action = self.env.action_space.sample()
            while not done and count < max_iter:
                [observation, reward, done, _, _] = self.env.step(action)  # get new state
                self.model.update(observation, action, reward)  # update q-matrix
                action = self.model.sample_action(observation, eps=0)  # get action
                count += 1


def main():
    env = gym.make('MountainCar-v0')
    feature_extract = FeatureExtract(env)
    model = Model(env, feature_extract, learning_rate=0.1)
    spin_me = SpinMe(env, model, n_episodes=5000)
    spin_me.run_multi_episode()


if __name__ == "__main__":
    main()
