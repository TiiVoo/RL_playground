from unittest import TestCase

import gym

from RBF_MountainCar import Model, FeatureExtract, run_episode


class TestModel(TestCase):
    def setUp(self):
        env = gym.make('MountainCar-v0')
        feature_extract = FeatureExtract(env)
        self.model = Model(env, feature_extract, learning_rate="constant")
        self.env = env

    def test_predict(self):
        observation = self.env.reset()[0]
        prediction = self.model.predict(observation)

        self.assertTrue((len(prediction.shape) == 2))

    def test_update(self):
        prev_observation = self.env.reset()[0]
        action = self.env.action_space.sample()
        G = 1
        self.model.update(prev_observation, action, G)

    def test_sample_action(self):
        observation = self.env.reset()[0]
        eps = 0.1
        action = self.model.sample_action(observation, eps)

        [observation, reward, done, _, _] = self.env.step(action)
        action = self.model.sample_action(observation, eps)


class Test(TestCase):
    def test_run_episode(self):
        env = gym.make('MountainCar-v0', render_mode="human")
        feature_extract = FeatureExtract(env)
        model = Model(env, feature_extract, learning_rate="constant")
        run_episode(env, model, eps=0.1, gamma=0.9)
