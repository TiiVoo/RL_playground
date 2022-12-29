from __future__ import print_function

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import RBF_cartpole



class SGDRegressor:
    def __init__(self,D , learning_rate=0.1):
        #self.w = np.random.randn(D) / np.sqrt(D)

        self.weights = tf.Variable(tf.random.normal(stddev=1.0 / D, shape=(D, 1)))
        self.lr =  learning_rate

    def partial_fit(self, x, y):
        Xy = tf.concat([x, y], axis=1)
        Xy = tf.random.shuffle(Xy)
        X, y = tf.split(Xy, [Xy.shape[1] - 1, 1], axis=1)

        self.w += self.lr * (y - x.dot(self.w)).dot(x)

        xw = tf.tensordot(self.w * x, 1)

        dy_dm = (y - x.dot(self.w)).dot(x)
        self.w.assign_add(self.lr * dy_dm)

    def predict(self, x):
        #return x.dot(self.w)
        return tf.tensordot(self.w * x, 1)


def main():
    env = gym.make('CartPole-v1')#, render_mode="human")
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
