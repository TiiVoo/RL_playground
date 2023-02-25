from __future__ import print_function

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import RBF_cartpole

class SGDRegressor:
    def __init__(self,D , learning_rate=0.1):
        #self.w = np.random.randn(D) / np.sqrt(D)

        self.w = tf.Variable(tf.random.normal(stddev=1.0 / D, shape=(D, 1)),name='w')
        lr =  learning_rate
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)

        # ops we want to call later
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat


        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)
    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        #return x.dot(self.w)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


if __name__ == "__main__":
    RBF_cartpole.SGDRegressor = SGDRegressor
    RBF_cartpole.main()
