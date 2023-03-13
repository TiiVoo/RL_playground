

##  https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQG_model:
    def __init__(self, env):

        self.env = env
        # Initialize attributes
        self._state_size = env.observation_space.shape[0]
        self._action_size = env.action_space.n

        self.experience_replay = deque(maxlen=2000)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        # Initialize discount and exploration rate
        self.gamma = 0.99

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def _build_compile_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(25, input_dim=self._state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(25, input_dim=self._state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)

        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    @tf.function
    def act(self, state,eps):
        if tf.random.uniform([]) <= eps:
            return tf.random.uniform([], minval=0, maxval=self._action_size, dtype=tf.int64)

        q_values = self.q_network(state)
        return tf.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        with tf.GradientTape() as tape:
            q_values = self.q_network(tf.concat(states, axis=0))

            target = tf.identity(q_values)
            updates = rewards + (1 - tf.cast(dones, tf.float32)) * self.gamma * tf.reduce_max(
                self.target_network(tf.concat(next_states, axis=0)), axis=-1)
            indices = tf.stack([tf.range(batch_size), actions], axis=-1)
            target = tf.tensor_scatter_nd_update(target, indices, updates)

            loss = tf.keras.losses.mean_squared_error(target, q_values)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))


def run_episode(env,model,eps):
    state = env.reset()[0]
    state = state.reshape((1, -1))
    max_iter = 2000
    count = 0
    totalreward = 0
    batch_size = 100

    done = False
    while not done and count < max_iter:

        action = model.act(state,eps)  # get action
        action = action.numpy()
        [next_state, reward, done, _, _] = env.step(action)  # get new state
        next_state = next_state.reshape((1, -1))

        if done:
            reward = -200

        model.store(state, action, reward, next_state, done)

        if len(model.experience_replay) > batch_size:
            model.retrain(batch_size)

        totalreward += reward
        count += 1
        state = next_state

    model.alighn_target_model()

    return totalreward
def main():
    env = gym.make('CartPole-v1')

    # Create the actor and critic networks

    model = DQG_model(env)
    model.q_network.summary()
    # Create the optimizer



    n_episodes = 50
    total_rewards = np.empty(n_episodes)
    for i in range(n_episodes):
        eps = 0.3 * (0.9 ** i)
        total_count = run_episode(env,model,eps)
        total_rewards[i] = total_count
        print("episode:", i, "total reward:", total_count)
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())

    plt.plot(total_rewards)
    plt.title("Rewards")

    from RBF_MountainCar import plot_running_avg
    plot_running_avg(total_rewards)
    plt.show()

    env = gym.make('CartPole-v1', render_mode="human")

    for i in range(10):
        total_count = run_episode(env,model,eps=0.001)
        print("episode:", i, "total reward:", total_count)

if __name__ == "__main__":
    main()