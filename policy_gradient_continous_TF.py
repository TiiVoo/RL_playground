

##not working##
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Define hyperparameters
learning_rate = 0.005

# Define the actor network
state_dim = 2
action_dim = 1
hidden_dim = 32


class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(state_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.mean = tf.keras.layers.Dense(action_dim, activation='tanh')
        self.stddev = tf.keras.layers.Dense(action_dim, activation='softplus')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mean = self.mean(x)
        stddev = self.stddev(x)
        return mean, stddev


# Define the critic network
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(state_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        value = self.dense3(x)
        return value


# Define the loss function
def compute_loss(actor, critic, states, actions, rewards):
    # Compute the advantages
    values = critic(states)
    advantages = rewards - values[:, 0]

    # Compute the actor loss
    means, stddevs = actor(states)
    normal_dist = tfp.distributions.Normal(loc=means, scale=stddevs)
    log_probs = normal_dist.log_prob(actions)
    actor_loss = -tf.reduce_mean(log_probs * advantages)

    # Compute the critic loss
    target_values = rewards
    critic_loss = tf.keras.losses.mean_squared_error(target_values, values)

    # Compute the total loss
    loss = actor_loss + critic_loss

    return loss


def run_episode(env,actor,critic,optimizer):
    state = env.reset()[0]
    max_iter = 2000
    count = 0
    totalreward = 0

    a_high= env.action_space.high
    a_low= env.action_space.low

    done = False
    while not done and count < max_iter:
        # Sample an action from the actor network
        mean, stddev = actor(np.array([state]))
        action_dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        action = action_dist.sample()
        action = tf.clip_by_value(action, a_low, a_high).numpy()

        # Take a step in the environment
        next_state, reward, done, _ , _ = env.step(action)

        # Update the actor and critic networks
        with tf.GradientTape() as tape:
            loss = compute_loss(actor, critic, np.array([state]), np.array([action]), np.array([reward]))
        gradients = tape.gradient(loss, actor.trainable_variables + critic.trainable_variables)
        optimizer.apply_gradients(zip(gradients, actor.trainable_variables + critic.trainable_variables))

        # Update the state and reward
        state = next_state.flatten()

        totalreward += reward
        count += 1


    return totalreward
def main():
    env = gym.make('MountainCarContinuous-v0',render_mode="human")

    # Create the actor and critic networks
    actor = ActorNetwork()
    critic = CriticNetwork()
    # Create the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)


    n_episodes = 30
    total_rewards = np.empty(n_episodes)
    for i in range(n_episodes):

        total_count = run_episode(env,actor,critic,optimizer)
        total_rewards[i] = total_count
        print("episode:", i, "total reward:", total_count)
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())

    plt.plot(total_rewards)
    plt.title("Rewards")

    from RBF_MountainCar import plot_running_avg
    plot_running_avg(total_rewards)
    plt.show()

    env = gym.make('MountainCarContinuous-v0',render_mode="human")

    for i in range(20):
        total_count = run_episode(env,actor,critic,optimizer)
        print("episode:", i, "total reward:", total_count)

if __name__ == "__main__":
    main()