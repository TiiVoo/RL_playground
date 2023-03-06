
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def calc_discounted_rewards(reward_lst, GAMMA):
    """function to calc dicounted rewards.
    Params:
        :reward_lst: 1d array/list of rewards and addes them up with the discount factor gamma each step.
    Returns
        Numpy array 1d with discounted reward for each reward position
    """
    # Q(k,t) = Sigma_i(gamma*reward_i) with t=step and k=episode
    prev_val = 0
    out = []
    for val in reward_lst:
        new_val = val + prev_val * GAMMA
        out.append(new_val)
        prev_val = new_val
    # remember to flip
    return np.array(out[::-1])

class PolicyGradient:
    def __init__(self,dim_states,dim_actions):
        # Define the network architecture
        input_shape = dim_states # Assuming the environment has 4 states
        output_shape = dim_actions  # Assuming there are 2 possible actions
        hidden_size = 300  # Number of hidden units in the network
        learning_rate = 0.005 # Learning rate for the optimizer

        # Define the placeholders for the input data
        self.states_ph = tf.placeholder(tf.float32, shape=(None, *input_shape))
        self.actions_ph = tf.placeholder(tf.float32, shape=(None, output_shape))
        self.rewards_ph = tf.placeholder(tf.float32, shape=(None,))

        # Define the network weights
        weights = {
            'hidden': tf.Variable(tf.random.normal([input_shape[0], hidden_size])),
            'output': tf.Variable(tf.random.normal([hidden_size, output_shape]))
        }
        biases = {
            'hidden': tf.Variable(tf.zeros([hidden_size])),
            'output': tf.Variable(tf.zeros([output_shape]))
        }

        # Define the network computation graph
        hidden = tf.nn.relu(tf.matmul(self.states_ph, weights['hidden']) + biases['hidden'])
        logits = tf.matmul(hidden, weights['output']) + biases['output']
        self.action_probs = tf.nn.softmax(logits)

        # Define the loss function
        neg_log_probs = -tf.reduce_sum(self.actions_ph * tf.log(self.action_probs), axis=1)
        loss = tf.reduce_mean(neg_log_probs * self.rewards_ph)

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(loss)

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def fit(self, states, actions ,rewards):
        feed_dict = {
            self.states_ph: states,
            self.actions_ph: actions,
            self.rewards_ph: rewards
        }
        self.session.run(self.train_op, feed_dict=feed_dict)

    def predict(self, state):

        return  self.session.run(self.action_probs, feed_dict={self.states_ph: state})




class Model:
    def __init__(self, env):
        self.env = env
        self.models = []
        self.actionspace =  env.action_space.n
        self.statespace = env.observation_space.shape
        self.models = PolicyGradient(self.statespace, self.actionspace)

    def predict(self, s):
        return self.models.predict(s)

    def update(self, s, a, g):
        self.models.fit(s, a, g)

    def sample_action(self, s, eps):

        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))



def run_episode(env, model, eps, gamma):
    observation = env.reset()[0]
    max_iter = 2000
    count = 0
    totalreward = 0
    rewards = []
    states = []
    actions = []
    actionspace=env.action_space.n
    statespace = env.observation_space.shape[0]
    done = False
    while not done and count < max_iter:

        action = model.sample_action(np.reshape(observation, [1, statespace]), eps)  # get action

        [next_observation, reward, done, _, _] = env.step(action)  # get new state

        states.append(observation)
        a_hot = [0] * actionspace
        a_hot[action] = 1
        actions.append(a_hot)

        if done:
            reward = -200
        rewards.append(reward)
        totalreward += reward

        count += 1
        observation = next_observation

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(calc_discounted_rewards(rewards,gamma))
    model.update(states,actions,rewards)


    return totalreward
def main():
    env = gym.make('CartPole-v1')
    model = Model(env)

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

    env = gym.make('CartPole-v1',render_mode="human")

    for i in range(20):
        eps = 0.001
        total_count = run_episode(env, model, eps, gamma=0.99)
        print("episode:", i, "total reward:", total_count)

if __name__ == "__main__":
    main()