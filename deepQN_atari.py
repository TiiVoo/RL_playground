

##  running but not converging.
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import pickle
from skimage.color import rgb2gray
from skimage.transform import resize

from collections import deque
import random
from PIL import Image
IM_SIZE = 84
AGENT_HISTORY = 4
def preprocess_frame(frame):
    frame = frame[31:194, 8:152]
    '''frame = Image.fromarray(frame)            # Convert frame to PIL image
    frame = frame.convert("L")                # Convert to grayscale
    frame = frame.resize((IM_SIZE, IM_SIZE), Image.ANTIALIAS)  # Resize to 84x84
    frame = np.array(frame, dtype=np.float32) / 255.0  # Convert back to NumPy array and normalize pixel values
    '''
    frame = np.uint8(
        resize(rgb2gray(frame), (84, 84), mode='constant') * 255)
    return frame
    #plt.imshow(frame, cmap='gray')


class ReplayMemory:
    def __init__(self, size, batch_size, frame_height=IM_SIZE, frame_width=IM_SIZE,
                 agent_history_length=AGENT_HISTORY):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number of transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer-encoded action
            frame: One grayscale frame of the game
            reward: reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


class DQGModel:
    def __init__(self, state_shape, n_actions, memory_size=250000, min_memory=0, batch_size=32, gamma=0.99, epsilon=0.1,
                 epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.005,update_target_freq=1000):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.memory = ReplayMemory(memory_size,batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.min_memory = min_memory
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_freq = update_target_freq
        self.steps_taken = 0

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides = 4, activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(64, (4, 4), strides= 2, activation='relu'))
        model.add(Conv2D(64, (3, 3), strides= 1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, action, next_frame, reward, done):
        self.memory.add_experience(action, next_frame, reward, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        q_values = self.model(state)
        return tf.argmax(q_values[0])

    def replay(self):
        if self.memory.count < self.batch_size or self.memory.count < self.min_memory:
            return

        states, actions, rewards, next_states, dones = self.memory.get_minibatch()
        target = rewards + np.invert(dones).astype(np.float32)*self.gamma * tf.reduce_max(self.model(next_states), axis=1)
        current_target_f = self.model(states)
        current_target_f = current_target_f.numpy()
        current_target_f[0][actions] = target
        self.model.fit(np.array(states), np.array(current_target_f), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_taken += 1
        if self.steps_taken % self.update_target_freq == 0:
            self.update_target_network()
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    def save_model(self, filepath):
        self.model.save((filepath+ ".h5"))
        with open((filepath+'.pkl'), 'wb') as file:
            pickle.dump(self.memory, file)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model((filepath+".h5"))
        self.update_target_network()
    def load_memory(self, filepath):
        with open((filepath+".pkl"), 'rb') as file:
            self.memory = pickle.load(file)

def run_episode(env,model):
    frame = env.reset()[0]
    frame = preprocess_frame(frame)
    state = np.stack([frame] * 4, axis=2)
    state = np.expand_dims(state, axis=0)
    max_iter = 2000
    count = 0
    total_reward = 0

    done = False

    while not done and count < max_iter:

        action = model.act(state)  # get action
        [next_frame, reward, done, _, _] = env.step(action)  # get new state

        next_frame = preprocess_frame(next_frame)
        assert next_frame.shape == (IM_SIZE, IM_SIZE)
        model.remember(action, next_frame, reward, done)

        next_frame = np.expand_dims(next_frame, axis=2)
        next_frame = np.expand_dims(next_frame, axis=0)
        next_state = np.append(state[:, :, :, 1:], next_frame, axis=3)


        total_reward += reward
        count += 1
        state = next_state
        model.replay()

    return total_reward
if __name__ == "__main__":
    MODEL_PATH = "./model/dQN_atari_model"
    env = gym.make('BreakoutDeterministic-v4', render_mode="human")
    state_shape = (IM_SIZE, IM_SIZE, AGENT_HISTORY) # 84x84 after rescaling and 4 steps history
    n_actions = env.action_space.n

    model = DQGModel(state_shape, n_actions)

    #model.load_model(MODEL_PATH)
    #model.load_memory(MODEL_PATH)


    n_episodes = 80
    total_rewards = np.empty(n_episodes)
    for i in range(n_episodes):

        total_count = run_episode(env,model)
        total_rewards[i] = total_count
        print("episode:", i, "total reward:", total_count)
        if i%500==0:
            model.save_model(MODEL_PATH)
        if i % 100 == 0:
            print("avg reward for last 100 episodes:", total_rewards[-100:].mean())

    plt.plot(total_rewards)
    plt.title("Rewards")

    from RBF_MountainCar import plot_running_avg
    plot_running_avg(total_rewards)
    plt.show()

