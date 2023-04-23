

##  https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from PIL import Image
def preprocess_frame(frame):
    frame = frame[31:194, 8:152]
    frame = Image.fromarray(frame)            # Convert frame to PIL image
    frame = frame.convert("L")                # Convert to grayscale
    frame = frame.resize((84, 84), Image.ANTIALIAS)  # Resize to 84x84
    frame = np.array(frame, dtype=np.float32) / 255.0  # Convert back to NumPy array and normalize pixel values
    return frame
    #plt.imshow(frame, cmap='gray')

class DQGModel:
    def __init__(self, state_shape, n_actions, memory_size=100000, batch_size=64, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides = 4, activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(32, (4, 4), strides= 2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_episode(env,model):
    state = env.reset()[0]
    state = preprocess_frame(state)
    state = np.stack([state] * 4, axis=2)
    state = np.expand_dims(state, axis=0)
    max_iter = 2000
    count = 0
    total_reward = 0

    done = False

    while not done and count < max_iter:

        action = model.act(state)  # get action
        [next_state, reward, done, _, _] = env.step(action)  # get new state

        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=2)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = np.append(state[:, :, :, 1:], next_state, axis=3)

        model.remember(state, action, reward, next_state, done)



        total_reward += reward
        count += 1
        state = next_state

    model.replay()

    return total_reward
if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4', render_mode="human")
    state_shape = (84, 84, 4) # 84x84 after rescaling and 4 steps history
    n_actions = env.action_space.n

    model = DQGModel(state_shape, n_actions)



    n_episodes = 100
    total_rewards = np.empty(n_episodes)
    for i in range(n_episodes):

        total_count = run_episode(env,model)
        total_rewards[i] = total_count
        print("episode:", i, "total reward:", total_count)
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())

    plt.plot(total_rewards)
    plt.title("Rewards")

    from RBF_MountainCar import plot_running_avg
    plot_running_avg(total_rewards)
    plt.show()

    env = gym.make('BreakoutDeterministic-v4', render_mode="human")

    for i in range(10):
        total_count = run_episode(env,model)
        print("episode:", i, "total reward:", total_count)

