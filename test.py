import gym
from gym import wrappers
import numpy as np


def run_cartpole(weights):
    env.reset()
    count = 0
    done = False
    action = 1
    max_iter = 10000
    while not done and count < max_iter:
        [observation, reward, done, _, _] = env.step(action)

        if np.dot(observation, weights) > 0:
            action = 1
        else:
            action = 0
        count += 1
    return count


def run_multiple(weights, iter):
    count_list = np.empty(iter)

    for t in range(iter):
        count = run_cartpole(weights)
        count_list[t] = count
    return count_list


def random_search(search_iterations):
    best_count = 0
    best_weights = np.zeros(4)
    best_count = 0
    count_mean = np.empty(search_iterations)

    for i in range(search_iterations):
        weights = np.random.rand(4, 1) * 2 - 1

        count_list = run_multiple(weights, 10)
        count_mean[i] = count_list.mean()
        if count_mean[i] > best_count:
            best_weights = weights
            best_count = count_mean[i]
    return best_weights, best_count, count_mean


if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    env = gym.make('CartPole-v0')  # ,  render_mode="human")
    env.reset()
    env = wrappers.Monitor(env, "/home/tivo/01_RLstuff/videos")
    example_weights = np.random.rand(4, 1) * 2 - 1
    goodweight = [[0.56477445], [0.52499204], [0.94437464], [0.48111971]]
    print(run_cartpole(goodweight))
    count_list = run_multiple(example_weights, 10)
    print(count_list)
    best_weights, best_count, count_mean = random_search(1000)

    print(run_cartpole(best_weights))
    # env.close()
