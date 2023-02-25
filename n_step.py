
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import RBF_MountainCar
from RBF_MountainCar import plot_cost_to_go, FeatureExtract, Model, plot_running_avg


def run_episode(env, model, eps, gamma):
    observation = env.reset()[0]
    max_iter = 10000
    count = 1
    totalreward = 0
    n_steps = 3
    n_step_return = 0

    rewards = []
    states = []
    actions = []
    multiplier = np.array([gamma]*n_steps) ** np.arange(n_steps)

    done = False
    while not done and count < max_iter:
        action = model.sample_action(observation, eps)  # get action

        [next_observation, reward, done, _, _] = env.step(action)  # get new state

        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        # Calculate the n-step return
        if count >=  n_steps:
            prediction = model.predict(next_observation)
            n_step_return = multiplier.dot(rewards[(-n_steps):])
            g = n_step_return + gamma ** n_steps * np.max(prediction[0])
            model.update(states[-n_steps], actions[-n_steps], g)  # update q-matrix

        count += 1
        observation = next_observation
        totalreward += reward
        # print("step:", count, "reward:", reward, "total reward:", totalreward)
        # plot_cost_to_go(env, model)

    return totalreward

if __name__ == "__main__":
    RBF_MountainCar.run_episode = run_episode
    RBF_MountainCar.main()
