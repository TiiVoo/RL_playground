import gym

env = gym.make('CartPole-v1')

print(env.reset())

done = False
while not done:
    [observation, reward, done, _, _] = env.step(env.action_space.sample())



