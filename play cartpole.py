import gym
from ai import Dqn

env = gym.make('CartPole-v0')
observation = env.reset()
input_size = len(observation)
nb_actions = env.action_space.n

brain = Dqn(input_size, nb_actions, 0.99)

"""
for i_episode in range(20):
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
"""
rewards = [] #list of scores per game
for i in range(10000):
    done = False
    reward = 0
    sum_reward = 0
    state = env.reset()
    while not done:
        #env.render() #remove the first # to view the game as it learns (slows 
                      #down process)
        action = brain.update(reward, state)
        new_state, new_reward, done, info = env.step(int(action))
        state, reward = new_state, new_reward
        sum_reward += reward
    rewards.append(sum_reward)
    print(i, sum_reward)
    if i%100 == 0:
        brain.save()