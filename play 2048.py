import pickle
from math import log2

import numpy as np

import Game2048
from ai import Dqn
grid_length = 6
brain = Dqn(grid_length**2, 4, 0.9) #36 input neurons, 4 output neurons
game = Game2048.Game2048(grid_length)
state = game.reset()
reward = 0

def logger(grid):
    #used to convert all numbers to between 0 and 1
    max_num = log2(2048)
    grid2 = [log2(i)/max_num if i!=0 else 0 for i in grid]
    return grid2

#brain.load()
saved_grids = []
rewards = []
for i in range(1000):
    state = game.reset()
    reward = 0
    game_over = False
    sum_rewards = 0
    grids = []
    while not game_over:
        grids.append(np.array(state).reshape(6, 6))
        state = logger(state)
        reward2 = reward/2048
        action = brain.update(reward2, state)
        new_state, new_reward, game_over = game.step(int(action))
        state, reward = new_state, new_reward
        sum_rewards += reward
    rewards.append(sum_rewards)
    
    #if sum_rewards>4999:
    #    print(np.array(grids))

    print(i, sum_rewards)
    
    #if i%100 == 0:
        #print("saving")
        #brain.save()
        #with open("rewards", "wb") as f:
        #    pickle.dump(rewards, f)
print(sum(rewards)/len(rewards))

test_rewards = []
for i in range(10):
    state = game.reset()
    reward = 0
    sum_rewards = 0
    game_over = False
    while not game_over:
        action = brain.update(reward, state)
        new_state, new_reward, game_over = game.step(int(action))
        state, reward = new_state, new_reward
        sum_rewards += reward
    test_rewards.append(sum_rewards)

print(sum(test_rewards)/len(test_rewards))
