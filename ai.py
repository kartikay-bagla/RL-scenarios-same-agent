# AI for Cartpole

# Importing the libraries
import numpy as np
import random
import os
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Activation
from keras import backend as K

# Creating the loss function (which I just copy pasted from the internet and modified a bit)
HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    #y_true is something like [0, 0.8] there is a 0 there because we dont know 
    #the q_value of that action
    #
    #y_pred would be like [0.4, 0.6]
    
    act_ind = np.argmax(y_true) #since we only know the q_value of only one 
                                #action, we only take that value to calculate 
                                #the loss
    
    #This is the smoothL1 loss function.
    x = K.abs(y_true[act_ind] - y_pred[act_ind])
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, 
                 HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)


def mse(y_true, y_pred):
    #The mean squared error function, with the same logic as above applied
    #But when i use mse, i get this: "ValueError: Error when checking target: 
    #expected out to have shape (2,) but got array with shape (1,)"
    #Now I have no idea why this happens.
    
    act_ind = np.argmax(y_true)
    return K.mean(K.square(y_pred[act_ind] - y_true[act_ind]), axis=-1)

class Network():
    def __init__(self, input_size, nb_action):
        self.input_size = input_size
        self.nb_action = nb_action
        #Model Definition
        self.model = Sequential()
        self.model.add(Dense(16, activation="relu", name = "H1", 
                             input_shape=(input_size, )))
        self.model.add(Dense(nb_action, activation="softmax", name = "out"))
        self.model.compile("adam", mse) #change loss function to smoothL1 and it works
        
    def forward(self, state):
        q_values = self.model.predict(state)
        return q_values

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return samples

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = "adam"
        self.last_state = np.zeros((1, input_size))
        self.last_action = 0
        self.last_reward = 0
        self.size = input_size
        self.nb_action = nb_action
    
    def select_action(self, state):
        state = np.array(state).reshape(1, self.size)
        probs = self.model.forward(state)
        
        #An attempt to add random exploration in the agent
        if random.random() > 0.9:
            action = random.randrange(0, self.nb_action)
        else:
            action = np.argmax(probs)
        return action
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        batch_state = np.array(batch_state).reshape(len(batch_action), 
                               self.size) #reshaping everything into size
        batch_next_state= np.array(batch_next_state).reshape(len(batch_action), 
                                   self.size)
        outputs = self.model.forward(batch_state)
        
        outputs = [outputs[i][batch_action[i]] for i in range(len(outputs))]
        #selecting the q_values of those actions which were originally taken 
        #by the model
        
        next_outputs = self.model.forward(batch_next_state)
        next_outputs = np.array([max(i) for i in next_outputs]) 
        #keeping only the max value for q function
        
        targets = (self.gamma * next_outputs) + batch_reward
        
        y_actual = []
        for i in range(targets.shape[0]):
            #a is list of 0 with number of 0's = no of actions possible and 
            #then we replace the 0 with the index of the action taken by the 
            #model with its q_value and hence it becomes the y_actual vector 
            #for that move
            #So each vector is like [0, q_value] or [q_value, 0], the 0 is 
            #there since we don't know the actual q_value of that action to 
            #train the model
            a = [0 for i in range(self.nb_action)]
            a[batch_action[i]] = targets[i]
            y_actual.append(np.array(a))
        
        y_actual = np.array(y_actual)
        self.model.model.fit(batch_state, targets, verbose = 0)
    
    def update(self, reward, new_signal):
        
        new_state = np.array(new_signal).reshape(1, self.size)
        self.memory.push((self.last_state, new_state, self.last_action, self.last_reward))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 64:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(64)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        #save_file name is common for all games
        self.model.model.save_weights("brain.brain")
    
    def load(self):
        #save file name is common for all games
        if os.path.isfile('brain.brain'):
            print("=> loading checkpoint... ")
            self.model.model.load_weights("brain.brain")
            print("done !")
        else:
            print("no checkpoint found...")
