from keras.models import Sequential
from keras.layers import Activation, InputLayer, Dense
from keras.optimizers import Nadam
from keras.models import load_model
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy


class Q_agent:
    def __init__(self, env, training = True):
        self.env = env
        self.training = training
        
    def act(self, state):
        action = self.get_action(state)
        next_state = self.env.get_next_state(state,action)
        reward, done = self.env.get_reward(next_state)
        return action, next_state, done, reward
    
    def buildmodel(self, N, learning_rate):
        model = Sequential()
        model.add(InputLayer(input_shape=(N[0],)))
        for layer in range(len(N)-2):
            model.add(Dense(N[layer+1], activation='relu'))
        model.add(Dense(N[-1], activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(lr=learning_rate), metrics=['mae'])
        print("Finished building the model")
        return model

    def get_action(self, state):
        ## Exploration vs. Exploitation
        if self.train == False:
            self.epsilon = 0
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(1, self.env.action_size)
        else:
            act_values = self.model.predict(state.reshape(1,np.prod(self.env.state_shape)))
        ## Constrain
        ind = self.env.get_constrain(state)
        act_values[0,ind] = -1000
        action = np.argmax(act_values)
        return action

    def save_model(self, fname):     
        '''
        Stores the model at the given path
        '''
        self.model.save(fname)

    def set_model(self,fname):
        '''
        Loads the model from a given path
        '''
        model = load_model(fname)
        self.model = model

    def train(self, 
                      N,                    #ADDED FOR LAYERS & ACTIVATIONS
                      episodes          =   100,
                      epsilon           =   1, 
                      epsilon_min       =   0.01, 
                      epsilon_decay     =   0.9999, 
                      batch_size        =   50, 
                      gamma             =   0.95,
                      learning_rate     =   0.001, 
                      memory_length     =   1000,
                      breaks            =   10):
        self.epsilon = epsilon
        self.model = self.buildmodel(N, learning_rate)
        ## envionment settings
        memory = []
        scores = []
        start_time = time.time()
        for e in range(episodes):
            state = self.env.get_initial_state()
            turn = 0
            if self.env.players == 2:
                old_state = 0; old_action = 0
                action, next_state, done, reward = self.act(state)
                
                while True:
                    if turn != 0:
                        memory.append((old_state,old_action,reward,next_state,done))
                    turn += 1
                    old_action = deepcopy(action); old_state = deepcopy(state)
                    state = self.env.switch_state(next_state)
                    action, next_state, done, reward = self.act(state)
                    if len(memory)>memory_length:
                        del memory[0]
                    if done:
                        memory.append((state,action,reward,next_state,done))
                        if reward == self.env.reward_win:
                            reward = -reward
                        memory.append((old_state,old_action,reward,next_state,done))
                        break
            
            else:    # If 1 player game
                while True:
                    turn += 1
                    action, next_state, done, reward = self.act(state)
                    memory.append((state,action,reward,next_state,done))
                    if len(memory)>memory_length:
                        del memory[0]
                    state = next_state.copy()
                    if done:
                        break
    
            ## train netwerk
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    if not done:
                        target = reward + gamma * np.amax(self.model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])
                    else:
                        target = reward
                    target_f = self.model.predict(state.reshape(1,np.prod(self.env.state_shape)))
                    target_f[0][action] = target
                    stats = self.model.fit(state.reshape(1,np.prod(self.env.state_shape)), target_f, epochs=1, verbose=0)
                    
#                print('End game number {} | epsilon={}'.format(e, self.epsilon))
            if e % breaks == 0 and e > 0:
                if self.epsilon > epsilon_min:
                    self.epsilon *= epsilon_decay
                scores.append(self.env.test_skill(self.model))
                break_time = time.time() - start_time
                time_left = break_time*(float(episodes)/float(e)-1)
                print("eps:",round(self.epsilon,2),scores[-1],e,'/',episodes,'ETA: '+'%02d'%(int(time_left)/3600)+":"+'%02d'%((int(time_left)%3600)/60)+":"+'%02d'%int(time_left%60))
                print(stats.history['mean_absolute_error'])
        plt.plot(scores)
        