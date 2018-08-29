from keras.models import Sequential
from keras.layers import Activation, InputLayer, Dense
from keras.optimizers import Adam
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt


class Q_agent:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions


    def buildmodel(self, learning_rate):
        '''
        Neural network
        :param learning_rate:
        :return: model object
        '''

        model = Sequential()
        model.add(InputLayer(batch_input_shape=self.env.state_shape))
        model.add(Dense(9, activation='relu'))
        #model.add(Dense(25, activation='relu'))
        model.add(Dense(self.actions, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['mae'])

        print("We finish building the model")
        return model

    def get_next_state(self, state, A, player=0):
        next_state = self.env.get_state(state, A, player)
        return next_state

    def get_reward(self, state,player=0):
        reward, final_state = self.env.get_reward(state,player)
        return reward, final_state

    def get_action(self, state,player=0):
        ## Exploration vs. Exploitation
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(1, self.actions)
        else:
            act_values = self.model.predict(state)
        ##contrain
        ind = self.env.get_constrain(state, act_values, player)
        act_values[0,ind] = -1000
        ## Action
        A = np.argmax(act_values)
        return A

    def save_weights(self,fname):
        self.model.save_weights(fname)

    def set_weights(self,fname):
        self.model.load_weights(fname, by_name=False)

    def train_1player(self, episodes, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995, batch_size=50, gamma=0.95,
                      learning_rate=0.01, memory_length=200):
        self.epsilon = epsilon
        ## NW
        self.model = self.buildmodel(learning_rate=learning_rate)
        ## envionment settings
        memory = []
        for e in range(episodes):
            state = self.env.get_initial_state()
            while True:
                A = self.get_action(state,0)
                next_state = self.get_next_state(state, A)
                reward, final_state = self.get_reward(next_state)
                ## Store: state, action, reward
                memory.append((state, A, reward, next_state, final_state))
                if len(memory)>memory_length:
                    del memory[0]
                ## next state
                state = next_state
                ## Terminate
                if final_state:
                    break
            ## train netwerk
            if len(memory) > batch_size * 1.2:
                minibatch = random.sample(memory, batch_size)
            else:
                minibatch = memory
            for current_state, action, reward, next_state, final_state in minibatch:
                if not final_state:
                    target = reward + gamma * np.argmax(self.model.predict(next_state))
                else:
                    target = reward
                target_f = self.model.predict(current_state)
                target_f[0][action] = target
                stats = self.model.fit(current_state, target_f, epochs=1, verbose=0)
                if self.epsilon > epsilon_min:
                    self.epsilon *= epsilon_decay
            print('End game number {} | epsilon={} | acc={}'.format(e, self.epsilon, stats.history['mean_absolute_error']))

    def train_2players(self, episodes, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, gamma=0.95, learning_rate=0.01, memory_length=10):
        self.epsilon = epsilon
        ## NW
        self.model = self.buildmodel(learning_rate=learning_rate)
        ## envionment settings
        memory = []
        for e in range(episodes):
            state = self.env.state
            while True:
                ## player 1
                A = self.get_action(state,0)
                intermediate_state = self.get_next_state(state, A, 0)
                reward, final_state = self.get_reward(intermediate_state, 0)
                ## player 2
                if not final_state:
                    A_player2 = self.get_action(intermediate_state,1)
                    next_state = self.get_next_state(intermediate_state, A_player2, 1)
                    ## Get reward player 1
                    reward, final_state = self.get_reward(next_state, 0)
                else:
                    break
                ## Store: state, action, reward
                memory.append((state, A, reward, next_state, final_state))
                if len(memory)>memory_length:
                    del memory[0]
                ## next state
                state = next_state
                ## Terminate
                if final_state:
                    break
            ## train netwerk
            if len(memory) > batch_size * 1.2:
                minibatch = random.sample(memory, batch_size)
            else:
                minibatch = memory
            for current_state, action, reward, next_state, final_state in minibatch:
                if not final_state:
                    target = reward + gamma * np.argmax(self.model.predict(next_state))
                else:
                    target = reward
                target_f = self.model.predict(current_state)
                target_f[0][action] = target
                stats = self.model.fit(current_state, target_f, epochs=1, verbose=0)
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
            print('End game number {} | acc={}'.format(e, stats.history['mean_absolute_error']))








