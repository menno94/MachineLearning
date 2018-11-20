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
        ## use frozen network with a delay for the target value
        self.delay_model = True
        self.double_q = True
        
    def act(self, state):
        action = self.get_action(state)
        next_state = self.env.get_next_state(state,action)
        reward, done = self.env.get_reward(next_state)
        return action, next_state, done, reward
    
    def buildmodel(self, N, learning_rate):
        model = Sequential()
        model.add(InputLayer(input_shape = (np.prod(self.env.state_shape),) )) # Vreeeeeemd
        for layer in range(len(N)):
            model.add(Dense(N[layer], activation='relu'))
        model.add(Dense(self.env.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(lr=learning_rate), metrics=['mae'])
        return model

    def get_action(self, state):
        ## Exploration vs. Exploitation
        if self.training == False:
            self.epsilon = 0
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(1, self.env.action_size)
        else:
            act_values = self.model.predict(state.reshape(1,np.prod(self.env.state_shape)))
        ## Constrain
        ind = self.env.get_constrain(state)
        act_values[0,ind] = -1000
        action = np.argmax(act_values)
        if self.training:
            return action
        else:
            return action, act_values

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
    
    def create_model(self,N, learning_rate):
        '''
        Create model from scratch.
        N = list with the neurons in the hidden layers
        '''
        self.model = self.buildmodel(N, learning_rate)
        ## delay
        if self.delay_model:
            self.previous_model = self.buildmodel(N, learning_rate)
        if self.double_q:
            self.value_model = self.buildmodel(N, learning_rate)
            
        print("Finished building the model")
        

    def train(self, 
                      episodes          =   100,
                      epsilon           =   1, 
                      epsilon_min       =   0.01, 
                      epsilon_decay     =   0.9999, 
                      batch_size        =   50, 
                      gamma             =   0.95, 
                      memory_length     =   1000,
                      breaks            =   10):
        ## set epsilion
        self.epsilon = epsilon
        ## lists for sats
        scores      = []
        accur       = []
        loss        = []
        Qtest       = []
        epoch_axis  = []
        eps_hist = []
        R = []
        avg_reward = np.zeros(2)
        ## envionment settings
        memory = []
        start_time = time.time()
        ## save initial weights
        self.model.save_weights('temp_previous.h5')
        self.model.save_weights('temp_value_model.h5')
        model_update_freq = 10
        for e in range(episodes):
            state = self.env.get_initial_state()
            total_reward = 0
            turn = 0
# =============================================================================
#       Two player game
# =============================================================================
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
                    total_reward += reward
                    if len(memory)>memory_length:
                        del memory[0]
                    if done:
                        memory.append((state,action,reward,next_state,done))
                        if reward == self.env.reward_win:
#                            reward = -reward
                            reward = self.env.reward_notdone
                        memory.append((old_state,old_action,reward,next_state,done))
                        break
# =============================================================================
#           One player game
# =============================================================================
            else:    # If 1 player game
                while True:
                    turn += 1
                    action, next_state, done, reward = self.act(state)
                    total_reward += reward
                    memory.append((state,action,reward,next_state,done))
                    if len(memory)>memory_length:
                        del memory[0]
                    state = next_state.copy()
                    if done:
                        break
            ## mean stats
            avg_reward[1] = avg_reward[0] + (total_reward-avg_reward[0])/(e+1)
            if self.double_q:
                if e % model_update_freq == 0:
                    self.value_model.load_weights('temp_value_model.h5')
                    self.model.save_weights('temp_value_model.h5')
                ## train netwerk
            if len(memory) >= batch_size:
                ## model from previous episode
                self.previous_model.load_weights('temp_previous.h5')
                minibatch = random.sample(memory, batch_size)
                errortmp    = []
                losstmp     = []
                for state, action, reward, next_state, done in minibatch:
                    if not done:
                        if self.double_q:
                            if self.delay_model:
                                target_move = np.argmax(self.previous_model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])                        
                            else:
                                target_move = np.argmax(self.model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])
                            target = reward + gamma * self.value_model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0][target_move]
                        else:
                            if self.delay_model:
                                target = reward + gamma * np.amax(self.previous_model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])
                            else:
                                target = reward + gamma * np.amax(self.model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])
                    else:
                        target = reward
                    target_f = self.model.predict(state.reshape(1,np.prod(self.env.state_shape)))
                    target_f[0][action] = target
                    stats = self.model.fit(state.reshape(1,np.prod(self.env.state_shape)), target_f, epochs=1, verbose=0)
                ## update averaged error based
                errortmp.append(stats.history['mean_absolute_error'][0])
                losstmp.append(stats.history['loss'][0])
                ## save model weights
                self.model.save_weights('temp_previous.h5')
                ## adjust opsilon
                
                if e % breaks == 0:
                    ## test values
                    if self.epsilon > epsilon_min:
                        self.epsilon *= epsilon_decay
                    
#                    Qtesttmp = np.zeros(2)
#                    for i, item in enumerate(self.env.test_states):
#                        value = self.model.predict(item.reshape(1, np.prod(self.env.state_shape)))
#                        Qtesttmp[1] = Qtesttmp[0] + (np.amax(value) - Qtesttmp[0]) / (i + 1)
                    Qtest.append(0)
                    
                    ## test scores
                    scores.append(self.env.test_skill(self.model))
#                    scores.append(0)
                    ## stats
                    accur.append(np.mean(errortmp))
                    loss.append(np.mean(losstmp))
                    R.append(avg_reward[1])
                    avg_reward[:] = 0
                    eps_hist.append(round(self.epsilon,2))
                    epoch_axis.append(e)
                    ## print results
                    break_time = time.time() - start_time
                    time_left = break_time * (float(episodes)/float(e+1)) - break_time
                    eta = '%02d'%(int(time_left)/3600)+":"+'%02d'%((int(time_left)%3600)/60)+":"+'%02d'%int(time_left%60)
                    print('#### {} % | eps={} | score={} | ETA={} ####'.format(round(e/episodes*100,1), round(self.epsilon,2),scores[-1], eta ))
#                    print( 'MAE={} loss={} R={}, Qmean={}'.format(accur[-1], loss[-1],R[-1], Qtest[-1] ))
                    print( 'MAE={} loss={} R={}'.format(accur[-1], loss[-1],R[-1]))
                    avg_reward[:] = 0

### Alle regels hieronder Cntrl+1

        ## plot results
        plt.figure(figsize=[10,10])
        plt.suptitle("Results", fontsize=16)

        ax1 = plt.subplot(6,1,1)
        ax1.plot(epoch_axis, scores,'.-')
        plt.title('Scores')
        plt.grid('on')
        ax1.set_xticklabels([])

        ax2 = plt.subplot(6,1,2)
        ax2.plot(epoch_axis,eps_hist,'.-')
        plt.title('Epsilon')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(6,1,3)
        ax2.plot(epoch_axis,R,'.-')
        plt.title('Total (averaged) reward')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(6,1,4)
        ax2.plot(epoch_axis,Qtest,'.-')
        plt.title('Averaged Q value')
        plt.grid('on')
        ax2.set_xticklabels([])
       
        ax2 = plt.subplot(6,1,5)
        ax2.plot(epoch_axis,accur,'.-')
        plt.title('Accuracy')
        plt.grid('on')
        ax2.set_xticklabels([])
              
        ax = plt.subplot(6,1,6)
        ax.plot(epoch_axis,loss,'.-')
        ax.set_yscale('log')
        plt.title('loss')
        plt.grid('on')
        plt.xlabel('Episode')
        plt.savefig('results.png')
        #plt.close()
        