from keras.models import Sequential
from keras.layers import Activation, InputLayer, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Nadam, SGD, RMSprop
from keras.models import load_model, clone_model
from keras import callbacks
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import os



class Q_agent:
    def __init__(self, env, training = True):
        ## environment
        self.env = env
        ## switch between prediction and training
        self.training = training
        ## use frozen network with a delay for the target value
        self.delay_model = True
        ## double q
        self.double_q = True
        ## store data about training
        self.analyse = True
        ## store all states (works only if self.analyse=True)
        self.analyse_full = False
        ## apply a optimal agent as opponent
        self.opponent_optimal = False
        ## prioritized replay
        self.prioritized_replay = True
        
        self.conv = False
        
    def act(self, state, turn = 0, agent_nr =0):
        '''
        Returns the action, next state, final state and reward based on current state
        '''
        action          = self.get_action(state, turn, agent_nr)
        next_state      = self.env.get_next_state(state,action)
        reward, done    = self.env.get_reward(next_state)
        return action, next_state, done, reward
    
    def buildmodel(self, N, learning_rate):
        '''
        Build the model with len(N) layers and N[i] neurons per layer.
        '''
        ## calback for tensorboard
        self.tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=100, write_graph=True, write_images=False, write_grads=True)
        
        ## create model
        model = Sequential()
        model.add(InputLayer(input_shape = (np.prod(self.env.state_shape),) ))
        # np.prod is not need when using flatten: model.add(InputLayer(input_shape = (self.env.state_shape) )) 
        for layer in range(len(N)):
            model.add(Dense(N[layer], activation='relu'))
        # np.prod is not need when using flatten: model.add(Flatten())
        model.add(Dense(self.env.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(lr=learning_rate), metrics=['mae'])
        return model

    def buildmodel_conv(self, learning_rate):
        '''
        Build the model with len(N) layers and N[i] neurons per layer.
        '''
        ## calback for tensorboard
        self.tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=100, write_graph=True, write_images=False, write_grads=True)
        
        ## create model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1),
                 activation='relu',
                 input_shape=(self.env.state_shape) )) 
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(Conv2D(64, (1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.env.action_size, activation='softmax'))

        model.compile(loss='mse', optimizer=Nadam(lr=learning_rate), metrics=['mae'])
        return model

    def get_action(self, state, turn = 0, agent_nr = 0):
        '''
        Get action for current state
        '''
        ## no randonness during prediction
        if not self.training:
                self.epsilon = 0
        if turn % 2 == 0:           
            ## Exploration vs. Exploitation
            if np.random.rand() <= self.epsilon:
                act_values = np.random.rand(1, self.env.action_size)
            else:
                act_values = self.model.predict(self.state_to_predict(state))
        else:
            ## optimal qagent
            if self.opponent_optimal:
                act_values = self.env.optimalAgent(state)
            else:      
                act_values = self.list_agents[agent_nr].predict(self.state_to_predict(state))
        ## Constrain
        ind = self.env.get_constrain(state)
        act_values[0,ind] = -1000
        action = np.argmax(act_values)
        if self.training:
            return action
        else:
            return action, act_values

            
    def state_to_predict(self,state):
        if self.conv:
            return state[np.newaxis,:]
        else:
            return state.reshape(1,np.prod(self.env.state_shape))
            ## np.prod is not need when using flatten: return state[np.newaxis,:]
            
            
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
        #model.load_weights('temp_previous.h5')
        self.model = model
    
    def create_model(self,N, learning_rate):
        '''
        Create model from scratch.
        N = list with the neurons in the hidden layers
        '''
        if self.conv:
            self.model = self.buildmodel_conv(learning_rate)
        else:
            self.model = self.buildmodel(N, learning_rate)
        ## delay model
        if self.delay_model:
            if self.conv:
                self.previous_model = self.buildmodel_conv(learning_rate)
            else:
                self.previous_model = self.buildmodel(N, learning_rate)
        if self.double_q:
            if self.conv:
                self.value_model = self.buildmodel_conv(learning_rate)
            else:
                self.value_model = self.buildmodel(N, learning_rate)
        print("Finished building the model")

    def evaluate(self, total_R, e, error_list, loss_list,P):
        ''' 
        Evaluate agent
        '''
        error = np.mean(error_list)
        loss  = np.mean(loss_list)
        ## mean Q score for test states
        Qtest_mean = np.zeros(2)
        for i, item in enumerate(self.env.test_states):
            value = self.model.predict(self.state_to_predict(item))
            Qtest_mean[1] = Qtest_mean[0] + (np.amax(value) - Qtest_mean[0]) / (i + 1)

        ## score from test_skill function
        score = self.env.test_skill(self.model)
        
        ##
        mean_p  = np.mean(P)
        std_p   = np.std(P)
        
        ## open file
        f = open('stats.txt','a')
        f.write('e={}\teps={}\tQ={}\tscore={}\ttotal_R={}\tloss={}\tacc={}\tmeanP={}\tstdP={}\n'.format(e, self.epsilon, Qtest_mean[1], score, total_R, loss, error,mean_p,std_p))
        f.close()

        ## print results
        break_time = time.time() - self.start_time
        time_left = break_time * (float(self.episodes) / float(e + 1)) - break_time
        eta = '%02d' % (int(time_left) / 3600) + ":" + '%02d' % ((int(time_left) % 3600) / 60) + ":" + '%02d' % int(
            time_left % 60)
        print('#### {} % | eps={} | score={} | ETA={} ####'.format(round(e / self.episodes * 100, 1), round(self.epsilon, 2),score, eta))
        print('MAE={} loss={} R={}'.format(error, loss, total_R))

        #return evaluation
        total_R = 0
        return total_R
        

    def train(self, 
                      episodes          =   100,
                      epsilon           =   1, 
                      epsilon_min       =   0.01, 
                      epsilon_decay     =   0.9999, 
                      batch_size        =   50, 
                      gamma             =   0.95, 
                      memory_length     =   1000,
                      breaks            =   10,
                      model_update_freq =   10,
                      opponent_freq     =   100,
                      max_agent_pool    =   10):
        '''
        Train the Qagent
        '''
        ## set epsilion
        self.epsilon = epsilon
        self.episodes = episodes
        ## set stats dict
        f = open('stats.txt', 'w')
        f.close()
        total_reward = 0
        ## envionment settings
        memory = []
        self.start_time = time.time()
        ## save initial weights and set epoch number
        self.model.save_weights('temp_previous.h5')
        self.model.save_weights('temp_value_model.h5')
        current_epoch = 0
        self.list_agents = [clone_model(self.model)]         
            
        for e in range(episodes):
            state       = self.env.get_initial_state() 
            
# =============================================================================
#       Two player game
# =============================================================================
            if self.env.players == 2:
                agent_nr    = random.randint(0,len(self.list_agents)-1)
                turn        = random.randint(0,1)
                if turn == 1:
                    action2, next_state2, done2, reward2 = self.act(state, turn, agent_nr)
                    state = self.env.switch_state(next_state2)
                
                ## first move
                #action, next_state, done, reward = self.act(state,turn,agent_nr)
                
                while True:
                    ## player x (move)
                    action, next_state, done, reward = self.act(state,turn,agent_nr)
                    ## update total reward
                    total_reward += reward
                    ## if done (memory append next_state) only win or draw
                    if done:
                        memory.append((state,action, reward, next_state,True,0))
                        break
                    ## change turn
                    turn += 1
                    ## player o 
                    state2 = self.env.switch_state(next_state)
                    action2, next_state2, done2, reward2 = self.act(state2, turn, agent_nr)
                    
                    if done2:
                        if reward2==self.env.reward_win:
                            reward = self.env.reward_lose
                        state2 = self.env.switch_state(next_state2)
                        memory.append((state,action, reward,state2,done2,0))
                        break
                        
                    
                    state2 = self.env.switch_state(next_state2)

                    memory.append((state,action, reward,state2,done2,0))
                    state = state2
                    turn += 1
                    ## ---
                    while len(memory)>memory_length:
                            del memory[0]
                    #save to analse
                    if self.analyse:
                        ## save cumulative state
                        if e==0:
                            state_sum = state * 0
                        ## save first state as zero
                        else:
                            state_sum = state_sum + state
                        ## 
                        if self.analyse_full:
                            if not os.path.exists('analyse'):
                                os.makedirs('analyse')
                            np.save('analyse/{:010.0f}_{:010.0f}'.format(e,turn),state)
# =============================================================================
#           One player game
# =============================================================================
            else:    # If 1 player game
                turn = 0
                while True:
                    action, next_state, done, reward = self.act(state)
                    total_reward += reward
                    memory.append((state,action,reward,next_state,done,0))
                    if len(memory)>memory_length:
                        del memory[0]
                    state = next_state.copy()
                    ##save to analse
                    if self.analyse:
                        ## save cumulative state
                        if e>0:
                            state_sum = state_sum + state
                        ## save first state as zero
                        else:
                            state_sum = state * 0
                    if done:
                        break

            ## double q learning
            if self.double_q and e % model_update_freq == 0:
                self.value_model.load_weights('temp_value_model.h5')
                self.model.save_weights('temp_value_model.h5')
                
            if e % opponent_freq == 0 and e > 0:
                # if len(agent_pool) > max_agent_pool
                if len(self.list_agents) > max_agent_pool:
                    random.shuffle(self.list_agents)
                    self.list_agents[-1].set_weights(self.model.get_weights()) 
                else:
                    self.list_agents.append(clone_model(self.model))
                    self.list_agents[-1].set_weights(self.model.get_weights())                
                
# =============================================================================
#           Train netwerk
# =============================================================================
            if len(memory) >= batch_size:
                ## model from previous episode
                if not self.double_q and self.delay_model:
                    self.previous_model.load_weights('temp_previous.h5')
                ## distribution of memory entries 
                p = np.zeros(len(memory))
                ## prioritized replay
                if self.prioritized_replay:
                    for ii in range(len(memory)):
                        p[ii] = memory[ii][-1]
                    ## set values of 0 to mean loss
                    I       = np.where(p==0)
                    p[I]    = np.mean(p)
                ## if all entries are 0 set to 1.
                if np.sum(p) == 0:
                    p[:] = 1
                ## normalise p
                p = p/np.sum(p)
                ## choose entries from memory given a distribution p
                index = np.random.choice(len(memory),batch_size,False,p)
                minibatch = []
                for i in range(len(index)):
                    minibatch.append(memory[index[i]])
                # minibatch = random.sample(memory, batch_size)
                error_list    = []
                loss_list     = []
                
                for jj, (state, action, reward, next_state, done, _) in enumerate( minibatch):
                    if not done:
                        if self.double_q:
                            target_move = np.argmax(
                                self.model.predict(self.state_to_predict(state))[0])
                            target = reward + gamma * self.value_model.predict(self.state_to_predict(state))[0][target_move]

                        else:
                            if self.delay_model:
                                target = reward + gamma * np.amax(self.previous_model.predict(self.state_to_predict(state))[0])
                            else:
                                target = reward + gamma * np.amax(self.model.predict(self.state_to_predict(state))[0])
                    else:
                        target = reward
                    target_f = self.model.predict(self.state_to_predict(state))
                    target_f[0][action] = target

                    stats = self.model.fit(self.state_to_predict(state), target_f,batch_size=1,
                                           epochs=current_epoch + 1, initial_epoch=current_epoch, verbose=0)
                    #callback# stats = self.model.fit(state.reshape(1,np.prod(self.env.state_shape)), target_f, epochs=current_epoch+1, initial_epoch=current_epoch, verbose=0,callbacks=[self.tbCallBack], validation_split = 0.2)
                    ## update averaged error based
                    error_list.append(stats.history['mean_absolute_error'][0])
                    current_loss = stats.history['loss'][0]
                    loss_list.append(current_loss)
                    ## update weights in memory needed for prioritized learning 
                    temp                = list(memory[index[jj]])
                    temp[-1]            = current_loss
                    memory[index[jj]]   = tuple(temp)

                ## update epoch after full memory
                current_epoch = current_epoch + 1
                ## save model weights
                if not self.double_q and self.delay_model:
                    self.model.save_weights('temp_previous.h5')
                ## adjust opsilon
                if self.epsilon > epsilon_min:
                    self.epsilon *= epsilon_decay
                if e % breaks == 0 and e>0:
                    ## stats
                    total_reward = self.evaluate(total_reward,e,error_list,loss_list,p)




### Alle regels hieronder Cntrl+1
        ##
        if self.analyse:
            np.save('sum_state.npy',state_sum)


                    
        with open('stats.txt','r') as f:
            lines = f.readlines()

        episode = []
        epsilon = []
        Qtest = []
        score = []
        totalR = []
        loss = []
        acc = []
        mean_p = []
        std_p = []
        for line in lines:
            for item in line.split('\t'):
                var, number = item.split('=')
                number = float(number.split('\n')[0])
                if var =='e':
                    episode.append(number)
                if var =='eps':
                    epsilon.append(number)
                if var =='Q':
                    Qtest.append(number)
                if var =='score':
                    score.append(number)
                if var=='total_R':
                    totalR.append(number)
                if var =='loss':
                    loss.append(number)
                if var =='acc':
                    acc.append(number)
                if var =='meanP':
                    mean_p.append(number)
                if var =='stdP':
                    std_p.append(number)

        N = max(1,int(len(episode)/5))
        ## plot results
        plt.figure(figsize=[12,12])
        plt.suptitle("Results", fontsize=16)

        ax1 = plt.subplot(7,1,1)
        ax1.plot(episode, score,'.-')
        ax1.plot(episode, np.convolve(score, np.ones((N,)) / N, mode='same'))

        plt.ylabel('Scores')
        plt.title('Scores based on test skill function')
        plt.grid('on')
        ax1.set_xticklabels([])

        ax2 = plt.subplot(7,1,2)
        ax2.plot(episode, epsilon,'.-')
        plt.ylabel('Epsilon')
        plt.title('Epsilon per epoch')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(7,1,3)
        ax2.plot(episode,totalR,'.-')
        ax2.plot(episode, np.convolve(totalR, np.ones((N,))/N, mode='same'))
        plt.ylabel('Total reward') ## total reward 
        plt.title('Total reward within the breaks interval')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(7,1,4)
        ax2.plot(episode,Qtest,'.-')
        ax2.plot(episode, np.convolve(Qtest, np.ones((N,)) / N, mode='same'))
        plt.title('Averaged Q value for given test states')
        plt.ylabel('averaged Q')
        plt.grid('on')
        ax2.set_xticklabels([])
       
        ax2 = plt.subplot(7,1,5)
        ax2.plot(episode,acc,'.-')
        ax2.plot(episode, np.convolve(acc, np.ones((N,)) / N, mode='same'))
        plt.ylabel('Accuracy')
        plt.title('Mean accuracy over all batches')
        plt.grid('on')
        ax2.set_xticklabels([])
              
        ax = plt.subplot(7,1,6)
        ax.plot(episode,loss,'.-')
        ax.plot(episode, np.convolve(loss, np.ones((N,)) / N, mode='same'))
        ax.set_yscale('log')
        plt.ylabel('loss')
        plt.title('Mean loss over all batches')
        plt.grid('on')
        plt.xlabel('Episode')
        
        ax = plt.subplot(7,1,7)
        ax.plot(episode,mean_p,'.-')
        ax.plot(episode,std_p,'.-')
        ax.plot(episode, np.convolve(mean_p, np.ones((N,)) / N, mode='same'))
        
        ax.plot(episode, np.convolve(std_p, np.ones((N,)) / N, mode='same'))
        plt.ylabel('P')
        plt.legend(['mean','std'])
        plt.title('stats of prioritized replay function')
        plt.grid('on')
        plt.xlabel('Episode')        

        plt.xlabel('Episode')
        plt.savefig('results.png')
        #plt.close()
        