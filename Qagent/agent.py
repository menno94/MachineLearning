from keras.models import Sequential
from keras.layers import Activation, InputLayer, Dense
from keras.optimizers import Nadam
from keras.models import load_model
from keras import callbacks
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy


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
        ##
        self.analyse = False
        
    def act(self, state):
        '''
        Returns the action, next state, final state and reward based on current state
        '''
        action = self.get_action(state)
        next_state = self.env.get_next_state(state,action)
        reward, done = self.env.get_reward(next_state)
        return action, next_state, done, reward
    
    def buildmodel(self, N, learning_rate):
        '''
        Build the model with len(N) layers and N[i] neurons per layer.
        '''
        ## calback for tensorboard
        self.tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=100, write_graph=True, write_images=False, write_grads=True)
        
        ## create model
        model = Sequential()
        model.add(InputLayer(input_shape = (np.prod(self.env.state_shape),) )) # Vreeeeeemd
        for layer in range(len(N)):
            model.add(Dense(N[layer], activation='relu'))
        model.add(Dense(self.env.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(lr=learning_rate), metrics=['mae'])
        return model

    def get_action(self, state):
        '''
        Get action for current state
        '''
        ## no randonness during prediction
        if self.training == False:
            self.epsilon = 0
        ## Exploration vs. Exploitation
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
        ## delay model
        if self.delay_model:
            self.previous_model = self.buildmodel(N, learning_rate)
        if self.double_q:
            self.value_model = self.buildmodel(N, learning_rate)
        print("Finished building the model")

    def evaluate(self, total_R, e, error, loss, loss_std):


        ## score
        Qtesttmp = np.zeros(2)
        for i, item in enumerate(self.env.test_states):
            value = self.model.predict(item.reshape(1, np.prod(self.env.state_shape)))
            Qtesttmp[1] = Qtesttmp[0] + (np.amax(value) - Qtesttmp[0]) / (i + 1)
        #tmp = evaluation['Q']
        #tmp.append(Qtesttmp[1])
        #evaluation['Q'] = tmp

        ## test scores
        #tmp = evaluation['score']
        score = self.env.test_skill(self.model)
        #tmp.append(score)
        #evaluation['score'] = tmp

        ## epsilon
        #tmp = evaluation['epsilon']
        #tmp.append(self.epsilon)
        #evaluation['epsilon'] = tmp

        ## average reward
        #tmp = evaluation['total_R']
        #tmp.append(total_R)
        #evaluation['total_R'] = tmp

        ## loss
        #tmp = evaluation['loss']
        #tmp.append(loss)
        #evaluation['loss'] = tmp

        ## loss
        #tmp = evaluation['loss_std']
        #tmp.append(loss_std)
        #evaluation['loss_std'] = tmp
        ## acc
        #tmp = evaluation['acc']
        #tmp.append(error)
        #evaluation['acc'] = tmp

        ## acc
        #tmp = evaluation['episode']
        #tmp.append(e)
        #evaluation['episode'] = tmp

        ## open file

        f = open('stats.txt','a')
        f.write('e={}\teps={}\tQ={}\tscore={}\ttotal_R={}\tloss={}\tacc={}\n'.format(e, self.epsilon, Qtesttmp[1], score, total_R, loss, error))
        f.close()

        ## print results
        break_time = time.time() - self.start_time
        time_left = break_time * (float(self.episodes) / float(e + 1)) - break_time
        eta = '%02d' % (int(time_left) / 3600) + ":" + '%02d' % ((int(time_left) % 3600) / 60) + ":" + '%02d' % int(
            time_left % 60)
        print('#### {} % | eps={} | score={} | ETA={} ####'.format(round(e / self.episodes * 100, 1), round(self.epsilon, 2),score, eta))
        print('MAE={} loss={} R={}'.format(error, loss, total_R))

        #return evaluation

    def train(self, 
                      episodes          =   100,
                      epsilon           =   1, 
                      epsilon_min       =   0.01, 
                      epsilon_decay     =   0.9999, 
                      batch_size        =   50, 
                      gamma             =   0.95, 
                      memory_length     =   1000,
                      breaks            =   10,
                      model_update_freq=10):
        '''
        Train the Qagent
        '''
        ## set epsilion
        self.epsilon = epsilon
        self.episodes = episodes
        ## set stats dict
        ### oud: evaluation = {'score':[],'epsilon':[],'total_R':[],'avg_q':[],'acc':[],'loss':[],'episode':[],'Q':[], 'loss_std':[]}
        f = open('stats.txt', 'w')
        f.close()
        ## envionment settings
        memory = []
        self.start_time = time.time()
        ## save initial weights and set epoch number
        self.model.save_weights('temp_previous.h5')
        self.model.save_weights('temp_value_model.h5')
        current_epoch = 0

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
                    ##save to analse
                    if self.analyse and e>0:
                        if e>0:
                            temp = np.load('state.npy')
                            temp = temp + state
                            np.save('state',temp)
                    else:
                        temp = state * 0
                        np.save('state',temp)
                        
                        
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

            ## double q learning
            if self.double_q and e % model_update_freq == 0:
                self.value_model.load_weights('temp_value_model.h5')
                self.model.save_weights('temp_value_model.h5')
# =============================================================================
#           Train netwerk
# =============================================================================
            if len(memory) >= batch_size:
                ## model from previous episode
                self.previous_model.load_weights('temp_previous.h5')
                minibatch = random.sample(memory, batch_size)
                errortmp    = []
                losstmp     = []
                for state, action, reward, next_state, done in minibatch:
                    if not done:
                        if self.double_q:
                            target_move = np.argmax(
                                self.model.predict(next_state.reshape(1, np.prod(self.env.state_shape)))[0])
                            target = reward + gamma * self.value_model.predict(next_state.reshape(1, np.prod(self.env.state_shape)))[0][target_move]
                        else:
                            if self.delay_model:
                                target = reward + gamma * np.amax(self.previous_model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])
                            else:
                                target = reward + gamma * np.amax(self.model.predict(next_state.reshape(1,np.prod(self.env.state_shape)))[0])
                    else:
                        target = reward
                    target_f = self.model.predict(state.reshape(1,np.prod(self.env.state_shape)))
                    target_f[0][action] = target

                    stats = self.model.fit(state.reshape(1, np.prod(self.env.state_shape)), target_f,
                                           epochs=current_epoch + 1, initial_epoch=current_epoch, verbose=0)
                    #callback# stats = self.model.fit(state.reshape(1,np.prod(self.env.state_shape)), target_f, epochs=current_epoch+1, initial_epoch=current_epoch, verbose=0,callbacks=[self.tbCallBack], validation_split = 0.2)
                    ## update averaged error based
                    errortmp.append(stats.history['mean_absolute_error'][0])
                    losstmp.append(stats.history['loss'][0])

                ## update epoch after full memory
                current_epoch = current_epoch + 1
                ## save model weights
                self.model.save_weights('temp_previous.h5')
                ## adjust opsilon
                if self.epsilon > epsilon_min:
                    self.epsilon *= epsilon_decay
                if e % breaks == 0:
                    ## stats
                    self.evaluate(total_reward,e,np.mean(errortmp),np.mean(losstmp),np.std(losstmp))




### Alle regels hieronder Cntrl+1
        with open('stats.txt','r') as f:
            lines = f.readlines()

        episode = []
        epsilon = []
        Qtest = []
        score = []
        totalR = []
        loss = []
        acc = []
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



        N = max(1,int(len(episode)/5))
        ## plot results
        plt.figure(figsize=[10,10])
        plt.suptitle("Results", fontsize=16)

        ax1 = plt.subplot(6,1,1)
        ax1.plot(episode, score,'.-')
        ax1.plot(episode, np.convolve(score, np.ones((N,)) / N, mode='same'))

        plt.title('Scores')
        plt.grid('on')
        ax1.set_xticklabels([])

        ax2 = plt.subplot(7,1,2)
        ax2.plot(episode, epsilon,'.-')
        plt.title('Epsilon')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(7,1,3)
        ax2.plot(episode,totalR,'.-')
        ax2.plot(episode, np.convolve(totalR, np.ones((N,))/N, mode='same'))
        plt.title('Total reward')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(7,1,4)
        ax2.plot(episode,Qtest,'.-')
        ax2.plot(episode, np.convolve(Qtest, np.ones((N,)) / N, mode='same'))
        plt.title('Averaged Q value')
        plt.grid('on')
        ax2.set_xticklabels([])
       
        ax2 = plt.subplot(7,1,5)
        ax2.plot(episode,acc,'.-')
        ax2.plot(episode, np.convolve(acc, np.ones((N,)) / N, mode='same'))
        plt.title('Accuracy')
        plt.grid('on')
        ax2.set_xticklabels([])
              
        ax = plt.subplot(7,1,6)
        ax.plot(episode,loss,'.-')
        ax.plot(episode, np.convolve(loss, np.ones((N,)) / N, mode='same'))
        ax.set_yscale('log')
        plt.title('loss')
        plt.grid('on')
        plt.xlabel('Episode')

        ax = plt.subplot(7,1,7)
        #ax.plot(episode,evaluation['loss_std'],'.-')
        #ax.plot(eepisode, np.convolve(evaluation['loss_std'], np.ones((N,)) / N, mode='same'))
        plt.title('std(loss)')
        plt.grid('on')
        plt.xlabel('Episode')
        plt.savefig('results.png')
        #plt.close()
        