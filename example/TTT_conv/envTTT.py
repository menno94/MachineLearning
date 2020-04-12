# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:38:40 2018

@author: Arjen 
"""

import numpy as np
from random import randint
from operator import xor
#from optimal_agent import optimal_agent

class env_TTT:
    def __init__(self):
        # first dimension is player 1. Second dimension player 2
        self.state_shape = (2,3,3)
        # moves are numpad (1 - 9)
        self.action_size = 9
        self.players = 2
        # De reward structure 1/0/-1 was de meest stabiele, ook omdat dit zero-sum is, wat ik nog niet helemaal honderd procent snap
        self.reward_win = 1
        self.reward_draw = 0
        self.reward_notdone = 0
        self.reward_lose = -1
        self.create_test_states()

    def optimalAgent(self,state):
        move = optimal_agent(state)
        tmp = np.zeros((1,self.action_size))
        tmp[0,move] = 10
        return tmp

    def create_test_states(self):

        state1 = self.get_initial_state()
        state1 = state1.reshape(2,9)

        state2 = self.get_initial_state()
        state2 = state2.reshape(2,9)
        
        state2[0,1] = 1

        state3 = self.get_initial_state()
        state3 = state3.reshape(2,9)
        state3[0,6] = 1
        state3[1, 0] = 1
        state3[0, 7] = 1

        state4 = self.get_initial_state()
        state4 = state4.reshape(2,9)
        state4[0,3] = 1
        state4[1, 6] = 1
        state4[0, 5] = 1

        state6 = self.get_initial_state()
        state6 = state6.reshape(2,9)
        state6[0,8] = 1
        state6[1, 4] = 1
        state6[0, 5] = 1
        state6[1, 1] = 1

        state5 = self.get_initial_state()
        state5 = state5.reshape(2,9)
        state5[0,0] = 1
        state5[1, 4] = 1
        state5[0, 1] = 1
        state5[1, 8] = 1
        
        state6 = state6.reshape(2,3,3)
        state5 = state5.reshape(2,3,3)
        state4 = state4.reshape(2,3,3)
        state3 = state3.reshape(2,3,3)
        state2 = state2.reshape(2,3,3)
        state1 = state1.reshape(2,3,3)
        
        self.test_states = [state1, state2, state3, state4, state5, state6]

    def check_win(self,state):
        state = state.reshape(2,9)
        wins = np.array([[7,8,9], [4,5,6], [1,2,3],[7,4,1],[8,5,2],[9,6,3],[7,5,3],[9,5,1]]) - 1  
        for i in range(np.shape(wins)[0]):
            if np.sum(state[0][wins[i]]) == 3:
                return True
        return False
    
    def check_draw(self,state):
        if np.sum(state) == 9:
            return True
        return False
    
    def get_constrain(self,state):
        tmp_state = state.copy()
        # if np.sum(tmp_state) == 0:
        #     a = [2,3,5,6,7,8]
        #     tmp_state[0,a] = 1
        temp = np.sum(tmp_state,axis=0)
        temp = temp.reshape(1,9)
        ind = np.where(temp == 1)
        return ind

    def get_initial_state(self, random = False):
        state = np.zeros(self.state_shape)
        #state = state.reshape(2,9)
        if random:
            # Start every game with a random first move to train/check general skill
            pass
        return state

    def get_next_state(self, state, action):
        #turn = int(np.sum(state)%2)  # If even then turn==0 ('x')
        temp = state.copy()
        temp = temp.reshape(2,9)
        temp[0,action] = 1
        temp = temp.reshape(2,3,3)
        return temp

    def get_reward(self, state):
        win = self.check_win(state)
        if win:
            reward = self.reward_win
            done = True
        else:
            draw = self.check_draw(state)
            if draw:
                reward = self.reward_draw
                done = True
            else:
                reward = self.reward_notdone
                done = False
        return reward, done
    
    def save_model(self, name):
        self.model.save(name)
    
    def switch_state(self, state):
        switched_state = np.zeros_like(state)
        switched_state[0,:] += state[1,:]
        switched_state[1,:] += state[0,:]
        return switched_state
    
    def test_skill(self, model):
        win = 0; draw = 0; lose = 0;
        for j in range(12):
            # for i in range(3):          # loop was van iets ouds, lekker laten zo!
            #     state = self.get_initial_state()
            #     while True:  
            #         ind = self.get_constrain(state)
            #         act_values = np.random.rand(1, self.action_size)
            #         act_values[0,ind] = -1000
            #         action = np.argmax(act_values)
            #         next_state = self.get_next_state(state,action)
            #         if self.check_win(next_state):
            #             lose += 1
            #             # score -= 1
            #             break
            #         if self.check_draw(next_state):
            #             draw += 1
            #             break                
            #         state = self.switch_state(next_state)
            #         ind = self.get_constrain(state)
            #         act_values = model.predict(state.reshape(1,18))
            #         act_values[0,ind] = -1000
            #         action = np.argmax(act_values)
            #         next_state = self.get_next_state(state, action)
            #         if self.check_win(next_state):
            #             win += 1
            #             break
            #         if self.check_draw(next_state):
            #             draw += 1
            #             break
            #         state = self.switch_state(next_state)
                    
            for i in range(6):
                state = self.get_initial_state()
                while True:
                    ind = self.get_constrain(state)
                    act_values = model.predict(state[np.newaxis,:])
                    #opt_action = optimal_agent(state)
                    #act_values[0,opt_action] = 1000
                    act_values[0,ind] = -1000
                    action = np.argmax(act_values)
                    next_state = self.get_next_state(state, action)

                    if self.check_win(next_state):
                        win += 1
                        break
                    if self.check_draw(next_state):
                        draw += 1
                        break
                    state = self.switch_state(next_state)
                    ind = self.get_constrain(state)
                    act_values = np.random.rand(1, self.action_size)
                    act_values[0,ind] = -1000
                    action = np.argmax(act_values)
                    next_state = self.get_next_state(state,action)
                    if self.check_win(next_state):
                        # score -= 1
                        lose += 1
                        break
                    if self.check_draw(next_state):
                        draw += 1
                        break                
                    state = self.switch_state(next_state)
        print('win={} draw={} lose={}'.format(win, draw, lose))
        score = lose/(12*6)
        return score
            
    def value_move(self,state,action):
        scores = []
        for i in range(self.action_size):
            potential_action = i
            potential_state = self.get_next_state(state, potential_action)
            scores.append(self.value_state(potential_state))
        missed_value = max(scores) - scores[action]
        return missed_value
        
    def value_state(self, state):
        wins = np.array([[7,8,9], [4,5,6], [1,2,3],[7,4,1],[8,5,2],[9,6,3],[7,5,3],[9,5,1]]) - 1  
        x_score = 0
        if self.check_win(state):
            return 10
        for i in range(np.shape(wins)[0]):
            if xor(np.sum(state[0][wins[i]]) == 0,np.sum(state[1][wins[i]]) == 0):
                x_score += np.sum(state[0][wins[i]]) - np.sum(state[1][wins[i]])
        return x_score
                