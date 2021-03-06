# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:38:40 2018

@author: Arjen
"""

import numpy as np

class env_Vier:
    def __init__(self, stats_name):
        self.state_shape = (1,6,7,1)       # (2 players ,field 7x6)
        self.action_size = 7            # X
        self.players = 2                # (1 or 2)
        self.reward_win = 1             # Default 10
        self.reward_draw = 0            # Default none if no draw possible
        self.reward_notdone = 0         # Default -1
        self.reward_lose = -1
        self.create_test_states()
        self.stats_name = stats_name
        
    def check_win(self,state):
        # Define if state is winning where state[0]=player who last played
        # Return True or False
        temp = state.flatten()
        ## Horizontal
        win = np.array([0,1,2,3])
        for i in range(6):
            if temp[3+7*i]==1:
                for j in range(4):
                    if np.sum(temp[win+7*i+j]) == 4:
                        return True
        ## Vertical
        win = np.array([0,7,14,21])
        for i in range(7):
            if temp[14+i]==1:
                for j in range(3):
                    if np.sum(temp[win+7*j+i]) == 4:
                        return True
        ## Diagonals
        win = np.array([[0,8,16,24],[3,9,15,21]])
        for a in range(2):
            for i in range(4):
                for j in range(3):
                    if np.sum(temp[win[a]+i+7*j])==4:
                        return True
                
        return False
    
    def check_draw(self,state):
        # Define if state is drawn
        # Return True or False
        if np.sum(np.abs(state)) == 42:
            return True
        return False    # False by default
    
    def create_test_states(self):

        state1 = self.get_initial_state()
        self.test_states = [state1]
    
    def get_constrain(self,state):
        # Define which actions are out of bounds
        # Return 'ind' = np.where()
        temp = state.flatten()
        temp = np.abs(temp[0:self.state_shape[2]])
        ind = np.where(temp == 1)
        return ind
    
    def get_initial_state(self):
        # Might be used to expand on initializing of states
        state = np.zeros(self.state_shape)
        return state

    def get_next_state(self, state, action):
        # Input action to the state, where state[0] = player at play
        # Returns action as one-hot-encoded by default

        for i in range(self.state_shape[1]):
            if state[0,self.state_shape[1]-1-i,action,0]==0:
                state[0,self.state_shape[1]-1-i,action,0]=1
                return state

    def get_reward(self, state):
        if self.check_win(state):
            return self.reward_win, True            # Win, done = True
        else:
            if self.check_draw(state):
                return self.reward_draw, True       # Draw, done = True
            else:
                return self.reward_notdone, False   # No reward, done = False
            
    def print_state(self, state):
        board = state.astype('object')
        # board = np.flip(board.reshape(6,7),0).astype('object')
        board[board==1] = 'x'
        board[board==-1] = 'o'
        board[board==0] = ' '
        print(board)
        
    def switch_state(self, state):
        switched_state = state*-1
        # switched_state[0,:] += state[1,:]
        # switched_state[1,:] += state[0,:]
        return switched_state
    
    def test_skill(self, model):
        # Define metric of current acquired skill 
        # Return scalar score value
        win = 0; draw = 0; lose = 0; loseo=0
        losing_sequences = []
        for j in range(12):
            for i in range(6):          # loop was van iets ouds, lekker laten zo!
                state = self.get_initial_state()           
                action_list = []
                while True:
                    ind = self.get_constrain(state)
                    act_values = np.random.rand(1, self.action_size)
                    # opt_action = optimal_agent(state)
                    # act_values[0,opt_action] = 1000
                    act_values[0,ind] = -1000
                    action = np.argmax(act_values)
                    next_state = self.get_next_state(state,action)
                    action_list.append(action+1)
                    if self.check_win(next_state):
                        lose += 1
                        loseo += 1
                        temp = action_list.copy()
                        losing_sequences.append(temp)
                        # score -= 1
                        break
                    if self.check_draw(next_state):
                        draw += 1
                        break                
                    state = self.switch_state(next_state)
                    ind = self.get_constrain(state)
                    act_values = model.predict(state)

                    act_values[0,ind] = -1000
                    action = np.argmax(act_values)
                    action_list.append(action+1)
                    next_state = self.get_next_state(state, action)
                    if self.check_win(next_state):
                        win += 1
                        break
                    if self.check_draw(next_state):
                        draw += 1
                        break
                    state = self.switch_state(next_state)
                    
            for i in range(6):
                state = self.get_initial_state()
                while True:
                    ind = self.get_constrain(state)
                    act_values = model.predict(state)
                
                    
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
                    # opt_action = optimal_agent(state)
                    # act_values[0,opt_action] = 1000
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
        print('win={} draw={} losex={} loseo={}'.format(win, draw, lose-loseo, loseo))
        score = lose
        if not losing_sequences==[]:
            print(losing_sequences[-1])
        return score
        return 0
            
                