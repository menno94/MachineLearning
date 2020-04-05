# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:38:40 2018

@author: Arjen
"""

import numpy as np

class env_Vier:
    def __init__(self, stats_name, dim=[6,7], connect=4):
        self.state_shape_dual = (1,dim[0],dim[1],1)   # (2 players ,field 7x6)
        self.action_size = dim[1]       # X
        self.row_size = dim[0]
        self.state_size = self.action_size*self.row_size
        self.state_shape = (2,self.state_size)
        self.connect = connect          # Connect this amount to win 'connect_four'
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
        winh = np.arange(self.connect)
        for i in range(self.row_size):
            for j in range(self.action_size-self.connect+1):
                if np.sum(temp[winh+self.action_size*i+j]) == self.connect:
                    return True
        ## Vertical
        winv = winh*self.action_size
        for i in range(self.action_size):
            for j in range(self.row_size-self.connect+1):
                if np.sum(temp[winv+self.action_size*j+i]) == self.connect:
                    return True
        ## Diagonals
        win = np.array([winv+winh,winv-winh+self.connect-1])
        for a in range(2):
            for i in range(self.action_size-self.connect+1):
                for j in range(self.row_size-self.connect+1):
                    if np.sum(temp[win[a]+i+self.action_size*j])==self.connect:
                        return True
        return False
    
    def check_draw(self,state):
        # Define if state is drawn
        # Return True or False
        if np.sum(np.abs(state)) == self.row_size*self.action_size:
            return True
        return False    # False by default
    
    def create_test_states(self):

        state1 = self.get_initial_state()
        self.test_states = [state1]
        
    def state_reshape(self,state):
        temp = np.array(state.reshape(self.state_shape_dual))
        return temp
    
    def dual_state(self,state):
        temp = np.zeros((2,self.row_size*self.action_size))
        ind0 = np.where(state==1)
        ind1 = np.where(state==-1)
        temp[0,ind0] = 1
        temp[1,ind1] = 1
        return temp
        
    def solo_state(self,state):
        temp = self.get_initial_state()
        state0 = state[0,:]
        state1 = state[1,:]
        ind0 = np.where(state0==1)
        ind1 = np.where(state1==1)
        temp[0][ind0] = 1
        temp[0][ind1] = -1
        return temp
    
    def state_flatten(self,state):
        return np.ndarray.flatten(state)
    
    def get_constrain(self,state):
        # Define which actions are out of bounds
        # Return 'ind' = np.where()
        temp = state.flatten()
        temp = np.abs(temp[0:self.action_size])
        ind = np.where(temp == 1)
        return ind
    
    def get_initial_state(self):
        # Might be used to expand on initializing of states
        state = np.zeros(self.state_shape_dual)
        return state

    def get_next_state(self, state, action):
        # Input action to the state, where state[0] = player at play
        # Returns action as one-hot-encoded by default
        for i in range(self.row_size):
            if state[0,self.row_size-1-i,action,0]==0:
                state[0,self.row_size-1-i,action,0]=1
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
                    act_values = model.predict(self.dual_state(state).reshape(1,2*self.state_size))
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
                    act_values = model.predict(self.dual_state(state).reshape(1,2*self.state_size))
                
                    
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
            
                