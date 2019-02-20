# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:38:40 2018

@author: Arjen
"""

import numpy as np

class env_Vier:
    def __init__(self):
        self.state_shape = (2,42)       # (2 players ,field 7x6)
        self.action_size = 7            # X
        self.players = 2                # (1 or 2)
        self.reward_win = 10             # Default 10
        self.reward_draw = 3            # Default none if no draw possible
        self.reward_notdone = 0         # Default -1
        self.create_test_states()
        
    def check_win(self,state):
        # Define if state is winning where state[0]=player who last played
        # Return True or False
        temp = state[0]
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
        if np.sum(state) == 42:
            return True
        return False    # False by default
    
    def create_test_states(self):

        state1 = self.get_initial_state()
        self.test_states = [state1]
    
    def get_constrain(self,state):
        # Define which actions are out of bounds
        # Return 'ind' = np.where()
        temp = state[0][35:42] + state[1][35:42]
        ind = np.where(temp == 1)
        return ind
    
    def get_initial_state(self):
        # Might be used to expand on initializing of states
        state = np.zeros(self.state_shape)
        return state

    def get_next_state(self, state, action):
        # Input action to the state, where state[0] = player at play
        # Returns action as one-hot-encoded by default
        col = np.array([0,7,14,21,28,35]) + action
        count = int(np.sum(state[0][col])+np.sum(state[1][col]))
        state[0,action + count*7] = 1
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
        board = state[0] - state[1]
        board = np.flip(board.reshape(6,7),0).astype('object')
        board[board==1] = 'x'
        board[board==-1] = 'o'
        board[board==0] = ' '
        print(board)
        
    def switch_state(self, state):
        switched_state = np.zeros_like(state)
        switched_state[0,:] += state[1,:]
        switched_state[1,:] += state[0,:]
        return switched_state
    
    def test_skill(self, model):
        # Define metric of current acquired skill 
        # Return scalar score value
        return 0
            
                