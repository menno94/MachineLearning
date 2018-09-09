# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:38:40 2018

@author: Arjen
"""

import numpy as np

class env_DEFAULT:
    def __init__(self):
        self.state_shape = 0            # (X,X)
        self.action_size = 0            # X
        self.players = 0                # (1 or 2)
        self.reward_win = 0             # Default 10
        self.reward_draw = 0            # Default none if no draw possible
        self.reward_notdone = 0         # Default -1

    def check_win(self,state):
        # Define if state is winning where state[0]=player who last played
        # Return True or False
        pass 
    
    def check_draw(self,state):
        # Define if state is drawn
        # Return True or False
        return False    # False by default
    
    def get_constrain(self,state):
        # Define which actions are out of bounds
        # Return 'ind' = np.where()
        pass
    
    def get_initial_state(self):
        # Might be used to expand on initializing of states
        state = np.zeros(self.state_shape)
        return state

    def get_next_state(self, state, action):
        # Input action to the state, where state[0] = player at play
        # Returns action as one-hot-encoded by default
        state[0,action] = 1
        return state

    def get_reward(self, state):
        if self.check_win(state):
            return self.reward_win, True            # Win, done = True
        else:
            if self.check_draw(state):
                return self.reward_draw, True       # Draw, done = True
            else:
                return self.reward_notdone, False   # No reward, done = False
    
    def switch_state(self, state):
        switched_state = np.zeros_like(state)
        switched_state[0,:] += state[1,:]
        switched_state[1,:] += state[0,:]
        return switched_state
    
    def test_skill(self, model):
        # Define metric of current acquired skill 
        # Return scalar score value
        pass
            
                