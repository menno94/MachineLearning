# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:38:40 2018

@author: Arjen
"""

import numpy as np
from random import randint

class env_Simple:
    def __init__(self):
        self.state_shape = (1,7)            # (X,X)
        self.action_size = 7            # X
        self.players = 1                # (1 or 2)
        self.reward_win = 10             # Default 10
        self.reward_draw = None            # Default none if no draw possible
        self.reward_notdone = -5         # Default -1

    def check_win(self,state):
        # Define if state is winning where state[0]=player who last played
        # Return True or False
        if np.argmax(state)>4:
            return True
        return False
    
    def check_draw(self,state):
        # Define if state is drawn
        # Return True or False
        return False    # False by default
    
    def get_constrain(self,state):
        # Define which actions are out of bounds
        # Return 'ind' = np.where()
        #                  0,1,2,3,4,5,6
        moves = np.array([[0,1,0,1,1,1,0],
                          [1,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0],
                          [1,0,1,0,0,0,0],
                          [1,0,0,0,0,0,1]])
#        print(state)
        current_position = np.argmax(state)
#        print(current_position)
        ind = np.where(moves[current_position]==0)
        return ind
    
    def get_initial_state(self):
        # Might be used to expand on initializing of states
        state = np.zeros(self.state_shape)
        start = randint(0,4)
        state[0,start] = 1
        return state

    def get_next_state(self, state, action):
        # Input action to the state, where state[0] = player at play
        # Returns action as one-hot-encoded by default
        state = np.zeros_like(state)
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
        counter = 0
        for i in range(5):
            short_count = 0
            state = np.zeros(self.state_shape)
            state[0,i] = 1
            while True:
                counter += 1
                act_values = model.predict(state.reshape(1,np.prod(self.state_shape)))
                ind = self.get_constrain(state)
                act_values[0,ind] = -1000
                action = np.argmax(act_values)
                if short_count == 0:
                    print(i,action)
                next_state = self.get_next_state(state,action)
                reward, done = self.get_reward(next_state)
                state = next_state
                short_count += 1
                if done or counter > 50:
                    break
        return counter
            
                