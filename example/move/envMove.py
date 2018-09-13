import numpy as np
from random import randint

class env_move:

    def __init__(self):
        self.field_shape = (1,3)
        self.state_shape = (2,np.prod(self.field_shape))            # (X,X)
        self.action_size = 4            # X
        self.players = 1                # (1 or 2)
        self.reward_win = 10             # Default 10
        self.reward_draw = None            # Default none if no draw possible
        self.reward_notdone = -1         # Default -1
        self.counter = 0
        self.max_counter = 19
    
    def check_win(self, state):
        '''
        Determine wheter there are 3 in a row. Player must be -1 or 1.
        '''
        s     = state.reshape(self.state_shape).copy()
        temp1 = s[0,:]
        temp2 = s[1,:]
        for i in range(len(temp1)):
            if temp1[i]==temp2[i]==1:
                return True
        return False

    def check_draw(self,state):
        # Define if state is drawn
        # Return True or False
        return False    # False by default
    
    def get_constrain(self,state):
        temp = state.reshape(self.state_shape)[0].copy().reshape(self.field_shape)       #goed naar kijken!!!
        ind = []
        if np.max(temp[:,0]) ==1:
            ind.append(3)
        if np.max(temp[:,-1]) ==1:
            ind.append(1)
        if np.max(temp[0,:]) ==1:
            ind.append(0)
        if np.max(temp[-1,:]) ==1:
            ind.append(2)

        return ind

    def get_initial_state(self):
        state = np.zeros(self.state_shape)
        ind = randint(0,self.state_shape[1]-1)
        state[1,ind] = 1

        while True:
            ind = randint(0,self.state_shape[1]-1)
            if state[1,ind]==1:
                pass
            else:
                state[0,ind] = 1
                break
        self.counter = 0
        return state


    def get_next_state(self, state, action):
        '''
        Returns the new state based on the action. Player must be -1 or 1.
        '''
        s = state
        temp = s[0,:].copy().reshape(self.field_shape)
        row, col = np.where(temp==1)
        row, col = row[0],col[0]
        temp[:] = 0
        if action == 0:
            row = row-1
        elif action==1:
            col = col +1
        elif action==2:
            row = row +1
        elif action==3:
            col =col - 1

        temp[row,col] = 1
        s[0,:] = temp.flatten()
        self.counter += 1
        return s

    def get_reward(self, state):
        if self.counter > self.max_counter:
            return self.reward_notdone, True
        if self.check_win(state):
            return self.reward_win, True            # Win, done = True
        else:
            if self.check_draw(state):
                return self.reward_draw, True       # Draw, done = True
            else:
                return self.reward_notdone, False   # No reward, done = False

    def test_skill(self, model):
        # Define metric of current acquired skill 
        # Return scalar score value
        score = 0 #Perfect network scores 0 on test
        for i in range(50):
            count = 0
            state = self.get_initial_state()
            player = np.argmax(state[0,:])
            treasure = np.argmax(state[1,:])
            row_p = player//self.field_shape[1]; col_p = player%self.field_shape[1]
            row_t = treasure//self.field_shape[0]; col_t = treasure%self.field_shape[0]
            optimal_count = abs(col_p-col_t) + abs(row_p-row_t)   
            while True:
                count += 1
                act_values = model.predict(state.reshape(1,np.prod(self.state_shape)))
                ind = self.get_constrain(state)
                act_values[0,ind] = -1000
                action = np.argmax(act_values)
                next_state = self.get_next_state(state, action)
                if self.check_win(next_state):
#                    count += 1000000
                    break
                if count > 20:
#                    count = 1000 + optimal_count
                    break
            score += count - optimal_count
        return score