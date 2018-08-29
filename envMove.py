import numpy as np
from random import randint

class env_move:

    def __init__(self):
        # first dimension is player. Second dimension the treasure
        self.state_shape = (1,32)
        # 0=up 1=right 2=down 3=left
        self.actions = 4

    def check_winner(self, state, player):
        '''
        Determine wheter there are 3 in a row. Player must be -1 or 1.
        '''
        state = state.reshape(2,16).copy()
        temp1 = state[0,:]
        temp2 = state[1,:]
        win = False
        for i in range(len(temp1)):
            if temp1[i]==temp2[i]==1:
                win = True
                break
        return win

    def get_constrain(self,state,actions,player):
        temp = state.reshape(2,16)[0].copy().reshape(4,4)
        ind = []
        if np.max(temp[:,0]) ==1:
            ind.append(3)
        if np.max(temp[:,-1]) ==1:
            ind.append(1)
        if np.max(temp[0,:]) ==1:
            ind.append(0)
        if np.max(temp[-1,:]) == 1:
            ind.append(2)

        return ind

    def get_initial_state(self):
        state = np.zeros(self.state_shape)
        ind = randint(0,15)
        state = state.reshape(2,16)
        state[1,ind] = 1

        while True:
            ind = randint(0,15)
            if state[1,ind]==1:
                pass
            else:
                state[0,ind] = 1
                break
        return state.reshape(1,32)


    def get_state(self, state, action, player):
        '''
        Returns the new state based on the action. Player must be -1 or 1.
        '''
        s = state.reshape(2,16).copy()
        temp = s[0,:].copy().reshape(4,4)
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

        return s.reshape(1,32)

    def get_reward(self, state, player):
        final_state = False
        win = self.check_winner(state, player)
        if win:
            reward = 10
            final_state = True
        else:
            reward = -1
        return reward, final_state
