import numpy as np

class env_tictactoe:

    def __init__(self):
        self.state = np.zeros((2,9))

    def check_winner(self, temp, player):
        '''
        Determine wheter there are 3 in a row. Player must be -1 or 1.
        '''
        temp = temp[player,:]
        temp = temp.reshape(3, 3)
        win = False
        ## vertical
        for j in range(3):
            check = 0
            # horizontal
            for i in range(2):
                if temp[j, i] == temp[j, i + 1] and temp[j, i] == 1:
                    check = check + 1
                    if check == 2:
                        win = True
                        break

        ## horzontal line
        for j in range(3):
            check = 0
            # vertical
            for i in range(2):
                if temp[i, j] == temp[i + 1, j] and temp[i, j] == 1:
                    check = check + 1
                    if check == 2:
                        win = True
                        break

        if temp[0, 0] == 1 and temp[1, 1] == 1 and temp[2, 2] == 1:
            win = True
        if temp[0, 2] == 1 and temp[1, 1] == 1 and temp[0, 2] == 1:
            win = True
        return win

    def check_tie(self, temp):
        '''
        Determine wheter there is a tie.
        '''
        tie = False
        if np.sum(abs(temp)) == 9:
            tie = True
        return tie

    def get_state(self, state, action,player):
        '''
        Returns the new state based on the action. Player must be -1 or 1.
        '''
        s = state.copy()
        s[player, action] = 1
        return s

    def get_reward(self, state, player):
        final_state = False
        win = self.check_winner(state, player)
        tie = self.check_tie(state)

        if win:
            reward = 10
            final_state = True
        elif tie:
            reward = 0
            final_state = True
        else:
            reward = -1
        return reward, final_state
