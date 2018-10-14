import numpy as np

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Qagent'))
from agent import Q_agent
from envTTT import env_TTT

if __name__ == '__main__':
    env = env_TTT()
    Q = Q_agent(env, training=False)
    Q.set_model('TTT.h5')

    ## test
    stop = 'no'
    while stop != 'q':
        state = env.get_initial_state()
        Q.training = False
        state = np.zeros((2,9))
        board = np.zeros((1,9))
        while True:
            print('----------------------------')
            ind = np.where(state[0,:]==1)[0]
            ind2 = np.where(state[1,:]==1)[0]
            board[0,ind] = 1
            board[0,ind2] = -1
            print(board.reshape((3,3)))
            stop = input("Press Enter to continue or q to quite")
            A, values = Q.get_action(state)
            print('Values: up={} right={} down={} left={}'.format(round(values[0, 0], 2), round(values[0, 1], 2),
                                                                  round(values[0, 2], 2), round(values[0, 3], 2)))
            next_state = env.get_next_state(state, A)
            state = next_state
            board[:] = 0
            r, final_state = env.get_reward(state)
            if final_state == True:
                print('+++++++ new game +++++++')
                break
            if stop == 'q':
                break