import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envMove import env_move

if __name__ =='__main__':
    env = env_move()
    Q = Q_agent(env,training=False)
    Q.set_model('Move_v1')
    
    ## test
    stop = 'no'
    while stop!='q':
        state = env.get_initial_state()
        Q.training = False
        board = np.zeros((1,env.state_shape[1]))
        while True:
            print('----------------------------')
            ind = np.where(state.reshape(env.state_shape)[0,:]==1)
            ind2 = np.where(state.reshape(env.state_shape)[1,:]==1)
            board[0,ind] = 1
            board[0,ind2] = -1
            print(board.reshape(env.field_shape))
            stop = input("Press Enter to continue or q to quite")
            print(stop=='q')
            A, value = Q.get_action(state)
            print('Value={}'.format(value))
            next_state = env.get_next_state(state, A)
            state = next_state
            board[:] = 0
            r, final_state = env.get_reward(state)
            if final_state==True:
                print('+++++++ new game +++++++')
                break
            if stop=='q':
                break
            

