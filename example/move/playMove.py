import datetime
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','agent'))
from agent import Q_agent
from envMove import env_move

if __name__ =='__main__':
    env = env_move()
    Q = Q_agent(env)
    now = datetime.datetime.now()
    Q.train(N                   =   [np.prod(env.state_shape) ,env.action_size],
            episodes            =   5000,
            epsilon             =   1, 
            epsilon_min         =   0.2, 
            epsilon_decay       =   0.96, 
            batch_size          =   32, 
            gamma               =   0.7,
            learning_rate       =   1e-3, 
            memory_length       =   1000,
            breaks              =   50)  # Show feedback every X episodes
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))


    Q.save_weights('test1.w')
    ## test
    for i in range(10):
        state = env.get_initial_state()
        board = np.zeros((1,16))
        while True:
            print('----------------------------')
            ind = np.where(state.reshape(2,16)[0,:]==1)
            ind2 = np.where(state.reshape(2,16)[1,:]==1)
            print(ind,ind2)
            board[0,ind] = 1
            board[0,ind2] = -1
            print(board.reshape(4,4))
            input("Press Enter to continue...")
            Q.training = False
            A = Q.get_action(state)
            next_state = env.get_next_state(state, A)
            state = next_state
            board[:] = 0
            r, final_state = env.get_reward(state)
            if final_state==True:
                print('+++++++ new game +++++++')
                break
