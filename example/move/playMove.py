import datetime
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envMove import env_move

if __name__ =='__main__':
    env = env_move()
    Q = Q_agent(env)
    now = datetime.datetime.now()
    Q.create_model(N = [9],
                   learning_rate = 1e-3)
#    Q.set_model('Move_v1')
    Q.train(N                   =   [9],
            episodes            =   5000,
            epsilon             =   1, 
            epsilon_min         =   0.15, 
            epsilon_decay       =   0.96, 
            batch_size          =   1, 
            gamma               =   0.9,
            memory_length       =   10000,
            breaks              =   1)  # Show feedback every X episodes
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    Q.save_model('Move_v1')
#    Q.save_weights('test1.w')
    ## test
    for i in range(1):
        state = env.get_initial_state()
        Q.training = False
        board = np.zeros((1,env.state_shape[1]))
        while True:
            print('----------------------------')
            ind = np.where(state.reshape(env.state_shape)[0,:]==1)
            ind2 = np.where(state.reshape(env.state_shape)[1,:]==1)
            print(ind,ind2)
            board[0,ind] = 1
            board[0,ind2] = -1
            print(board.reshape(env.field_shape))
            input("Press Enter to continue...")
            A = Q.get_action(state)
            next_state = env.get_next_state(state, A)
            state = next_state
            board[:] = 0
            r, final_state = env.get_reward(state)
            if final_state==True:
                print('+++++++ new game +++++++')
                break
