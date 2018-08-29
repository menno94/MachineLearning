import datetime
import numpy as np

from agent import Q_agent
from envMove import env_move

if __name__ =='__main__':
    env = env_move()
    Q = Q_agent(env)

    now = datetime.datetime.now()
    Q.train_1player(episodes=100)
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    Q.save_weights('test1.w')
    ## test
    for i in range(10):
        state = env.get_initial_state()
        board = np.zeros((1,16))
        while True:
            print('----------------------------')
            ind = np.where(state[0,:]==1)
            ind2 = np.where(state[1,:]==1)
            print(ind,ind2)
            board[0,ind] = 1
            board[0,ind2] = -1
            print(board.reshape(4,4))
            input("Press Enter to continue...")
            A = Q.get_action(state)
            next_state = Q.get_next_state(state, A)
            state = next_state
            board[:] = 0
            r, final_state = Q.get_reward(state,0)
            if final_state==True:
                print('+++++++ new game +++++++')
                break