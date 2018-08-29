import datetime

from agent import Q_agent
from envTicTacToe import env_tictactoe

if __name__ =='__main__':
    tic_tac_toe = env_tictactoe()
    Q = Q_agent(tic_tac_toe)

    now = datetime.datetime.now()
    Q.train(episodes=200000)
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    Q.save_weights('test1.w')
    ## test
    for i in range(10):
        state = np.zeros((2,9))
        board = np.zeros((1,9))
        while True:
            print('----------------------------')
            ind = np.where(state[0,:]==1)[0]
            ind2 = np.where(state[1,:]==1)[0]
            board[0,ind] = 1
            board[0,ind2] = -1
            print(board.reshape(3,3))
            A = int(input('Input:'))
            r, final_state = tic_tac_toe.get_reward(state,0)
            state[0][A]=1
            print('Yor move: {}; Reward:{}'.format(A, r))
            A = Q.get_action(state,0)
            r, final_state = tic_tac_toe.get_reward(state,1)
            print('Pc move: {}; Reward:{}'.format(A,r))
            state[1][A] = 1
            if final_state==True:
                print('+++++++ new game +++++++')
                break