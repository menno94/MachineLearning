import numpy as np


#from optimal_agent import optimal_agent
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Qagent'))
from agent import Q_agent
from envTTT import env_TTT

if __name__ == '__main__':
    env = env_TTT()
    Q = Q_agent(env, training=False)
    Q.set_model('TTT.h5')
    Q.conv = True

    ## test
    stop = 'no'
    while stop != 'q':
        state = env.get_initial_state()
        Q.training = False
        state = np.zeros((2,3,3))
        board = np.zeros((1,3,3))
        turn_player         = True
        first_turn_player   = 0     #True
        start = input("[P]layer or [A]i")
        if start == 'q':
            break
        if start == "A":
            turn_player         = False
            first_turn_player   = 1 #False
        
        while True:
            print('----------------------------')
            
            if turn_player:              
                
                action = input("Action selection: [1-9] numpad order (First move 1/2/5)")
                if action == 'q':
                    stop = 'q'
                    break
                action = int(action)
                if action < 4:
                    action+= 5
                elif action > 6:
                    action-=7
                else:
                    action-=1
            else:
                action, values = Q.get_action(state)  
                #action = optimal_agent(state)
                print(values.reshape(3,3))
            
            
            next_state = env.get_next_state(state,action)
            ind     = np.where(next_state[first_turn_player  ,:,:]==1)
            ind2    = np.where(next_state[1-first_turn_player,:,:]==1)
            board[0,ind[0],ind[1]] = 1
            board[0,ind2[0],ind2[1]] = -1  
            r, final_state = env.get_reward(next_state)
            state = env.switch_state(next_state)
            turn_player = not turn_player
            first_turn_player = 1 - first_turn_player
            print(board.reshape((3,3)))
            if final_state == True:
                print('+++++++ new game +++++++')
                break
