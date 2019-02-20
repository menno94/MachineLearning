import numpy as np

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Qagent'))
from agent import Q_agent
from envVier import env_Vier

if __name__ == '__main__':
    env = env_Vier()
    Q = Q_agent(env, training=False)
    Q.set_model('Vier.h5')

    ## test
    stop = 'no'
    while stop != 'q':
        state = env.get_initial_state()
        Q.training = False
        turn_player         = True
        first_turn_player   = 0     #True
        start = input("[P]layer or [A]i")
        if start == "A":
            turn_player         = False
            first_turn_player   = 1 #False
        
        while True:
            print('----------------------------')
            
            if turn_player:              
                action = input("Action selection: [1-7]")
                if action == 'q':
                    stop = 'q'
                    break
                action = int(action) - 1
            else:
                action, values = Q.get_action(state)            
            
            
            next_state = env.get_next_state(state,action)
            r, final_state = env.get_reward(next_state)
            state = env.switch_state(next_state)
            turn_player = not turn_player
            first_turn_player = 1 - first_turn_player
            if first_turn_player == 1:
                env.print_state(next_state)
            else:
                env.print_state(state)
            if final_state == True:
                print('+++++++ new game +++++++')
                break
