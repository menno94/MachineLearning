import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Qagent'))
from agent import Q_agent
from envVier import env_Vier

if __name__ == '__main__':
    env = env_Vier()
    Q = Q_agent(env, training=False)
    Q.set_model('TTTtemp.h5')

    ## test
    stop = 'no'
    show_game = True
    if show_game:
        action_sequence = [3, 1, 7, 1, 1, 1, 2, 1, 1, 2, 7, 2, 6, 2, 4, 3, 7, 3, 3, 3, 5]
    while stop != 'q':
        state = env.get_initial_state()
        Q.training = False
        move_count = 0
        turn_player         = True
        first_turn_player   = 0     #True
        
        if not show_game:
            start = input("[P]layer or [A]i")
            if start == "A":
                turn_player         = False
                first_turn_player   = 1 #False
        
        while True:
            print('----------------------------')
            
            if show_game:
                action = action_sequence[move_count] - 1
                move_count+=1
            else:
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
            if show_game:
                if first_turn_player ==1:
                    env.print_state(next_state)
                else:
                    env.print_state(state)
                stop = input("Press Enter for next move...")
                if stop == 'q':
                    break
            else:
                if first_turn_player == 1:
                    env.print_state(next_state)
                else:
                    env.print_state(state)
            if final_state == True:
                print('+++++++ new game +++++++')
                break
