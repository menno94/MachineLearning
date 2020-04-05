import datetime
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envVier3 import env_Vier

if __name__ =='__main__':
    stats_name = 'stats_vier_test.txt'
    env = env_Vier(stats_name, dim=[4,4], connect=3)
    Q = Q_agent(env)
    Q.analyse = True
    # continue_learning = input("Continue learning or reset?")
    continue_learning = True
    now = datetime.datetime.now()
    episodes = 100000
    epsilon_min = 0.05
    percentage  = 0.3

    if continue_learning:
        Q.set_model('test.h5')
        # Q.set_model('model_temp.h5')   # If aborted learning midway through
        epsilon = 0.5
    else:
        # Create or reset the stats file, if continue_learning: append to existing stats_name.txt
        Q.f = open(stats_name, 'w')
        Q.f.close()
        Q.create_model( N = [500,500,500,500],
                        learning_rate = 1e-4)
        epsilon = 1
    
    decay = epsilon_min**(1/(percentage*episodes))
    Q.train(episodes            =   episodes,
            epsilon             =   epsilon, 
            epsilon_min         =   epsilon_min,
            epsilon_decay       =   decay, 
            epsilon_opponent    =   0.2,
            batch_size          =   32, 
            gamma               =   0.9,
            memory_length       =   5000,
            breaks              =   20,
            model_update_freq   =   20,
            opponent_freq       =   300,
            max_agent_pool      =   1)
    
    Q.save_model('test.h5')
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))
     