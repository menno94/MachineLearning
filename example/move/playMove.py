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
    Q.create_model(N = [5],
                   learning_rate = 1e-3)
#    Q.set_model('Move_v1')
    Q.train(episodes            =   200,
            epsilon             =   1, 
            epsilon_min         =   0.15, 
            epsilon_decay       =   0.96, 
            batch_size          =   10, 
            gamma               =   0.9,
            memory_length       =   50,
            breaks              =   10)  # Show feedback every X episodes
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    Q.save_model('Move_v1')

