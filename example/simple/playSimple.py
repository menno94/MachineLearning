import datetime
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','agent'))
from agent import Q_agent
from envSimple import env_Simple

'''
Simpel spel, met nodes. 
Ben je in node 5 of 6 dan win je
Moves:  0 --> 1, 3, 4, 5
        1 --> 0
        2 --> 3
        3 --> 0, 2
        4 --> 0, 6        
Een score van 9 is optimaal!!
'''

if __name__ =='__main__':
    env = env_Simple()
    Q = Q_agent(env)
    now = datetime.datetime.now()
    Q.train(N                   =   [np.prod(env.state_shape), env.action_size],
            episodes            =   200,
            epsilon             =   1, 
            epsilon_min         =   0.4, 
            epsilon_decay       =   0.9, 
            batch_size          =   32, 
            gamma               =   1,
            learning_rate       =   1e-3, 
            memory_length       =   100,
            breaks              =   10)
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))
      