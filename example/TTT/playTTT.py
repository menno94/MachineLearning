import datetime
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envTTT import env_TTT

if __name__ =='__main__':
    env = env_TTT()
    Q = Q_agent(env)
    now = datetime.datetime.now()
    Q.create_model(N = [10],
                   learning_rate = 1e-3)
    Q.train(episodes            =   5000,
            epsilon             =   1, 
            epsilon_min         =   0.02,
            epsilon_decay       =   0.97, 
            batch_size          =   32, 
            gamma               =   0.55,
            memory_length       =   100,
            breaks              =   50)
    
    Q.save_model('TTT.h5')
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))
    