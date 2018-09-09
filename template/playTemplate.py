import datetime
import numpy as np

from agent import Q_agent
from envDEFAULT import env_DEFAULT

if __name__ =='__main__':
    env = env_DEFAULT()
    Q = Q_agent(env)
    now = datetime.datetime.now()
    Q.train(N                   =   [np.prod(env.state_shape), 18, env.action_size],
            episodes            =   1000,
            epsilon             =   1, 
            epsilon_min         =   0.01, 
            epsilon_decay       =   0.95, 
            batch_size          =   32, 
            gamma               =   0.9,
            learning_rate       =   1e-3, 
            memory_length       =   200,
            breaks              =   20)
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))
      