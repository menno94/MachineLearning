import datetime
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envVier import env_Vier

if __name__ =='__main__':
    env = env_Vier()
    Q = Q_agent(env)
    now = datetime.datetime.now()
    Q.create_model(N = [42,21],
                   learning_rate = 1e-3)
    episodes = 10000
    epsilon_min = 0.05
    percentage = 0.4
    decay = epsilon_min**(1/(percentage*episodes))
    Q.train(episodes            =   episodes,
            epsilon             =   1, 
            epsilon_min         =   epsilon_min,
            epsilon_decay       =   decay, 
            batch_size          =   32, 
            gamma               =   0.8,
            memory_length       =   5000,
            breaks              =   100,
            model_update_freq   =   10)
    
    Q.save_model('Vier.h5')
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))
     