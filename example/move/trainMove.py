import datetime
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envMove import env_move

if __name__ =='__main__':
    env = env_move()
    Q = Q_agent(env)
    Q.analyse = True
    now = datetime.datetime.now()
    Q.create_model(N = [10],
                   learning_rate = 1e-3)
#    Q.set_model('Move_v1')
    Q.train(episodes            =   1000,
            epsilon             =   1, 
            epsilon_min         =   0.02, 
            epsilon_decay       =   0.96, 
            batch_size          =   32,
            gamma               =   0.9,
            memory_length       =   250,
            breaks              =   10,
            model_update_freq   =   10)  # Show feedback every X episodes
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    Q.save_model('Move_v1')
    
    state = np.load('state.npy')
    plt.figure()
    plt.subplot(2,1,1)
    plt.pcolor(state[0].reshape((3,3))/np.sum(state[0]))
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.pcolor(state[1].reshape((3,3))/np.sum(state[1]))
    plt.colorbar()    

