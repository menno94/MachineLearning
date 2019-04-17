import datetime
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envTTT import env_TTT

if __name__ =='__main__':
    env = env_TTT()
    Q = Q_agent(env)
    Q.analyse = True
    now = datetime.datetime.now()
    Q.create_model(N = [32],
                   learning_rate = 1e-3)
    episodes = 10000
    epsilon_min = 0.05
    percentage = 0.2
    decay = epsilon_min**(1/(percentage*episodes))
    Q.train(episodes            =   episodes,
            epsilon             =   1, 
            epsilon_min         =   epsilon_min,
            epsilon_decay       =   decay, 
            batch_size          =   32,
            gamma               =   0.8,
            memory_length       =   200,
            breaks              =   50,
            model_update_freq   =   1000)
   
    Q.save_model('TTT.h5')
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    
# =============================================================================
# Menno, dit is jouw shizzle
# =============================================================================
#    state = np.load('state.npy')
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.pcolor(state[0].reshape((3,3))/np.sum(state[0]))
#    plt.colorbar()
#    plt.subplot(2,1,2)
#    plt.pcolor(state[1].reshape((3,3))/np.sum(state[1]))
#    plt.colorbar()    
#    
    

