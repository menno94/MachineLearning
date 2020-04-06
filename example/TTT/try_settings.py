import datetime
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envTTT import env_TTT

if __name__ =='__main__':
    delay_model         = [False,   True,   False,  False]
    double_q            = [True,    False,  False,  False]
    prioritized_replay  = [False,   False,  False,  True]
    names = ['Double','Delay','-','prio']
    for ii in range(len(delay_model)):
        env = env_TTT()
        Q = Q_agent(env)
        Q.analyse           = False
        Q.analyse_full      = False
        Q.opponent_optimal  = True
        Q.prioritized_replay= prioritized_replay[ii]
        Q.double_q          = double_q[ii]
        Q.delay_model       = delay_model[ii]
        now = datetime.datetime.now()
        Q.create_model(N = [80,60,40],
                       learning_rate = 1e-4)
        episodes    = 5000
        epsilon_min = 0.05
        percentage  = 0.4
        decay = epsilon_min**(1/(percentage*episodes))
        Q.train(episodes            =   episodes,
                epsilon             =   1, 
                epsilon_min         =   epsilon_min,
                epsilon_decay       =   decay, 
                batch_size          =   32,
                gamma               =   0.5,
                memory_length       =   100,
                breaks              =   10,
                model_update_freq   =   250,
                opponent_freq       =   2000,
                max_agent_pool      =   10)
    
       
        Q.save_model('TTT{}.h5'.format(ii))
        
        with open('stats.txt') as f:
            with open('test{}.txt'.format(ii),'w') as f1:
                for line in f:
                    f1.write(line)           
        f.close()
        f1.close()
        
        ##
    color = ['r','b','g','m']
    for ii in range(len(delay_model)):
        with open('test{}.txt'.format(ii),'r') as f:
            lines = f.readlines()

        episode = []
        epsilon = []
        Qtest = []
        score = []
        totalR = []
        loss = []
        acc = []
        mean_p = []
        std_p = []
        for line in lines:
            for item in line.split('\t'):
                var, number = item.split('=')
                number = float(number.split('\n')[0])
                if var =='e':
                    episode.append(number)
                if var =='eps':
                    epsilon.append(number)
                if var =='Q':
                    Qtest.append(number)
                if var =='score':
                    score.append(number)
                if var=='total_R':
                    totalR.append(number)
                if var =='loss':
                    loss.append(number)
                if var =='acc':
                    acc.append(number)
                if var =='meanP':
                    mean_p.append(number)
                if var =='stdP':
                    std_p.append(number)

        N = max(1,int(len(episode)/5))
        ## plot results
        plt.figure(1,figsize=[12,12])
        plt.suptitle("Results", fontsize=16)

        ax1 = plt.subplot(3,1,1)
        ax1.plot(episode, score,'.-',color=color[ii])
        ax1.plot(episode, np.convolve(score, np.ones((N,)) / N, mode='same'),'--',color=color[ii])

        plt.ylabel('Scores')
        plt.title('Scores based on test skill function')
        plt.grid('on')
        ax1.set_xticklabels([])

        ax2 = plt.subplot(3,1,2)
        ax2.plot(episode, epsilon,'.-',color=color[ii])
        plt.ylabel('Epsilon')
        plt.title('Epsilon per epoch')
        plt.grid('on')
        ax2.set_xticklabels([])

        ax2 = plt.subplot(3,1,3)
        ax2.plot(episode,totalR,'.-',color=color[ii])
        ax2.plot(episode, np.convolve(totalR, np.ones((N,))/N, mode='same'),'--',color=color[ii])
        plt.ylabel('Total reward') ## total reward 
        plt.title('Total reward within the breaks interval')
        plt.grid('on')
        ax2.set_xticklabels([])
        
        plt.figure(2,figsize=[12,12])
        ax2 = plt.subplot(3,1,1)
        ax2.plot(episode,Qtest,'.-',color=color[ii])
        ax2.plot(episode, np.convolve(Qtest, np.ones((N,)) / N, mode='same'),'--',color=color[ii])
        plt.title('Averaged Q value for given test states')
        plt.ylabel('averaged Q')
        plt.grid('on')
        ax2.set_xticklabels([])
       
        ax2 = plt.subplot(3,1,2)
        ax2.plot(episode,acc,'.-',color=color[ii])
        ax2.plot(episode, np.convolve(acc, np.ones((N,)) / N, mode='same'),'--',color=color[ii])
        plt.ylabel('Accuracy')
        plt.title('Mean accuracy over all batches')
        plt.grid('on')
        ax2.set_xticklabels([])
              
        ax = plt.subplot(3,1,3)
        ax.plot(episode,loss,'.-',label=names[ii],color=color[ii])
        ax.plot(episode, np.convolve(loss, np.ones((N,)) / N, mode='same'),'--',label='_nolegend_',color=color[ii] )
        ax.set_yscale('log')
        plt.ylabel('loss')
        plt.title('Mean loss over all batches')
        plt.grid('on')
        plt.xlabel('Episode')
        plt.legend(names)
        
        plt.figure(3,figsize=[12,12])
        ax = plt.subplot(1,1,1)
        ax.plot(episode,mean_p,'.-',color=color[ii])
        ax.plot(episode,std_p,'.-',color=color[ii])
        ax.plot(episode, np.convolve(mean_p, np.ones((N,)) / N, mode='same'),'--',color=color[ii])
        
        ax.plot(episode, np.convolve(std_p, np.ones((N,)) / N, mode='same'),'--',color=color[ii])
        plt.ylabel('P')
        plt.legend(['mean','std'])
        plt.title('stats of prioritized replay function')
        plt.grid('on')
        plt.xlabel('Episode')        

        plt.xlabel('Episode')
    #plt.savefig('results.png')
    
    

    

    
    

