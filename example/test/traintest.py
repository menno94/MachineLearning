import datetime
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','Qagent'))
from agent import Q_agent
from envtest import env_test
import random
import matplotlib.pyplot as plt

if __name__ =='__main__':
    env = env_test()
    Q = Q_agent(env)
    # Q.analyse = True
    now = datetime.datetime.now()
    Q.create_model(N = [64,64],
                   learning_rate = 1e-3)
    episodes = 100000
    # epsilon_min = 0.08
    # percentage = 0.3
    # decay = epsilon_min**(1/(percentage*episodes))
    n = 1200; percent = 0.15;
    pi = 3.1415
    breaks = 5000
    x = np.linspace(0,8*pi,n) 
    random.shuffle(x)
    noise = np.random.normal(0,0.1,n)
    y = np.sin(x) + noise
    memory = []
    for i in range(n):
        memory.append(([x[i]],[y[i]]))
    memory_test = memory[int((1-percent)*n):None]
    memory = memory[:int((1-percent)*n)]
    x_test = []; y_test = []
    for i in range(len(memory_test)):
        x_test.append(memory_test[i][0])
        y_test.append(memory_test[i][1])
    batch_size = 1
    # Q.model.save_weights('temp_previous.h5')  #Niet nodig om op te slaan en te laden, scheelt tijd factor 8!!!!!
    for e in range(episodes):
        # Q.previous_model.load_weights('temp_previous.h5')
        minibatch = random.sample(memory, batch_size)
        for x, y in minibatch:
            stats = Q.model.fit(x, y,batch_size=32,epochs=e+1, initial_epoch=e, verbose=0)
        # Q.model.save_weights('temp_previous.h5')
        if e % breaks == 0:
            print(e)
            y_test_hat = []
            for i in range(int(percent*n)):
                y_test_hat.append(Q.model.predict(memory_test[i][0])[0][0])
    plt.close("all")
    plt.figure(2)
    plt.scatter(x_test,y_test) 
    plt.scatter(x_test,y_test_hat)       
    
    # Q.save_model('test.h5')
    
    dt = (datetime.datetime.now()-now).total_seconds()
    print('Total time: {}'.format(dt))

    
# =============================================================================
# Menno, dit is jouw shizzle
# =============================================================================
    # import matplotlib.pyplot as plt
    # state = np.load('state.npy')
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.pcolor(state[0].reshape((3,3))/np.sum(state[0]))
    # plt.colorbar()
    # plt.subplot(2,1,2)
    # plt.pcolor(state[1].reshape((3,3))/np.sum(state[1]))
    # plt.colorbar()    
    
    

