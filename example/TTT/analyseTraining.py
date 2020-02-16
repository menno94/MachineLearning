import numpy as np
import os
import glob

fnames = glob.glob('analyse/*')

start_episode = 8000


count = 0
for fname in fnames:

    
    state = np.load(fname)
    e, turn = os.path.splitext(os.path.basename(fname))[0].split( '_')
    
    if int(e)<start_episode:
        continue
    print('episode={} turn={}'.format(e,turn))
    
    board = np.zeros((1,9))
    
    ind     = np.where(state[0  ,:]==1)[0]
    ind2    = np.where(state[1,:]==1)[0]
    board[0,ind] = 1
    board[0,ind2] = -1
    if count%2==0:
        board = board *-1
    
    print(board.reshape((3,3)))
    
    action = input("next")
    
    count = count+1
    if action == 'q':
        break


