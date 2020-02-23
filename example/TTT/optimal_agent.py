import numpy as np

def optimal_agent(state):
    opponent = state[1].reshape((3,3))
    opponent_array = state[1]
    player = state[0].reshape((3,3))
    player_array = state[0]
    
    ## --- (1) move to win
    ## go throufh all the rows
    for ii in range(3):
        ## 1-1-0
        if player[ii,0] == 1 and player[ii,1] == 1:
            if opponent_array[ii*3+2]==0:
                return ii*3+2
        ## 0-1-1
        if player[ii,1] == 1 and player[ii,2] == 1:
            if opponent_array[ii*3]==0:
                return ii*3
        ## 1-0-1
        if player[ii,0] == 1 and player[ii,2] == 1:
            if opponent_array[ii*3+1]==0:
                return ii*3+1
    
    ## go throufh all the columns
    for ii in range(3):
        ## 1
        ## 1
        ## 0
        if player[0,ii] == 1 and player[1,ii] == 1:
            if opponent_array[6+ii]==0:
                return 6+ii
        ## 0
        ## 1
        ## 1
        if player[1,ii] == 1 and player[2,ii] == 1:
            if opponent_array[ii]==0:
                return ii
        ## 1
        ## 0
        ## 1
        if player[0,ii] == 1 and player[2,ii] == 1:
            if opponent_array[3+ii ]==0:
                return 3+ii      
    ## win by a diagonal
    if   player[0,0] == 1 and player[2,2] == 1 and opponent[1,1]==0:  
        return 4
    if   player[0,2] == 1 and player[2,0] == 1 and opponent[1,1]==0:  
        return 4   
    if   player[0,0] == 1 and player[1,1] == 1 and opponent[2,2]==0:  
        return 8   
    if   player[1,1] == 1 and player[2,2] == 1 and opponent[0,0]==0:  
        return 0  
    if   player[0,2] == 1 and player[1,1] == 1 and opponent[2,0]==0:  
        return 6
    if   player[2,0] == 1 and player[1,1] == 1 and opponent[0,2]==0:  
        return 2
    
    ## --- (2) move to defend
    ## go throufh all the rows
    for ii in range(3):
        ## 1-1-0
        if opponent[ii,0] == 1 and opponent[ii,1] == 1:
            if player_array[ii*3+2]==0:
                return ii*3+2
        ## 0-1-1
        if opponent[ii,1] == 1 and opponent[ii,2] == 1:
            if player_array[ii*3]==0:
                return ii*3
        ## 1-0-1
        if opponent[ii,0] == 1 and opponent[ii,2] == 1:
            if player_array[ii*3+1]==0:
                return ii*3+1
    
    ## go throufh all the columns
    for ii in range(3):
        ## 1
        ## 1
        ## 0
        if opponent[0,ii] == 1 and opponent[1,ii] == 1:
            if player_array[6+ii]==0:
                return 6+ii
        ## 0
        ## 1
        ## 1
        if opponent[1,ii] == 1 and opponent[2,ii] == 1:
            if player_array[ii]==0:
                return ii
        ## 1
        ## 0
        ## 1
        if opponent[0,ii] == 1 and opponent[2,ii] == 1:
            if player_array[3+ii]==0:
                return 3+ii      
    ## prevent lose by a diagonal
    if   opponent[0,0] == 1 and opponent[2,2] == 1  and player[1,1]==0:  
        return 4
    if   opponent[0,2] == 1 and opponent[2,0] == 1  and player[1,1]==0:  
        return 4
    if   opponent[0,0] == 1 and opponent[1,1] == 1  and player[2,2]==0:  
        return 8   
    if   opponent[1,1] == 1 and opponent[2,2] == 1  and player[0,0]==0:  
        return 0  
    if   opponent[0,2] == 1 and opponent[1,1] == 1  and player[2,0]==0:  
        return 6
    if   opponent[2,0] == 1 and opponent[1,1] == 1  and player[0,2]==0:  
        return 2    
    
    ## first move as x (middle)
    if np.sum(state)==0:
        return 4
    ## first move as o (middle or -corner)
    if np.sum(state)==1:
        if state[1][4]==0:
            return 4
        else:
            I = np.random.randint(4)
            tmp = [0,2,6,8]
            return tmp[I]
      
        
    ## second move as x (corner)
    if np.sum(state[0])==1 and np.sum(state[1])==1:
        tmp = [0,2,6,8]
        I = np.random.randint(4)
        while True:
            if state[1][tmp[I]]==0:
                return tmp[I]
            I = np.random.randint(4)



    ## other move
    I = np.where(np.logical_and(state[0]==0,state[1]==0))
    return I[0][0]
    



if __name__=='__main__':
    state = np.array([[1,1,0,0,0,0,0,1,1],[1,0,0,1,1,0,0,0,0]])
    
    print(state[0].reshape((3,3)))
    print(state[1].reshape((3,3)))
    print(optimal_agent(state))