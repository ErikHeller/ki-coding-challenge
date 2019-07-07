
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline


# In[2]:

def count_column(state,column):
    return sum(state == column)


# In[3]:

def count_column2(state,column,last_e):
    res = 0
    for i in range(last_e+1):
        if state[i] == column:
            res +=1
    return res


# In[4]:

def last_element(state):
    for i in range(len(state)):
        if state[i] == 0:
            return i-1
    return len(state)-1


# In[5]:

def terminate(state):
    if sum(state!=0) < 9:
        return False
    last_e1,_ =  findlastvalues(state)    
    color = last_e1%2
    #0 für rot, 1 für gelb
    n = max(1,int(last_e1/2))
    #print(n)
    for k in range(n): # geht jeden gelben/roten stein der farbe durch
        param = 0
        while True:
            last_e = last_e1-k*2-param
            #print(last_e)
            if last_e < 8: return False
            elif state[last_e] != -1 and state[last_e] != 13:
                break
            else: 
                param+=2
        #print(last_e)
        counter = 0
        column = state[last_e]
        height = count_column2(state,column,last_e)
        #print(height)
        #horizontal
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column-i-1))[0]
            if( len(indices) >= height and len(indices)>0):
                if (check_left and column-i-1 > 0 and indices[height-1] %2 == color):
                    counter += 1
                else:
                    check_left = False
            else:
                check_left = False
            indices = np.where(state == int(column+i+1))[0]
            if(len(indices) >= height and len(indices)>0):
                if (check_right and column+i < 12 and indices[height-1] %2 == color):
                    counter += 1
                else:
                    check_right= False
            else:
                check_right= False
            #print(counter,'r')
            if(counter == 4):
                #print('h',last_e)
                return True
        #vertikal
        counter = 0
        check_bot = True

        for i in range(4):
            indices = np.where(state == int(column))[0]
            #if(and len(indices)>0)
            if (check_bot and height-i-1 > 0 and indices[height-i-2] %2 == color):
                counter += 1
            else:
                check_bot = False
        #print(counter)
        if(counter == 4):
            #print('v',last_e)
            return True

        #diag top left, bot right
        counter = 0
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column-i-1))[0]
            if (len(indices) >= height+1+i ):
                if (check_left and column-i-1 > 0 and indices[height+i] %2 == color):
                    counter += 1
                else:
                    check_left = False
            else:
                check_left = False

            indices = np.where(state == int(column+i+1))[0]
            if(len(indices) >= height-1-i and height-1-i >0):
                if (check_right and column+i+1 < 12 and indices[height-i-2] %2 == color):
                    counter += 1
                else:
                    check_right= False
            else:
                check_right= False
            #print(counter,'r')
            if(counter == 4):
                #print('diag_tl',last_e)
                return True

        #diag bot left, top right
        counter = 0
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column-i-1))[0]
            if (len(indices) >= height-1-i and height-1-i >0):
                if (check_left and column-i-1 > 0 and indices[height-i-2] %2 == color):
                    counter += 1
                else:
                    check_left = False
            else:
                    check_left = False

            indices = np.where(state == int(column+i+1))[0]
            if (len(indices) >= height+1+i):
                if (check_right and column+i+1 < 12 and indices[height+i] %2 == color):
                    counter += 1
                else:
                    check_right= False

            else:
                check_right= False
            #print(counter,'r')
            if(counter == 4):
                #print('diag_bl',last_e)
                return True
    return False


# In[6]:

def utility2(state):
    last_e,last_e2 = findlastvalues(state)
    color = last_e%2
    res = color
    lastvals = findminusones(state)
    #0 für rot, 1 für gelb
    state_before = state.copy()
    state_before[last_e] = 0
    #print(state_before)
    counter = [0,0]    
    a,b = 2,1
    for c in range(len(counter)):        
        if c == 1:
            last_e =  last_e2    
        #print(last_e)
        color = last_e%2
        column = state[last_e]
        height = count_column(state,column)

        #horizontal
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column-i-1))[0]
            if(len(indices) >= height and len(indices)>0 and check_left and column-i-1 > 0):
                if (indices[height-1] %2 == color):
                    counter[c] += a
                else:
                    check_left = False
            elif(check_left and column-i-1 > 0 and len(indices) < height):
                counter[c] += b
            else:
                check_left = False
            #print(counter,'lh')
            indices = np.where(state == int(column+i+1))[0]
            if(len(indices) >= height and len(indices)>0 and check_right and column+i+1 < 12):
                if (indices[height-1] %2 == color):
                    counter[c] += a
                else:
                    check_right = False
            elif (check_right and column+i+1 < 12 and len(indices) < height):
                counter[c] += b
            else:
                check_right= False
            #print(counter,'rh')
        
        #vertikal
        check_bot = True
        for i in range(4):
            indices = np.where(state == int(column))[0]
            #if(and len(indices)>0)
            if (check_bot and height-i-1 > 0 and indices[height-i-2] %2 == color):
                counter[c] += a
            else:
                check_bot = False
        counter[c] += b*(min(4,8-height))
        #print(counter,'v')
        
        #diag top left, bot right
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column-i-1))[0]
            if (len(indices) >= height+1+i and check_left and column-i-1 > 0):
                if (indices[height+i] %2 == color):
                    counter[c] += a
                else:
                    check_left = False
            elif(check_left and column-i-1 > 0 and len(indices) < height+1+i):
                counter[c] += b
            else:
                check_left = False

            indices = np.where(state == int(column+i+1))[0]
            if(len(indices) >= height-1-i and height-1-i >0 and check_right and column+i+1 < 12):
                if (indices[height-i-2] %2 == color):
                    counter[c] += a
                else:
                    check_right = False
            elif(check_right and column+i+1 < 12 and len(indices) < height-1-i and height-1-i >0):
                counter[c] += b
            else:
                check_right= False
        #print(counter,'diag1')
            
        #diag bot left, top right
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column-i-1))[0]
            if (len(indices) >= height-1-i and height-1-i >0 and check_left and column-i-1 > 0):
                if (indices[height-i-2] %2 == color):
                    counter[c] += a
                else:
                    check_left = False
            elif(check_left and column-i-1 > 0 and len(indices) < height-1-i and height-1-i >0):
                counter[c] += b
            else:
                    check_left = False

            indices = np.where(state == int(column+i+1))[0]
            if (len(indices) >= height+1+i and check_right and column+i+1 < 12):
                if (indices[height+i] %2 == color):
                    counter[c] += a
                else:
                    check_right = False
            elif(check_right and column+i+1 < 12 and len(indices) < height+1+i):
                counter[c] += b
            else:
                check_right= False
            #print(counter,'r')
        
        counter[c] -= 50*lastvals[c] 
    #print(res)
    #print(counter)
    #print()
    if res == 0:
        return counter[0]-counter[1]
    return counter[1]-counter[0]


# In[7]:

def utility(state):
    res = state.copy()
    n = len(res)
    i = 0
    v = utility2(res)
    while n-4-2*i >= 0 and i <3:
        #print(v,res)
        a,b = findlastvalues(res)
        if a == 0 or b == 0:
            break
        res[a] = -1
        res[b] = -1
        v += utility2(res)
        i +=1
    return v        


# In[8]:

#1: red, 2: yellow
#%matplotlib inline
def plotboard(ax,state):
    for i in range(12):
        ax.vlines(i+0.5,0.5,8.5)
    ax.hlines(0.5,0.5,11.5)
    counter = 0
    height = np.zeros(11)
    for j in state:
        if(j == 0):
            break
        elif j == 13 or j == -1:
            counter += 1
            continue
        elif counter % 2 == 0:
            ax.plot(j,height[int(j-1)]+1,'.',markersize = 45,color = 'red')
        else:
            ax.plot(j,height[int(j-1)]+1,'.',markersize = 45,color = 'yellow')
        height[int(j-1)] += 1
        counter += 1
    plt.show()


# In[9]:

def findlastvalues(state):
    a,b = 0,0
    last_e = last_element(state)
    color = last_e%2
    for i in range(last_e+1):
        if state[last_e-i] != -1 and (last_e-i)%2 == color and state[last_e-i] != 13:
            a = last_e-i
            break
    for i in range(last_e+1):
        if state[last_e-i] != -1 and (last_e-i)%2 != color and state[last_e-i] != 13:
            b = last_e-i
            break
    return a,b


# In[10]:

def findminusones(state):
    a,b = 0,0
    last_e = last_element(state)
    color = last_e%2
    for i in range(last_e+1):
        if (i)%2 == color and state[i] == -1:
            a += 1
            
    for i in range(last_e+1):
        if (i)%2 != color and state[i] == -1:
            b += 1
    return a,b


# In[11]:

def field():
    return np.zeros(90)
#spieler 1 
def do_action(state,column = None,flip = False):        
    return_state = np.copy(state)
    if flip:
        last_e = last_element(state)
        if last_e %2 == 1: #rot flippt
            for i in range(last_e+1):
                return_state[i+1] = state[last_e-i]
            return_state[0] = 13
            return return_state
        else: #gelb flippt
            for i in range(last_e+1):
                return_state[i] = state[last_e-i]
            return_state[last_e+1] = 13
            return return_state
    for i in range(len(return_state)):
        if return_state[i] == 0 :
            if count_column(return_state,column) < 8:
                return_state[i] = column
            else:
                return_state[i] = -1
            return return_state
            
    return return_state


def newaction(state,flip = False):
    if flip:
        res = np.zeros((12,90))
        res[11] = do_action(state,flip = True)
    else:
        res = np.zeros((11,90))
    order = [6,5,7,8,4,3,9,10,2,1,11]
    for i in range(len(order)):
        res[i] = (do_action(state,order[i]))
    #np.random.shuffle(res)
    return res

def minval1(state,flip,num_it,alpha,beta,player):
    if ((not terminate(state) and sum(state != 0) ==9) and (state[6] == 3 or state[6] == 7) ):
        print(state)
    if terminate(state):
        last_e =  last_element(state)
        color = last_e%2
        #if (color == 0 and player == 1) or (color == 1 and player == 2):
        #if state[5] == 3 or state[5] == 7:
            #print(state)
        if color == 0:
            return 900
        return -1000

    if num_it == 0:
        return utility(state)
    v = 1000
    actions = newaction(state,flip[1])
    #print(actions)
    for i in actions:
        if i[0] != state[0]:
            v = min(v,maxval1(i,[flip[0],False],num_it-1,alpha,beta,otherplayer(player)))
        else:
            v = min(v,maxval1(i,flip,num_it-1,alpha,beta,otherplayer(player)))
        if v <= alpha:
            return v
        beta = min(v,beta)
    return v

def maxval1(state,flip,num_it,alpha,beta,player):
    if terminate(state):
        last_e =  last_element(state)
        color = last_e%2
        #if (color == 0 and player == 1) or (color == 1 and player == 2):
        if color == 0:
            return 1000
        return -1000
    
    if num_it == 0:
        return utility(state)
    v = -1000
    actions = newaction(state,flip[0])
    
    for i in actions:
        if i[0] != state[0]:
            v = max(v,minval1(i,[False,flip[1]],num_it-1,alpha,beta,otherplayer(player)))
        else:
            v = max(v,minval1(i,flip,num_it-1,alpha,beta,otherplayer(player)))
        if v >= beta:
            return v
        alpha = max(v,beta)            
    return v

def findbestmove1(state,flip,num_it):
    if state[3] == 0:
        return do_action(state,6),False,flip
    #if state[] nicht nacheinander flippen 
    actions = newaction(state,flip[0])    
    v = -10000
    index = np.random.uniform(1,12)
    beststate = do_action(state,index)  
    for i in actions:
        if i[0] != state[0]:
            newv = minval1(i,[False,flip[1]],num_it-1,-1000,1000)
        else:
            newv = minval1(i,flip,num_it-1,-1000,1000)
        #print(newv)
        print(newv,i)
        if newv > v:            
            v = newv
            
            beststate = i
    
    if beststate[0] != state[0]:
        return beststate, terminate(beststate),[False,flip[1]]
    return beststate, terminate(beststate),flip


# In[12]:

def otherplayer (player):
    if player == 1:
        return 2
    return 1


# In[13]:

#player1 fängt an, dann player2
def findbestmove2(state,flip,player,num_it):
    if state[3] == 0:
        return do_action(state,6),False,flip
    actions = newaction(state,flip[0])    
    v = -10000 if player == 1 else 10000
    index = np.random.uniform(1,12)
    beststate = do_action(state,index)
    
    for i in actions:
        last_e = last_element(i)
        if i[0] != state[0] and player == 1:
            newv = minval1(i,[False,flip[1]],num_it-1,-1000,1000,otherplayer(player))
        elif i[last_e] == 13:
            newv = maxval1(i,[flip[0],False],num_it-1,-1000,1000,otherplayer(player))
        elif player == 2:
            newv = maxval1(i,flip,num_it-1,-1000,1000,otherplayer(player))
        else:
            newv = minval1(i,flip,num_it-1,-1000,1000,otherplayer(player))
        #print(newv)
        print(newv,i)
        if newv > v and player == 1:            
            v = newv            
            beststate = i
        if newv < v and player == 2:
            v = newv
            beststate = i
    
    if sum(beststate == 13) != sum(state == 13) and player == 1:
        return beststate, terminate(beststate),[False,flip[1]]
    if sum(beststate == 13) != sum(state == 13) and player == 2:
        return beststate, terminate(beststate),[flip[0],False]
    return beststate, terminate(beststate),flip


# In[ ]:

def play5():
    state = field()
    start = int(input('Wähle einen Spieler(1 oder 2): '))
    if start == 1:
        new_input = int(input('Setze einen Stein (1-11): '))
        state = do_action(state,new_input)
        fig, ax = plt.subplots(figsize = (8,4),dpi = 130)
        plotboard(ax,state)      
    game_over = False
    #flip = [True,True]
    flip = [False,False]
    new_input = 100
    depth = 4
    
    comp = 2 if start == 1 else 1
    #print(comp)
    while not game_over:
        if new_input == 55:
            state,game_over,flip = findbestmove2(state,[False,False],comp,depth)
        else:
            state,game_over,flip = findbestmove2(state,flip,comp,depth)
        #print(v)
        #print(cboard.shape)
        fig, ax = plt.subplots(figsize = (8,4),dpi = 130)
        plotboard(ax,state)
        if game_over:
            break
        new_input = int(input('Setze einen Stein (1-11): '))
        if new_input == 55:
            state = do_action(state,flip = True)
            flip[1] = False
        else:
            state = do_action(state,new_input)
        #clear_output()
        fig, ax = plt.subplots(figsize = (8,4),dpi = 130)
        plotboard(ax,state)
        print(state)
        game_over = terminate(state)


# In[ ]:

play5()


# In[ ]:

r = np.array([6,6,5,6,4,3,7])
print(terminate(r))
fig, ax = plt.subplots()
plotboard(ax,r)


# In[ ]:

#t = np.array([5.  ,6.  ,4.  ,6.  ,3.,0,0,0,0,0,0,0,0])
6.  6.  4.  6.  3.  5.  2.  6.  1.
t = field()
t = do_action(t,6)
t = do_action(t,6)
t = do_action(t,5)
t = do_action(t,6)
t = do_action(t,4)
t = do_action(t,7)
t = do_action(t,3)
fig, ax = plt.subplots()
plotboard(ax,t)
res = newaction(t,True)

#for i in res:
 #   print(terminate(i))
#np.flat
res1 = []
for i in res:

    res1.append(newaction(i,True))
c = 0
for i in res1:
    for j in i:
        if terminate(j):

            print(j)
            c+=1
#print(terminate(t))
#fig, ax = plt.subplots()
#plotboard(ax,t)
#print(len(res1))
print(c)


# t = np.array([ 6.,  6 , 6. , 5. , 6.,  6.,  6.,  5. , 6. , 3.,  5.,  4.,  5. ,  4. , 5. , 3. , 5.])#, 5.,6.,  0.])
# t2 = np.array([6,6,6,6,4,6,5,7,3,2])
# t3 = np.array([ 13.,   5.   ,6.,   4.,   6. ,  5. ,  6. ,  7. ,  8.,   4. ,  6. ,  6. ,  6. ,  6. ,  6.,
#    4.,  11.])
# #print(terminate(t))
# print(terminate(t2))
# fig,ax = plt.subplots()
# plotboard(ax,t2)
# #print(count_column(t3,11))
# #print(count_column2(t3,11,len(t3)-1))
# #print(t3[last_element(t3)])
# #print(len(t3))

# In[ ]:

2
len(t2)


# In[ ]:

test = np.array([6,6,6,6,6,6,6,6,-1])
test1 = np.array([13,10,6,3,6,4,6,0,0])
test2 = np.array([6,4,6,3,6,10,6,6])
print(utility(test))


# In[ ]:

def play5():
    state = field()
    game_over = False
    while not game_over:
        state,game_over,v = findbestmove1(state,3)
        #print(v)
        #print(cboard.shape)
        fig, ax = plt.subplots(figsize = (8,4),dpi = 130)
        plotboard(a6
                  x,state)
        state,game_over,v = findbestmove2(state,3)
        if game_over:break
        #clear_output()
        fig, ax = plt.subplots(figsize = (8,4),dpi = 130)
        plotboard(ax,state)


# test = field()
# test = do_action(test,6)
# test = do_action(test,3)
# test = do_action(test,6)
# test = do_action(test,4)
# test = do_action(test,6)
# test = do_action(test,6)
# test = do_action(test,4)
# test = do_action(test,5)
# test = do_action(test,4)
# test = do_action(test,2)
# fig, ax = plt.subplots()
# plotboard(ax,test)
# print(terminate2(test))
# print(test)
