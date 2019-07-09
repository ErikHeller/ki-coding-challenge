# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import os

def field():
    return np.zeros(90)


# #######------------------#############
def openHelperFile():
    with open('Helperfile.txt') as f:
        lines = f.readlines()
        firstline = lines[0].strip()
        if firstline == 'C':
            weareplayer, theotherplayer = open5GewinntState()
        elif firstline == 'A':
            weareplayer = 'A'
            theotherplayer = 'B'
        elif firstline == 'B':
            weareplayer = 'B'
            theotherplayer = 'A'
        else:
            raise ValueError("Unable to read helper file")
        ourflip = (lines[1] == 'True\n')
        enemyflip = (lines[2] == 'True\n' or lines[2] == 'True')
        fliplist = [ourflip, enemyflip]
    return weareplayer, theotherplayer, fliplist


def openOurState(weareplayer, fliplist):
    with open('OurState.txt', 'r') as f:
        #content = f.read()
        state = np.loadtxt(f)
        currentposition = len(state != 0)
        enemyflipallowed = fliplist[1]
        if state[0] == 0:
            if weareplayer == 'A':
                enemyflipallowed = True
                return state, enemyflipallowed
            elif openLastAction() != 'flip': 
                #enemyflipallowed = True
                return do_action(state, column=openLastAction()), enemyflipallowed
            elif openLastAction() == 'flip':
                enemyflipallowed = False
                return do_action(state, flip = True), enemyflipallowed
            
        elif openLastAction() != 'flip': 
            #enemyflipallowed = True
            return do_action(state, column=openLastAction()), enemyflipallowed
        elif openLastAction() == 'flip':
            enemyflipallowed = False
            return do_action(state, flip=True), enemyflipallowed


def writeNewState(state, fliplist, weareplayer, ourmove):
    np.savetxt('OurState.txt', state, fmt = '%1d')
    with open('Helperfile.txt', 'w') as f:
        f.write(weareplayer + '\n')
        f.write(str(fliplist[0]) + '\n')
        f.write(str(fliplist[1]))
    f = open('LastAction_Player' + weareplayer + '.txt', "w")
    if ourmove != 13:
        f.write(str(int(ourmove)))
    elif ourmove == 13:
        f.write('flip')
    f.close()


def open5GewinntState():
    with open('5GewinntState.txt') as f:
        lines = f.readlines()
        if lines[1][11] == 'A':
            weareplayer = 'B'
            theotherplayer = 'A'
        elif lines[1][11] != 'A':
            weareplayer = 'A'
            theotherplayer = 'B'
        else:
            raise ValueError("Unable to read state file")
    return weareplayer, theotherplayer


#def openLastAction():
#    weareplayer, theotherplayer, _ = openHelperFile()
#    with open('LastAction_Player' + theotherplayer + '.txt', 'r') as f:
#        lastmove = f.read()
#    return lastmove

def openLastAction():
    with open('5GewinntState.txt') as f:
            lines = f.readlines()
            if len(lines[1]) == 15:
                lastmove = int(lines[1][-2])
            elif len(lines[1]) == 16:
                lastmove = int(lines[1][-3] + lines[1][-2])
            else:
                lastmove = 'flip'
    return lastmove
        
# #######------------------#############


# 1: red, 2: yellow
# %matplotlib inline
def plotboard(ax, state):
    for i in range(12):
        ax.vlines(i + 0.5, 0.5, 8.5)
    ax.hlines(0.5, 0.5, 11.5)
    counter = 0
    height = np.zeros(11)
    for j in state:
        if j == 0:
            break
        elif j == 13 or j == -1:
            counter += 1
            continue
        elif counter % 2 == 0:
            ax.plot(j, height[int(j - 1)] + 1, '.', markersize=45, color='red')
        else:
            ax.plot(j, height[int(j - 1)] + 1, '.', markersize=45, color='yellow')
        height[int(j - 1)] += 1
        counter += 1
    plt.show()


def last_element(state):
    for i in range(len(state)):
        if state[i] == 0:
            return i - 1
    return len(state) - 1


def findlastvalues(state):
    a, b = 0, 0
    last_e = last_element(state)
    color = last_e % 2
    for i in range(last_e + 1):
        if state[last_e - i] != -1 and (last_e - i) % 2 == color and state[last_e - i] != 13:
            a = last_e - i
            break
    for i in range(last_e + 1):
        if state[last_e - i] != -1 and (last_e - i) % 2 != color and state[last_e - i] != 13:
            b = last_e - i
            break
    return a, b


def count_column(state, column):
    # count = state == column
    counter = 0
    for x in state:
        if x == column:
            counter += 1
    return counter


def count_column2(state, column, last_e):
    res = 0
    for i in range(last_e + 1):
        if state[i] == column:
            res += 1
    return res


def terminate(state):
    if sum(state != 0) < 9:
        return False
    last_e1, _ = findlastvalues(state)
    color = last_e1 % 2
    # 0 für rot, 1 für gelb
    n = max(1, int(last_e1 / 2))
    # print(n)
    for k in range(n):  # geht jeden gelben/roten stein der farbe durch
        param = 0
        while True:
            last_e = last_e1 - k * 2 - param
            # print(last_e)
            if last_e < 8:
                return False
            elif state[last_e] != -1 and state[last_e] != 13:
                break
            else:
                param += 2
        # print(last_e)

        counter = 0
        column = state[last_e]
        height = count_column(state, column)

        # Configure the directions to look at and define if the counter determining subsequent identical stones
        # should be reset before checking this direction.
        # index1: vertical direction, index2: horizontal direction, index3: Reset counter
        # Always reset unless you have subsequent directions on the same axis that need to have a common counter.
        directions = [["bot", "left", True],
                      ["top", "right", False],
                      ["top", "left", True],
                      ["bot", "right", False],
                      ["bot", "", True],
                      ["", "left", True],
                      ["", "right", False]]

        for direction in directions:
            vert = direction[0]
            horiz = direction[1]
            reset = direction[2]

            # Reset counter
            if reset:
                counter = 0

            # Check the neighborhood with iteratively increasing distance for subsequent identical stones
            for i in range(4):
                # Set horizontal direction for column dependent variables
                if horiz == "left":
                    column_dir = column - i - 1
                    column_bounds_dir = column_dir > 0
                elif horiz == "right":
                    column_dir = column + i + 1
                    column_bounds_dir = column_dir < 12
                else:
                    # Don't move horizontally - no column bounds need to be checked
                    column_dir = column
                    column_bounds_dir = True
                indices_dir = np.where(state == int(column_dir))[0]

                # Set vertical direction for height dependent variables
                if vert == "bot":
                    height_dir = height - i - 1
                elif vert == "top":
                    height_dir = height + i + 1
                else:
                    # Don't move vertically
                    height_dir = height

                # Read color of stone in looked up direction.
                # Check if position of stone is outside the vertical board boundaries
                # and if so stop looking in that direction
                if len(indices_dir) >= height_dir > 0:
                    color_dir = indices_dir[height_dir - 1] % 2
                else:
                    break

                # Increase counter if a stone with the same color has been found in the looked up direction.
                # If no fitting stone has been found or the stone is outside the horizontal board boundaries
                # don't look in that direction for more stones
                if column_bounds_dir and color_dir == color:
                    counter += 1
                else:
                    break

                # If four additional subsequent stones with the same color have been found, terminate
                if counter == 4:
                    return True

    return False


def findminusones(state):
    a, b = 0, 0
    last_e = last_element(state)
    color = last_e % 2
    for i in range(last_e + 1):
        if i % 2 == color and state[i] == -1:
            a += 1
        elif i % 2 != color and state[i] == -1:
            b += 1

    return a, b


def utility2_new(state):
    full_column_penalty = -50
    win_bonus = 100
    space_penalty = 100

    last_e, last_e2 = findlastvalues(state)
    color = last_e % 2
    res = color
    full_columns = findminusones(state)
    # 0 für rot, 1 für gelb
    state_before = state.copy()
    state_before[last_e] = 0
    # print(state_before)
    counter = [0, 0]
    a, b = 2, 1
    for c in range(len(counter)):
        if c == 1:
            last_e = last_e2
            # print(last_e)
        color = last_e % 2
        combo = 0
        column = state[last_e]
        height = count_column2(state, column, last_e)

        # Configure the directions to look at
        # index1: vertical direction, index2: horizontal direction
        directions = [["bot", "left", True],
                      ["top", "right", False],
                      ["top", "left", True],
                      ["bot", "right", False],
                      ["bot", "", True],
                      ["", "left", True],
                      ["", "right", False]]

        for direction in directions:
            vert = direction[0]
            horiz = direction[1]
            reset = direction[2]

            # Reset combos
            if reset:
                combo = 0

            # Check the neighborhood with iteratively increasing distance
            for i in range(4):
                # Set horizontal direction for column dependent variables
                if horiz == "left":
                    column_dir = column - i - 1
                    column_bounds_dir = column_dir > 0
                elif horiz == "right":
                    column_dir = column + i + 1
                    column_bounds_dir = column_dir < 12
                else:
                    # Don't move horizontally - no column bounds need to be checked
                    column_dir = column
                    column_bounds_dir = True
                indices_dir = np.where(state == int(column_dir))[0]

                # Set vertical direction for height dependent variables
                if vert == "bot":
                    height_dir = height - i - 1
                elif vert == "top":
                    height_dir = height + i + 1
                else:
                    # Don't move vertically
                    height_dir = height

                # TODO: Edit comment
                # Increase counter if a stone with the same color has been found in the looked up direction.
                # If no fitting stone has been found or the stone is outside the horizontal board boundaries
                # don't look in that direction for more stones
                if height_dir > 0 and column_bounds_dir:
                    if len(indices_dir) >= height_dir:
                        if indices_dir[height_dir - 1] % 2 == color:
                            combo += 1
                            counter[c] += a * combo
                        else:
                            break
                    else:
                        counter[c] += b
                else:
                    break

            # Add bonus for possible win
            if combo == 4:
                counter[c] += win_bonus

            # Penalty if there is less space than required for vertical move
            # TODO: Check diagonal axes
            if horiz == "":
                space_req = 4 - combo
                space_avail = 8 - height
                penalty = b * min(0, space_avail - space_req) * space_penalty
                counter[c] += penalty

        counter[c] += b * min(4, 8 - height)
        counter[c] += full_column_penalty * full_columns[c]

    if res == 0:
        return counter[0] - counter[1]
    return counter[1] - counter[0]


def utility(state):
    res = state.copy()
    n = len(res)
    i = 0
    v = utility2_new(res)
    while n - 4 - 2 * i >= 0 and i < 3:
        # print(v,res)
        a, b = findlastvalues(res)
        if a == 0 or b == 0:
            break
        res[a] = -1
        res[b] = -1
        v += utility2_new(res)
        i += 1
    return v


# spieler 1
def do_action(state, column=None, flip=False):
    return_state = np.copy(state)
    if flip:
        last_e = last_element(state)
        if last_e % 2 == 1:  # rot flippt
            for i in range(last_e + 1):
                return_state[i + 1] = state[last_e - i]
            return_state[0] = 13
            return return_state
        else:  # gelb flippt
            for i in range(last_e + 1):
                return_state[i] = state[last_e - i]
            return_state[last_e + 1] = 13
            return return_state
    for i in range(len(return_state)):
        if return_state[i] == 0:
            if count_column(return_state, column) < 8:
                return_state[i] = column
            return return_state

    return return_state


def newaction(state, flip=False):
    if flip:
        res = np.zeros((12, 90))
        res[11] = do_action(state, flip=True)
    else:
        res = np.zeros((11, 90))

    # Order new actions starting from last action of the same player outwards in the board
    last_action = state[last_element(state) - 1]
    order = [last_action,
             ((last_action - 2) % 11) + 1, (last_action % 11) + 1,
             ((last_action - 3) % 11) + 1, ((last_action + 1) % 11) + 1,
             ((last_action - 4) % 11) + 1, ((last_action + 2) % 11) + 1,
             ((last_action - 5) % 11) + 1, ((last_action + 3) % 11) + 1,
             ((last_action - 6) % 11) + 1, ((last_action + 4) % 11) + 1]

    for i in range(len(order)):
        res[i] = (do_action(state, order[i]))
    # np.random.shuffle(res)
    return res
def utility2_new(state):
    full_column_penalty = -50
    win_bonus = 100
    space_penalty = 100

    last_e, last_e2 = findlastvalues(state)
    color = last_e % 2
    res = color
    full_columns = findminusones(state)
    # 0 für rot, 1 für gelb
    state_before = state.copy()
    state_before[last_e] = 0
    # print(state_before)
    counter = [0, 0]
    a, b = 2, 1
    for c in range(len(counter)):
        if c == 1:
            last_e = last_e2
            # print(last_e)
        color = last_e % 2
        combo = 0
        column = state[last_e]
        height = count_column2(state, column, last_e)

        # Configure the directions to look at
        # index1: vertical direction, index2: horizontal direction
        directions = [["bot", "left", True],
                      ["top", "right", False],
                      ["top", "left", True],
                      ["bot", "right", False],
                      ["bot", "", True],
                      ["", "left", True],
                      ["", "right", False]]

        for direction in directions:
            vert = direction[0]
            horiz = direction[1]
            reset = direction[2]

            # Reset combos
            if reset:
                combo = 0

            # Check the neighborhood with iteratively increasing distance
            for i in range(4):
                # Set horizontal direction for column dependent variables
                if horiz == "left":
                    column_dir = column - i - 1
                    column_bounds_dir = column_dir > 0
                elif horiz == "right":
                    column_dir = column + i + 1
                    column_bounds_dir = column_dir < 12
                else:
                    # Don't move horizontally - no column bounds need to be checked
                    column_dir = column
                    column_bounds_dir = True
                indices_dir = np.where(state == int(column_dir))[0]

                # Set vertical direction for height dependent variables
                if vert == "bot":
                    height_dir = height - i - 1
                elif vert == "top":
                    height_dir = height + i + 1
                else:
                    # Don't move vertically
                    height_dir = height

                # TODO: Edit comment
                # Increase counter if a stone with the same color has been found in the looked up direction.
                # If no fitting stone has been found or the stone is outside the horizontal board boundaries
                # don't look in that direction for more stones
                if height_dir > 0 and column_bounds_dir:
                    if len(indices_dir) >= height_dir:
                        if indices_dir[height_dir - 1] % 2 == color:
                            combo += 1
                            counter[c] += a * combo
                        else:
                            break
                    else:
                        counter[c] += b
                else:
                    break

            # Add bonus for possible win
            if combo == 4:
                counter[c] += win_bonus

            # Penalty if there is less space than required for vertical move
            # TODO: Check diagonal axes
            if horiz == "":
                space_req = 4 - combo
                space_avail = 8 - height
                penalty = b * min(0, space_avail - space_req) * space_penalty
                counter[c] += penalty

        counter[c] += b * min(4, 8 - height)
        counter[c] += full_column_penalty * full_columns[c]

    if res == 0:
        return counter[0] - counter[1]
    return counter[1] - counter[0]


def utility(state):
    res = state.copy()
    n = len(res)
    i = 0
    v = utility2_new(res)
    while n - 4 - 2 * i >= 0 and i < 3:
        # print(v,res)
        a, b = findlastvalues(res)
        if a == 0 or b == 0:
            break
        res[a] = -1
        res[b] = -1
        v += utility2_new(res)
        i += 1
    return v


# spieler 1
def do_action(state, column=None, flip=False):
    return_state = np.copy(state)
    if flip:
        last_e = last_element(state)
        if last_e % 2 == 1:  # rot flippt
            for i in range(last_e + 1):
                return_state[i + 1] = state[last_e - i]
            return_state[0] = 13
            return return_state
        else:  # gelb flippt
            for i in range(last_e + 1):
                return_state[i] = state[last_e - i]
            return_state[last_e + 1] = 13
            return return_state
    for i in range(len(return_state)):
        if return_state[i] == 0:
            if count_column(return_state, column) < 8:
                return_state[i] = column
            return return_state

    return return_state


def newaction(state, flip=False):
    if flip:
        res = np.zeros((12, 90))
        res[11] = do_action(state, flip=True)
    else:
        res = np.zeros((11, 90))

    # Order new actions starting from last action of the same player outwards in the board
    last_action = state[last_element(state) - 1]
    order = [last_action,
             ((last_action - 2) % 11) + 1, (last_action % 11) + 1,
             ((last_action - 3) % 11) + 1, ((last_action + 1) % 11) + 1,
             ((last_action - 4) % 11) + 1, ((last_action + 2) % 11) + 1,
             ((last_action - 5) % 11) + 1, ((last_action + 3) % 11) + 1,
             ((last_action - 6) % 11) + 1, ((last_action + 4) % 11) + 1]

    for i in range(len(order)):
        res[i] = (do_action(state, order[i]))
    # np.random.shuffle(res)
    return res



def minval1(state, flip, num_it, alpha, beta, player):
    #if (not terminate(state) and sum(state != 0) == 9) and (state[6] == 3 or state[6] == 7):
        #print(state)
    if terminate(state):
        last_e = last_element(state)
        color = last_e % 2
        # if (color == 0 and player == 1) or (color == 1 and player == 2):
        # if state[5] == 3 or state[5] == 7:
        # print(state)
        if color == 0:
            return 900
        return -1000

    if num_it == 0:
        return utility(state)
    v = 1000
    actions = newaction(state, flip[1])
    for i in actions:
        if i[0] != state[0]:
            v = min(v, maxval1(i, [flip[0], False], num_it - 1, alpha, beta, otherplayer(player)))
        else:
            v = min(v, maxval1(i, flip, num_it - 1, alpha, beta, otherplayer(player)))
        if v <= alpha:
            return v
        beta = min(v, beta)
    return v


def maxval1(state, flip, num_it, alpha, beta, player):
    if terminate(state):
        last_e = last_element(state)
        color = last_e % 2
        # if (color == 0 and player == 1) or (color == 1 and player == 2):
        if color == 0:
            return 1000
        return -1000

    if num_it == 0:
        return utility(state)
    v = -1000
    actions = newaction(state, flip[0])

    for i in actions:
        if i[0] != state[0]:
            v = max(v, minval1(i, [False, flip[1]], num_it - 1, alpha, beta, otherplayer(player)))
        else:
            v = max(v, minval1(i, flip, num_it - 1, alpha, beta, otherplayer(player)))
        if v >= beta:
            return v
        alpha = max(v, beta)
    return v


def otherplayer(player):
    if player == 'A':
        return 'B'
    return 'A'


# player1 fängt an, dann player2
def findbestmove2(state, flip, player, num_it):
    if state[3] == 0:
        return do_action(state, 6), False, flip
    actions = newaction(state, flip[0])
    v = -10000 if player == 'A' else 10000
    index = np.random.uniform(1, 12)
    beststate = do_action(state, index)

    for i in actions:
        last_e = last_element(i)
        if i[0] != state[0] and player == 'A':
            newv = minval1(i, [False, flip[1]], num_it - 1, -1000, 1000, otherplayer(player))
        elif i[last_e] == 13:
            newv = maxval1(i, [flip[0], False], num_it - 1, -1000, 1000, otherplayer(player))
        elif player == 'B':
            newv = maxval1(i, flip, num_it - 1, -1000, 1000, otherplayer(player))
        else:
            newv = minval1(i, flip, num_it - 1, -1000, 1000, otherplayer(player))
        # print(newv)
        print(newv, i)
        if newv > v and player == 'A':
            v = newv
            beststate = i
        if newv < v and player == 'B':
            v = newv
            beststate = i

    if sum(beststate == 13) != sum(state == 13) and player == 'A':
        return beststate, terminate(beststate), [False, flip[1]]
    if sum(beststate == 13) != sum(state == 13) and player == 'B':
        return beststate, terminate(beststate), [flip[0], False]
    return beststate, terminate(beststate), flip



def play5():
    new_input = 100
    depth = 3


    weareplayer, theotherplayer, fliplist = openHelperFile()  
    comp = 'A' if weareplayer == 'A' else 'B'
    print(comp)
    newstate, enemyflipallowed = openOurState(weareplayer, fliplist)
    if enemyflipallowed == False:
        fliplist[1] = False
    state, _, fliplist = findbestmove2(newstate, fliplist, comp, depth)
    ourmove = state[max(np.where(state != 0)[-1])]
    if ourmove.any() == 13:
        ourmove = 'flip'
    writeNewState(state, fliplist, weareplayer, ourmove)




if __name__ == '__main__':
    play5()
