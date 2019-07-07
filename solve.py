# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt


def field():
    return np.zeros(90)


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
    a, b = -1, -1
    last_e = last_element(state)
    color = last_e % 2
    for i in range(last_e):
        if a != -1 and b != -1:
            break

        if state[last_e - i] != -1 and state[last_e - i] != 13:
            if (last_e - i) % 2 == color:
                a = last_e - i
            else:
                b = last_e - i

    # TODO: Check if necessary
    if a == -1:
        a = 0
    if b == -1:
        b = 0

    return a, b


def count_column(state, column):
    # count = state == column
    count = [(x == column) for x in state]
    return sum(count)


def count_column2(state,column,last_e):
    res = 0
    for i in range(last_e+1):
        if state[i] == column:
            res +=1
    return res


def do_action(state, column=None, flip=False):
    return_state = np.copy(state)
    if flip:
        last_e = last_element(state)

        for i in range(last_e + 1):
            return_state[i + 1] = state[last_e - i]

        if last_e % 2 == 1:
            return_state[0] = 13
            return return_state
        else:
            return_state[last_e + 1] = 13
            return return_state

    for i in range(len(return_state)):
        if return_state[i] == 0:
            if count_column(return_state, column) < 8:
                return_state[i] = column
            return return_state

    return return_state


def terminate(state):
    if sum([(x != 0) for x in state]) < 9:
        return False
    last_el, _ = findlastvalues(state)
    color = last_el % 2
    # 0 für rot, 1 für gelb

    n = max(1, int(last_el / 2))
    # print(n)
    for k in range(n):  # geht jeden gelben/roten stein der farbe durch
        param = 0
        while True:
            last_e = last_el - k * 2 - param
            # print(last_e)
            if last_e < 8:
                return False
            elif state[last_e] != -1 and state[last_e] != 13:
                break
            else:
                param += 2

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


def utility(state):
    last_e, last_e2 = findlastvalues(state)
    # color = last_e % 2
    # 0 für rot, 1 für gelb
    state_before = state.copy()
    state_before[last_e] = 0
    # print(state_before)
    counter = [0, 0]
    a, b = 3, 1
    for c in range(len(counter)):
        if c == 1:
            last_e = last_e2
            # print(last_e)
        color = last_e % 2
        column = state[last_e]
        height = count_column(state, column)

        # horizontal
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column - i - 1))[0]
            if len(indices) >= height and len(indices) > 0 and check_left and column - i - 1 > 0:
                if indices[height - 1] % 2 == color:
                    counter[c] += a
                else:
                    check_left = False
            elif check_left and column - i - 1 > 0 and len(indices) < height:
                counter[c] += b
            else:
                check_left = False
            # print(counter,'lh')
            indices = np.where(state == int(column + i + 1))[0]
            if len(indices) >= height and len(indices) > 0 and check_right and column + i + 1 < 12:
                if indices[height - 1] % 2 == color:
                    counter[c] += a
                else:
                    check_right = False
            elif check_right and column + i + 1 < 12 and len(indices) < height:
                counter[c] += b
            else:
                check_right = False
            # print(counter,'rh')

        # vertikal
        check_bot = True
        for i in range(4):
            indices = np.where(state == int(column))[0]
            # if(and len(indices)>0)
            if check_bot and height - i - 1 > 0 and indices[height - i - 2] % 2 == color:
                counter[c] += a
            else:
                check_bot = False
        counter[c] += b * (min(4, 8 - height))
        # print(counter,'v')

        # diag top left, bot right
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column - i - 1))[0]
            if len(indices) >= height + 1 + i and check_left and column - i - 1 > 0:
                if indices[height + i] % 2 == color:
                    counter[c] += a
                else:
                    check_left = False
            elif check_left and column - i - 1 > 0 and len(indices) < height + 1 + i:
                counter[c] += b
            else:
                check_left = False

            indices = np.where(state == int(column + i + 1))[0]
            if len(indices) >= height - 1 - i > 0 and check_right and column + i + 1 < 12:
                if indices[height - i - 2] % 2 == color:
                    counter[c] += a
                else:
                    check_right = False
            elif check_right and column + i + 1 < 12 and len(indices) < height - 1 - i and height - 1 - i > 0:
                counter[c] += b
            else:
                check_right = False
        # print(counter,'diag1')

        # diag bot left, top right
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column - i - 1))[0]
            if len(indices) >= height - 1 - i > 0 and check_left and column - i - 1 > 0:
                if indices[height - i - 2] % 2 == color:
                    counter[c] += a
                else:
                    check_left = False
            elif check_left and column - i - 1 > 0 and len(indices) < height - 1 - i and height - 1 - i > 0:
                counter[c] += b
            else:
                check_left = False

            indices = np.where(state == int(column + i + 1))[0]
            if len(indices) >= height + 1 + i and check_right and column + i + 1 < 12:
                if indices[height + i] % 2 == color:
                    counter[c] += a
                else:
                    check_right = False
            elif check_right and column + i + 1 < 12 and len(indices) < height + 1 + i:
                counter[c] += b
            else:
                check_right = False
            # print(counter,'r')
    return counter[0] - counter[1]


def newaction(state, flip=False):
    if flip:
        res = np.zeros((12, 90))
        res[11] = do_action(state, flip=True)
    else:
        res = np.zeros((11, 90))
    for i in range(11):
        res[i] = (do_action(state, i + 1))
    np.random.shuffle(res)
    return res


def maxval1(state, flip, num_it, alpha, beta):
    if terminate(state):
        last_e = last_element(state)
        color = last_e % 2
        if color == 0:
            return 1000
        return -1000

    if num_it == 0:
        return utility(state)
    v = -1000
    actions = newaction(state, flip[0])

    for i in actions:
        if i[0] != state[0]:
            v = max(v, minval1(i, [False, flip[1]], num_it - 1, alpha, beta))
        else:
            v = max(v, minval1(i, flip, num_it - 1, alpha, beta))
        if v >= beta:
            return v
        alpha = max(v, beta)
    return v


def minval1(state, flip, num_it, alpha, beta):
    if terminate(state):
        last_e = last_element(state)
        color = last_e % 2
        if color == 0:
            return 1000
        return -1000

    if num_it == 0:
        return utility(state)
    v = 1000
    actions = newaction(state, flip[1])
    # print(actions)
    for i in actions:
        if i[0] != state[0]:
            v = min(v, maxval1(i, [flip[0], False], num_it - 1, alpha, beta))
        else:
            v = min(v, maxval1(i, flip, num_it - 1, alpha, beta))
        if v <= alpha:
            return v
        beta = min(v, beta)
    return v


def findbestmove1(state, flip, num_it):
    if state[2] == 0:
        return do_action(state, 6), False, flip
    # if state[] nicht nacheinander flippen
    actions = newaction(state, flip[0])
    v = -10000
    index = np.random.uniform(1, 12)
    beststate = do_action(state, index)
    for i in actions:
        if i[0] != state[0]:
            newv = minval1(i, [False, flip[1]], num_it - 1, -1000, 1000)
        else:
            newv = minval1(i, flip, num_it - 1, -1000, 1000)
        # print(newv)
        if newv > v:
            v = newv
            # print(i)
            beststate = i
    if beststate[0] != state[0]:
        return beststate, terminate(beststate), [False, flip[1]]
    return beststate, terminate(beststate), flip


def play5():
    state = field()
    game_over = False
    flip = [True, True]
    while not game_over:
        state, game_over, flip = findbestmove1(state, flip, 4)
        # print(v)
        # print(cboard.shape)
        fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
        plotboard(ax, state)
        new_input = int(input('Setze einen Stein (1-11): '))
        if new_input == 55:
            state = do_action(state, flip=True)
            flip[1] = False
        else:
            state = do_action(state, new_input)
        # clear_output()
        fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
        plotboard(ax, state)
        print(state)


if __name__ == '__main__':
    play5()
