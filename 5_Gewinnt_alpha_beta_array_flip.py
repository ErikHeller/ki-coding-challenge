# coding: utf-8
import numpy as np

from matplotlib import pyplot as plt

# player1 = 1, player2 = 2
print(80 * 0.9994 ** 3000)
fig, ax = plt.subplots(figsize=(6, 4))
n = 5000
start = 0.5
fac = 0.999
x = np.linspace(0, n, 300)
ax.plot(x, 100 * start * fac ** x)
ax.grid()
plt.show()


def field():
    return np.zeros(90)


def plotboard(ax, state):
    for i in range(12):
        ax.vlines(i + 0.5, 0.5, 8.5)
    ax.hlines(0.5, 0.5, 11.5)
    counter = 0
    height = np.zeros(11)
    for j in state:
        if (j == 0):
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


def findlastvalues(state):
    a, b = 0, 0
    last_e = last_element(state)
    color = last_e % 2
    for i in range(last_e):
        if state[last_e - i] != -1 and (last_e - i) % 2 == color and state[last_e - i] != 13:
            a = last_e - i
            break
    for i in range(last_e):
        if state[last_e - i] != -1 and (last_e - i) % 2 != color and state[last_e - i] != 13:
            b = last_e - i
            break
    return a, b


# spieler 1


def do_action(state, column=None, flip=False):
    return_state = np.copy(state)
    if flip:
        last_e = last_element(state)
        if last_e % 2 == 1:
            for i in range(last_e + 1):
                return_state[i + 1] = state[last_e - i]
            return_state[0] = 13
            return return_state
        else:
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


def count_column(state, column):
    return sum(state == column)


def last_element(state):
    for i in range(len(state)):
        if state[i] == 0:
            return i - 1
    return len(state) - 1


def terminate(state):
    if sum(state != 0) < 9:
        return False
    last_e, _ = findlastvalues(state)
    color = last_e % 2
    # 0 für rot, 1 für gelb

    counter = 0
    column = state[last_e]
    height = count_column(state, column)

    # horizontal
    check_left = True
    check_right = True
    for i in range(4):
        indices = np.where(state == int(column - i - 1))[0]
        if len(indices) >= height and len(indices) > 0:
            if check_left and column - i - 1 > 0 and indices[height - 1] % 2 == color:
                counter += 1
            else:
                check_left = False
        else:
            check_left = False
        indices = np.where(state == int(column + i + 1))[0]
        if len(indices) >= height and len(indices) > 0:
            if check_right and column + i < 12 and indices[height - 1] % 2 == color:
                counter += 1
            else:
                check_right = False
        else:
            check_right = False
        # print(counter,'r')
        if (counter == 4):
            return True
    # vertikal
    counter = 0
    check_bot = True

    for i in range(4):
        indices = np.where(state == int(column))[0]
        # if(and len(indices)>0)
        if check_bot and height - i - 1 > 0 and indices[height - i - 2] % 2 == color:
            counter += 1
        else:
            check_bot = False
    # print(counter)
    if (counter == 4):
        return True

    # diag top left, bot right
    counter = 0
    check_left = True
    check_right = True
    for i in range(4):
        indices = np.where(state == int(column - i - 1))[0]
        if len(indices) >= height + 1 + i:
            if check_left and column - i - 1 > 0 and indices[height + i] % 2 == color:
                counter += 1
            else:
                check_left = False
        else:
            check_left = False

        indices = np.where(state == int(column + i + 1))[0]
        if len(indices) >= height - 1 - i and height - 1 - i > 0:
            if check_right and column + i + 1 < 12 and indices[height - i - 2] % 2 == color:
                counter += 1
            else:
                check_right = False
        else:
            check_right = False
        # print(counter,'r')
        if (counter == 4):
            return True

    # diag bot left, top right
    counter = 0
    check_left = True
    check_right = True
    for i in range(4):
        indices = np.where(state == int(column - i - 1))[0]
        if len(indices) >= height - 1 - i and height - 1 - i > 0:
            if check_left and column - i - 1 > 0 and indices[height - i - 2] % 2 == color:
                counter += 1
            else:
                check_left = False
        else:
            check_left = False

        indices = np.where(state == int(column + i + 1))[0]
        if len(indices) >= height + 1 + i:
            if check_right and column + i + 1 < 12 and indices[height + i] % 2 == color:
                counter += 1
            else:
                check_right = False

        else:
            check_right = False
        # print(counter,'r')
        if (counter == 4):
            return True
    return False


def utility(state):
    last_e, last_e2 = findlastvalues(state)
    color = last_e % 2
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
            if (len(indices) >= height and len(indices) > 0 and check_left and column - i - 1 > 0):
                if (indices[height - 1] % 2 == color):
                    counter[c] += a
                else:
                    check_left = False
            elif (check_left and column - i - 1 > 0 and len(indices) < height):
                counter[c] += b
            else:
                check_left = False
            # print(counter,'lh')
            indices = np.where(state == int(column + i + 1))[0]
            if (len(indices) >= height and len(indices) > 0 and check_right and column + i + 1 < 12):
                if (indices[height - 1] % 2 == color):
                    counter[c] += a
                else:
                    check_right = False
            elif (check_right and column + i + 1 < 12 and len(indices) < height):
                counter[c] += b
            else:
                check_right = False
            # print(counter,'rh')

        # vertikal
        check_bot = True
        for i in range(4):
            indices = np.where(state == int(column))[0]
            # if(and len(indices)>0)
            if (check_bot and height - i - 1 > 0 and indices[height - i - 2] % 2 == color):
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
            if (len(indices) >= height + 1 + i and check_left and column - i - 1 > 0):
                if (indices[height + i] % 2 == color):
                    counter[c] += a
                else:
                    check_left = False
            elif (check_left and column - i - 1 > 0 and len(indices) < height + 1 + i):
                counter[c] += b
            else:
                check_left = False

            indices = np.where(state == int(column + i + 1))[0]
            if (len(indices) >= height - 1 - i and height - 1 - i > 0 and check_right and column + i + 1 < 12):
                if (indices[height - i - 2] % 2 == color):
                    counter[c] += a
                else:
                    check_right = False
            elif (check_right and column + i + 1 < 12 and len(indices) < height - 1 - i and height - 1 - i > 0):
                counter[c] += b
            else:
                check_right = False
        # print(counter,'diag1')

        # diag bot left, top right
        check_left = True
        check_right = True
        for i in range(4):
            indices = np.where(state == int(column - i - 1))[0]
            if (len(indices) >= height - 1 - i and height - 1 - i > 0 and check_left and column - i - 1 > 0):
                if (indices[height - i - 2] % 2 == color):
                    counter[c] += a
                else:
                    check_left = False
            elif (check_left and column - i - 1 > 0 and len(indices) < height - 1 - i and height - 1 - i > 0):
                counter[c] += b
            else:
                check_left = False

            indices = np.where(state == int(column + i + 1))[0]
            if (len(indices) >= height + 1 + i and check_right and column + i + 1 < 12):
                if (indices[height + i] % 2 == color):
                    counter[c] += a
                else:
                    check_right = False
            elif (check_right and column + i + 1 < 12 and len(indices) < height + 1 + i):
                counter[c] += b
            else:
                check_right = False
            # print(counter,'r')
    return counter[0] - counter[1]
test = field()
test = do_action(test, 6)
test = do_action(test, 5)
# test = do_action(test,7)
test = do_action(test, 8)
fig, ax = plt.subplots()
# for i in actions:
#   print(i)
#  print(utility(i))
# print(findlastvalues(test))


plotboard(ax, test)


print(utility(test))


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
# 1: red, 2: yellow
# %matplotlib inline

actions = newaction(test, True)

param = True


for i in range(5):
    if i == 1 and param:
        print(1, 'i')
    elif i == 2 or i == 3:
        print(i)
    else:
        param = False
        print(9)


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

