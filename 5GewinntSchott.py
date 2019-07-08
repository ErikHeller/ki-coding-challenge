import os
import sys
import time
import signal
import matplotlib.pyplot as plt


# Kill a process
def check_kill_process(pstring):
    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)


# Add a coin
def addCoin(state, pos, player):
    pos = pos - 1
    if pos < 0:
        print("E1: Not allowed move by " + str(player) + " at position " + str(pos))
        return False
    if pos >= len(state):
        print("E2: Not allowed move by " + str(player) + " at position " + str(pos))
        return False
    if state[pos][0] != 'e':
        print("E3: Not allowed move by " + str(player) + " at position " + str(pos) + " since board was full")
        return False

    for y in range(len(state[pos]) - 1, -1, -1):
        if state[pos][y] == 'e':
            state[pos][y] = player
            return True
    print("Not allowed move by " + str(player))
    return False


# Flips the board
def flip(state):
    oldState = [['e' for y in range(8)] for x in range(11)]
    for y in range(len(state[0])):
        for x in range(len(state)):
            oldState[x][y] = state[x][y]
            state[x][y] = 'e'
    for x in range(len(state)):
        flag = False
        yPos = len(state[0]) - 1
        for y in range(len(state[0])):
            if oldState[x][y] != 'e':
                flag = True
            if flag:
                state[x][yPos] = oldState[x][y]
                yPos = yPos - 1
    return True


# test if there is a 5er in down-direction starting from x,y
def testPositionDown(state, x, y):
    player = state[x][y]
    maxX = len(state)
    maxY = len(state[0])
    if y > maxY - 5:
        return ''
    for ny in range(y + 1, y + 5):
        if player != state[x][ny]:
            return ''
    return player


# test if there is a 5er in right-direction starting from x,y
def testPositionRight(state, x, y):
    player = state[x][y]
    maxX = len(state)
    maxY = len(state[0])
    if x > maxX - 5:
        return ''
    for nx in range(x + 1, x + 5):
        if player != state[nx][y]:
            return ''
    return player


# test if there is a 5er down-right-diagonal starting from x,y
def testPositionDownRight(state, x, y):
    player = state[x][y]
    maxX = len(state)
    maxY = len(state[0])
    if x > maxX - 5 or y > maxY - 5:
        return ''
    ny = y + 1
    for nx in range(x + 1, x + 5):
        if player != state[nx][ny]:
            return ''
        ny = ny + 1
    return player


# test if there is a 5er up-right-diagonal starting from x,y
def testPositionUpRight(state, x, y):
    player = state[x][y]
    maxX = len(state)
    maxY = len(state[0])
    if x > maxX - 5 or y < 4:
        return ''
    ny = y - 1
    for nx in range(x + 1, x + 5):
        if player != state[nx][ny]:
            return ''
        ny = ny - 1
    return player


# test if there is a 5er starting from a given x,y
def testPosition(state, x, y):
    player = state[x][y]
    # x=0,y=0 is upper left corner
    maxX = len(state)
    maxY = len(state[0])
    if player == 'e':
        return ''
    if testPositionDown(state, x, y) != '':
        return player
    if testPositionRight(state, x, y) != '':
        return player
    if testPositionDownRight(state, x, y) != '':
        return player
    if testPositionUpRight(state, x, y) != '':
        return player
    return ''


# Test is somebody has one, by testing each position
def hasWon(state):
    player = ''
    for x in range(len(state)):
        for y in range(len(state[0])):
            player = testPosition(state, x, y)
            if player != '':
                return player
    return ''


# Test if there is a draw in the game
def isDraw(state):
    for x in range(len(state)):
        for y in range(len(state[0])):
            if state[x][y] == 'e':
                return False
    print("Draw Game")
    return True


# Read Move From File
def readMove(filename):
    try:
        f = open(filename, "r")
        move = f.readline()
        f.close()
        return str(move)
    except FileNotFoundError:
        return '-999'


# Read State-File to disk
def readStateSpace(filename):
    state = [['e' for y in range(8)] for x in range(11)]
    f = open(filename, "r")
    bflag = False
    nY = 0
    for line in f:
        if bflag:
            nX = 0
            for c in line:
                if nX < len(state):
                    state[nX][nY] = c
                nX = nX + 1
            nY = nY + 1
        if line.find("Current-State") != -1:
            bflag = True
    f.close()
    return state


# Save State-File to disk
def writeStateSpace(filename, nMove, sPlayer, sMove, state):
    f = open(filename, "w")
    f.write("Moves-Played: " + str(nMove) + "\n")
    f.write("Last-Move: " + str(sPlayer) + "	" + str(sMove) + "\n")
    f.write("\n")
    f.write("Current-State:\n")
    for y in range(len(state[0])):
        for x in range(len(state)):
            f.write(str(state[x][y]))
        f.write("\n")
    f.close()


# Print State on Screen
def printState(state):
    print("State")
    for y in range(len(state[0])):
        for x in range(len(state)):
            c = '.'
            if state[x][y] != 'e':
                c = state[x][y]
            print(c),
        print('')


def drawState(state):
    global fig
    global ax
    ax.cla()
    circle = plt.Circle((6, 5), 40, color='b')
    ax.add_artist(circle)
    for y in range(len(state[0])):
        for x in range(len(state)):
            scol = 'w'
            if state[x][y] == 'A':
                scol = 'r'
            if state[x][y] == 'B':
                scol = 'y'
            circle1 = plt.Circle((x + 1, len(state[0]) - y), 0.4, color=scol)
            ax.add_artist(circle1)
    plt.axis([0, 12, 0, 10])
    fig.savefig('5GewinntState.png')


# Lets Player X do its Move
def movePlayer(programname, player, state):
    global nMOVES
    global lMOVES
    global bFlippedPlayerA
    global bFlippedPlayerB
    os.system(programname + ' 5GewinntState.txt LastAction_Player' + player + '.txt &')
    time.sleep(1)
    check_kill_process('LastAction_Player' + player + '.txt')
    move = readMove('LastAction_Player' + player + '.txt')
    if move == '-999':
        return "Player " + player + " did not provide information on its next move in time - disqualified"
    os.remove('LastAction_Player' + player + '.txt')
    nMOVES = nMOVES + 1
    if move == 'flip' or move == 'FLIP' or move == 'Flip':
        if (player == 'A' and bFlippedPlayerA == True) or (player == 'B' and bFlippedPlayerB == True):
            return "Player " + player + " tried to flip board a second time - disqualified"
        print("Player " + player + " flips board")
        flip(state)
        lMOVES.append('Player ' + player + ': flipped')
        if player == 'A':
            bFlippedPlayerA = True
        if player == 'B':
            bFlippedPlayerB = True
    else:
        print("Player " + player + " puts coin to row " + move)
        lMOVES.append('Player ' + player + ': ' + move)
        if not addCoin(state, int(move), player):
            return "Player " + player + " tried to put a coin where one shouldnt - disqualified"
    writeStateSpace("5GewinntState.txt", nMOVES, 'A', move, state)
    printState(state)
    return ''


##########################################
# Main Program
# Usage: python 5Gewinnt.py PROGRAM_PLAYER_A PROGRAM_PLAYER_B 
##########################################
PROGRAM_PLAYER_A = 'python 5GewinntDummyPlayer.py'
PROGRAM_PLAYER_B = 'python 5GewinntB.py'

nMOVES = 0
lMOVES = []
bFlippedPlayerA = False
bFlippedPlayerB = False

# fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot

# Create Empty Board
state = [['e' for y in range(8)] for x in range(11)]
writeStateSpace("5GewinntState.txt", nMOVES, 'nobody', -1, state)
bFinished = False
while bFinished == False:
    # Player A
    result = movePlayer(PROGRAM_PLAYER_A, 'A', state)
    if result != '':
        print(result)
        sys.exit()
    if isDraw(state):
        print("Draw Game")
        sys.exit()
    if hasWon(state) != '':
        print("Player A won")
        sys.exit()

    # Player B
    result = movePlayer(PROGRAM_PLAYER_B, 'B', state)
    if result != '':
        print(result)
        sys.exit()
    if isDraw(state):
        print("Draw Game")
        sys.exit()
    if hasWon(state) != '':
        print("Player B won")
        sys.exit()

    # Something went wrong and we better stop
    if nMOVES == 100:
        bFinished = True

sys.exit()
