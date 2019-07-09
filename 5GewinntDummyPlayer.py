import random
import sys
import time

if len(sys.argv) < 3:
    print("5GewinntDummyPlayer.py STATEFILE MOVEFILE")
    sys.exit()

move = str(random.randint(1, 12))
if move == '12':
    move = 'flip'
    #move = '4'

f = open(sys.argv[2], "w")
f.write(move)
f.close()

time.sleep(20)
