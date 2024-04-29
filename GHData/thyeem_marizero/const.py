class Stone:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

class Player:
    HUMAN = 1
    SOFIAI = 2
    MARIAI = 3
    MARIZERO = 4

#-----------------------------------
BCF = True
API = True 
N = 19
WP = 10
GOAL = 5
INF = 1.0e7
D_MINIMAX = 2
BRD_DATA = './__data__'
API_SOFIA = './bin/sofia'
API_MARIA = './bin/maria'

CI = 10
H1 = 32
H2 = 64
H4 = 128
LEARNING_RATE = 1e-3
L2_CONST = 1e-4
GAMMA = 0.99
N_EPISODE = 1
N_STEPS = 10
SIZE_BATCH = 1024
SIZE_DATA = 50000
RATIO_OVERTURN = 0.55
C_PUCT = 1.0
N_SEARCH = 400
