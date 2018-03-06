# -*- coding: utf-8 -*-
"""
Config for Gomoku Alpha Zero.
"""

# params of the board and the game
SIZE = (9, 9)
PIECE = 5

# params for policy value network

K = 2
STEP = 3
DIM = (2 * STEP + 1, SIZE[0], SIZE[1])
KERNELS = (3, 3)
FILTERS = 32

# train params for policy value network during self-play

TRAIN_EPOCHS = 50
BATCH = 20
SELF_PLAY_EPOCHS = 500

# params for MCTS
c_puct = 5
N_SIMULATE = 500


# param for game AI
FIRST = 0