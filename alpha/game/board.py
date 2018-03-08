# -*- coding: utf-8 -*-
"""
A checkerboard of the Gomoku game.
"""
from .. import config as c
import numpy as np


class Board(object):
    """
    CheckerBoard
    """

    def __init__(self, size, piece, start_player):
        """
        # Arguments
            size Integer: height and width of checkerboard.
            piece: Integer, number of piece to win.
            start_player: Integer, which player is the first to start.
        """
        self.size = size
        self.piece = piece
        self.states = []    # save the position of piece
        self.start_player = start_player
        self.current_player = start_player

        if self.start_player not in [1, 2]:
            raise Exception('Start player must be 1 or 2')

        if self.size[0] < self.piece or self.size[1] < self.piece:
            raise Exception('Board size can not less than %d' % self.nums)

    def _convert_position(self, p, t):
        """Convert position of piece.
        Convert position of piece between 2D check board and 1D states.

        # Arguments
            p: Integer/Tuple, position of piece.
            t: String, convert type(m: states->board, 's': board->states)

        # Returns
            m: Integer/Tuple,position of piece.
        """
        if t == 'm':
            row = p // self.size[0]
            col = p % self.size[0]

            return row, col

        if t == 's':
            m = p[0] * self.size[0] + p[1]

            return m

    def get_current_states(self):
        """Get current states on the board.

        # Returns
            states_matrix: ndarray(height * width * (period + 1)),
                states on the board with in period.
        """
        states_matrix = np.zeros((c.STEP * 2 + 1, self.size[0], self.size[1]))

        cur = [self.states[i] for i in range(len(self.states) - 2, -1, -2)]
        opp = [self.states[i] for i in range(len(self.states) - 1, -1, -2)]
        temp = [cur, opp]

        for i in range(2):
            x = temp[i]
            p = len(x) if len(x) < c.STEP else c.STEP

            if i == 0:
                index = list(range(0, c.STEP * 2, 2))
            else:
                index = list(range(1, c.STEP * 2, 2))

            for j in range(p):
                x_step = x[j:]
                for t in x_step:
                    row, col = self._convert_position(t, 'm')
                    states_matrix[index[j]][row][col] = 1

        if self.current_player == self.start_player:
            states_matrix[c.STEP * 2] = np.ones((self.size[0], self.size[1]))

        return states_matrix

    def get_all_movements(self):
        """Get all movements for palyers.

        # Returns
            move: Dict, all movements for palyers.
        """
        move = {"first": [], "second": []}

        for i in range(0, len(self.states)):
            pos = self.states[i]
            row, col = self._convert_position(pos, 'm')
            if i % 2 == 0:
                move["first"].append([row, col])
            else:
                move["second"].append([row, col])

        return move

    def move(self, move):
        """Move piece on the board.
        The current player put the piece on the board.

        # Arguments
            move: Integer/tuple, position of checkerboard.

        # Returns
            Whether to move successfully.
        """
        if isinstance(move, tuple):
            move = self._convert_position(move, 's')

        if move in self.states:
            pos = 0
        else:
            pos = 1
            self.states.append(move)

        return pos

    def change_player(self):
        """Change current player.
        """
        self.current_player = 1 if self.current_player == 2 else 2

    def get_current_player(self):
        """Get current player.

        # Returns
            Integer, current player
        """
        return self.current_player

    def get_availables(self):
        """Get availables position on the board.

        # Returns
            availables: ndarray, availables position
        """
        availables = np.zeros(self.size[0] * self.size[1])

        for i in range(len(availables)):
            if i in self.states:
                availables[i] = 1

        availables = np.argwhere(availables == 0).reshape(1, -1)[0]

        return availables

    def _win(self, cps):
        """Check winner.
        Check if the current player win the Gomoku game.

        # Arguments
            cps: List, position of piece for one player.

        # Returns
            win: Integer, game win or not.(1: win, -1: continue)
        """
        win = -1

        cur_piece = np.array(self._convert_position(cps[0], 'm'))

        direct = np.array([[[-1, 0], [1, 0]],
                           [[0, -1], [0, 1]],
                           [[-1, -1], [1, 1]],
                           [[1, -1], [-1, 1]]])
        for d in direct:
            count = 1
            for x in d:
                flag = True
                temp = cur_piece
                while flag:
                    temp = temp + x
                    s = self._convert_position(temp, 's')
                    if s in cps and 0 <= temp[0] < self.size[0] and 0 <= temp[1] < self.size[0]:
                        count += 1
                    else:
                        flag = False

            if count >= self.piece:
                win = 1
                break

        return win

    def get_game_status(self):
        """Check the game result.

        # Returns
            Competition win or not.(0: draw, 1: win, -1: continue), winner
        """
        if len(self.states) > 8:
            cur = [self.states[i] for i in range(len(self.states) - 1, -1, -2)]
            other = [x for x in self.states if x not in cur]
            other.reverse()

            a = len(self.states) == self.size[0] * self.size[1]
            win_c = self._win(cur)
            win_o = self._win(other)

            if win_c == 1 and win_o != 1:
                return win_c, self.current_player
            elif win_o == 1 and win_c != 1:
                winner = 1 if self.current_player == 2 else 2
                return win_o, winner
            elif win_c == -1 and win_o == -1 and a:
                return 0, 0
            elif win_c == -1 and win_o == -1:
                return -1, 0
            else:
                print("Wrong: {0}, {1}".format(win_c, win_o))
                return -1, -1
        else:
            return -1, 0
