# -*- coding: utf-8 -*-
"""
Game process of Gomoku.
"""
import numpy as np
from .board import Board


class Game(object):
    """
    Play Game
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
        self.start_player = start_player
        self._restart_game()

    def _restart_game(self):
        """Reset the game status.
        """
        self.board = Board(self.size, self.piece, self.start_player)

    def play(self, players, click):
        """Start the match in players.

        # Arguments
            players: List, Players.
            click: click event.

        # Returns
            win, winner, movements: win or draw, winner of game, movements.
        """
        current_player = self.board.get_current_player()
        player_in_turn = players[current_player - 1]

        if player_in_turn.id == 'ai':
            move = player_in_turn.get_action(self.board)
        else:
            move = player_in_turn.get_action(click)

        flag = self.board.move(move)
        if not flag:
            return -1, current_player

        win, winner = self.board.get_game_status()

        movements = self.board.get_all_movements()
        self.board.change_player()

        return win, winner, movements

    def self_play(self, player):
        """Start the match between player1 and player2.

        # Arguments
            player: Player, player built with pv network.

        # Returns
            states: List, Input states for training.
            move_probs: List, output policy for training.
            values: List, output value for training.
        """
        self._restart_game()
        winner, win = 0, 0
        states, move_probs, cor_players = [], [], []

        while True:
            states.append(self.board.get_current_states())

            move, probs = player.get_action(self.board, 1)
            flag = self.board.move(move)

            if not flag:
                return -1

            move_probs.append(probs)
            current_player = self.board.get_current_player()
            cor_players.append(current_player)

            win, winner = self.board.get_game_status()

            if win in [0, 1]:
                break

            self.board.change_player()

        if win == 1:
            values = np.array([1 if p == winner else -1 for p in cor_players])
        else:
            values = np.zeros(len(move_probs))

        return np.array(states), np.array(move_probs), values
