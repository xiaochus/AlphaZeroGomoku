# -*- coding: utf-8 -*-
"""
Player of Gomoku.
"""
from abc import ABCMeta, abstractmethod

from .. import config as c
from .policy_mcts import MCTS as PolicyMCTS
from .model import PolicyValueNet

import numpy as np
from keras.utils.vis_utils import plot_model


class Player(metaclass=ABCMeta):
    """
    Abstract class for game player.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, states, probs=False):
        pass


class HumanPlayer(Player):
    """HumanPlayer.
    """

    def __init__(self, grid):
        self.id = 'human'
        self.grid = grid

    def get_action(self, event):
        """Get the next move.

        # Arguments
            event: click event.
        # Returns
            pos: piece position.
        """
        pos = event.pos
        pos = (int(round(pos[1] / (self.grid + .0))) - 1,
               int(round(pos[0] / (self.grid + .0))) - 1)

        return pos


class AlphaZeroPlayer(Player):
    """
    AlphaZeroPlayer consisting of PolicyValue net and MCTS.
    """

    def __init__(self, selfplay=0, init=0):
        """Init.

        # Arguments
            selfplay: Boolean, if self play.
            init: Boolean, if load the model.
        """
        self.id = 'ai'
        self.selfplay = selfplay
        self.model = PolicyValueNet(c.DIM, c.K,
                                    c.FILTERS, c.KERNELS).get_model()
        if not init:
            self.model.load_weights('alpha\data\pvmodel.h5')
        self.mcts = PolicyMCTS(c.c_puct, c.N_SIMULATE)

        plot_model(self.model, to_file='images/PolicyValueNet.png', show_shapes=True)

    def _get_value_policy(self, states):
        """Get the value output and policy output.

        # Arguments
            states: states for network input.
        # Returns
            value: Integer, value output.
            policy: ndarray, policy output.
        """
        states = np.expand_dims(states, axis=0)
        predicted = self.model.predict(states)
        value, policy = predicted

        return value[0], policy[0]

    def get_action(self, board, return_prob=0):
        """Get the next move and total move_probs.

        # Arguments
            board: check board.
            return_prob, if return the probs.
        # Returns
            move: Integer, piece position.
            move_probs: policy
        """
        value, policy = self._get_value_policy(board.get_current_states())
        availables = board.get_availables()

        act_probs = zip(availables, policy[availables])

        acts, probs = self.mcts.get_move_probs(board, act_probs, value)

        move_probs = np.zeros(len(policy))
        move_probs[list(acts)] = probs

        if self.selfplay:
            # add Dirichlet Noise for exploration (for self-play training)
            p = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            move = np.random.choice(acts, p=p)
            self.mcts.update_with_move(move)
        else:
            # choosing the move with the highest prob
            move = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move

    def update(self, states, values, probs):
        """Updata the policy value network with pre states.

        # Arguments
            states: ndarray, states for network input.
            value: ndarray, value output.
            policy: ndarray, policy output.

        # Returns
            loss: Double, train loss per self-play update.
            val_loss: Double, val loss per self-play update.
        """
        loss = []

        h = self.model.fit(
            states, {'value_output': values, 'policy_output': probs},
            verbose=0, batch_size=c.BATCH, epochs=c.TRAIN_EPOCHS)

        df = h.history
        loss.append(df['loss'][-1])
        loss.append(df['value_output_loss'][-1])
        loss.append(df['policy_output_loss'][-1])

        return loss

    def save_model(self):
        """Save the current policyvalue network model.
        """
        self.model.save_weights('alpha\data\pvmodel.h5')
