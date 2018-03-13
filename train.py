# -*- coding: utf-8 -*-
"""
Reinforcement Learning the PolicyValue Network.
"""
import warnings
import numpy as np
import pandas as pd
import alpha.config as c
from alpha.game.game import Game
from alpha.model.player import AlphaZeroPlayer


def augment_data(states, values, probs):
    """Augment the train data.

    # Arguments
        states: ndarray, states for network input.
        values: ndarray, value output.
        policy: ndarray, policy output.

    # Returns
        states: ndarray, augmented states for network input.
        values: ndarray, augmented value output.
        policy: ndarray, augmented policy output.
    """
    extends = {0: [], 1: [], 2: []}
    for state, prob, value in zip(states, probs, values):
        # rotate counterclockwise
        for i in [1, 2, 3, 4]:
            e_state = np.array([np.rot90(s, i) for s in state])
            e_prob = np.rot90(prob.reshape(c.SIZE[0], c.SIZE[0]), i)
            extends[0].append(e_state)
            extends[1].append(value)
            extends[2].append(e_prob.flatten())

        # flip horizontally
        for flip in [np.fliplr, np.flipud]:
            f_state = np.array([flip(s) for s in state])
            f_prob = flip(prob.reshape(c.SIZE[0], c.SIZE[0]))

            extends[0].append(f_state)
            extends[1].append(value)
            extends[2].append(f_prob.flatten())

    return np.array(extends[0]), np.array(extends[1]), np.array(extends[2])


def train(augment=0):
    """
    Train the model with self-play.

    # Arguments
        augment: Boolean, use augment or not.
    """
    warnings.filterwarnings("ignore")

    player = AlphaZeroPlayer(selfplay=1, init=c.INIT)
    game = Game(c.SIZE, c.PIECE, 1)

    record = {"loss": [], "value_output_loss": [], "policy_output_loss": []}
    for i in range(c.SELF_PLAY_EPOCHS):
        states, move_probs, values = game.self_play(player)

        if augment:
            states, values, move_probs = augment_data(states, values, move_probs)

        print("Self-play turn {0}".format(i + 1))

        loss = player.update(states, values, move_probs)
        print("Network update >> loss:{0}, value_loss:{1}, policy_loss:{2}".format(loss[0], loss[1], loss[2]))

        record["loss"].append(loss[0])
        record["value_output_loss"].append(loss[1])
        record["policy_output_loss"].append(loss[2])

        if i % 20 == 0:
            player.save_model()

    player.save_model()
    df = pd.DataFrame.from_dict(record)
    df.to_csv('alpha/data/loss.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    train(1)
    """
    from os import system
    import time

    time.sleep(60)
    system("shutdown -s -t 0")
    """
