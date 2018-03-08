# -*- coding: utf-8 -*-
"""
Reinforcement Learning the PolicyValue Network.
"""
import warnings
import pandas as pd
import alpha.config as c
from alpha.game.game import Game
from alpha.model.player import AlphaZeroPlayer


def train():
    """
    Train the model with self-play.
    """
    warnings.filterwarnings("ignore")

    player = AlphaZeroPlayer(selfplay=1, init=1)
    game = Game(c.SIZE, c.PIECE, 1)

    record = {"loss": [], "value_output_loss": [], "policy_output_loss": []}
    for i in range(c.SELF_PLAY_EPOCHS):
        states, move_probs, values = game.self_play(player)
        print("Self-play turn {0}".format(i + 1))

        loss = player.update(states, values, move_probs, 1)
        print("Network update >> loss:{0}, value_loss:{1}, policy_loss:{2}".format(loss[0], loss[1], loss[2]))

        record["loss"].append(loss[0])
        record["value_output_loss"].append(loss[1])
        record["policy_output_loss"].append(loss[2])

        if i % 50 == 0:
            player.save_model()

    player.save_model()
    df = pd.DataFrame.from_dict(record)
    df.to_csv('alpha/data/loss.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    train()
