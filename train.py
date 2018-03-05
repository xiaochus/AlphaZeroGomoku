# -*- coding: utf-8 -*-
"""
Reinforcement Learning the PolicyValue Network.
"""
import pandas as pd
import alpha.config as c
from alpha.game.game import Game
from alpha.model.player import AlphaZeroPlayer


def train():
    """
    Train the model with self-play.
    """
    player = AlphaZeroPlayer(selfplay=1, init=1)
    game = Game(c.SIZE, c.PIECE, 1)

    record = {"loss": [], "val_loss": []}
    for i in range(c.SELF_PLAY_EPOCHS):
        states, move_probs, values = game.self_play(player)
        print("Self-play turn {0}".format(i + 1))

        loss, val_loss = player.update(states, values, move_probs)
        print("Network update >> loss:{0} val_loss:{1}".format(loss, val_loss))

        record["loss"].append(loss)
        record["val_loss"].append(val_loss)

        if i % 10 == 0:
            player.save_model()

    player.save_model()
    df = pd.DataFram.from_dict(record)
    df.to_csv('alpha/data/loss.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    train()
