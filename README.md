# AlphaZeroGomoku
Pygame and Keras implementation of Gomoku game with simple AlphaZero.


## Requirement
- Python 3.6    
- Tensorflow-gpu 1.2.0  
- Keras 2.1.3
- pygame 1.9.3

## Train the model

**Run command below to train the model:**

```
python train.py
```

The `alpha/config.py` file is used to config the parameters of PolicyValue network, MCTS, game rules and train process.

## Run the game

**Run command below to run the game:**

```
python gomoku.py
```

## Policy Value Network used

![PolicyValueNet](/images/PolicyValueNet.png)

## Experiment

TODO

## Reference

	@article{AlphaZero,  
	  title=Mastering the Game of Go without Human Knowledge},  
	  author={David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel & Demis Hassabis},
	  journal={Nature,550 (7676):354-359},
	  year={2017}
	}

	@article{AlphaZero,  
	  title={Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm},  
	  author={David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis},
	  journal={arXiv preprint arXiv:1712.01815v1)
	  year={2017}
	}


## Copyright
See [LICENSE](LICENSE) for details.
