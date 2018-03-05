# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a output policy of
policy-value network to guide the tree search and evaluate the leaf nodes.
"""
import copy
import numpy as np


class TreeNode(object):
    """
    A node in the MCTS tree.
    """

    def __init__(self, parent, prior_p):
        """Init.

        # Arguments
            parent: parent node.
            prior_p, probs of node.
        """
        self.parent = parent  # parent node
        self.children = {}  # child node
        self.visited = 0    # visit times
        self.v = 0  # own value
        self.p = prior_p  # prior policy from pvnet
        self.u = 0  # visit-count-adjusted prior score

    def expand(self, action_priors):
        """Expand tree by creating new child node.

        # Arguments
            action_priors: move action and corresponding prob.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c):
        """Select action among children that gives maximum action value.

        # Arguments
            c: Integer, a number controlling the relative impact of
                values, v, and prior probability p on this node's score.
        # Returns
            r: tuple, (action, next_node)
        """
        r = max(self.children.items(), key=lambda node: node[1].get_value(c))

        return r

    def update(self, leaf_value):
        """Update node values from leaf evaluation.

        # Arguments
            leaf_value: the value of subtree evaluation from the
                current player's perspective.
        """
        self.visited += 1
        # Update v, a running average of values for all visits.
        self.v += 1.0 * (leaf_value - self.v) / self.visited

    def update_recursive(self, leaf_value):
        """Applied recursively for all ancestors.

        # Arguments
            leaf_value: the value of subtree evaluation from the
                current player's perspective.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c):
        """Calculate and return the value for this node.

        # Arguments
        c: Integer, a number controlling the relative impact of
            values, v, and prior probability p on this node's score.

        # Returns
            a combination of leaf evaluations v and this node's prior
            adjusted for its visit count u.
        """
        self.u = c * self.p * np.sqrt(self.parent.visited) / (1 + self.visited)

        return self.v + self.u

    def is_leaf(self):
        """Check if leaf node.
        """
        return self.children == {}

    def is_root(self):
        """Check if root node.
        """
        return self.parent is None


class MCTS(object):
    """
    A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, c_put, n_simulate):
        """Init.

        # Arguments
        c_put: Integer, a number controlling the relative impact of
            values, v, and prior probability p on this node's score.
        n_simulate: Integer, simulate times.
        """
        self.root = TreeNode(None, 1.0)
        self.c_put = c_put
        self.n_simulate = n_simulate

    def _simulate(self, board, policy, value):
        """Simluation.

        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents. State is modified
        in-place, so a copy must be provided.

        # Arguments
        board: Board, a copy of current check board.
        policy: tuple, (action, prob) from policy value network.
        value: Double， value from policy value network.
        """
        node = self.root

        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self.c_put)
            board.move(action)
            board.change_player()

        # Check for end of game.
        win, winner = board.get_game_status()

        if win == -1:
            node.expand(policy)
        else:
            # for end state，return the "true" leaf_value.
            if win == 0:
                value = 0.0
            else:
                value = 1.0 if winner == board.get_current_player() else - 1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-value)

    def get_move_probs(self, board, policy, value):
        """Get all move probs
        Runs all simluation sequentially and returns the available actions
        and their corresponding probabilities.

        # Arguments
            board: Board, current check board.
            policy: tuple, (action, prob) from policy value network.
            value: Double， value from policy value network.
        """
        temp = 1e-3

        for n in range(self.n_simulate):
            board_copy = copy.deepcopy(board)
            self._simulate(board_copy, policy, value)

        """
        calc the move probabilities based on the visit counts at
        the root node
        """
        act_visits = [(a, n.visited) for a, n in self.root.children.items()]

        acts, visits = zip(*act_visits)
        act_probs = self.softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree.
        keeping everything we already know about the subtree.

        last_move: Integer, last action move.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def softmax(self, x):
        """Softmax

        # Arguments
            x: ndarray, input x.
        """
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)

        return probs
