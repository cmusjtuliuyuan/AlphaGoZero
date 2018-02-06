# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS) in AlphaGo style
The original verion is written by:
@author: Junxiao Song
@github: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/MCTS_Pure.py

It is modified to Upper Confidence Bounds for Trees (UCT) http://mcts.ai/index.html version by:
@author: Yuan Liu
@github: https://github.com/cmusjtuliuyuan/AlphaGoZero/blob/master/MCTS_Pure.py
"""
import numpy as np
import copy
import random
import math

def fake_NN(board, node):
    # Get v
    board = copy.deepcopy(board)
    while len(board.get_moves()) != 0: # while state is non-terminal
        board.do_move(random.choice(board.get_moves()))
    v = board.get_result(node._player_just_moved)
    # Get p
    p = np.ones(board.width * board.height)
    return p, v

def fake_expand(board, node):
    p, v = fake_NN(board, node)
    move_list = get_untried_moves(board, node)
    move_priorP_pairs = []
    for move in move_list:
        move_priorP_pairs.append((move, p[move]))
    return move_priorP_pairs, v

def get_untried_moves(board, node):
    return  set(board.get_moves())-node.get_already_moved()


class TreeNode(object):
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        We need player_just_moved because n_wins depends on it.
    """

    def __init__(self, parent, move, prior_p, player_just_moved):
        self._parent = parent
        self._player_just_moved = player_just_moved
        self._move = move # the move that got us to this node - "None" for the root node
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0.0 # N(self._parent.board, self._move)
        self._Q = 0.0 # Q(self._parent.board, self._move)
        self._P = prior_p # NN_P(self._parent.board)[self._move]

    def expand(self, move_priorP_pairs, next_player):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        for move, prior_p in move_priorP_pairs:
            new_node = TreeNode(self, move, prior_p, next_player)
            self._children[move] = new_node

    def UCT_select(self, c_puct):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.Q+c.U to vary the amount of exploration versus exploitation.
        """
        return max(self._children.iteritems(), key=lambda act_node: act_node[1].get_UCT_value(c_puct))

    def get_UCT_value(self, c_puct):
        U = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + U

    def update(self, leaf_value):
        """ Update this node - one additional visit and result additional wins.
            result must be from the viewpoint of playerJustmoved.
        """
        self._Q = (self._n_visits * self._Q + leaf_value) / (self._n_visits + 1.0)
        self._n_visits += 1.0

    def get_already_moved(self):
        return set(self._children.keys())

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return len(self._children) == 0

    def __repr__(self):
        return "[M:" + str(self._move) +\
               " Q:" + str(self._Q) +\
               " M:" + str(self.get_already_moved()) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c_move, c_node in self._children.iteritems():
             s += c_node.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

def UCT(root_board, n_iteration, temp=1.0, c_puct=5):
    """ Conduct a UCT search for n_iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = TreeNode(parent=None, move=None, prior_p=1.0,
                        player_just_moved=root_board.get_player_just_moved())

    for i in range(n_iteration):
        node = rootnode
        board = copy.deepcopy(root_board)

        # Selection: Starting at root node R, recursively select optimal child nodes (explained below)
        # until a leaf node L is reached.
        while not node.is_leaf(): #node is fully expanded and non-terminal
            move, node = node.UCT_select(c_puct)
            board.do_move(move)

        # Expansion: If L is a not a terminal node (i.e. it does not end the game) then create one
        # or more child nodes and select one C.
        move_priorP_pairs, leaf_value = fake_expand(board, node)
        end, winner = board.game_end()
        if not end:
            node.expand(move_priorP_pairs, next_player=board.get_current_player())
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == node._player_just_moved else -1.0

        # Backpropagate: Update the current move sequence with the simulation result.
        while node != None: # backpropagate from the expanded node and work back to the root node
            # state is terminal. Update node with result from POV of node._player_just_moved
            node.update(leaf_value)
            leaf_value = - leaf_value
            node = node._parent
    # Output some information about the tree - can be omitted
    # print rootnode.TreeToString(0)
    # return the move and prob pairs
    def softmax(x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs
    move_visits = [(move, node._n_visits) for move, node in rootnode._children.iteritems()]
    moves, visits = zip(*move_visits)
    move_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
    return moves, move_probs


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, n_iteration=400):
        self._n_iteration=n_iteration

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board, temp=1e-3, dirichlet_weight=.0):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            moves, move_probs = UCT(board, self._n_iteration, temp)
            move = np.random.choice(moves, p=(1-dirichlet_weight)*move_probs \
                + dirichlet_weight*np.random.dirichlet(0.3*np.ones(len(move_probs))))
            print 'output position:', move
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
