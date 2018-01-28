# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
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

class TreeNode(object):
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        We need player_just_moved because n_wins depends on it.
    """

    def __init__(self, parent, move, board):
        self._parent = parent
        self._player_just_moved = board.current_player
        self._move = move # the move that got us to this node - "None" for the root node
        # We use copy here because it dictionary is passed by reference
        self._untried_moves = copy.copy(board.get_moves())
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0.0
        self._n_wins = 0.0

    def expand(self, move, new_board):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        self._untried_moves.remove(move)
        new_node = TreeNode(self, move, new_board)
        self._children[move] = new_node
        return new_node

    def UCT_select(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(c.parent.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        return max(self._children.iteritems(), key=lambda act_node: act_node[1].get_UCT_value())

    def get_UCT_value(self):
        value = self._n_wins/self._n_visits + math.sqrt(2*math.log(self._parent._n_visits)/self._n_visits)
        return value

    def update(self, result):
        """ Update this node - one additional visit and result additional wins.
            result must be from the viewpoint of playerJustmoved.
        """
        self._n_visits += 1.0
        self._n_wins += result
    
    def get_moves(self):
        """Get avaliable moves
        """
        return self._untried_moves

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return len(self._children) == 0

    def is_all_tried(self):
        """Check whether we have already tired all moves
        """
        return len(self._untried_moves) == 0

    def __repr__(self):
        return "[M:" + str(self._move) +\
               " W/V:" + str(self._n_wins) + "/" + str(self._n_visits) +\
               " U:" + str(self._untried_moves) + "]"

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

    def ChildrenToString(self):
        s = ""
        for c_move, c_node in self._children.iteritems():
             s += str(c_node) + "\n"
        return s

def UCT(root_board, n_iteration):
    """ Conduct a UCT search for n_iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = TreeNode(parent=None, move=None, board=root_board)

    for i in range(n_iteration):
        node = rootnode
        board = copy.deepcopy(root_board)

        # Selection: Starting at root node R, recursively select optimal child nodes (explained below)
        # until a leaf node L is reached.
        while node.is_all_tried() and not node.is_leaf(): #node is fully expanded and non-terminal
            move, node = node.UCT_select()
            board.do_move(move)

        # Expansion: If L is a not a terminal node (i.e. it does not end the game) then create one
        # or more child nodes and select one C.
        if not node.is_all_tried(): # if we can expand
            move = random.choice(node.get_moves())
            board.do_move(move)
            node = node.expand(move, board)

        # Simulation: Run a simulated playout from C until a result is achieved.
        while len(board.get_moves()) != 0: # while state is non-terminal
            board.do_move(random.choice(board.get_moves()))

        # Backpropagate: Update the current move sequence with the simulation result.
        while node != None: # backpropagate from the expanded node and work back to the root node
            # state is terminal. Update node with result from POV of node._player_just_moved
            node.update(board.get_result(node._player_just_moved)) 
            node = node._parent
    # Output some information about the tree - can be omitted
    print rootnode.TreeToString(0)
    # return the move that was most visited
    move, _ = max(rootnode._children.iteritems(), key=lambda act_node: act_node[1]._n_visits)
    return move

class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, n_iteration=400):
        self._n_iteration=n_iteration
    
    def set_player_ind(self, p):
        self.player = p
    
    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = UCT(board, self._n_iteration)
            print 'output position:', move
            return move
        else:            
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)  
