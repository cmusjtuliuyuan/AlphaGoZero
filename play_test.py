# -*- coding: utf-8 -*-
"""
Input your move in the format: 2,3
The original verison is written by:
@author: Junxiao Song
@github: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/play_human.py
"""

from __future__ import print_function
from game import Board, Game
from MCTS_Pure import MCTSPlayer
from MCTS_AlphaGo_Style import AlphaGoPlayer
from model import PolicyValueNet
import argparse
import torch

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]  # for python3
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move, None

    def __str__(self):
        return "Human {}".format(self.player)


def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--player1', default='AlphaGo', help='player1 tpye')
    parser.add_argument('--player2', default='MCTS', help='player2 tpye')
    parser.add_argument('--self_play', default=0, type=int, help='1 means self play, 0 means not')
    args = parser.parse_args()

    n = 4
    width, height = 6, 6
    AlphaGoNet = PolicyValueNet(width, height)
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        if args.self_play:
            player = AlphaGoPlayer(NN_fn=AlphaGoNet.policy_value_fn)
            game.AlphaGo_self_play(player, is_shown=1)
        else:
            if args.player1 == 'human':
                player1 = Human()
            if args.player1 == 'MCTS':
                player1 = MCTSPlayer()
            if args.player1 == 'AlphaGo':
                AlphaGoNet.policy_value_net.load_state_dict(torch.load('model/current_best.mdl'))
                player1 = AlphaGoPlayer(NN_fn=AlphaGoNet.policy_value_fn, n_iteration=1000)

            if args.player2 == 'human':
                player2 = Human()
            if args.player2 == 'MCTS':
                player2 = MCTSPlayer()
            if args.player2 == 'AlphaGo':
                player2 = AlphaGoPlayer(NN_fn=AlphaGoNet.policy_value_fn)
            # set start_player=0 for human first
            game.start_play(player1, player2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    main()