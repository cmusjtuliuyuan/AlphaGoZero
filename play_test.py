# -*- coding: utf-8 -*-
"""
Input your move in the format: 2,3
The original verison is written by:
@author: Junxiao Song
@github: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/play_human.py
"""

from __future__ import print_function
from game import Board, Game
#from MCTS_Pure import MCTSPlayer
from MCTS_AlphaGo_Style import MCTSPlayer
import argparse

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
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--player1', default='MCTS', help='player1 tpye')
    parser.add_argument('--player2', default='human', help='player2 tpye')
    args = parser.parse_args()

    n = 3
    width, height = 5, 5
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        player1 = Human() if args.player1=='human' else MCTSPlayer()
        player2 = Human() if args.player2=='human' else MCTSPlayer()

        # set start_player=0 for human first
        game.start_play(player1, player2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    main()