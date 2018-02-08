from collections import deque, defaultdict
from game import Board, Game
from MCTS_Pure import MCTSPlayer
from MCTS_AlphaGo_Style import AlphaGoPlayer
from model import PolicyValueNet
import numpy as np
import random
import argparse
from ShowProcess import ShowProcess
import cPickle as pickle
import torch

def get_data(game, player, ReplayMempory, num_episode):
    process_bar = ShowProcess(num_episode, 'Gathering data: ')
    for i in range(num_episode):
        process_bar.show_process()
        winner, play_data = game.AlphaGo_self_play(player, is_shown=0)
        play_data = get_equi_data(play_data)
        ReplayMempory.extend(play_data)
    process_bar.close()

def get_equi_data(play_data):
    """
    augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]"""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [0,1,2,3]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s,i) for s in state])
            equi_mcts_prob = np.rot90(mcts_prob)
            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
    return extend_data

def evaluate(game, player1, player2, n_game):
    win_cnt = defaultdict(int)
    process_bar = ShowProcess(n_game, 'Evaluating: ')
    for i in range(n_game):
        winner = game.start_play(player1, player2, start_player=i%2, is_shown=0)
        win_cnt[winner] += 1
        process_bar.show_process()
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1])/n_game
    print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[2], win_cnt[-1]))
    process_bar.close()
    return win_ratio

def update(ReplayMempory, AlphaGoNet, batch_size, train_epoch, lr):
    for _ in range(0, len(ReplayMempory)*train_epoch, batch_size):
        mini_batch = random.sample(ReplayMempory, batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])
        AlphaGoNet.train_batch(state_batch, mcts_probs_batch, winner_batch, lr)

def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--replay_memory_size', default=40000,
                type=int, help='replayMemory_size to store training data')
    parser.add_argument('--batch_size', default=32,
                type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4,
                type=float, help='learning_rate')
    parser.add_argument('--evaluate_freq', default=200,
                type=int, help='evaluate once every #evaluate_freq games')
    parser.add_argument('--train_epoch', default=4,
                type=int, help='train #train_epoch times replay mempry within each train')
    parser.add_argument('--n_game', default=10,
                type=int, help='number of games during one evaluation')
    parser.add_argument('--width', default=6, type=int)
    parser.add_argument('--height', default=6, type=int)
    parser.add_argument('--n_in_row', default=4, type=int)
    args = parser.parse_args()


    width, height = args.width, args.height
    board = Board(width=width, height=height, n_in_row=args.n_in_row)
    game = Game(board)
    # Prepare train and eval model
    AlphaGoNet_train = PolicyValueNet(width, height)
    AlphaGoNet_best = PolicyValueNet(width, height)
    torch.save(AlphaGoNet_train.policy_value_net.state_dict(), 'model/init.mdl')
    AlphaGoNet_best.policy_value_net.load_state_dict(torch.load('model/init.mdl'))

    #Replay is used to store training data:
    ReplayMempory = deque(maxlen=args.replay_memory_size)
    player = AlphaGoPlayer(NN_fn=AlphaGoNet_train.policy_value_fn)

    # First evaluate once
    eval_player = AlphaGoPlayer(NN_fn=AlphaGoNet_best.policy_value_fn)
    max_win_ratio = -1.0

    for i in range(20):
        print 'Iteration NO.:', i
        get_data(game, player, ReplayMempory, args.evaluate_freq)
        # Train:
        update(ReplayMempory, AlphaGoNet_train,
            batch_size=32, train_epoch=args.train_epoch, lr=args.learning_rate)
        win_ratio = evaluate(game, player, eval_player, 20)

        if win_ratio > max_win_ratio:
            print 'Get Current Best!!'
            max_win_ratio = win_ratio
            torch.save(AlphaGoNet_train.policy_value_net.state_dict(), 'model/current_best.mdl')
            AlphaGoNet_best.policy_value_net.load_state_dict(torch.load('model/current_best.mdl'))
        else:
            AlphaGoNet_train.policy_value_net.load_state_dict(torch.load('model/current_best.mdl'))


main()