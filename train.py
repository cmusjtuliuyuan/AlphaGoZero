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

def get_data_once(game, player, ReplayMemory):
    winner, play_data = game.AlphaGo_self_play(player, is_shown=0)
    play_data = get_equi_data(play_data)
    ReplayMemory.extend(play_data)

def burn_in(game, player, ReplayMemory, num_episode):
    process_bar = ShowProcess(num_episode, 'Burning in: ')
    for _ in range(num_episode):
        process_bar.show_process()
        get_data_once(game, player, ReplayMemory)
    process_bar.close()

def train_one_iteration(game,
                        player,
                        ReplayMemory,
                        AlphaGoNet_train,
                        batch_size,
                        learning_rate,
                        train_freq,
                        evaluate_freq):
    process_bar = ShowProcess(evaluate_freq, 'Train one iteration: ')
    for j in range(evaluate_freq):
        process_bar.show_process()
        get_data_once(game, player, ReplayMemory)
        # Train:
        if len(ReplayMemory)>batch_size and j%train_freq==0:
            update_once(ReplayMemory, AlphaGoNet_train, batch_size, learning_rate)
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
        process_bar.show_process()
        winner = game.start_play(player1, player2, start_player=i%2, is_shown=0)
        win_cnt[winner] += 1
    process_bar.close()
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1])/n_game
    print("win: {}, lose: {}, tie:{}, win_ratio:{}".format(win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio))
    return win_ratio

def update_once(ReplayMemory, AlphaGoNet, batch_size, lr):
    mini_batch = random.sample(ReplayMemory, batch_size)
    state_batch = np.array([data[0] for data in mini_batch])
    mcts_probs_batch = np.array([data[1] for data in mini_batch])
    winner_batch = np.array([data[2] for data in mini_batch])
    AlphaGoNet.train_batch(state_batch, mcts_probs_batch, winner_batch, lr)

def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--replay_memory_size', default=50000,
                type=int, help='replayMemory_size to store training data')
    parser.add_argument('--batch_size', default=512,
                type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-3,
                type=float, help='learning_rate')
    parser.add_argument('--evaluate_freq', default=50,
                type=int, help='evaluate once every #evaluate_freq games')
    parser.add_argument('--train_freq', default=1,
                type=int, help='train #train_epoch times replay mempry within each train')
    parser.add_argument('--n_eval_game', default=10,
                type=int, help='number of games during one evaluation')
    parser.add_argument('--n_burn_in', default=10,
                type=int, help='number of games to burn in the replay memory')
    parser.add_argument('--n_iteration', default=20,
                type=int, help='number of train iteration')
    parser.add_argument('--width', default=6, type=int)
    parser.add_argument('--height', default=6, type=int)
    parser.add_argument('--n_in_row', default=4, type=int)
    args = parser.parse_args()


    width, height = args.width, args.height
    board = Board(width=width, height=height, n_in_row=args.n_in_row)
    game = Game(board)
    # Prepare train and eval model
    AlphaGoNet_train = PolicyValueNet(width, height)
    #AlphaGoNet_best = PolicyValueNet(width, height)
    #torch.save(AlphaGoNet_train.policy_value_net.state_dict(), 'model/init.mdl')
    AlphaGoNet_train.policy_value_net.load_state_dict(torch.load('model/current.mdl'))

    # Replay is used to store training data:
    ReplayMemory = deque(maxlen=args.replay_memory_size)
    player = AlphaGoPlayer(NN_fn=AlphaGoNet_train.policy_value_fn)
    #eval_player = AlphaGoPlayer(NN_fn=AlphaGoNet_best.policy_value_fn)
    eval_player = MCTSPlayer()
    max_win_ratio = .0

    # Burn in
    burn_in(game, player, ReplayMemory, args.n_burn_in)

    for i in range(args.n_iteration):
        print 'Iteration NO.:', i
        train_one_iteration(game, player, ReplayMemory, AlphaGoNet_train,
                args.batch_size, args.learning_rate, args.train_freq, args.evaluate_freq)
        win_ratio = evaluate(game, player, eval_player, args.n_eval_game)

        if win_ratio > max_win_ratio:
            print('Get current_best model!')
            max_win_ratio = win_ratio
            torch.save(AlphaGoNet_train.policy_value_net.state_dict(), 'model/current_best.mdl')
        else:
            print('Save current model')
            torch.save(AlphaGoNet_train.policy_value_net.state_dict(), 'model/current.mdl')


main()