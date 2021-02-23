# 训练五子棋AI

from __future__ import print_function
import random
import numpy as np
# deque 是一个双端队列
from collections import defaultdict, deque
from Game import Game
from mcts import MCTSPlayer # AlphaGo方式的AI
from node import state as State
from node import node

class TrainPipeline():
    def __init__(self, init_model=None):
        # 设置棋盘和游戏的参数
        self.node1 = node({'cpu':150, 'memory':150, 'gpu':20})
        self.node2 = node({'cpu':150, 'memory':150, 'gpu':0})
        self.node3 = node({'cpu':100, 'memory':100, 'gpu':80})
        self.node_dict = {'node1':self.node1, 'node2':self.node2, 'node3':self.node3}
        self.weight = {'cpu':0.3, 'memory':0.2, 'gpu':0.5}
        self.jobs = [{'cpu': 5, 'memory': 2, 'gpu': 1}, {'cpu': 5, 'memory': 2, 'gpu': 0}]
        self.state = State(self.node_dict)
        self.game = Game(self.jobs, self.node_dict, self.weight)
        # 设置训练参数
        self.learn_rate = 2e-3 # 基准学习率
        self.lr_multiplier = 1.0  # 基于KL自动调整学习倍速
        self.temp = 1.0  # 温度参数
        self.n_playout = 2000  # 每下一步棋，模拟的步骤数
        self.c_puct = 5 # exploitation和exploration之间的折中系数
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size) #使用 deque 创建一个双端队列
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02 # 早停检查
        self.check_freq = 50 # 每50次检查一次，策略价值网络是否更新
        self.game_batch_num = 500 # 训练多少个epoch
        self.best_win_ratio = 0.0 # 当前最佳胜率，用他来判断是否有更好的模型
        # 弱AI（纯MCTS）模拟步数，用于给训练的策略AI提供对手
        self.pure_mcts_playout_num = 1000
        # AI Player，设置is_selfplay=1 自我对弈，因为是在进行训练
        self.mcts_player = MCTSPlayer(c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        

    # 收集自我对弈数据，用于训练
    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            # 与MCTS Player进行对弈
            jobs, moves1, moves2, credit1, credit2, resourse_last1, resourse_last2  = self.game.start_self_play(self.mcts_player, temp=self.temp)
            job_dealed_by_algorithm = len(moves1)
            job_dealed_by_trandition = len(moves2)
            a_win_t_by_job = job_dealed_by_algorithm - job_dealed_by_trandition
            a_win_t_by_credit = credit1 - credit2
            print('i: ', i)
            print('job_dealed_by_algorithm: ', job_dealed_by_algorithm)
            print('job_dealed_by_trandition: ', job_dealed_by_trandition)
            print('resourse_last1: ', resourse_last1)
            print('resourse_last2: ', resourse_last2)
            print('a_win_t_by_job: ', a_win_t_by_job)
            print('a_win_t_by_credit: ', a_win_t_by_credit)
            # 增加数据 play_data
            self.data_buffer.extend(moves1)
            

    def run(self):
        # 开始训练
        try:
            # 训练game_batch_num次，每个batch比赛play_batch_size场
            for i in range(self.game_batch_num):
                # 收集自我对弈数据
                self.collect_selfplay_data(self.play_batch_size)
                #print(self.data_buffer)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
