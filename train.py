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
import pickle

class TrainPipeline():
    def __init__(self, init_model=None):
        # 设置棋盘和游戏的参数
        self.node1 = node({'cpu':20, 'memory':20, 'gpu':10})
        self.node2 = node({'cpu':20, 'memory':20, 'gpu':0})
        self.node3 = node({'cpu':50, 'memory':50, 'gpu':50})
        #按比例应该是越大越明显
        self.node_dict = {'node1':self.node1, 'node2':self.node2, 'node3':self.node3}
        self.weight = {'cpu':0.3, 'memory':0.2, 'gpu':0.5}
        self.state = State(self.node_dict)
        self.game = Game(self.node_dict, self.weight)
        # 设置训练参数
        self.n_playout = 1000  # 每下一步棋，模拟的步骤数
        self.c_puct = 1 # exploitation和exploration之间的折中系数
        self.game_batch_num = 10
        # AI Player，设置is_selfplay=1 自我对弈，因为是在进行训练
        self.mcts_player = MCTSPlayer(c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        self.f = open(r'.\temp.pkl', 'wb')           

    def run(self):
        # 开始训练
        try:
            result = {}
            result['job_dealed_by_algorithm'] = {}
            result['job_dealed_by_max'] = {}
            result['job_dealed_by_cos'] = {}
            for self.c_puct in [0.01,0.03,0.1,0.3,1,3]:
            # 训练game_batch_num次，每个batch比赛play_batch_size场
                for i in range(self.game_batch_num):
                    # 收集自我对弈数据
                    print("self.c_puct", self.c_puct)
                    print(i + 1)
                    jobs, moves1, moves2, credit1, credit2, credit3, resourse_last1, resourse_last2, resourse_last3  = self.game.start_self_play(self.mcts_player)
                    job_dealed_by_algorithm = len(moves1)
                    job_dealed_by_max = len(moves2)
                    job_dealed_by_cos = len(moves2)
                    #a_win_t_by_job = job_dealed_by_algorithm - job_dealed_by_trandition
                    #a_win_t_by_credit = credit1 - credit2
                    #print('i: ', i)
                    result['job_dealed_by_algorithm'][(self.c_puct, i)] = job_dealed_by_algorithm
                    result['job_dealed_by_max'][(self.c_puct, i)] = job_dealed_by_max
                    result['job_dealed_by_cos'][(self.c_puct, i)] = job_dealed_by_cos
                    print('job_dealed_by_algorithm: ', job_dealed_by_algorithm)
                    print('job_dealed_by_max: ', job_dealed_by_max)
                    print('job_dealed_by_cos: ', job_dealed_by_cos)
                    '''
                    print('resourse_last1: ', resourse_last1)
                    print('resourse_last2: ', resourse_last2)
                    print('a_win_t_by_job: ', a_win_t_by_job)
                    print('a_win_t_by_credit: ', a_win_t_by_credit)
                    '''
            pickle.dump(result,self.f)
            self.f.close()
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
