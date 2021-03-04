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
from numba import jit
class TrainPipeline():
    def __init__(self, init_model=None):
        # 设置棋盘和游戏的参数
        
        
        self.node1 = node({'cpu':20, 'memory':20, 'gpu':0})
        self.node2 = node({'cpu':20, 'memory':20, 'gpu':0})
        self.node3 = node({'cpu':50, 'memory':50, 'gpu':50})
        self.node_dict = {'node1':self.node1, 'node2':self.node2, 'node3':self.node3}
        '''
        self.node1 = node({'cpu':30, 'memory':30, 'gpu':30, 'fpga':0})
        self.node2 = node({'cpu':30, 'memory':30, 'gpu':0, 'fpga':30})
        self.node3 = node({'cpu':50, 'memory':50, 'gpu':50, 'fpga':50})
        self.node4 = node({'cpu':30, 'memory':30, 'gpu':0, 'fpga':0})
        self.node5 = node({'cpu':30, 'memory':30, 'gpu':0, 'fpga':0})
        #按比例应该是越大越明显
        self.node_dict = {'node1':self.node1, 'node2':self.node2, 'node3':self.node3, 'node4':self.node4, 'node5':self.node5}
        '''

        #self.weight = {'cpu':0.3, 'memory':0.2, 'gpu':0.5}
        self.weight = None
        self.state = State(self.node_dict)
        self.game = Game(self.node_dict, self.weight)
        # 设置训练参数
        self.n_playout = 1000  # 每下一步棋，模拟的步骤数
        self.c_puct = 1 # exploitation和exploration之间的折中系数
        self.game_batch_num = 3
        self.n_job_thread = 6 #0
        self.probability_1 = 0 #0
        self.probability_2 = 0.2 #0.2
        #self.path = r'D:\科研\论文\High effient resource scheduling for cloud based on modified MCTS\programing\parameter_check_on_have_fpga.pkl'
        # AI Player，设置is_selfplay=1 自我对弈，因为是在进行训练
        self.mcts_player = MCTSPlayer(c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)          

    @jit
    def run(self):
    # 开始训练
        result = {}
        result['job_dealed_by_algorithm'] = {}
        result['job_dealed_by_max'] = {}
        result['job_dealed_by_cos'] = {}
        epoc = 1
        for self.c_puct in [0.03,0.3,3]:
            for self.n_job_thread in [0,5]:
                self.path = (r'D:\科研\论文\High effient resource scheduling for cloud based on modified MCTS\programing\parameter_check_on_standard') + 'c_puct'+ str(0.03) + 'n_job_thread' +str(0) +'.pkl'
                for self.probability_1 in [0,0.03,0.3]:
                    for self.probability_2 in [0.3,0.6,0.9]:
        #for self.c_puct in [0.01]:
        # 训练game_batch_num次，每个batch比赛play_batch_size场
                        for i in range(self.game_batch_num):
                            # 收集自我对弈数据
                            #print("self.c_puct", self.c_puct)
                            print("epoc:", epoc)
                            print("i+1:", i + 1)
                            self.jobs, moves1, moves2, moves3  = self.game.start_self_play(self.mcts_player, self.n_job_thread, self.probability_1, self.probability_2)
                            job_dealed_by_algorithm = len(moves1)
                            job_dealed_by_cos = len(moves2)
                            job_dealed_by_max = len(moves3)
                            #a_win_t_by_job = job_dealed_by_algorithm - job_dealed_by_trandition
                            #a_win_t_by_credit = credit1 - credit2
                            #print('i: ', i)
                            result['job_dealed_by_algorithm'][(self.c_puct, self.n_job_thread, self.probability_1, self.probability_2, i+1)] = job_dealed_by_algorithm
                            result['job_dealed_by_max'][(self.c_puct, self.n_job_thread, self.probability_1, self.probability_2, i+1)] = job_dealed_by_max
                            result['job_dealed_by_cos'][(self.c_puct, self.n_job_thread, self.probability_1, self.probability_2, i+1)] = job_dealed_by_cos
                            print('job_dealed_by_algorithm: ', job_dealed_by_algorithm)
                            print('job_dealed_by_max: ', job_dealed_by_max)
                            print('job_dealed_by_cos: ', job_dealed_by_cos)
                            epoc += 1

        #print("result:",result)
        result_f = open(self.path, 'wb') 
        pickle.dump(result,result_f)
        result_f.close()


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
