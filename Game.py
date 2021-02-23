# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from node import state as State
import random
from select_the_max import node_select_value, node_select_value_cos
#找到jobs的位置 23行
#应该做两套，一套是自我对弈，job是随机的，另一套是人机对弈，jobs是真实的

class Game(object):
    def __init__(self, jobs, node_dict, weight):
        self.jobs = jobs
        self.node_dict = node_dict
        self.weight = weight

    '''
    def random_job(self, state):
        state_resource_names = state.get_state_resource_name()
        job = {}
        for resource_name in state_resource_names:
            job[resource_name] = random.randint(0,10)
        return job
    '''

    def random_job(self, state):
        state_resource_names = state.get_state_resource_name()
        job = {}
        for resource_name in state_resource_names:
            if resource_name != 'gpu':
                job[resource_name] = random.randint(10,15)
            else:
                job[resource_name] = random.randint(0,1)
        if np.random.random() < 0.3:
            job['gpu'] += random.randint(5,8)
        return job 

    '''
    # AI自我对弈，存储自我对弈数据 用于训练 self-play data: (state, mcts_probs, z)
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        # 初始化棋盘
        self.state = State(self.node_dict)
        # 记录该局对应的数据：states, mcts_probs, current_players
        moves, jobs, states, mcts_probs = [], [], [], []
        n = 1
        # 一直循环到比赛结束
        while True:
            # 得到player的下棋位置
            # 加入job
            jobs.append(self.random_job(self.state))
            self.state.job(jobs[-1])
            end = self.state.game_end()
            if end:
                # 记录该局对弈中的每步分值，胜1，负-1，平局0
                # 重置MCTS根节点 reset MCTS root node
                resourse_last = self.state.get_state_resource_now()
                player.reset_player()
                credit = self.get_algorithm_credit(self.state)#返回该局得分
                # 返回获胜方，self-play数据: (state, mcts_probs, z)
                return moves, jobs, credit
            move, move_probs = player.get_action(self.state)
            # 存储数据
            states.append(self.state.get_state_resource_now()) #棋盘状态
            mcts_probs.append(move_probs)
            moves.append(move)
            # 按照move来下棋
            self.state.scheduling(move)
            if is_shown:
                pass
            # 判断游戏是否结束end，统计获胜方 winner
            print('第%d次实验:'%n)
            print('job: ', jobs[-1])
            print('move: ',move)
            n += 1
    '''
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        # 初始化棋盘
        self.state1 = State(self.node_dict, self.weight)
        self.state2 = State(self.node_dict, self.weight)
        # 记录该局对应的数据：states, mcts_probs, current_players
        jobs = []
        moves1, states1, mcts_probs1 = [], [], []
        moves2, states2, mcts_probs2 = [], [], []
        n1 = 1
        n2 = 1
        # 一直循环到比赛结束
        while True:
            # 得到player的下棋位置
            # 加入job
            jobs.append(self.random_job(self.state1))
            self.state1.job(jobs[-1])
            self.state2.job(jobs[-1])
            end1 = self.state1.game_end()
            end2 = self.state2.game_end()
            if end1 and end2:
                resourse_last1 = self.state1.now()
                credit1 = self.get_algorithm_credit(self.state1)#返回该局得分
                player.reset_player()

                resourse_last2 = self.state2.now()
                credit2 = self.get_algorithm_credit(self.state2)#返回该局得分
                #print('credit1: ',credit1)
                #print('credit2: ',credit2)

                return jobs, moves1, moves2, credit1, credit2, resourse_last1, resourse_last2
            if not end1:
                #move1, move_probs1 = player.get_action(self.state1)
                move1, move_probs1 = node_select_value(self.state1)
                # 存储数据                states1.append(self.state1.get_state_resource_now()) #棋盘状态
                mcts_probs1.append(move_probs1)
                moves1.append(move1)
                # 按照move来下棋
                self.state1.scheduling(move1)
                if is_shown:
                    pass
                # 判断游戏是否结束end，统计获胜方 winner
                #print('第%d次实验:'%n1)
                #print('job: ', jobs[-1])
                #print('move1: ',move1)
                n1 += 1
            if not end2:
                move2, move_probs2 = node_select_value_cos(self.state2)
                states2.append(self.state2.get_state_resource_now()) #棋盘状态
                mcts_probs2.append(move_probs2)
                moves2.append(move2)
                # 按照move来下棋
                self.state2.scheduling(move2)
                if is_shown:
                    pass
                # 判断游戏是否结束end，统计获胜方 winner
                #print('第%d次实验:'%n2)
                #print('job: ', jobs[-1])
                #print('move2: ',move2)
                n2 += 1
                '''
                if (move1 != move2) or (n1 != n2):
                    #print('第%d次任务:'%n2)
                    print('n1: ',n1-1, 'n2', n2-1)
                    print('move1: ',move1, 'move2', move2)
                '''    
                

    def get_algorithm_credit(self, state):
        state_resource_all = state.all()
        state_all_resource_name = state_resource_all.keys()
        state_weight = []
        state_now = []
        state_all = []
        for name in state_all_resource_name:
            state_weight.append(state.weight()[name])
            state_now.append(state.now()[name])
            state_all.append(state.all()[name])
        state_weight = np.array(state_weight)
        state_now = np.array(state_now)
        state_all = np.array(state_all)

        leaf_value = 3 * (np.sum(state_weight * np.square(state_now -
                                                        state_all) / np.square(state_all)) / np.sum(state_weight))
        return leaf_value

