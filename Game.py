# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from node import state as State
import random
#找到jobs的位置 23行
#应该做两套，一套是自我对弈，job是随机的，另一套是人机对弈，jobs是真实的

class Game(object):
    def __init__(self, jobs, node_dict):
        self.jobs = jobs
        self.node_dict = node_dict

    def random_job(self, state):
        state_resource_names = state.get_state_resource_name()
        job = {}
        for resource_name in state_resource_names:
            job[resource_name] = random.randint(0,10)
        return job

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
                credit = self.get_algorithm_credit(self.state)
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

        leaf_value = 1-(np.sum(state_weight * np.square(state_now -
                                                        state_all) / np.square(state_all)) / np.sum(state_weight))
        return leaf_value

