# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from node import state as State
#找到jobs的位置 23行
#应该做两套，一套是自我对弈，job是随机的，另一套是人机对弈，jobs是真实的

class Game(object):
    def __init__(self, jobs, node_dict):
        self.jobs = jobs
        self.node_dict = node_dict

    # AI自我对弈，存储自我对弈数据 用于训练 self-play data: (state, mcts_probs, z)
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        # 初始化棋盘
        self.state = State(self.node_dict)
        # 记录该局对应的数据：states, mcts_probs, current_players
        states, mcts_probs = [], []
        # 一直循环到比赛结束
        while True:
            # 得到player的下棋位置
            # 加入job
            self.state.job(self.jobs[0])
            move, move_probs = player.get_action(self.state)
            # 存储数据
            states.append(self.state.get_state_resource_now()) #棋盘状态
            mcts_probs.append(move_probs)
            # 按照move来下棋
            self.state.scheduling(move)
            if is_shown:
                pass
            # 判断游戏是否结束end，统计获胜方 winner
            end = self.state.game_end()
            if end:
                # 记录该局对弈中的每步分值，胜1，负-1，平局0
                # 重置MCTS根节点 reset MCTS root node
                resourse_last = self.state.get_state_resource_now()
                player.reset_player()
                # 返回获胜方，self-play数据: (state, mcts_probs, z)
                return resourse_last, zip(states, mcts_probs)
