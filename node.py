import numpy as np 
import mcts

class node:
    def __init__(self, resource_origin):
        '''
        资源是一个字典，例如:
        {cpu:15, memory:10, gpu:1}
        '''
        self.resource_origin = resource_origin 

class state:
    def __init__(self, resource):
        self.resource_origin = resource #原始资源是一个字典，每一个key就是node的名字
        self.n_node = len(resource.keys())
        self.state_log = []

    def job(self, resource_needed):
        '''
        接收新的任务
        '''
        self.resource_needed = resource_needed #一组资源消耗字典，形如：{cpu:15, memory:10, gpu:1}

    def scheduling(self):
        '''
        执行调度
        '''
        node_name = mcts.scheduling(self.resource_needed)#返回是调度到的节点名称

        for i in self.resource_origin[node_name].keys():
            self.resource_origin[node_name][i] -= self.resource_needed[i]#状态更新

        self.state_log.append(self.resource_origin)

        return node_name


    def show(self):
        pass


def main():

    node1 = node({'cpu':15, 'memory':10, 'gpu':1})
    node2 = node({'cpu':15, 'memory':10, 'gpu':1})
    node3 = node({'cpu':15, 'memory':10, 'gpu':1})

    node_list = {'node1':node1, 'node2':node2, 'node3':node3}

    jobs = [{},{}]

    state1 = state(node_list)

    for job in jobs:
        state1.job(job)
        state1.scheduling()
    state1.show()
    
