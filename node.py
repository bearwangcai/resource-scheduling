import numpy as np 
import mcts
from copy import deepcopy

class node:
    def __init__(self, resource_origin):
        '''
        资源是一个字典，例如:
        {'cpu':15, 'memory':10, 'gpu':1}
        '''
        self.node_resource_origin = resource_origin
        self.node_resource_now = deepcopy(self.node_resource_origin)

    def get_node_resource_origin(self):
        return self.node_resource_origin

    def get_node_resource_now(self):
        return self.node_resource_now

    def node_resource_consume(self, job):
        '''
        job:资源消耗请求,形如：{'cpu':15, 'memory':10, 'gpu':1}
        '''
        for key in job.keys():
            try:
                self.node_resource_now[key] -= job[key]
            except:
                print('无此资源')

    def node_resource_release(self, job_end):
        '''
        job_end:被释放的资源,形如：{'cpu':15, 'memory':10, 'gpu':1}
        '''
        for key in job_end.keys():
            try:
                self.node_resource_now[key] += job_end[key]
            except:
                print('无此资源')

class state:
    def __init__(self, resource, resource_weight = None):
        self.state_resource_origin = resource #原始资源是一个字典，每一个key就是node的名字
        self.state_resource_now = deepcopy(self.state_resource_origin) #现有资源
        self.state_resource_name = self.state_resource_origin[self.state_resource_origin.keys()[
            0]].keys()
        # 资源权重,形如{'cpu':0.3, 'memory':0.2, 'gpu':0.5}
        self.n_node = len(resource.keys()) #节点个数
        self.state_log = []
        if resource_weight == None:
            self.resource_weight = {}
            for key in self.state_resource_name:
                self.resource_weight[key] = 1
        else:
            self.resource_weight = resource_weight

    def get_state_resource_name(self):
        '''
        返回资源名称
        '''
        return self.state_resource_name

        
    def get_state_resource_origin(self):
        '''
        返回资源总数
        '''
        return self.state_resource_origin

    def get_state_resource_now(self):
        '''
        返回现有资源数
        '''
        return self.state_resource_now

    def calculate_resource(self,resource,whose):
        '''
        计算资源总数
        '''
        resources_all = {}
        for node_value in resource.values():
            #node_values:node({'cpu':15, 'memory':10, 'gpu':1})
            if whose == 'all':
                node_resource = node_value.get_node_resource_origin()
            else:
                node_resource = node_value.get_node_resource_now()
            for resource_key in node_resource.keys():
                if resource_key in resources_all.keys():
                    resources_all[resource_key] += node_resource[resource_key]
                else:
                    resources_all[resource_key] = node_resource[resource_key]

        return resources_all

    def all(self): #总资源数，即原始资源数
        resources_all = self.calculate_resource(self.state_resource_origin, 'all')
        return resources_all

    def now(self):  #现有资源数
        resources_now = self.calculate_resource(self.state_resource_now, 'now')
        return resources_now

    def weight(self): #返回资源权重
        return self.resource_weight

    def job(self, resource_needed):
        '''
        接收新的任务
        '''
        self.resource_needed = resource_needed  # 一组资源消耗字典，形如：{'cpu':15, 'memory':10, 'gpu':1}

    def get_job(self):
        return self.resource_needed

    def get_avaliable(self):
        node_keys = []
        for node_key in self.state_resource_now.keys():
            flag = 1
            for resource_key in self.state_resource_name:
                if self.state_resource_now[node_key][resource_key] < self.resource_needed[resource_key]:
                    flag = 0
                    break
            if flag:
                node_keys.append(node_key)
        return node_keys

    def game_end(self):
        node_keys = self.get_avaliable()
        return True if node_keys==[] else False
    
    
    def scheduling(self, action):
        '''
        执行调度
        '''
        node_name = action#返回是调度到的节点名称
        #node_name = 'node1'
        
        self.state_resource_now[node_name].node_resource_consume(
            self.resource_needed)  # 状态更新

        self.state_log.append(deepcopy(self.state_resource_now))

        return node_name
    

    def show(self):
        pass


def main():

    node1 = node({'cpu':15, 'memory':10, 'gpu':1})
    node2 = node({'cpu':15, 'memory':10, 'gpu':1})
    node3 = node({'cpu':15, 'memory':10, 'gpu':1})

    node_dict = {'node1':node1, 'node2':node2, 'node3':node3}

    jobs = [{'cpu': 5, 'memory': 2, 'gpu': 1}, {'cpu': 5, 'memory': 2, 'gpu': 0}]

    state1 = state(node_dict)
    resources_origin_all = state1.all()
    resources_origin_now = state1.now()


    
    for job in jobs:
        state1.job(job)
        action = 'node1'
        state1.scheduling(action)
        resources_origin_all = state1.all()
        resources_origin_now = state1.now()
    
    state1.show()
    
main()
