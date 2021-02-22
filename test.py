import numpy as np
state_weight = np.array([0.3,0.2,0.5])
node_job = np.array([10,10,1])
nodes_now = [np.array([100,100,10]),np.array([100,100,10]),np.array([100,100,100])]
for node_now in nodes_now:   
    prob = (np.sum(state_weight * np.square(node_job - node_now) / np.square(node_now)) / np.sum(state_weight)) #节点执行完任务后省的越多越好
    print(prob)