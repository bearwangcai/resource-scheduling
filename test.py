import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''
state_weight = np.array([0.3,0.2,0.5])
node_job = np.array([10,10,1]).reshape(1,-1)
#nodes_now = [np.array([100,100,10]).reshape(1,-1),np.array([100,100,10]).reshape(1,-1),np.array([100,100,100]).reshape(1,-1)]
nodes_now = np.array([[100,100,10],[100,100,10],[100,100,100]]).reshape(3,-1)
prob = cosine_similarity(node_job, nodes_now)
for node_now in nodes_now:   
    #prob = (np.sum(state_weight * np.square(node_job - node_now) / np.square(node_now)) / np.sum(state_weight)) #节点执行完任务后省的越多越好
    prob = cosine_similarity(node_job, node_now)
    print(prob)

prob = cosine_similarity(np.array([[10,10,1],[100,100,100]]))
print(prob)
'''
a = [np.array([1]).reshape(1,-1),np.array([0.5]).reshape(1,-1),np.array([0.8]).reshape(1,-1)]
b = np.argmax(a)
