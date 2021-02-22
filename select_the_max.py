import numpy as np
def node_select_value(state):
    #jobs = random_job(state)
    #state.job(jobs)
    jobs = state.get_job()
    state_avaliable = state.get_avaliable() #可选节点
    resource_needed = state.get_job() #资源需求
    action_probs = [] #选择概率

    if state_avaliable == [] :
        return None, 0

    for node_name in state_avaliable:
    #node_values:node({'cpu':15, 'memory':10, 'gpu':1})
        state_weight = []
        node_job = []
        node_now = []
        node_value = state.get_state_resource_now()[node_name]
        node_resource_now = node_value.get_node_resource_now()
        for name in resource_needed.keys():
            state_weight.append(state.weight()[name])
            node_now.append(node_resource_now[name])
            node_job.append(jobs[name])
        state_weight = np.array(state_weight)
        node_job = np.array(node_job)
        node_now = np.array(node_now)

        #prob = 1-(np.sum(state_weight * np.square(node_job - node_now) / np.square(node_now)) / np.sum(state_weight)) 节点执行完任务后省的越少越好
        prob = (np.sum(state_weight * np.square(node_job - node_now) / np.square(node_now)) / np.sum(state_weight)) #节点执行完任务后省的越多越好
        action_probs.append(prob)
    action = state_avaliable[np.argmax(action_probs)]
    return action, action_probs