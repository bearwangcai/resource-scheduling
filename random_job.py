import numpy as np
import random
def random_job(state, n_job, n_job_thread = 0, probability_1 = 0, probability_2 = 0.2):
    '''
    state:分配状态
    n_job:是该状态下第多少个任务
    n_job_thread:任务资源比例，例如在n_job_thread后出现异构资源任务
    probability_1:在n_job_thread之前出现异构任务的比例
    probability_2:在n_job_thread之后出现异构任务的比例
    '''
    state_resource_names = state.get_state_resource_name()
    job = {}
    for resource_name in state_resource_names:
        if ((resource_name == 'cpu') or (resource_name == 'memory')):
            job[resource_name] = random.randint(5,8)
            
        
        else:
            #job[resource_name] = random.randint(0,1)
            job[resource_name] = 0
    if n_job > n_job_thread:   
        for resource_name in state_resource_names:
            if ((not (resource_name == 'cpu')) and (not (resource_name == 'memory'))):     
                if np.random.random() < probability_2:#按比例越小越明显，但不宜太小，否则有可能不生成含gpu任务
                    job[resource_name] += random.randint(5,8)
                    
    else:
        for resource_name in state_resource_names:
            if ((not (resource_name == 'cpu')) and (not (resource_name == 'memory'))):     
                if np.random.random() < probability_1:#按比例越小越明显，但不宜太小，否则有可能不生成含gpu任务
                    job[resource_name] += random.randint(5,8)
                    
    return job
"""
理想化测试版
def random_job(state, n_job, n_job_thread = 0, probability_1 = 0, probability_2 = 0.2):
    '''
    state:分配状态
    n_job:是该状态下第多少个任务
    n_job_thread:任务资源比例，例如在n_job_thread后出现异构资源任务
    probability_1:在n_job_thread之前出现异构任务的比例
    probability_2:在n_job_thread之后出现异构任务的比例
    '''
    state_resource_names = state.get_state_resource_name()
    job = {}
    for resource_name in state_resource_names:
        if ((resource_name == 'cpu') or (resource_name == 'memory')):
            #job[resource_name] = random.randint(5,8)
            job[resource_name] = 10
        
        else:
            #job[resource_name] = random.randint(0,1)
            job[resource_name] = 0
    if n_job > n_job_thread:   
        for resource_name in state_resource_names:
            if ((not (resource_name == 'cpu')) and (not (resource_name == 'memory'))):     
                if np.random.random() < probability_2:#按比例越小越明显，但不宜太小，否则有可能不生成含gpu任务
                    #job[resource_name] += random.randint(5,8)
                    job[resource_name] += 10
    else:
        for resource_name in state_resource_names:
            if ((not (resource_name == 'cpu')) and (not (resource_name == 'memory'))):     
                if np.random.random() < probability_1:#按比例越小越明显，但不宜太小，否则有可能不生成含gpu任务
                    #job[resource_name] += random.randint(5,8)
                    job[resource_name] += 10
    return job
"""