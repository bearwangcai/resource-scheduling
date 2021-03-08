import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
plt.rc('font',family='Times New Roman', size=8)

def show(MCTS, maximum):
    x = [i+1 for i in range(len(MCTS))]
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    plt.bar(x,np.array(MCTS)-np.array(maximum))
    plt.xlabel('Experiment scenarios')
    plt.ylabel('The number of jobs')
    ax = plt.gca()
    x_major_locator=MultipleLocator(50)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(1)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig(r'D:\科研\论文\High effient resource scheduling for cloud based on modified MCTS\programing\experiment.png',dpi=500)
    #plt.title('')
    plt.show()

f1 = open(r'D:\科研\论文\High effient resource scheduling for cloud based on modified MCTS\programing\parameter_check_on_gpu_cpuct_3_njobthread_5.pkl', "rb")
f2 = open(r'D:\科研\论文\High effient resource scheduling for cloud based on modified MCTS\programing\parameter_check_on_have_fpgac_puct0.03n_job_thread0.pkl', "rb")
result1 = pickle.load(f1)
result2 = pickle.load(f2)
f1.close()
f2.close()
c_puct_list = [0.03,0.3,3]
n_job_thread_list1 = [0,5]
n_job_thread_list2 = [0,6,12]
probability_1_list1 = [0,0.03,0.3]
probability_1_list2 = [0,0.03,0.3]
#probability_1_list2 = [0.03,0.3,3]
probability_2_list = [0.3,0.6,0.9]
game_batch_num = 3
n = 0
MCTS = []
maximum = []
for index,result in enumerate([result1, result2]):
    for c_puct in c_puct_list:
        for probability_2 in probability_2_list:
            for i in range(game_batch_num):
                if  index == 0:
                    for n_job_thread in n_job_thread_list1:
                        for probability_1 in probability_1_list1:
                            MCTS.append(result['job_dealed_by_algorithm'][(c_puct, n_job_thread, probability_1, probability_2, i+1)])
                            maximum.append(result['job_dealed_by_max'][(c_puct, n_job_thread, probability_1, probability_2, i+1)])
                else:
                    for n_job_thread in n_job_thread_list2:
                        for probability_1 in probability_1_list2:
                            MCTS.append(result['job_dealed_by_algorithm'][(c_puct, n_job_thread, probability_1, probability_2, i+1)])
                            maximum.append(result['job_dealed_by_max'][(c_puct, n_job_thread, probability_1, probability_2, i+1)])
show(MCTS, maximum)
