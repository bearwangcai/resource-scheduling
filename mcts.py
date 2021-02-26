import numpy as np
import copy
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from select_the_max import node_select_value_cos

# 快速走子策略：随机走子


def random_job(state):
    state_resource_names = state.get_state_resource_name()
    job = {}
    for resource_name in state_resource_names:
        if ((resource_name == 'cpu') or (resource_name == 'memory')):
            job[resource_name] = random.randint(5,8)
        
        else:
            #job[resource_name] = random.randint(0,1)
            job[resource_name] = 0
        
    if np.random.random() < 0.1:#按比例越小越明显，但不宜太小，否则有可能不生成含gpu任务
        job['gpu'] += random.randint(5,8)
    return job

def rollout_policy_fn(state):
    state_avaliable = state.get_avaliable()
    # 随机走，从棋盘中可以下棋的位置中随机选一个
    action_probs = np.random.rand(len(state_avaliable))
    return zip(state_avaliable, action_probs)


# policy_value_fn 考虑了棋盘状态，输出一组(action, probability)和分数[-1,1]之间
def policy_value_fn(state):
    # 对于pure MCTS来说，返回统一的概率，得分score为0
    state_avaliable = state.get_avaliable()
    action_probs = np.ones(len(state_avaliable)) / len(state_avaliable)
    return zip(state_avaliable, action_probs), 0

def node_select_value(state):
    #jobs = random_job(state)
    #state.job(jobs)
    jobs = state.get_job()
    #state_avaliable = state.get_avaliable() #可选节点
    resource_node_name = state.get_resource_node_name()
    resource_needed = state.get_job() #资源需求
    action_probs = [] #选择概率
    
    '''
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
        '''
    #for node_name in state_avaliable:
    for node_name in resource_node_name:
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
        node_job = np.array(node_job).reshape(1,-1)
        node_now = np.array(node_now).reshape(1,-1)

        #prob = 1-(np.sum(state_weight * np.square(node_job - node_now) / np.square(node_now)) / np.sum(state_weight)) 节点执行完任务后省的越少越好
        #prob = (np.sum(state_weight * np.square(node_job - node_now) / np.square(node_now)) / np.sum(state_weight)) #节点执行完任务后省的越多越好
        prob = cosine_similarity(node_job, node_now)[0,0]
        action_probs.append(prob)
    #action = state_avaliable[np.argmax(action_probs)]
    #return action, action_probs
    #return zip(state_avaliable, action_probs), 0
    return zip(resource_node_name, action_probs), 0


# MCTS树节点，每个节点都记录了自己的Q值，先验概率P和 UCT值第二项，即调整后的访问次数u（用于exploration）
class TreeNode(object):
    # 节点初始化
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # Action到TreeNode的映射map
        self._n_visits = 0   # 访问次数
        self._Q = 0         # 行动价值
        self._u = 0   # UCT值第二项，即调整后的访问次数（exploration）
        self._P = prior_p    # 先验概率

    # Expand，展开叶子节点（新的孩子节点）
    def expand(self, action_priors):
        for action, prob in action_priors:
            # 如果不是该节点的子节点，那么就expand 添加为子节点
            if action not in self._children:
                # 父亲节点为当前节点self,先验概率为prob
                self._children[action] = TreeNode(self, prob)

    # Select步骤，在孩子节点中，选择具有最大行动价值UCT，通过get_value(c_puct)函数得到
    def select(self, actions_available, c_puct):
        # 每次选择可被选择的节点中最大UCT值的节点，返回(action, next_node)
        
        node_available = {}
        for action in actions_available:
            if action in self._children.keys():
                node_available[action] = self._children[action]
        return max(node_available.items(),
            key=lambda act_node: act_node[1].get_value(c_puct))
        
        #return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    # 从叶子评估中，更新节点Q值和访问次数
    def update(self, leaf_value):
        # 节点访问次数+1
        self._n_visits += 1
        # 更新Q值，变化的Q对于所有访问次数进行平均
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    # 递归的更新所有祖先，调用self.update
    def update_recursive(self, leaf_value):
        # 如果不是根节点，就需要先调用父亲节点的更新
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    # 计算节点价值 UCT值 = Q值 + 调整后的访问次数（exploitation + exploration）
    def get_value(self, c_puct):
        # 计算调整后的访问次数
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
        '''
        self._u = (c_puct * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u + self._P
        '''

    # 判断是否为叶子节点
    def is_leaf(self):
        return self._children == {}

    # 判断是否为根节点
    def is_root(self):
        return self._parent is None


# MCTS：Monte Carlo Tree Search 实现了蒙特卡洛树的搜索 
class MCTS(object):
    # policy_value_fn 考虑了棋盘状态，输出一组(action, probability)和分数[-1,1]之间(预计结束时的比分期望)
    # c_puct exploitation和exploration之间的折中系数
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0) # 根节点
        self._policy = policy_value_fn   # 策略状态，考虑了棋盘状态，输出一组(action, probability)和分数[-1,1]之间
        self._c_puct = c_puct # exploitation和exploration之间的折中系数
        self._n_playout = n_playout

    # 从根节点到叶节点运行每一个playout，获取叶节点的值（胜负平结果1，-1,0），并通过其父节点将其传播回来
    # 状态是就地修改的，所以需要保存副本
    def _playout(self, state, n):
        # 设置当前节点
        node = self._root
        state_contrast = copy.deepcopy(state)
        # 统计资源都有什么
        state_resource_all = state.all()
        state_all_resource_name = state_resource_all.keys() #资源名称，应该是个list
        job_list = []
        # 必须要走到叶子节点
        n_=0
        while(1):
            if node.is_leaf():
                break
            '''
            if n==0 :
            # 基于贪心算法 选择下一步
                action_available = state.get_avaliable()
                try:
                    action, node = node.select(action_available, self._c_puct)
                # action 被选择的节点，是一个string
                    state.scheduling(action)
                except:
                    pass
                n+=1
            else:
                state.job(random_job(state))
                try:
                    action, node = node.select(action_available, self._c_puct)
                # action 被选择的节点，是一个string
                    state.scheduling(action)
                except:
                    pass
                n+=1
            '''
            if n_==0 :
            # 基于贪心算法 选择下一步
                job_list.append(state.get_job())
                action_available = state.get_avaliable()
                action, node = node.select(action_available, self._c_puct)
                # action 被选择的节点，是一个string
                state.scheduling(action)
                n_+=1
            else:
                job_list.append(random_job(state))
                state.job(job_list[-1])
                end = state.game_end()
                if end:
                    break
                else:
                    action_available = state.get_avaliable()
                    action, node = node.select(action_available, self._c_puct)
                    # action 被选择的节点，是一个string
                    state.scheduling(action)
                    n_+=1
            # 对于current player，根据state 得到一组(action, probability) 和分数v [-1,1]之间（比赛结束时的预期结果）
        action_probs, leaf_value = self._policy(state)
        # 检查游戏是否结束
        end = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            #新leaf_value计算方式
            n_contrast = 0
            for job_contrast in job_list:
                state_contrast.job(job_contrast)
                if state_contrast.game_end():#是在do_action之前,记得修改
                    break
                else:
                    move_contrast, move_probs_contrast = node_select_value_cos(state_contrast)
                    state_contrast.scheduling(move_contrast)
                    n_contrast += 1
            while 1:
                job_list.append(random_job(state_contrast))
                state_contrast.job(job_list[-1])
                if state_contrast.game_end():
                    break
                else:
                    move_contrast, move_probs_contrast = node_select_value_cos(state_contrast)
                    state_contrast.scheduling(move_contrast)
                    n_contrast += 1                    
            leaf_value = n_ - n_contrast
            
            """
            '''
            记得要做资源查验 即node_resource_now满足资源请求的才可以被调度
            '''

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

            leaf_value = 1 * np.sum(state_weight * np.square(state_now - state_all) / np.square(state_all)) / np.sum(state_weight)
            #剩的越少越好
            '''
            state_weight 每一项资源的权重系数，是一个array
            state_now    每一项资源的现有量，是一个array
            state_all    每一项资源的原始存量，是一个array
            '''
            """

        # 将子节点的评估值反向传播更新父节点(所有)
        node.update_recursive(leaf_value)


    def get_move_probs(self, state, temp=1e-3):
        # 运行_n_playout次 _playout
        for n in range(self._n_playout):
            # 在进行_playout之前需要保存当前状态的副本，因为状态是就地修改的
            state_copy = copy.deepcopy(state)
            #state_copy.job(self.random_job(state_copy))
            self._playout(state_copy,n)
            #print(n)

        # 基于节点的访问次数，计算move probabilities
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        # 基于节点的_Q值，计算move probabilities
        #act_visits = [(act, node._Q) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        # 基于节点的访问次数，通过softmax计算概率
        act_probs = self.softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    # 在树中前进一步
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

# 基于MCTS的AI Player
class MCTSPlayer(object):
    def __init__(self, policy_value_function = node_select_value,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        # 使用MCTS进行搜索
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        
    # 设置player index
    def set_player_ind(self, p):
        self.player = p

    # 重置MCTS树
    def reset_player(self):
        self.mcts.update_with_move(-1)
        



    '''
    def get_action(self, state, temp=1e-3, return_prob=0):
        # 获取所有可能的下棋位置
        state.job(self.random_job(state))
        sensible_moves = state.get_avaliable()
        # MCTS返回的pi向量，基于alphaGo Zero论文
        move_probs = np.zeros(len(sensible_moves))
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(state, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 为探索添加Dirichlet噪声(需要进行自我训练)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点，重新使用搜索树
                self.mcts.update_with_move(move)
            else:
                # 默认temp=1e-3, 几乎等同于选择概率最大的那一步
                move = np.random.choice(acts, p=probs)
                # 重置根节点 reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
    '''
    def get_action(self, state):
        end = state.game_end()
        if not end:
            actions, probs = self.mcts.get_move_probs(state)
            action = actions[np.argmax(probs)]
            self.mcts.update_with_move(action)
            return action, probs
        else:
            print("WARNING: the state is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


def scheduling(resource_needed):
    rules = pickle.load('')
    node_name = rules
    return node_name
