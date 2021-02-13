import numpy as np
import copy
import pickle

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
    def select(self, c_puct):
        # 每次选择最大UCT值的节点，返回(action, next_node)
        return max(self._children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct))

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
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    # 计算节点价值 UCT值 = Q值 + 调整后的访问次数（exploitation + exploration）
    def get_value(self, c_puct):
        # 计算调整后的访问次数
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

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
    def _playout(self, state):
        # 设置当前节点
        node = self._root
        # 必须要走到叶子节点
        while(1):
            if node.is_leaf():
                break
            # 基于贪心算法 选择下一步
            action, node = node.select(self._c_puct)
            state.do_move(action)

            # 对于current player，根据state 得到一组(action, probability) 和分数v [-1,1]之间（比赛结束时的预期结果）
            action_probs, leaf_value = self._policy(state)
            # 检查游戏是否结束
            end, winner = state.game_end()
            if not end:
                node.expand(action_probs)
            else:
                '''
                记得要做资源查验 即node_resource_now满足资源请求的才可以被调度
                '''
                leaf_value = np.sum(state.weight() * np.square(state.now() - state.all()) / np.square(state.all())) / state.weight()
                '''
                state.weight() 每一项资源的权重系数，是一个array
                state.now()    每一项资源的现有量，是一个array
                state.all()    每一项资源的原始存量，是一个array
                '''

            # 将子节点的评估值反向传播更新父节点(所有)
            node.update_recursive(-leaf_value)


    def get_move_probs(self, state, temp=1e-3):
        # 运行_n_playout次 _playout
        for n in range(self._n_playout):
            # 在进行_playout之前需要保存当前状态的副本，因为状态是就地修改的
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 基于节点的访问次数，计算move probabilities
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
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
    def __init__(self, policy_value_function,
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
        
    # 获取AI下棋的位置
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 获取所有可能的下棋位置
        sensible_moves = board.availables
        # MCTS返回的pi向量，基于alphaGo Zero论文
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
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

    def __str__(self):
        return "MCTS {}".format(self.player)


def scheduling(resource_needed):
    rules = pickle.load('')
    node_name = rules
    return node_name
