import numpy as np

class PerceptionPosition:
    def __init__(self, random_range):
        '''
        根据随机范围，随机返回位置的值
        :param random_range: 随机范围, shape = 1
        '''
        self.random_range = np.array([random_range, random_range, 0])
        self.q = 0

        # 记录历史数据
        self._history = []
    
    def updata_actrual_q(self, q):
        '''
        设置实际的位置
        :param q: 实际的位置, shape = (n, 3)
        '''
        self.q = q if type(q) == np.ndarray else np.array(q)
        
        # 向量化生成随机数
        # self.random_range.shape = (3, )
        # self.random_num.shape = (n, 3)
        self.random_num = np.random.normal(0, self.random_range, self.q.shape)

        # 记录历史数据
        self._history.append(self.q.copy() + self.random_num.copy())
        

    def get_observation(self):
        '''
        获取观测值
        :return: 观测值
        '''
        return self._history[-1]
    
    def __call__(self):
        return self.get_observation()
    
    @property
    def history(self) -> list:
        '''
        观测历史数据, shape = (round, n, 3)
        '''
        return self._history