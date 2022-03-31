import numpy as np
import pandas as pd
import random

class Data():
    """  分析数据使用 """

    def __init__(self, data_frame):
        self.data_frame = data_frame
        data = data_frame.values
        self.row, self.col = np.shape(data)
        self.col -= 1
        self.Xmat = data[:, 0:-1]
        self.Xmat = np.concatenate((np.ones((self.row, 1)), self.Xmat), axis=1)
        self.Price = np.transpose(np.matrix(data[:, -1]))
        self.list = data_frame.columns.values
        self.n_mat = np.ones((self.col + 1, 1))
        self.w = np.zeros((self.col + 1, 1))

    def true_line(self):
        self.w = np.zeros((self.col + 1, 1))
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(self.Xmat.T, self.Xmat)), self.Xmat.T), self.Price)
        return self.w

    def grad_line(self, n=0.00000001, max_t=10000000, min_change=0.0001):
        t = 0  # 迭代计数器
        """
        np.random.seed(0)
        self.w = np.random.rand(self.col + 1, 1)
        """
        self.w = np.zeros((self.col+1, 1))
        if n>0.00000001:
            derta_n = n/max_t
        else:
            derta_n = 0
        of_ow = np.zeros((self.col + 1, 1))  # 目标函数值f对系数w的偏导矩阵
        old_f = 0
        new_f = 0  # 前后两次的目标函数值f
        while (t < max_t):
            of_ow = np.dot(np.transpose(self.Xmat), np.dot(self.Xmat, self.w) - self.Price)
            self.w -= n * of_ow
            new_f = np.square(np.dot(self.Xmat, self.w) - self.Price).sum()
            if abs(new_f - old_f) < min_change:
                break;
            old_f = new_f
            n -= derta_n
            t += 1
        print("Runing times:")
        print(t)
        return self.w

    def get_f(self):
        return np.square(np.dot(self.Xmat, self.w) - self.Price).sum()

    def max_min_change(self):
        for i in range(1,self.col):
            max_num = max(self.Xmat[:, i])
            min_num = min(self.Xmat[:, i])
            self.Xmat[:, i:i+1] = (self.Xmat[:, i:i+1] - min_num*np.ones((self.row, 1))) / (max_num - min_num)

    def normol_change(self):
        self.Xmat = (StandardScaler().fit_transform(self.data_frame))
        self.Xmat = np.matrix(self.Xmat[:, 0:-1])
        self.Xmat = np.concatenate((np.ones((self.row, 1)), self.Xmat), axis=1)
