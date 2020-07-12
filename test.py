import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set()
# np.random.seed(0)
# uniform_data = np.random.rand(5, 12)
# ax = sns.heatmap(uniform_data)
# plt.show()


# -*-coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# -*-coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return np.sum(s,axis=0)/len(s)

A=np.random.randn(3,5)
print(A)
# [1]使用自定义softmax
s1 = softmax(A)
print("s1:{}".format(s1))
