from numpy import concatenate, ones, random, sqrt
from utils import *

"""
直接使用了tutorial_minst_fnn-numpy-exercise.ipynb中已经写好的代码
这里输出恒定为1，去掉了softmax层
"""
# noinspection PyAttributeOutsideInit
class SimpleFNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.W1 = random.normal(
            scale=sqrt(2. / (input_size + 1)),
            size=[input_size + 1, hidden_size]
        )
        self.W2 = random.normal(
            scale=sqrt(2. / (hidden_size + 1)),
            size=[hidden_size + 1, 1]
        )


        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        bias = ones(shape=[x.shape[0], 1])
        x = concatenate([x, bias], axis=1)

        self.h1 = self.mul_h1.forward(x, self.W1)  # shape(5, 4)
        self.h1_relu = self.relu.forward(self.h1)
        self.h1_relu_bias = concatenate([self.h1_relu, bias], axis=1)
        self.h2 = self.mul_h2.forward(self.h1_relu_bias, self.W2)

        return self.h2

    def backward(self, grad_output):
        self.h2_grad, self.W2_grad = self.mul_h2.backward(grad_output)
        self.h1_relu_grad = self.relu.backward(self.h2_grad[:, :-1])
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)
