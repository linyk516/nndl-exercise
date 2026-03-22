import numpy as np

"""
直接使用了tutorial_minst_fnn-numpy-exercise.ipynb中已经写好的代码
"""
class Matmul:
    def __init__(self):
        self.mem = {}

    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem = {'x': x, 'W': W}
        return h

    def backward(self, grad_y):
        """
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        """
        x = self.mem['x']
        W = self.mem['W']

        ####################
        '''计算矩阵乘法的对应的梯度'''
        ####################

        grad_x = np.matmul(grad_y, W.T)
        grad_W = np.matmul(x.T, grad_y)

        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))

    # noinspection PyMethodMayBeStatic
    def backward(self, grad_y):
        """
        grad_y: same shape as x
        """
        ####################
        '''计算relu 激活函数对应的梯度'''
        ####################
        x = self.mem['x']
        grad_x = np.where(x > 0, grad_y, 0)
        return grad_x


class Softmax:
    """
    softmax over last dimension
    """

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        """
        x: shape(N, c)
        """
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (partition + self.epsilon)

        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y):
        """
        grad_y: same shape as x
        """
        s = self.mem['out']
        # noinspection SpellCheckingInspection
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)  # (N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp + grad_y * s
        return tmp


class Log:
    """
    softmax over last dimension
    """

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        """
        x: shape(N, c)
        """
        out = np.log(x + self.epsilon)

        self.mem['x'] = x
        return out

    def backward(self, grad_y):
        """
        grad_y: same shape as x
        """
        x = self.mem['x']

        return 1. / (x + 1e-12) * grad_y

class Function:
    def __init__(self):
        self.input_dim = None
        self.output_dim = None

    @staticmethod
    def forward(X):
        return X


def _ensure_feature_batch(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def _ensure_label_batch(y):
    y = np.asarray(y)
    if y.ndim == 0:
        return y.reshape(1, 1)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y

"""
采样函数输入输出对
按照比例分为两个集合
"""
def sample(func: Function, num_samples=100, train_ratio=0.8, bound = (-10, 10)):

    X = np.random.uniform(low = bound[0], high = bound[1], size=[num_samples, func.input_dim])
    y = _ensure_label_batch(func.forward(X))
    X_train, X_test = np.split(X, [int(num_samples * train_ratio)], axis=0)
    y_train, y_test = np.split(y, [int(num_samples * train_ratio)], axis=0)
    return X_train, X_test, y_train, y_test

"""
函数拟合任务使用均方误差损失
"""
def compute_loss(predictions, labels):
    predictions = _ensure_label_batch(predictions)
    labels = _ensure_label_batch(labels)
    return np.mean(np.square(predictions - labels))


def compute_accuracy(predictions, labels):
    predictions = _ensure_label_batch(predictions)
    labels = _ensure_label_batch(labels)
    residual = float(np.sum(np.square(labels - predictions)))
    total = float(np.sum(np.square(labels - np.mean(labels))))
    if total < 1e-12:
        return float(np.allclose(predictions, labels))
    return 1.0 - residual / total


def compute_loss_grad(predictions, labels):
    predictions = _ensure_label_batch(predictions)
    labels = _ensure_label_batch(labels)
    return 2. * (predictions - labels) / predictions.shape[0]


def _prepare_batch(x, y):
    x = _ensure_feature_batch(x)
    y = _ensure_label_batch(y)
    return x, y


def train_one_step(model, x, y, lr=1e-3):
    x, y = _prepare_batch(x, y)
    model.forward(x)
    model.backward(compute_loss_grad(model.h2, y))
    model.W1 -= lr * model.W1_grad
    model.W2 -= lr * model.W2_grad
    loss = compute_loss(model.h2, y)
    accuracy = compute_accuracy(model.h2, y)
    return loss, accuracy


def test(model, x, y):
    x, y = _prepare_batch(x, y)
    model.forward(x)
    loss = compute_loss(model.h2, y)
    accuracy = compute_accuracy(model.h2, y)
    return loss, accuracy