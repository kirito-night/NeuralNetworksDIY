import numpy as np
from utils import *
class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return 2 * (yhat-y )


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return 1 - (yhat * y).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        return yhat - y


class LogSoftmax(Loss):
    def forward(self, y, yhat):
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        log_sum_exp = yhat_max + np.log(
            np.sum(np.exp(yhat - yhat_max), axis=1, keepdims=True)
        )
        loss = -np.sum(y * (yhat - log_sum_exp), axis=1)
        return np.mean(loss)

    def backward(self, y, yhat):
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        log_sum_exp = yhat_max + np.log(
            np.sum(np.exp(yhat - yhat_max), axis=1, keepdims=True)
        )
        softmax = np.exp(yhat - log_sum_exp)
        return -(y - softmax) / y.shape[0]

# class CELoss(Loss):
#     def forward(self, y, yhat): 
#         yhat = yhat - np.max(yhat, axis=1, keepdims=True)
#         m = np.arange(len(yhat))
#         self.exp = np.exp(yhat)
#         return -yhat[m, y] + np.log(np.sum(self.exp, axis=1, keepdims=True))
#     def backward(self, y, yhat):
#         exp  = self.exp
#         m = np.arange(len(yhat))
#         s = np.sum(exp, axis=1)
#         M  = exp / s[:, np.newaxis]
#         M[m, y]  = M[m, y] - 1
#         return M 
       
class CELoss(Loss):
    def forward(self, y, yhat):
        yhat = yhat - np.max(yhat, 1)[:,np.newaxis] # evite les valeurs infinies
        return -yhat[:, y] + np.log(np.sum(np.exp(yhat), 1))

    def backward(self, y, yhat):
        yhat = yhat - np.max(yhat, 1)[:,np.newaxis] # evite les valeurs infinies
        M = np.zeros(yhat.shape) + np.log(np.sum(np.exp(yhat), 1)).reshape(-1,1)
        M[:,y] += -1
        return M


class BCELoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        y = np.clip(y, 1e-15, 1 - 1e-15)
        yhat = np.clip(yhat, 1e-15, 1 - 1e-15)
        res = -(y * np.maximum(-100, np.log(yhat))+ (1 - y) * np.maximum(-100, np.log(1 - yhat)))
        #print(f"foward : res, {res} \n")

        return res
    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            f"dimension mismatch, y and yhat must of same dimension. "
            f"Here it is {y.shape} and {yhat.shape}"
        )
        eps = 1e-8
        y = np.clip(y, eps, 1 - eps)
        yhat = np.clip(yhat, eps, 1 - eps)
        res = -((y / yhat) - (1 - y) / (1 - yhat))
        #print(f" backward res :  {res} \n")
        return res 
