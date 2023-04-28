from projet_etu import Module
import numpy as np
from utils import *
from loss import *

class TanH(Module):
    def __init__(self):
        super(TanH, self).__init__()


    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        # yhat = self.forward(input)
        # return delta * (1-yhat**2)
        return delta * (1 - np.tanh(input)**2)
    def update_parameters(self, gradient_step=0.001):
        pass
    
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
      

    def forward(self, X):
        return 1/(1+np.exp(-X))

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        yhat = self.forward(input)
        return delta * yhat * (1-yhat)
    
    def update_parameters(self, gradient_step=0.001):
        pass

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, X):
        return np.maximum(X, 0)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (input > 0)
    def update_parameters(self, gradient_step=0.001):
        pass

class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()


    def forward(self, X):
        X_exp = np.exp(X)
        return X_exp / X_exp.sum(axis=1, keepdims=True)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input ,delta):
        softmax = self.forward(input)
        return delta * (softmax * (1 - softmax))
    def update_parameters(self, gradient_step=0.001):
        pass
