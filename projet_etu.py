import numpy as np
from utils import *
from loss import *


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        self._gradient = np.zeros_like(self._gradient)

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class Linear(Module):
    def __init__(self, input, output, w= None ,b=None, biais=True):
        super(Linear, self).__init__()

        if(biais):
            input = input + 1
        self._gradient = np.zeros((input, output))

        if w is None:
            self._parameters = np.random.randn(input, output) / np.sqrt(input)  
        else :
            if(biais):
                self._parameters = np.vstack((w, np.zeros((1, output))))
            else:
                self._parameters = w     
        self.biais = biais

    def forward(self, X):
        if self.biais:
            X = proj_biais(X)
        try : 
            assert X.shape[1] == self._parameters.shape[0], f"Erreur de dimension, X.shape[1] = {X.shape[1]} et self._parameters.shape[0] = {self._parameters.shape[0]}"
            return X @ self._parameters 
        except AssertionError:
            raise AssertionError

    def backward_update_gradient(self, input, delta):
        if self.biais:
            input = proj_biais(input)
        self._gradient += input.T @ delta/len(input)

    def backward_delta(self, input, delta):
        if self.biais:
            input = proj_biais(input)
        return delta @ self._parameters[:-1,:].T
    



class Sequential(Module): 
    def __init__(self, args):
        super(Sequential, self).__init__()
        self._layers = args
        self.input = None

    def forward(self, X):
        self.input = [X]
        for layer in self._layers:
            X = layer.forward(X)
            self.input.append(X)
        return X

    def backward(self, input, delta, gradient_step = 1e-3):
        self.input.pop()

        for layer in reversed(self._layers):
            input = self.input.pop()
            layer.backward_update_gradient(input, delta)
            delta = layer.backward_delta(input, delta)

            layer.update_parameters(gradient_step=gradient_step)
            layer.zero_grad()

class Optim(object):
    def __init__(self, net, loss, eps=1e-3, lam= 1e-3):
        self._net, self._loss = net, loss
        self.eps = eps    
    
    def step(self, batch_x, batch_y):
        yhat = self._net.forward(batch_x)
        
        C = self._loss.forward(batch_y, yhat)
        delta = self._loss.backward(batch_y, yhat)
        self._net.backward(batch_x, delta , gradient_step = self.eps)
        
        for m in self._net._layers:
            m.zero_grad()
            m.update_parameters(self.eps)
        return C


def SGD(net, loss, datax, datay, predict=None, xtest = None , ytest = None,  batch_size = 50, eps = 1e-3, max_iter = 100, split = 0.8):
    optimisation = Optim(net, loss, eps)
    c = int(datax.shape[0] * split)
    indices = np.random.permutation(datax.shape[0])
    if xtest is None and ytest is None:
        xtrain, ytrain = datax[indices[:c]], datay[indices[:c]]
        xtest, ytest = datax[indices[c:]], datay[indices[c:]]
    else : 
        xtrain, ytrain = datax[indices[:c]], datay[indices[:c]]      

    Lerror = []
    Lscore = []
    #Lscore_train = []
    for iteration in range(max_iter):
        print("Epoch: {0} / {1}".format(iteration, max_iter))
        # descente de gradient mini-batch
        for index in make_minibatch(batch_size, xtrain, ytrain):
            batch_x, batch_y = xtrain[index], ytrain[index] 
            optimisation.step(batch_x, batch_y)
        # calcule du cout
        cost = loss.forward(datay, net.forward(datax)).mean()
        print("Loss " ,round(cost, 3))
        Lerror.append(cost)
        if predict is not  None:
            #score_train = np.mean(predict(xtrain) == ytrain)
            score_test = np.mean(predict(xtest) == ytest)
            Lscore.append(score_test)
            #Lscore_train.append(score_train)
            print("score " ,round(Lscore[-1], 3))
            #print("score: {} (train) {} (test)".format(round(score_train, 3), round(score_test, 3)))
    return np.array(Lerror), np.array(Lscore)




# class Conv1D(Module):
#     def __init__(self, k_size, chan_in, chan_out , stride=1):
#         """(k_size,chan_in,chan_out)"""
#         super().__init__()
#         self._k_size = k_size  # taille du filtre
#         self._chan_in = chan_in  # C
#         self._chan_out = chan_out  # nombre de filtres
#         self._stride = stride
#         self._parameters = np.random.rand(k_size, chan_in, chan_out)  # filtres

#     def forward(self, X):
#         """Performe une convolution en 1D sans boucles for.
#         Parameters
#         ----------
#         X : ndarray (batch, length, chan_in)
#         Returns
#         -------
#         ndarray (batch, (length-k_size)/stride + 1, chan_out)"""
#         batch_size, length, chan_in = X.shape
#         assert chan_in == self._chan_in, f"X must have {self._chan_in} channels. Here X have {chan_in} channels." 
        
#         batch_stride, length_stride, chan_stride = X.strides

#         out_size = int((length - self._k_size) / self._stride + 1)
#         new_shape = (batch_size, out_size, chan_in, self._k_size)
#         new_strides = (
#             batch_stride,
#             self._stride * length_stride,
#             chan_stride,
#             length_stride,
#         )

#         X_windows = np.lib.stride_tricks.as_strided(X, new_shape, new_strides)

#         self.inputs = X, X_windows
#         output = np.einsum("blck,kcf->blf", X_windows, self._parameters)
#         return output

#     def backward_update_gradient(self, input, delta):
#         """TO DO"""
#         batch, length, chan_in = input.shape
#         assert chan_in == self._chan_in
#         batch_size, out_size, chan_in, k_size = self.inputs[1].shape
#         input_windows = np.lib.stride_tricks.as_strided(
#             input, (batch_size, out_size, chan_in, k_size), self.inputs[1].strides
#         )
#         print(input_windows.shape)
#         #self._gradient += np.einsum("blck,lcf->bkf", input_windows, delta) / batch

#     def backward_delta(self, input, delta):
#         """TO DO"""
#         np.einsum("", delta, self._parameters)
#         ...

#     def forward_loops(self, X):
#         """Performe une convolution en 1D avec des boucles for."""
#         batch, length, chan_in = X.shape
#         assert chan_in == self._chan_in

#         # Initialize the output array
#         out_size = int((length - self._k_size) / self._stride + 1)
#         out = np.zeros((batch, out_size, self._chan_out))

#         # Convolve for each batch element
#         for b in range(batch):
#             # Convolve for each output channel
#             for c_out in range(self._chan_out):
#                 # Convolve for each position in the output
#                 for i in range(out_size):
#                     # Compute the receptive field
#                     start = i * self._stride
#                     end = start + self._k_size

#                     # Compute the convolution
#                     out[b, i, c_out] = np.sum(
#                         X[b, start:end, :] * self._parameters[:, :, c_out]
#                     )

#         return out


class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1, init = 1e-1):
        """
        Initializes a new instance of the Conv1D class.
        Parameters:
        k_size (int): The size of the convolution kernel.
        chan_in (int): The number of input channels.
        chan_out (int): The number of output channels.
        stride (int): The stride of the convolution.
        init (float): The standard deviation of the normal distribution used to initialize the convolution kernel.
        """
        assert stride > 0
        self.stride = stride
        self._parameters = np.random.randn(k_size, chan_in, chan_out)*init
        self._gradient = np.zeros(self._parameters.shape)

    def forward(self, X):
        """
        Computes the forward pass of the convolution layer.

        Parameters:
        X (ndarray): The input tensor of shape (batch, length, chan_in).

        Returns:
        The output tensor of shape (batch, length_out, chan_out), where length_out = (length - k_size) // stride + 1.
        """
        assert X.ndim == 3
        assert X.shape[2] == self._parameters.shape[1]
        batch, length, _ = X.shape
        k_size, chan_in, chan_out = self._parameters.shape
        offset = length - k_size + 1
        # dimension k_size, batch, index, chan_in
        Map = [X[:, k:offset+k:self.stride, :] for k in np.arange(k_size)]
        res = np.einsum('knli,kio->nlo', Map, self._parameters)
        return res
        
    def backward_update_gradient(self, input, delta):

        # delta: batch, length_out , chan_out
        # input: batch, length , chan_in
        batch, length, _ = input.shape
        k_size, _, chan_out = self._parameters.shape

        offset = length - k_size + 1
        # dimension k_size, batch, index, chan_in
        Map = np.array([input[:, k:offset+k:self.stride, :] for k in np.arange(k_size)])
        self._gradient += np.einsum('inlj,nlc->ijc', Map, delta)/batch
        
    def backward_delta(self, input, delta):
        # delta: batch, length_out , chan_out
        # input: batch, length , chan_in
        batch, length, chan = input.shape
        k_size, c_in, c_out = self._parameters.shape
        offset = length - k_size + 1
        # dimension self.k_size, batch, index, chan
        res = np.zeros(input.shape)
        temp = np.einsum('nlo,kio->knli', delta, self._parameters)
        for k in range(k_size):
            res[:, k:offset+k:self.stride, :] += temp[k]
        return res




class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super(MaxPool1D, self).__init__()
        self.stride = stride
        self.k_size = k_size

    def forward(self, X):
        assert X.ndim == 3
        _, length, _ = X.shape

        offset = length - self.k_size + 1
        # dimension k_size, batch, index, chan_in
        M = [X[:, cx:offset+cx:self.stride, :] for cx in np.arange(self.k_size)]
        #M = X[:, np.arange(self.k_size)[:, None] + np.arange(offset)[None, :] * self.stride, :]
        return np.max(M, axis=0)

    def backward_delta(self, input, delta):
        assert len(input.shape) == 3
        batch, length, chan = input.shape
        offset = length - self.k_size + 1
        # dimension k_size, batch, l_out, c_in
        M = np.asarray([input[:, cx:offset+cx:self.stride, :] for cx in np.arange(self.k_size)])
        # batch, l_out, c_int
        idx = np.transpose(np.argmax(M, axis = 0), (0,2,1)) # batch, c_in, length_out
        row = np.arange(idx.shape[2]) # les indices
        idx = idx*self.stride + row # calcule les coordonnées des maximums
        idx =  np.transpose(idx, (0,2,1))
        #print("\nidx: ",idx)
        idx = idx.flatten()
        nb_move = delta.shape[1]
        res = np.zeros(input.shape)
        
        res[np.repeat(range(batch),chan*nb_move),idx,list(range(chan))*nb_move*batch] = delta.flatten()
        return res
    
    def update_parameters(self, gradient_step=0.001):
        pass




class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.ndim == 3 or x.ndim == 4
        self._input_cache = x
        if x.ndim == 3:
            batch_size, length, chan_in = x.shape
            return x.reshape(batch_size, length * chan_in)
        else:
            batch, h, w, chan_in = x.shape
            h = max(h,1)
            w = max(w,1)
            return np.reshape(x, (batch, h*w*chan_in))

    def backward_update_gradient(self, input, delta):
        pass  # No parameters to update

    def backward_delta(self, input, delta):
        if self._input_cache.ndim == 3:
            batch_size, length, chan_in = input.shape
            return delta.reshape(batch_size, length, chan_in)
        else:
            return np.reshape(delta, self._input_cache.shape)
    

    def update_parameters(self, gradient_step=0.001):
        pass





# class Flatten(Module):
#     def forward(self, X):
#         ## Calcule la passe forward
#         d = len(X.shape)
#         assert d==3 or d==4
#         self.shape = X.shape
#         if d == 3:
#             batch, length, chan_in = X.shape
#             return np.reshape(X, (batch, length*chan_in))
#         else:
#             batch, h, w, chan_in = X.shape
#             h = max(h,1)
#             w = max(w,1)
#             return np.reshape(X, (batch, h*w*chan_in))

#     def backward_delta(self, input, delta):
#         ## Deflat delta
#         return np.reshape(delta, self.shape)
    
#     def update_parameters(self, gradient_step=1e-3):
#         pass