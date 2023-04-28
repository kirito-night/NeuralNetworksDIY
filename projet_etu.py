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
            print("Erreur de dimension")
            print("X.shape[1] = ", X.shape[1])
            print("self._parameters.shape[0] = ", self._parameters.shape[0])
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
    
    xtrain, ytrain = datax, datay

    Lerror = []
    Lscore = []
    for iteration in range(max_iter):
        print("Iteration: {0} / {1}".format(iteration, max_iter))
        # descente de gradient mini-batch
        for index in make_minibatch(batch_size, xtrain, ytrain):
            batch_x, batch_y = xtrain[index], ytrain[index] 
            optimisation.step(batch_x, batch_y)
        # calcule du cout
        cost = loss.forward(datay, net.forward(datax)).mean()
        print("cost " ,round(cost, 3))
        Lerror.append(cost)
        if predict != None:
            Lscore.append(np.mean(predict(xtest) == ytest))
            print("score " ,round(Lscore[-1], 3))
            print()
    return Lerror, Lscore




class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out , stride=1):
        """(k_size,chan_in,chan_out)"""
        super().__init__()
        self._k_size = k_size  # taille du filtre
        self._chan_in = chan_in  # C
        self._chan_out = chan_out  # nombre de filtres
        self._stride = stride
        self._parameters = np.random.rand(k_size, chan_in, chan_out)  # filtres

    def forward(self, X):
        """Performe une convolution en 1D sans boucles for.
        Parameters
        ----------
        X : ndarray (batch, length, chan_in)
        Returns
        -------
        ndarray (batch, (length-k_size)/stride + 1, chan_out)"""
        batch_size, length, chan_in = X.shape
        assert chan_in == self._chan_in, f"X must have {self._chan_in} channels. Here X have {chan_in} channels." 
        
        batch_stride, length_stride, chan_stride = X.strides

        out_size = int((length - self._k_size) / self._stride + 1)
        new_shape = (batch_size, out_size, chan_in, self._k_size)
        new_strides = (
            batch_stride,
            self._stride * length_stride,
            chan_stride,
            length_stride,
        )

        X_windows = np.lib.stride_tricks.as_strided(X, new_shape, new_strides)

        self.inputs = X, X_windows
        output = np.einsum("blck,kcf->blf", X_windows, self._parameters)
        return output

    def backward_update_gradient(self, input, delta):
        """TO DO"""
        batch, length, chan_in = input.shape
        assert chan_in == self._chan_in
        batch_size, out_size, chan_in, k_size = self.inputs[1].shape
        input_windows = np.lib.stride_tricks.as_strided(
            input, (batch_size, out_size, chan_in, k_size), self.inputs[1].strides
        )
        print(input_windows.shape)
        #self._gradient += np.einsum("blck,lcf->bkf", input_windows, delta) / batch

    def backward_delta(self, input, delta):
        """TO DO"""
        np.einsum("", delta, self._parameters)
        ...

    def forward_loops(self, X):
        """Performe une convolution en 1D avec des boucles for."""
        batch, length, chan_in = X.shape
        assert chan_in == self._chan_in

        # Initialize the output array
        out_size = int((length - self._k_size) / self._stride + 1)
        out = np.zeros((batch, out_size, self._chan_out))

        # Convolve for each batch element
        for b in range(batch):
            # Convolve for each output channel
            for c_out in range(self._chan_out):
                # Convolve for each position in the output
                for i in range(out_size):
                    # Compute the receptive field
                    start = i * self._stride
                    end = start + self._k_size

                    # Compute the convolution
                    out[b, i, c_out] = np.sum(
                        X[b, start:end, :] * self._parameters[:, :, c_out]
                    )

        return out




class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        """Performe un max pooling en 1D.
        Parameters
        ----------
        X : ndarray (batch, length, chan_in)
        Returns
        -------
        (batch, (length-k_size)/stride + 1, chan_in)
        """
        batch, length, chan_in = X.shape

        out_size = int((length - self._k_size) / self._stride + 1)
        out = np.zeros((batch, out_size, chan_in))

        for b in range(batch):
            for c_in in range(chan_in):
                for i in range(out_size):
                    start = i * self._stride
                    end = start + self._k_size
                    out[b, i, c_in] = np.max(X[b, start:end, :])

        return out

    def backward_update_gradient(self, input, delta):
        """TO DO"""
        ...

    def backward_delta(self, input, delta):
        """TO DO"""
        ...




class Flatten(Module):
    def forward(self, X):
        ## Calcule la passe forward
        assert len(X.shape) == 3
        batch, self.length, self.chan_in = X.shape
        return np.reshape(X, (batch, self.length*self.chan_in))

    def backward_delta(self, input, delta):
        ## Deflat delta
        return np.reshape(delta, (delta.shape[0], self.length,self.chan_in))
    
    def update_parameters(self, gradient_step=1e-3):
        pass