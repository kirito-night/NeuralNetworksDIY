from projet_etu import *

class Conv2D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1, init = 1e-1):
        assert stride > 0
        self.stride = stride
        self._parameters = np.random.randn(k_size, k_size, chan_in, chan_out)*init
        self._gradient = np.zeros(self._parameters.shape)

    def forward(self, X):
        assert len(X.shape) == 4
        assert X.shape[3] == self._parameters.shape[2], f"X.shape[3] = { X.shape[3]} != {self._parameters.shape[2]}"
        batch, h, w, _ = X.shape
        k_size, _, chan_in, chan_out = self._parameters.shape
        dh = h - k_size + 1
        dw = w - k_size + 1
        # dimension k_size, batch, index, chan_in
        M = [[X[:, i:dh+i:self.stride, j:dw+j:self.stride] for j in range(k_size)] for i in range(k_size)]
        R = np.einsum('abncdi,abio->ncdo', M, self._parameters)
        return R
        
    def backward_update_gradient(self, input, delta):
        # delta: batch, length_out , chan_out
        # input: batch, length , chan_in
        batch, h, w, _ = input.shape
        k_size,_, chan_in, chan_out = self._parameters.shape
        dh = h - k_size + 1
        dw = w - k_size + 1
        # dimension k_size, batch, index, chan_in
        M = [[input[:, i:dh+i:self.stride, j:dw+j:self.stride,:] for j in range(k_size)] for i in range(k_size)]
        
        #for out in range(chan_out):  
        #    self._gradient[:,:,out] += (M*delta[:,:,out]).sum((2,3))/batch
        self._gradient += np.einsum('abncdi,ncdo->abio', M, delta)/batch
        
    def backward_delta(self, input, delta):
        # delta: batch, length_out , chan_out
        # input: batch, length , chan_in
        batch, h, w, _ = input.shape
        k_size,_, c_in, c_out = self._parameters.shape
        dh = h - k_size + 1
        dw = w - k_size + 1
        # dimension self.k_size, batch, index, chan
        R = np.zeros(input.shape)
        temp = np.einsum('ncdo,abio->abncdi', delta, self._parameters)
        for i in range(k_size):
            for j in range(k_size):
                R[:, i:dh+i:self.stride, j:dw+j:self.stride, :] += temp[i,j]
        return R

    
class Conv2D_T(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1, init = 1e-1):
        assert stride > 0
        self.stride = stride
        self._parameters = np.random.randn(k_size, k_size, chan_in, chan_out)*init
        self._gradient = np.zeros(self._parameters.shape)
        self._biais = np.random.randn(1, 1, 1, chan_out)*init
        self._biais_gradient = np.zeros(self._biais.shape)

    def forward(self, X):
        assert len(X.shape) == 4
        assert X.shape[3] == self._parameters.shape[2]
        batch, h, w, _ = X.shape
        k_size, _, chan_in, chan_out = self._parameters.shape
        stride = self.stride

        R = np.zeros((batch, k_size +h*stride-1, k_size+w*stride - 1, chan_out))
        temp = np.einsum("nxyi,abio->nxyabo",X, self._parameters)
        for i in range(h):
            for j in range(w):
                i_out = i*stride
                j_out = j*stride
                R[:,i_out:i_out+k_size, j_out:j_out+k_size, :] += temp[:,i,j,:]
        return R + self._biais
    
    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._gradient
        self._biais -= gradient_step*self._biais_gradient

        
    def backward_update_gradient(self, input, delta):
        batch, h, w, _ = input.shape
        k_size, _, chan_in, chan_out = self._parameters.shape
        stride = self.stride

        for i in range(h):
            for j in range(w):
                i_out = i*stride
                j_out = j*stride
                self._gradient += np.einsum("ni,nabo->abio", input[:,i,j,:], delta[:,i_out:i_out+k_size, j_out:j_out+k_size, :])
        self._gradient /= batch
        self._biais_gradient += np.mean(delta, axis=(0, 1, 2))

    def grad(self, input, delta):
        batch, h, w, _ = input.shape
        batch, ho, wo, _ = input.shape
        k_size, _, chan_in, chan_out = self._parameters.shape
        stride = self.stride

        # Reshape input and delta to avoid using einsum
        input_reshaped = input.reshape(batch*h*w, chan_in)
        delta_reshaped = delta.transpose(1, 2, 0, 3).reshape(ho*wo, batch*k_size*k_size, chan_out)

        # Compute gradient and bias gradient
        G= np.matmul(input_reshaped.T, delta_reshaped.reshape(h*w, batch*k_size*k_size*chan_out)).reshape(chan_in, k_size, k_size, chan_out)
        return G



        
    def backward_delta(self, input, delta):
        batch, h, w, _ = input.shape
        k_size,_, c_in, c_out = self._parameters.shape
        stride = self.stride

        R = np.zeros(input.shape)
        for i in range(h):
            for j in range(w):
                i_out = i*stride
                j_out = j*stride
                R[:, i,j, :] += np.einsum("nabo, abio->ni", delta[:,i_out:i_out+k_size, j_out:j_out+k_size, :], self._parameters)
        return R
    

class MaxPool2D(Module):
    def __init__(self, k_size, stride):
        super(MaxPool2D, self).__init__()
        self.stride = stride
        self.k_size = k_size

    def forward(self, X):
        assert len(X.shape) == 4
        batch, h, w, _ = X.shape

        k_size = self.k_size
        dh = h - k_size + 1
        dw = w - k_size + 1
        # dimension k_size, batch, index, chan_in
        M = [[X[:, i:dh+i:self.stride, j:dw+j:self.stride,:] for j in range(k_size)] for i in range(k_size)]
        R = np.max(M, axis=(0,1))
        return R

    def backward_delta(self, input, delta):
        assert len(input.shape) == 4
        batch, h, w, chan = input.shape
        _, ho, wo, co = delta.shape

        k_size = self.k_size
        dh = h - k_size + 1
        dw = w - k_size + 1
        stride = self.stride

        # dimension k_size, batch, index, chan_in
        M = [input[:,i:dh+i:stride, j:dw+j:stride]  for i in range(k_size) for j in range(k_size)]
        M = np.array(M)
        #print("input: \n",input)

        x,y = np.unravel_index(np.argmax(M, axis = 0), (k_size, k_size)) # position dans filtre
        row = np.arange(ho) # les indices
        col = np.arange(wo) # les indices


        # n,c,y,x -> n,x,y,c
        # n,c,x,y -> n,x,y,c
        x = (np.transpose(x, (0,3,2,1))*stride + row).transpose((0,3,2,1)).ravel()
        y = (np.transpose(y, (0,3,1,2))*stride + col).transpose((0,2,3,1)).ravel()

        res = np.zeros(input.shape)
        nb_move = ho*wo
        
        """
        print(np.repeat(range(batch),nb_move*chan).shape)
        print(x.shape)
        print(y.shape)
        print(len(list(range(chan))*nb_move*batch))"""
        #R = input[np.repeat(range(batch),nb_move*chan),x,y, list(range(chan))*nb_move*batch]

        np.add.at(res, (np.repeat(range(batch),nb_move*chan),x,y, list(range(chan))*nb_move*batch), delta.flatten())
        return res.reshape(input.shape)

    
    def update_parameters(self, gradient_step=0.001):
        pass


