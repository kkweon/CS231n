import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MultiLayerConvNet(object):
    """
    (conv - relu - 2x2 max pool) * N -> (affine - relu)*M -> affine -> softmax
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32, 32], filter_size=[7,7,7],
                 hidden_dim=[100, 100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 use_batchnorm=False,
                 dtype=np.float32,
                 xavier_init=False
                ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.bn_params = dict()
        self.conv_layers = len(num_filters)
        self.fc_layers = len(hidden_dim)
        self.tot_layers = self.conv_layers + self.fc_layers + 1
        C, H, W = input_dim
        
        # W1 = randn
        # b1 = zeros
        # INPUT  = (N, C, H, W)
        # W1 = filters (Filters, Channels, WW, HH)
        # OUTPUT = (N, F, H_new_max_pool, W_new_max_pool)
        # W2 = (F, H_new, W_new,
        output_shape = None
        for i in xrange(self.tot_layers):
            idx = str(i+1)
            
            Wi = 'W' + idx
            bi = 'b' + idx
            
            if self.use_batchnorm:
                gammai = 'gamma' + idx
                betai = 'beta' + idx
            
            if i < self.conv_layers:
                # Convolution  Layers
                if i > 0:
                    C  = num_filters[i-1]
                    
                if xavier_init:
                    n = C * filter_size[i] * filter_size[i] + 1
                    weight_scale = np.sqrt(2./n)
                self.params[Wi] = np.random.randn(num_filters[i], C, filter_size[i], filter_size[i]) * weight_scale
                self.params[bi] = np.zeros(num_filters[i])

                if self.use_batchnorm:
                    self.params[gammai] = np.ones(num_filters[i])
                    self.params[betai]  = np.zeros(num_filters[i])

                # Output shape
                pad = (filter_size[i] - 1) / 2
                stride = 1
                H_new = 1 + (H + 2*pad - filter_size[i]) / stride # after conv layer, height
                W_new = 1 + (W + 2*pad - filter_size[i]) / stride # after conv layer, width
                H_new_max_pool = 1 + (H_new - 2) / 2           # 2 by 2 max pool w/ 2 strides
                W_new_max_pool = 1 + (W_new - 2) / 2           # 2 by 2 max pool w/ 2 strides

                output_shape = (num_filters[i], H_new_max_pool, W_new_max_pool)
                H = H_new_max_pool
                W = W_new_max_pool
            
            else:
                # Fully Connected Layers
                if i == self.tot_layers - 1:
                    # Final Layer
                    if xavier_init:
                        n = hidden_dim[-1] + num_classes
                        weight_scale = np.sqrt(2./n)
                    self.params[Wi] = np.random.randn(hidden_dim[-1], num_classes) * weight_scale
                    self.params[bi] = np.zeros(num_classes)
                else:
                    if i == self.conv_layers:
                        F, H, W = output_shape
                        prev_dim = F * H * W
                    else:
                        prev_dim = hidden_dim[i-self.conv_layers - 1]
                    next_dim = hidden_dim[i-self.conv_layers]
                    
                    if xavier_init:
                        n = prev_dim + next_dim
                        weight_scale = np.sqrt(2./n)
                    self.params[Wi] = np.random.randn(prev_dim, next_dim) * weight_scale
                    self.params[bi] = np.zeros(next_dim)

                    if self.use_batchnorm:
                        self.params[gammai] = np.ones(next_dim)
                        self.params[betai] = np.zeros(next_dim)                

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        
        if self.use_batchnorm:
            for i in xrange(self.conv_layers + self.fc_layers):
                self.bn_params[i] = {'mode': 'train'}            
            

    def loss(self, X, y=None):
        """
        """
        if y is None and self.use_batchnorm:
            for k, v in self.bn_params.iteritems():
                self.bn_params[k]['mode'] = 'test'
        if y is not None and self.use_batchnorm:
            for k, v in self.bn_params.iteritems():
                self.bn_params[k]['mode'] = 'train'
                
        ## Forward        
        all_cache = list()
        next_input = X
        for i in xrange(self.tot_layers):
            Wi, bi = self.params['W'+str(i+1)], self.params['b'+str(i+1)]
            
            if self.use_batchnorm and i < self.tot_layers - 1:
                gammai, betai = self.params['gamma' + str(i+1)], self.params['beta'+str(i+1)]
            
            if i < self.conv_layers:
                # Conv layer
                filter_size = Wi.shape[2]
                conv_param = {'stride':1,'pad':(filter_size-1)/2}
                pool_param = {'pool_height':2,'pool_width':2,'stride':2}
                if self.use_batchnorm:
                    next_input, cache = conv_relu_batchnorm_pool_forward(next_input, Wi, bi, conv_param, pool_param, 
                                                                         gammai, betai, self.bn_params.get(i, 'None'))
                else:
                    next_input, cache = conv_relu_pool_forward(next_input, Wi, bi, conv_param, pool_param)
                
            elif i < self.tot_layers - 1:
                # FC Layer
                if self.use_batchnorm:
                    next_input, cache = affine_batchnorm_relu_forward(next_input, Wi, bi, gammai, betai, self.bn_params[i])
                else:    
                    next_input, cache = affine_relu_forward(next_input, Wi, bi)
            else:
                # Final Layer
                scores, cache = affine_forward(next_input, Wi, bi)       
            
            # Save Cache
            all_cache.append(cache)
            
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        # regularization
        loss_reg = 0
        
        for i in xrange(self.tot_layers, 0, -1):
            Wi = 'W' + str(i)            
            bi = 'b' + str(i)
            if self.use_batchnorm and i < self.tot_layers:
                gammai = 'gamma' + str(i)
                betai  = 'beta' + str(i)
            loss_reg += 0.5 * self.reg * np.sum(np.square(self.params[Wi]))           
            cache = all_cache[i-1]
            
            if i == self.tot_layers:
                # Final Layers
                dout, grads[Wi], grads[bi] = affine_backward(dout, cache)
            
            elif i > self.conv_layers:                
                # FC Layer
                if self.use_batchnorm:
                    tmp = affine_batchnorm_relu_backward(dout, cache)
                    dout = tmp[0]
                    grads[Wi] = tmp[1]
                    grads[bi] = tmp[2]
                    grads[gammai] = tmp[3]
                    grads[betai] = tmp[4]
                    #dout, grads[Wi], grads[bi], grads[gammai], grads[betai] = affine_batchnorm_relu_backward(dout, cache)
                else:
                    dout, grads[Wi], grads[bi] = affine_relu_backward(dout, cache)
            else:
                # Conv Layer
                if self.use_batchnorm:
                    dout, grads[Wi], grads[bi], grads[gammai], grads[betai] = conv_relu_batchnorm_pool_backward(dout, cache)
                else:
                    dout, grads[Wi], grads[bi] = conv_relu_pool_backward(dout, cache)
            
            # regularization
            grads[Wi] += self.reg * self.params[Wi]
            
        loss += loss_reg
        return loss, grads
  
