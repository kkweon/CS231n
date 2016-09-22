import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    (conv - relu)*2N - 2x2 max pool) * N -> (affine - relu)*M -> affine -> softmax
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[(3,3), (3,3,3), (3,3,3)], filter_size=[(3, 3), (3,3,3), (3,3,3)],\
                 hidden_dim=[100, 100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,\
                 use_batchnorm=False, dtype=np.float32, dropout = 0, dropout_conv=0, seed = None, xavier_init=False):
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
        self.num_filters = num_filters #[(3,3), (3,3), ...]
        self.fc_layers = len(hidden_dim)
        self.tot_layers = self.conv_layers + self.fc_layers + 1
        self.use_dropout = dropout > 0
        self.dropout = dropout
        self.dropout_conv = dropout_conv
        self.dropout_param = {}
        
        C, H, W = input_dim
        #assert len(num_filters) == len(filter_size) and len(num_filters.flatten()) == len(filter_size.flatten()), "Check F"
        
        # W1 = randn
        # b1 = zeros
        # INPUT  = (N, C, H, W)
        # W1 = filters (Filters, Channels, WW, HH)
        # OUTPUT = (N, F, H_new_max_pool, W_new_max_pool)
        # W2 = (F, H_new, W_new,
        output_shape = None
        
        # num_filters = [(64, 64), (128, 128), ...]
        # filter_size = [(3, 3), (3, 3), ...]
        for i, v in enumerate(zip(num_filters, filter_size)):
            # v = ((64, 64), (3, 3))
            F, FS = v[0], v[1] # F = (64, 64), FS = (3, 3)
            for j, k in enumerate(zip(F, FS)):
                # k = (64, 3)
                # idx = (1-1), (1-2), (2-1), ...
                idx = str(i + 1) + '-' + str(j + 1)
                number_of_filters = k[0]
                f_size = k[1]
                
                Wi = 'W' + idx # W1-1, W1-2, ...
                bi = 'b' + idx # b1-1, b1-2, ...
                
                if self.use_batchnorm:
                    gammai = 'gamma' + idx # gamma1-1, gamma1-2, ...
                    betai  = 'beta'  + idx # beta1-1, beta1-2, ...
                
                
                if i > 0:
                    # After First Conv Layer
                    if j == 0:
                        C = num_filters[i-1][-1]
                    else:
                        C = F[j-1]
                else:
                    # First Conv Layer
                    if j == 0:
                        # Very First
                        # So we ues C from X (N,C,H,W)
                        pass
                    else:
                        # Not First
                        # Use the previous from F = (64, 64)
                        C = F[j-1]
                
                if xavier_init:
                    # Since Wi = (Number of F, Previous Filters or C, FilterSize, FilterSize)
                    # Input Dimension is (Previous Filters or C) * FilterSize ** 2
                    n = C * np.square(f_size)
                    weight_scale = np.sqrt(2.0/n)
                    
                self.params[Wi] = np.random.randn(number_of_filters, C, f_size, f_size) * weight_scale
                #print "{} init with shape {}".format(Wi, self.params[Wi].shape)
                self.params[bi] = np.zeros(number_of_filters)
                # Output Shape will be
                # if input = (N, C, H, W), Wi = (F, C, FS, FS)
                # output = (N, F, H, W) because automatic padding
                # pooling = (N, F, H/2, W/2)
                
                if self.use_batchnorm:
                    #print "{} init with shape {}".format(gammai, number_of_filters)
                    #print "{} init with shape {}".format(betai, number_of_filters)
                    self.params[gammai] = np.ones(number_of_filters)
                    self.params[betai]  = np.zeros(number_of_filters)
                    self.bn_params[idx] = {'mode': 'train'}
                
                output_shape = None
                
                pad    = (f_size - 1)/2 # Auto Padding so output is same
                stride = 1              # Stride 1 for convention
                H_new = 1 + (H + 2*pad - f_size) / stride # after conv layer, height remains the same
                W_new = 1 + (W + 2*pad - f_size) / stride # after conv layer, width remains the same
                
                if j == len(F)-1:
                    # If at the end of conv layer we do pooling
                    H_new = 1 + (H_new - 2) / 2           # 2 by 2 max pool w/ 2 strides
                    W_new = 1 + (W_new - 2) / 2           # 2 by 2 max pool w/ 2 strides
                
                output_shape = (number_of_filters, H_new, W_new)
                H = H_new
                W = W_new
                
        for i in xrange(self.fc_layers+1):
            # fc_layers = [100, 100]
            # then there are W1(input, 100), W2(100, 100), W3(100, classes)
            # so we're adding one to it
            # i = 0, ..., self.fc_layers
            # self.conv_layers > 0
            # idx = 3+0, 3+1, 3+2, ... 
            idx = self.conv_layers + i
            Wi = 'W' + str(idx)
            bi = 'b' + str(idx)
            
            if self.use_batchnorm and i < self.fc_layers:
                # Except for the final layer, batch normalization
                gammai = 'gamma' + str(idx)
                betai  = 'beta'  + str(idx)
            
            if i == self.fc_layers:
                # Final Layer W3(100, num_classes)
                prev_dim = hidden_dim[-1]
                next_dim = num_classes
                
                if xavier_init:
                    n = prev_dim
                    weight_scale = np.sqrt(2.0/n)
                
                self.params[Wi] = np.random.randn(prev_dim, next_dim) * weight_scale
                self.params[bi] = np.zeros(next_dim)
            
            else:
                # If Not Final Layer
                if i==0:
                    # Very First Layer Of FC
                    # We get input from the final conv layer
                    F, H, W = output_shape
                    prev_dim = F*H*W
                else:
                    # if middle layer of FC
                    # previous dimension is just the previous hidden dimension
                    prev_dim = hidden_dim[i-1]
                    
                # Current Dimension which will be the next previous one    
                next_dim = hidden_dim[i]
                
                if xavier_init:
                    n = prev_dim
                    weight_scale = np.sqrt(2.0/n)
                    
                self.params[Wi] = np.random.randn(prev_dim, next_dim) * weight_scale
                self.params[bi] = np.zeros(next_dim)
                
                if self.use_batchnorm:
                    self.params[gammai] = np.ones(next_dim)
                    self.params[betai]  = np.zeros(next_dim)
                    self.bn_params[str(idx)] = {'mode': 'train'}
                                 
            #print "init {} shape of {}".format(Wi, self.params[Wi].shape)
            

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)        
               
       
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def loss(self, X, y=None):
        """
        Calculate Loss and Gradients
        
        Parameters
        ========================        
        X (np.array) : (N, C, H, W) shape
        y (np.array) : (N, 1) shape
        
        
        Returns
        ========================
        if test mode:
            Scores (np.array) : (N, Classes) after softmax
        else:
            Loss   (float)    : Loss + Regularization Loss
            Grads  (dict)     : grads['w1'], grads['w2'], ...
            
        """
        
        mode = 'test' if y is None else 'train'
        
        if self.use_batchnorm:
            for k, v in self.bn_params.iteritems():
                self.bn_params[k]['mode'] = mode
        
        if self.use_dropout:
            self.dropout_param['mode'] = mode
            
        # Propagation Preparation       
        all_cache = dict()        # saves the cache during forward prop and used during back prop
        pool_cache = dict()       # saves the cache related to max pool (which were max)
        if self.use_dropout:      
            dropout_cache = dict()# saves the dropout caches (which were dropped)
        next_input = X            # Initial Input is X (N,C,H,W)
              
        
        # Convolution Layers Forward Propagation        
        for i in xrange(self.conv_layers):
            # i = 0, ..., self.conv_layers - 1
            # we use 2x2 max pool same stride for efficiency
            pool_param = {'pool_height':2,'pool_width':2,'stride':2}
            
            # self.num_filters =[(3,3), (3,3), ...]
            # self.num_filters[0] = (3, 3) and it len = 2
            for j in xrange(len(self.num_filters[i])):
                # j = 0, 1
                # idx = (1-1), (1-2), ...
                # INPUT -> CONV -> (BATCHNORM) -> RELU -> DROPOUT -> MAXPOOL
                idx = str(i + 1) + '-' + str(j + 1)
                Wi = self.params['W' + idx]
                bi = self.params['b' + idx]
                
                #print "X: {}  W{} forward: {} | b{} forward: {}".format(next_input.shape, idx, Wi.shape, idx, bi.shape) 
                if self.use_batchnorm:
                    gammai = self.params['gamma' + idx]
                    betai = self.params['beta' + idx]
                
                # Wi = (F, prev F or C, FS, FS)
                filter_size = Wi.shape[2]
                conv_param = {'stride':1,'pad':(filter_size-1)/2}
                
                
                if self.use_batchnorm:
                    #print "gamma{}.shape: {}".format(idx,gammai.shape)
                    #print "beta{}.shape: {}".format(idx, betai.shape)
                    next_input, cache = conv_relu_batchnorm_forward(next_input, Wi, bi, conv_param, pool_param, 
                                                                    gammai, betai, self.bn_params.get(idx, 'None'))               
                else:
                    next_input, cache = conv_relu_forward(next_input, Wi, bi, conv_param)
                
                if self.use_dropout and self.dropout_conv > 0:
                    self.dropout_param['p'] = self.dropout_conv
                    next_input, d_cache = dropout_forward(next_input, self.dropout_param)
                    dropout_cache[idx] = d_cache
                    
                all_cache[idx] = cache
                
            next_input, p_cache = max_pool_forward_fast(next_input, pool_param)
            pool_cache[i] = p_cache
        
        
        # Fully Connected Layer Forward Propagation
        for i in xrange(self.fc_layers+1):
            # i = 0, ..., self.fc_layers
            # idx = (self.conv_layers), (self.conv_layers + self.fc_layers)
            idx = str(self.conv_layers + i)
            Wi = self.params['W' + idx]
            bi = self.params['b' + idx]
            
            if i == self.fc_layers:
                # if final layer of FC
                # we just proceed to the scores
                scores, cache = affine_forward(next_input, Wi, bi)
            
            else:
                if self.use_batchnorm:
                    gammai = self.params['gamma' + idx]
                    betai  = self.params['beta' + idx]

                if self.use_batchnorm:
                    next_input, cache = affine_batchnorm_relu_forward(next_input, Wi, bi, gammai, betai, self.bn_params[idx])
                else:
                    next_input, cache = affine_relu_forward(next_input, Wi, bi)

                if self.use_dropout:
                    self.dropout_param['p'] = self.dropout
                    next_input, d_cache = dropout_forward(next_input, self.dropout_param)
                    dropout_cache[idx] = d_cache
                
            # Save Cache
            all_cache[idx] = cache
            
        if y is None:
            return scores
        
        # self.params[k] = self.grads[k]
        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)
        # regularization init
        loss_reg = 0
       
        # Fully Connected Layers Backward
        for i in xrange(self.fc_layers, -1, -1):
            # i = (self.fc_layers, ..., 0)
            # idx = (self.conv_layers + self.fc_layers, ..., self.conv_layers)
            idx = str(self.conv_layers + i)
            Wi = 'W' + idx
            bi = 'b' + idx
            
            # If Not the Final Layer, we have batchnorm
            if self.use_batchnorm and i < self.fc_layers:
                gammai = 'gamma' + idx
                betai  = 'beta'  + idx
            
            # add the regularization
            loss_reg += 0.5 * self.reg * np.sum(np.square(self.params[Wi]))           
            # take the cache which were saved during the forward process
            cache = all_cache[idx]
            
            # if used dropout, except last layer
            # Forward: ( FC -> Relu -> Dropout ) -> FC -> Score
            # Backward: dout -> (dropout) -> relu -> (batchnorm) -> fc
            if self.use_dropout and i < self.fc_layers:
                d_cache = dropout_cache[idx]
            
            # if final layer then
            if i == self.fc_layers:
                # Final Layers: Softmax Loss -> FC
                dout, grads[Wi], grads[bi] = affine_backward(dout, cache)
            else:
                # dout -> (dropout) -> relu -> (batchnorm) -> fc
                if self.use_dropout:                    
                    dout = dropout_backward(dout, d_cache)
                    
                if self.use_batchnorm:   
                    # Relu -> Batchnorm -> FC
                    dout, grads[Wi], grads[bi], grads[gammai], grads[betai] = affine_batchnorm_relu_backward(dout, cache)
                else:
                    # Relu -> FC
                    dout, grads[Wi], grads[bi] = affine_relu_backward(dout, cache)
                    
            # Add regularization
            grads[Wi] += self.reg * self.params[Wi]
            
        # Convoluation Layer Backward    
        for i in xrange(self.conv_layers-1, -1, -1):
            # i = (self.conv_layers-1, ..., 0)
            p_cache = pool_cache[i]
            dout = max_pool_backward_fast(dout, p_cache)
            
            # self.num_filters = [(3,3), (3,3), ...]
            # self.num_filters[i] = (3,3) and its len is 2
            for j in xrange(len(self.num_filters[i])-1, -1, -1):
                # j = len((F,F)) - 1, ..., 0
                # idx = (self.conv-layers - len((F,F))), ..., (1-1)
                idx = str(i + 1) + '-' + str(j + 1)
                Wi = 'W' + idx
                bi = 'b' + idx
                
                loss_reg += 0.5 * self.reg * np.sum(np.square(self.params[Wi]))
                cache = all_cache[idx]
                
                # forward: conv -> (batchnorm) -> relu -> (dropout)
                # backward: (dropout) -> relu -> (batchnorm) -> conv
                
                if self.use_dropout and self.dropout_conv > 0:
                    self.dropout_param['p'] = self.dropout_conv
                    d_cache = dropout_cache[idx]
                    dout = dropout_backward(dout, d_cache)
                    
                if self.use_batchnorm:
                    gammai = 'gamma' + idx
                    betai  = 'beta' + idx                    
                    dout, grads[Wi], grads[bi], grads[gammai], grads[betai] = conv_relu_batchnorm_backward(dout, cache)
                else:
                    dout, grads[Wi], grads[bi] = conv_relu_backward(dout, cache)
                
                           
                grads[Wi] += self.reg * self.params[Wi]
            
        loss += loss_reg
        return loss, grads
  



class MultiLayerConvNet(object):
    """
    (conv - relu)*2N - 2x2 max pool) * N -> (affine - relu)*M -> affine -> softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 16, 16], filter_size=[3, 3, 3],\
                 hidden_dim=[100, 100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,\
                 use_batchnorm=False, dtype=np.float32, dropout = 0, seed = None, xavier_init=False):
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
        self.use_dropout = dropout > 0
        
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
                    n = C * filter_size[i] * filter_size[i] 
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
                H_new = 1 + (H_new - 2) / 2           # 2 by 2 max pool w/ 2 strides
                W_new = 1 + (W_new - 2) / 2           # 2 by 2 max pool w/ 2 strides

                output_shape = (num_filters[i], H_new, W_new)
                H = H_new
                W = W_new
            
            else:
                # Fully Connected Layers
                if i == self.tot_layers - 1:
                    # Final Layer
                    if xavier_init:
                        n = hidden_dim[-1]
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
                        n = prev_dim 
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
       
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def loss(self, X, y=None):
        """
        """
        mode = 'test' if y is None else 'train'
        
        if self.use_batchnorm:
            for k, v in self.bn_params.iteritems():
                self.bn_params[k]['mode'] = mode
        
        if self.use_dropout:
            self.dropout_param['mode'] = mode
            
        ## Forward        
        all_cache = list()
        if self.use_dropout:
            drop_caches = list()
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
                
                if self.use_dropout:
                    next_input, dropout_cache = dropout_forward(next_input, self.dropout_param)
                    drop_caches.append(dropout_cache)
                    
            elif i < self.tot_layers - 1:
                # FC Layer
                if self.use_batchnorm:
                    next_input, cache = affine_batchnorm_relu_forward(next_input, Wi, bi, gammai, betai, self.bn_params[i])
                else:    
                    next_input, cache = affine_relu_forward(next_input, Wi, bi)
                    
                if self.use_dropout:
                    next_input, dropout_cache = dropout_forward(next_input, self.dropout_param)
                    drop_caches.append(dropout_cache)
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
            
            if self.use_dropout and i < self.tot_layers:
                drop_cache = drop_caches[i-1]
                            
            if i == self.tot_layers:
                # Final Layers
                dout, grads[Wi], grads[bi] = affine_backward(dout, cache)
            
            elif i > self.conv_layers:                
                if self.use_dropout:
                    dout = dropout_backward(dout, drop_cache)
                # FC Layer
                if self.use_batchnorm:
                    dout, grads[Wi], grads[bi], grads[gammai], grads[betai] = affine_batchnorm_relu_backward(dout, cache)
                else:
                    dout, grads[Wi], grads[bi] = affine_relu_backward(dout, cache)
            else:
                # Conv Layer
                if self.use_dropout:
                    dout = dropout_backward(dout, drop_cache)
                if self.use_batchnorm:
                    dout, grads[Wi], grads[bi], grads[gammai], grads[betai] = conv_relu_batchnorm_pool_backward(dout, cache)
                else:
                    dout, grads[Wi], grads[bi] = conv_relu_pool_backward(dout, cache)
            
            # regularization
            grads[Wi] += self.reg * self.params[Wi]
            
        loss += loss_reg
        return loss, grads
  
