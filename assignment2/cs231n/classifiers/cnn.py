import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                use_batchnorm=False,
               dtype=np.float32):
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
        C, H, W = input_dim
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # INPUT  = (N, C, H, W)
        # W1 = filters (Filters, Channels, WW, HH)
        # OUTPUT = (N, F, H_new_max_pool, W_new_max_pool)
        # W2 = (F, H_new, W_new, 
        for i in xrange(3):
            idx = str(i+1)
            if idx == '1':
                self.params['W' + idx] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
                self.params['b' + idx] = np.zeros(num_filters)
                if self.use_batchnorm:
                    self.params['gamma' + idx] = np.ones(num_filters)
                    self.params['beta' + idx] = np.zeros(num_filters)
            elif idx == '2':
                pad = (filter_size - 1) / 2
                stride = 1
                H_new = 1 + (H + 2*pad - filter_size) / stride # after conv layer, height
                W_new = 1 + (W + 2*pad - filter_size) / stride # after conv layer, width
                H_new_max_pool = 1 + (H_new - 2) / 2           # 2 by 2 max pool w/ 2 strides
                W_new_max_pool = 1 + (W_new - 2) / 2           # 2 by 2 max pool w/ 2 strides
                      
                self.params['W' + idx] = np.random.randn(num_filters * H_new_max_pool * W_new_max_pool, hidden_dim) * weight_scale
                self.params['b' + idx] = np.zeros(hidden_dim)
                
                #if self.use_batchnorm:
                #    self.params['gamma' + idx] = np.ones(C)
                #    self.params['beta' + idx] = np.zeros(C)
                    
            else:
                self.params['W' + idx] = np.random.randn(hidden_dim, num_classes) * weight_scale
                self.params['b' + idx] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        
        if self.use_batchnorm:
            self.bn_params['mode'] = 'train'
            
            

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        !! conv - relu - 2x2 max pool - affine - relu - affine - softmax !!
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if y is None and self.use_batchnorm:
            self.bn_params['mode'] = 'test'
        if y is not None and self.use_batchnorm:
            self.bn_params['mode'] = 'train'
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        if self.use_batchnorm:
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        out, cache1    = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param, self.use_batchnorm, gamma1, beta1, self.bn_params)
        out, cache2    = affine_relu_forward(out, W2, b2)
        scores, cache3 = affine_forward(out, W3, b3)
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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
        loss_reg = np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
        loss_reg *= self.reg * 0.5
        # add it to the loss
        loss += loss_reg
        # gradients
        dout, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache2)
        dout, grads['W1'], grads['b1'], grads_batch = conv_relu_pool_backward(dout, cache1, self.use_batchnorm)
        if self.use_batchnorm:
            grads['gamma1'], grads['beta1'] = grads_batch
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
  
