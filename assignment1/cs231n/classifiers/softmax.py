import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    def softmax(X: np.ndarray):
        """Calculates a softmax function
        
        Args:
            X (2-D Array): Logit array of shape (N, C)
        
        Returns:
            2-D Array: Softmax output of shape (N, C)
        """
        X -= np.max(X)
        
        numerator = np.exp(X)
        denominator = np.sum(numerator, axis=1).reshape(-1, 1) + 1e-30
        
        return numerator / denominator
    
    
    def d_softmax(softmax, y_labels):
        """Returns a derivative softmax
        
        Args:
            softmax (2-D Array): (N, C)
            y_labels (1-D Array): (N,)
        
        Returns:
            2-D Array: Gradients of shape (N, C)
            
        Notes:        
            softmax(x_i) = np.exp(x_i) / np.sum(np.exp(x_i), axis=1)
        """
        N, C = softmax.shape
        
        y_onehot = np.zeros(softmax.shape)
        y_onehot[np.arange(N), y_labels] = 1
        
        grad = softmax - y_onehot
        
        return grad
        
    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################    
    logit = np.dot(X, W)
    N, C = logit.shape
    
    probs = softmax(logit)
    
    correct_probs = probs[np.arange(N), y]
    negative_log_probs = - np.log(correct_probs)
    
    loss += np.mean(negative_log_probs) + reg * np.sum(W**2)
    
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################
    # (N, C)
    d_out = d_softmax(probs, y)
    dW = np.dot(X.T, d_out)
    dW /= N
    
    dW += 2 * reg * W
    
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    pass
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW

softmax_loss_vectorized = softmax_loss_naive
