import numpy as np
from random import shuffle

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
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in xrange(X.shape[0]):
        first_step = X[i].dot(W).T
        first_step -= np.max(first_step) # for numerical stability
        second_step = np.exp(first_step)
        nps = np.sum(second_step)
        npsinv = 1/nps
        third_step = second_step * npsinv
        L_i = - np.log(third_step[y[i]])
        loss += L_i
        
        #dW D x C
        dW[:, y[i]] +=  (second_step[y[i]]/nps - 1) * X[i]
        for j in xrange(W.shape[1]):
            if j == y[i]:
                continue
            dW[:, j] += second_step[j] / nps * X[i]
    
    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += 0.5 * reg * np.sum(np.square(W))
    
    dW += reg * W
                                                              
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    XW = X.dot(W)
    XW -= np.max(XW)
    XWexp = np.exp(XW)
    nps = np.sum(XWexp, axis=1)
    true_scores = XWexp[xrange(X.shape[0]), y] / nps
    true_scores_mlog = -1 * np.log(true_scores)
    true_scores_mean = np.mean(true_scores_mlog)
    
    loss = true_scores_mean + 0.5 * reg * np.sum(W**2)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    grad = XWexp / nps.reshape(-1, 1)
    grad[xrange(grad.shape[0]), y] -= 1
    
    dW = X.T.dot(grad)
    
    dW /= X.shape[0]
    
    dW += reg * W

    return loss, dW

