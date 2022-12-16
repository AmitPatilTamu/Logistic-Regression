#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):

    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, self.k))
        self.n_features = n_features
        y = np.zeros((n_samples, self.k))
        y[np.arange(n_samples), labels.astype(int)] = 1

        for i in range(self.max_iter):
            w_sum = np.zeros((n_features, self.k))
            if batch_size <= n_samples:
                subset_batch_size = np.random.randint(n_samples, size=batch_size)
            else:
                subset_batch_size = np.linspace(0, n_samples, n_samples - 1)
            for j in subset_batch_size:
                w_sum += self.learning_rate * (-self._gradient(X[j], y[j]))
            w_sum = w_sum / n_samples
            self.W += w_sum
        return self
        ### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        dl_dak = self.softmax(_x) - _y
        _g = np.matmul(_x.reshape(self.n_features,1), dl_dak.reshape(1,self.k))
        return _g
        ### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

        ### YOUR CODE HERE
        XW = np.matmul(x, self.W)
        exp = np.exp(XW)
        softmax = exp/np.sum(exp)
        return softmax
        ### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        ### YOUR CODE HERE
        n_samples = X.shape[0]
        preds = np.zeros(n_samples)
        for i in range(n_samples):
            preds[i] = np.argmax(self.softmax(X[i]))
        return preds
        ### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        ### YOUR CODE HERE
        preds = self.predict(X)
        score = np.sum(preds == labels)/X.shape[0]
        return score
        ### END YOUR CODE

