import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE
        self.assign_weights(np.zeros(n_features))
        for i in range(self.max_iter):
            w_sum = np.zeros(n_features)
            for j in range(n_samples):
                w_sum += self.learning_rate * (- self._gradient(X[j], y[j]))
            w_sum = w_sum / n_samples
            self.W += w_sum
        ### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))
        for i in range(self.max_iter):
            w_sum = np.zeros(n_features)
            if batch_size <= n_samples:
                subset_batch_size = np.random.randint(n_samples, size=batch_size)
            else:
                subset_batch_size = np.linspace(0, n_samples, n_samples - 1)
            for j in subset_batch_size:
                w_sum += self.learning_rate * (- self._gradient(X[j], y[j]))
            w_sum = w_sum / n_samples
            self.W += w_sum
        ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.assign_weights(np.zeros(n_features))
        for i in range(self.max_iter):
            j = np.random.randint(n_samples)
            w_sum = self.learning_rate * (- self._gradient(X[j], y[j]))/n_samples
            self.W += w_sum
        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        e_ywx = np.exp(-_y*np.dot(self.W, _x))
        _g = (e_ywx/(1+e_ywx))*(-_y)*_x
        return _g
        ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        ### YOUR CODE HERE
        preds = np.ones(X.shape[0])

        XW = np.matmul(X, self.W)  # array operation
        theta = 1 / (1 + np.exp(-XW))
        preds_proba[:, 0] = theta
        preds_proba[:, 1] = 1 - theta
        return preds_proba
        ### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        preds = np.ones(X.shape[0])
        XW = np.matmul(X,self.W) # array operation
        theta = 1/(1+np.exp(-XW))
        preds[theta<0.5] = -1

        return preds
        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE

        preds = self.predict(X)
        #array = np.array(y)
        #sub = np.subtract(preds, array.T)
        #score = np.count_nonzero(sub==0)/y.shape[0]
        score = np.sum(preds == y) / y.shape[0]
        return score
        ### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

