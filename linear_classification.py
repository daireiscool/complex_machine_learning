"""
Methods to perform linear classification.

I have a Complex method, and a Real method to compare against.

@author: Dáire Campbell <daireiscool@gmail.com>

Notes:
    - The process is very slow.
        Cannot perform gradient descent, so must perform Random Search Algorithim.
        Is there a stocastic version?
"""
import math
import numpy as np
import random
import tqdm

class ComplexLinearClassification():
    """
    Method to apply linear classification using complex coefficients.
    
    Activation function is sign(atan(z)).
    
    Using a manual Random Search Algorithim.
    
    """
    def __init__(
        self, 
        alpha = 1e-5,
        epoch = 1,
        tol = 1e-3,
        random_state = 0,
        n_iter = 10000,
        verbose = False
    ):
        """
        Initialising function.
        
        ::param alpha: (float) default = 1e-5
        ::param epoch: (int) default = 1
        ::param tol: (float) default = 1e-3
        ::random_state: (int) default = 0
        ::param n_iter: (int) default 10000
        ::param verbose: (boolean), default = False
        """
        self.alpha = alpha
        self.epoch = epoch
        self.tol = tol
        random.seed(random_state)
        self.loss = []
        self.weights_history = []
        self.n_iter = n_iter
        self.verbose = verbose
        self.weights = []

    def _print(self, *args):
        """
        Function to print, if verbose = True
        """
        if self.verbose:
            print(*args)


    def complex_weights(self, n):
        """
        Function get initial weights
        
        ::param n: (int)
        ::return: (numpy array[complex])
        """
        return np.array([
            random.randint(-10, 10) +
            random.randint(-10, 10)*1j
            for i in list(range(n))])

    def sigmoid(self, z):
        """
        Function to apply sigmoid function.
        
        ::param z: (complex)
        ::return: (boolean)
        """
        return 1/(1+np.exp(-1*z.imag/z.real))
    
    def activation(self, z):
        """
        Function to convert a complex number to a boolean.
        
        ::param z: (complex)
        ::return: (boolean)
        """
        return True if (z.imag/z.real) > 0 else False

    def error(self, y, y_pred):
        """
        Function to get error between y and prediction.
        
        ::param y: (numpy array)
        ::param y_pred: (numpy array)
        """
        return sum((y*1 - y_pred)**2)/len(y)

    def random_search_algorithim(self, X, y):
        """
        Random Search Algorithim for complex numbers polynomial.
        
        ::param X: (numpy ndarray)
        ::param y: (numpy array)
        ::param weights: (numpy array)
        """
        weights = self.weights
        
        dim = list(range(X.shape[1]))
        random.shuffle(dim)
        
        for i in dim:
            a = np.zeros((X.shape[1],)); 
            a[i] = 1
            
            temp_weights = weights + a*(0 + 0 * 1j)
            loss = self.error(y, self._predict(X, temp_weights))
            self._print(f"Initial Loss: {loss}")
            
            weights_alpha = [
                a*(0 + 1 * 1j) * self.alpha,
                a*(0 - 1 * 1j) * self.alpha,
                a*(1 + 0 * 1j) * self.alpha,
                a*(1 + 1 * 1j) * self.alpha,
                a*(1 - 1 * 1j) * self.alpha,
                a*(-1 + 0 * 1j) * self.alpha,
                a*(-1 + 1 * 1j) * self.alpha,
                a*(-1 - 1 * 1j) * self.alpha]
        
            for w in weights_alpha:
                if self.error(y, self._predict(X, temp_weights+w)) < loss:
                    temp_weights += w
                    self.weights_history += [temp_weights]
                    loss = self.error(y, self._predict(X, temp_weights))
                    self._print(f"Updated Loss: {loss}")
            
            self.loss += [loss]
            self.weights = temp_weights

    def fit(self, X, y):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param y: (numpy array)
        ::return: (complex)
        """
        X = np.c_[X, np.ones(len(X))]
        self.weights = self.complex_weights(X.shape[1])
        self._print("Initial weights: ", self.weights)
        
        for i in tqdm.tqdm(range(self.n_iter)):
            self.random_search_algorithim(X, y)

    def _predict(self, X, weights):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param training: (boolean)
        ::param weights: (numpy array)
        ::return: (complex)
        """
        return np.array([self.sigmoid(i) for i in X.dot(weights)])

    def predict(self, X):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param training: (boolean)
        ::param weights: (numpy array)
        ::return: (complex)
        """
        X = np.c_[X, np.ones(len(X))]
        weights = self.weights
        return np.array([self.activation(i) for i in X.dot(self.weights)])


class LinearClassification():
    """
    Method to apply linear classification using real coefficients.
    
    Activation function is sign(atan(z)).
    
    Using a manual Random Search Algorithim.
    
    """
    def __init__(
        self, 
        alpha = 1e-5,
        epoch = 1,
        tol = 1e-3,
        random_state = 0,
        n_iter = 10000,
        verbose = False
    ):
        """
        Initialising function.
        
        ::param alpha: (float) default = 1e-5
        ::param epoch: (int) default = 1
        ::param tol: (float) default = 1e-3
        ::random_state: (int) default = 0
        ::param n_iter: (int) default 10000
        ::param verbose: (boolean), default = False
        """
        self.alpha = alpha
        self.epoch = epoch
        self.tol = tol
        random.seed(random_state)
        self.loss = []
        self.weights_history = []
        self.n_iter = n_iter
        self.verbose = verbose
        self.weights = []

    def _print(self, *args):
        """
        Function to print, if verbose = True
        """
        if self.verbose:
            print(*args)


    def _weights(self, n):
        """
        Function get initial weights
        
        ::param n: (int)
        ::return: (numpy array[complex])
        """
        return np.array([
            random.randint(-10, 10)
            for i in list(range(n))])

    def sigmoid(self, z):
        """
        Function to apply sigmoid function.
        
        ::param z: (complex)
        ::return: (boolean)
        """
        return 1/(1+np.exp(-1*z))
    
    def activation(self, z):
        """
        Function to convert a complex number to a boolean.
        
        ::param z: (complex)
        ::return: (boolean)
        """
        return True if z > 0 else False

    def error(self, y, y_pred):
        """
        Function to get error between y and prediction.
        
        ::param y: (numpy array)
        ::param y_pred: (numpy array)
        """
        return sum((y*1 - y_pred)**2)/len(y)

    def random_search_algorithim(self, X, y):
        """
        Random Search Algorithim for complex numbers polynomial.
        
        ::param X: (numpy ndarray)
        ::param y: (numpy array)
        ::param weights: (numpy array)
        """
        temp_weights = self.weights
        
        dim = list(range(X.shape[1]))
        random.shuffle(dim)
        
        for i in dim:
            a = np.zeros((X.shape[1],)); 
            a[i] = 1
            
            loss = self.error(y, self._predict(X, temp_weights))
            self._print(f"Initial Loss: {loss}")
            weights_alpha = [
                a*(-1.0) * self.alpha,
                a*(1.0) * self.alpha]
        
            for w in weights_alpha:
                if self.error(y, self._predict(X, temp_weights+w)) < loss:

                    temp_weights = temp_weights + w
                    self.weights_history += [temp_weights]
                    loss = self.error(y, self._predict(X, temp_weights))
                    self._print(f"Updated Loss: {loss}")
            
            self.loss += [loss]
            self.weights = temp_weights

    def fit(self, X, y):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param y: (numpy array)
        ::return: (complex)
        """
        X = np.c_[X, np.ones(len(X))]
        self.weights = self._weights(X.shape[1])
        self._print("Initial weights: ", self.weights)
        
        for i in tqdm.tqdm(range(self.n_iter)):
            self.random_search_algorithim(X, y)

    def _predict(self, X, weights):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param training: (boolean)
        ::param weights: (numpy array)
        ::return: (complex)
        """
        return np.array([self.sigmoid(i) for i in X.dot(weights)])

    def predict(self, X):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param training: (boolean)
        ::param weights: (numpy array)
        ::return: (complex)
        """
        X = np.c_[X, np.ones(len(X))]
        weights = self.weights
        return np.array([self.activation(i) for i in X.dot(self.weights)])