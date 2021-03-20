"""
Methods to perform linear regression.

I have a Complex method, and a Real method to compare against.

@author: Dáire Campbell <daireiscool@gmail.com>

Notes:
    - The process is very slow.
        Cannot perform gradient descent, so must perform Random Search Algorithim.
        Is there a stocastic version?
"""
import numpy as np
import random
import tqdm

class ComplexLinearRegression():
    """
    Method to apply linear regression using complex coefficients.
    
    Using a manual Random Search Algorithim.
    
    """
    def __init__(
        self, 
        alpha = 1e-5,
        epoch = 1,
        tol = 1e-3,
        random_state = 0,
        n_iter = 10000,
        verbose = False,
        stochastic = False,
        p = 0.1,
        maximum = 10,
        minimum = 0.001,
        decay = 0.995
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
        self.stochastic = stochastic
        self.p = p
        self.maximum = maximum
        self.minimum = minimum
        self.decay = decay

    def _print(self, *args):
        """
        Function to print, if verbose = True
        """
        if self.verbose:
            print(*args)

    def decaying_function(self, n):
        """
        Function to decay the learning rate.

        ::param n: (int)
        """
        return self.maximum*(self.decay**n) + self.minimum
            
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

    def error(self, y, y_pred):
        """
        Function to get error between y and prediction.
        
        ::param y: (numpy array)
        ::param y_pred: (numpy array)
        """
        return sum((y*1 - y_pred)**2)/len(y)

    def random_search_algorithim(self, X, y, n):
        """
        Random Search Algorithim for complex numbers polynomial.
        
        ::param X: (numpy ndarray)
        ::param y: (numpy array)
        ::param weights: (numpy array)
        """
        weights = self.weights
        
        dim = list(range(X.shape[1]))
        random.shuffle(dim)
        decay = self.decaying_function(n)

        for i in dim:
            a = np.zeros((X.shape[1],)); 
            a[i] = 1
            
            temp_weights = weights + a*(0 + 0 * 1j)
            loss = self.error(y, self._predict(X, temp_weights))
            self._print(f"Initial Loss: {loss}")
            
            weights_alpha = [(j[0]+j[1]*1j) for j in 
                np.transpose([np.tile([0,1,-1], 3), np.repeat([0,1,-1], 3)])]

            for w in weights_alpha:
                change = a*w*decay
                error = self.error(y, self._predict(X, temp_weights+change))
                if error < loss:
                    temp_weights += change
                    loss = error
                    self._print(f"Updated Loss: {loss}")
            
            self.weights_history += [temp_weights]
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
        
        for n in tqdm.tqdm(range(self.n_iter)):
            self.random_search_algorithim(X, y, n)

    def _predict(self, X, weights):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param training: (boolean)
        ::param weights: (numpy array)
        ::return: (complex)
        """
        z = X.dot(weights)
        return np.abs(z)*np.sign(z.real)

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
        z = X.dot(weights)
        return np.abs(z)*np.sign(z.real)

class LinearRegression():
    """
    Method to apply linear regression using real coefficients.
        
    Using a manual Random Search Algorithim.
    
    """
    def __init__(
        self, 
        alpha = 1e-5,
        epoch = 1,
        tol = 1e-3,
        random_state = 0,
        n_iter = 10000,
        verbose = False,
        stochastic = False,
        p = 0.1,
        maximum = 10,
        minimum = 0.001,
        decay = 0.995

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
        self.stochastic = stochastic
        self.p = p
        self.maximum = maximum
        self.minimum = minimum
        self.decay = decay


    def _print(self, *args):
        """
        Function to print, if verbose = True
        """
        if self.verbose:
            print(*args)

    def decaying_function(self, n):
        """
        Function to decay the learning rate.

        ::param n: (int)
        """
        return self.maximum*(self.decay**n) + self.minimum

    def _weights(self, n):
        """
        Function get initial weights
        
        ::param n: (int)
        ::return: (numpy array[complex])
        """
        return np.array([
            random.randint(-10, 10)
            for i in list(range(n))])\
        .astype(float)

    def error(self, y, y_pred):
        """
        Function to get error between y and prediction.
        
        ::param y: (numpy array)
        ::param y_pred: (numpy array)
        """
        return sum((y*1 - y_pred)**2)/len(y)

    def random_search_algorithim(self, X, y, n):
        """
        Random Search Algorithim for complex numbers polynomial.
        
        ::param X: (numpy ndarray)
        ::param y: (numpy array)
        ::param weights: (numpy array)
        """
        temp_weights = self.weights
        
        dim = list(range(X.shape[1]))
        random.shuffle(dim)
        decay = self.decaying_function(n)

        for i in dim:
            a = np.zeros((X.shape[1],)); 
            a[i] = 1
            
            loss = self.error(y, self._predict(X, temp_weights))
            self._print(f"Initial Loss: {loss}")
 
            weights_alpha = [1,-1]

            for w in weights_alpha:
                change = a*w*decay
                if self.error(y, self._predict(X, temp_weights+change)) < loss:

                    temp_weights += change
                    loss = self.error(y, self._predict(X, temp_weights))
                    self._print(f"Updated Loss: {loss}")
            
            self.weights_history.append(np.array(temp_weights))
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
        
        for n in tqdm.tqdm(range(self.n_iter)):
            self.random_search_algorithim(X, y, n)

    def _predict(self, X, weights):
        """
        Function to fit data to a complex linear function.
        
        ::param X: (numpy ndarray)
        ::param training: (boolean)
        ::param weights: (numpy array)
        ::return: (complex)
        """
        return X.dot(weights)

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
        return X.dot(weights)