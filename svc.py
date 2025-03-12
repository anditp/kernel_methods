import numpy as np
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
from kernels import WeightedDegreeKernel

#%%



class KernelSVC:
    """
    This implementation of KernelSVC is specifically designed for the 
    SpectrumKernel class used in the final submission.
    """
    
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
        N = len(y)
        
        K = self.kernel(X)
        
        yK = np.outer(y, y) * K

        # Lagrange dual problem
        def loss(alpha):
            return 0.5 * alpha.T @ yK @ alpha  - np.sum(alpha)

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return yK @ alpha - np.ones(N)


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: np.sum(y * alpha)
        jac_eq = lambda alpha: y
        fun_ineq = lambda alpha: np.hstack((alpha, -alpha + self.C))
        jac_ineq = lambda alpha: np.vstack((np.eye(N), -np.eye(N)))
        
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})
        
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints,
                                   options = {'maxiter': 500, 'disp': True})
        self.alpha = optRes.x


        support_index = np.argwhere(~np.isclose(self.alpha, 0)).squeeze()
        margin_index = np.intersect1d(support_index, np.argwhere(~np.isclose(self.alpha, self.C)))
        
        if isinstance(X, list):
            margin_points = [X[i] for i in margin_index]
            self.support = [X[i] for i in support_index]
        else:
            margin_points = X[margin_index]
            self.support = X[support_index]
        
        self.alpha_support = self.alpha[support_index]
        self.y_support = y[support_index]
        
        self.kernel.fit_precompute(self.support, support_index)
        
        self.b = np.mean(y[margin_index] - self.separating_function(margin_points))
        self.norm_f = 2 * (loss(self.alpha) + np.sum(self.alpha))


    ### Implementation of the separting function $f$ 
    def separating_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(self.support, x)
        return (self.y_support * self.alpha_support).T @ K
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1
    
    
    
    
    
    