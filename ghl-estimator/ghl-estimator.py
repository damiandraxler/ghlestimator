import numpy as np
from scipy.optimize import minimize

def sgn(x):
    sig = np.sign(x)
    sig[sig == 0] = 1
    return sig

_log = lambda x: sgn(x) * np.log(1 + np.abs(x))
_loginv = lambda x: sgn(x) * (np.exp(np.abs(x)) - 1)
_loginvp = lambda x: np.exp(np.abs(x))

def _generalized_huber_loss_and_gradient(w, X, y, epsilon, link_dict):
    
        n_features = X.shape[1]
        fit_intercept = (n_features + 1 == w.shape[0])
        
        if fit_intercept:
            X = np.append(np.ones(len(y)).reshape(-1,1),X,axis=1)
    
        yhat = np.dot(X,w)
    
        g = link_dict['g']
        ginv = link_dict['ginv']
        ginvp = link_dict['ginvp']
    
        diff = g(y) - yhat
        absdiff = np.abs(diff)
    
        bool1_l = ((absdiff <= epsilon) & (diff < 0))
        bool1_r = ((absdiff <= epsilon) & (diff >= 0))
        bool2_l = ((absdiff > epsilon) & (diff < 0))
        bool2_r = ((absdiff > epsilon) & (diff >= 0))
    
        grad = np.zeros(len(y))
        loss = np.zeros(len(y))
    
        A = np.zeros(len(y))
        Ap = np.zeros(len(y))
        B = np.zeros(len(y))
    
        A[bool1_l] = ginv(yhat[bool1_l] - epsilon) - ginv(yhat[bool1_l])
        A[bool1_r] = ginv(yhat[bool1_r] + epsilon) - ginv(yhat[bool1_r])
        Ap[bool1_l] = ginvp(yhat[bool1_l] - epsilon) - ginvp(yhat[bool1_l])
        Ap[bool1_r] = ginvp(yhat[bool1_r] + epsilon) - ginvp(yhat[bool1_r])
    
        A[bool2_l] = ginv(yhat[bool2_l] - epsilon) - ginv(yhat[bool2_l])
        A[bool2_r] = ginv(yhat[bool2_r] + epsilon) - ginv(yhat[bool2_r])
        Ap[bool2_l] = ginvp(yhat[bool2_l] - epsilon) - ginvp(yhat[bool2_l])
        Ap[bool2_r] = ginvp(yhat[bool2_r] + epsilon) - ginvp(yhat[bool2_r])
    
        B[bool1_l] = y[bool1_l] - ginv(g(y[bool1_l]) + epsilon)
        B[bool1_r] = y[bool1_r] - ginv(g(y[bool1_r]) - epsilon)
        
        B[bool2_l] = y[bool2_l] - ginv(g(y[bool2_l]) + epsilon)
        B[bool2_r] = y[bool2_r] - ginv(g(y[bool2_r]) - epsilon)    
    
        # loss computation
        
        loss[bool1_l] = (y[bool1_l]-ginv(yhat[bool1_l]))**2 * \
                        (1/np.abs(A[bool1_l]) + 1/np.abs(B[bool1_l]))
        loss[bool1_r] = (y[bool1_r]-ginv(yhat[bool1_r]))**2 * \
                        (1/np.abs(A[bool1_r]) + 1/np.abs(B[bool1_r]))
        
        loss[bool2_l] = 4*np.abs(y[bool2_l] - ginv(yhat[bool2_l])) - \
                        (np.abs(A[bool2_l]) + np.abs(B[bool2_l]))
        loss[bool2_r] = 4*np.abs(y[bool2_r] - ginv(yhat[bool2_r])) - \
                        (np.abs(A[bool2_r]) + np.abs(B[bool2_r]))
        
        loss = np.sum(loss)
        
        # gradient computation
        
        grad[bool1_l] = -2*(y[bool1_l]-ginv(yhat[bool1_l]))*ginvp(yhat[bool1_l]) * \
                        (1/np.abs(A[bool1_l]) + 1/np.abs(B[bool1_l])) - \
                        (y[bool1_l]-ginv(yhat[bool1_l]))**2 * \
                        (1/(np.abs(A[bool1_l])**2))*sgn(A[bool1_l])*Ap[bool1_l]
    
        grad[bool1_r] = -2*(y[bool1_r]-ginv(yhat[bool1_r]))*ginvp(yhat[bool1_r]) * \
                        (1/np.abs(A[bool1_r]) + 1/np.abs(B[bool1_r])) - \
                        (y[bool1_r]-ginv(yhat[bool1_r]))**2 * \
                        (1/(np.abs(A[bool1_r])**2))*sgn(A[bool1_r])*Ap[bool1_r]    
    
        grad[bool2_l] = -4 * sgn(y[bool2_l] - ginv(yhat[bool2_l])) * ginvp(
                        yhat[bool2_l]) - sgn(A[bool2_l]) * Ap[bool2_l]
    
        grad[bool2_r] = -4 * sgn(y[bool2_r] - ginv(yhat[bool2_r])) * ginvp(
                        yhat[bool2_r]) - sgn(A[bool2_r]) * Ap[bool2_r]    
                
        grad = np.dot(grad.reshape(1,-1),X)
    
        del A ,Ap ,B ,bool1_l ,bool1_r ,bool2_l ,bool2_r 
        print(loss)
        
        return loss , grad

class GeneralizedHuberRegressor():
    
    def __init__(self,epsilon=1.0,max_iter=100,tol=1e-5, scale=10,
                 fit_intercept=True, link_dict={'g':_log,'ginv':_loginv,'ginvp':_loginvp}):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale        
        self.fit_intercept = fit_intercept
        self.link_dict = link_dict

    def fit(self, X, y):
        
        if self.fit_intercept:
            parameters = np.zeros(X.shape[1] + 1)
        else:
            parameters = np.zeros(X.shape[1])
                
        opt_res = minimize(
                _generalized_huber_loss_and_gradient,parameters, method="L-BFGS-B", jac=True,
                args=(X/self.scale, y, self.epsilon, self.link_dict),
                options={"maxiter": self.max_iter, "gtol": self.tol, "iprint": -1})

        parameters = opt_res.x
        
        if opt_res.status == 2:
            raise ValueError("HuberRegressor convergence failed:"
                             " l-BFGS-b solver terminated with %s"
                             % opt_res.message)
        
        self.n_iter_ = opt_res.nit        
        if self.fit_intercept:
            self.intercept_ = parameters[0]
            self.coef_ = parameters[1:1+X.shape[1]]/self.scale       
        else:            
            self.intercept_ = 0.0        
            self.coef_ = parameters[0:X.shape[1]]/self.scale
        return self
    
    def predict(self, X, y=None):
        return self.link_dict['ginv'](np.dot(X,self.coef_) + self.intercept_)
