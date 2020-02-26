import numpy as np
from scipy.optimize import minimize

def sgn(x):
    sig = np.sign(x)
    sig[sig == 0] = 1
    return sig

def _log(x): 
    return sgn(x) * np.log(1 + np.abs(x))

def _loginv(x):
    return sgn(x) * (np.exp(np.abs(x)) - 1)

def _loginvp(x): 
    return np.exp(np.abs(x))

def _generalized_huber_loss_and_gradient(w, X, y, epsilon, link_dict):
    """Returns the generalized Huber loss and the gradient.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Feature vector.
        w[:n_features] gives the coefficients if the intercept is not fit
        w[1:1+n_features] gives the coefficients and w[0] gives the intercept 
        if the intercept is fit.
    X : ndarray, shape (n_samples, n_features)
        Input data.
    y : ndarray, shape (n_samples,)
        Target vector.
    epsilon : float
        Parameter of the generalized Huber estimator.
    link_dict : dictionary
        Dictionary containing a link function 'g', it's inverse function 'ginv'
        and the derivative of the latter 'ginvp'. All three are callables of 
        the form fun(x) -> ndarray where both x and ndarray are 1-D arrays with
        shape (n_samples,).    
    Returns
    -------
    loss : float
        Generalized Huber loss.
    gradient : ndarray, shape (len(w))
        Returns the derivative of the generalized Huber loss with respect to 
        each coefficient and the intercept as a vector.
    """
    n_features = X.shape[1]
    fit_intercept = (n_features + 1 == w.shape[0])
    
    if fit_intercept:
        X = np.append(np.ones(len(y)).reshape(-1,1),X,axis=1)

    yhat = np.dot(X,w)
    
    # Define the link function (g), it's inverse (ginv) and the derivative of 
    # the latter (ginvp).
    g = link_dict['g']
    ginv = link_dict['ginv']
    ginvp = link_dict['ginvp']

    # Distinguish between values of abolut error smaller or larger than epsilon.
    # The distinction is done on the "link scale" defined by g(y).    
    diff = g(y) - yhat
    absdiff = np.abs(diff)

    bool1_l = ((absdiff <= epsilon) & (diff < 0))
    bool1_r = ((absdiff <= epsilon) & (diff >= 0))
    bool2_l = ((absdiff > epsilon) & (diff < 0))
    bool2_r = ((absdiff > epsilon) & (diff >= 0))

    # Compute the gradient and the loss.
    grad = np.zeros(len(y))
    loss = np.zeros(len(y))

    # Calculation of terms repeatedly needed in the loss and gradient computation.
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

    # loss calculation   
    loss[bool1_l] = (y[bool1_l]-ginv(yhat[bool1_l]))**2 * \
                    (1/np.abs(A[bool1_l]) + 1/np.abs(B[bool1_l]))
    loss[bool1_r] = (y[bool1_r]-ginv(yhat[bool1_r]))**2 * \
                    (1/np.abs(A[bool1_r]) + 1/np.abs(B[bool1_r]))
    
    loss[bool2_l] = 4*np.abs(y[bool2_l] - ginv(yhat[bool2_l])) - \
                    (np.abs(A[bool2_l]) + np.abs(B[bool2_l]))
    loss[bool2_r] = 4*np.abs(y[bool2_r] - ginv(yhat[bool2_r])) - \
                    (np.abs(A[bool2_r]) + np.abs(B[bool2_r]))
    
    loss = np.sum(loss)
    
    # gradient calculation    
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
    
    return loss , grad

class GeneralizedHuberRegressor():
    """Linear regression model that is robust to outliers and allows for a 
    link function.
    The Generalized Huber Regressor optimizes a term proportional to 
    ``(y - ginv(X'w/scale))**2`` for the samples where 
    ``|g(y) - (X'w/scale)| <= epsilon`` and a term proportional to 
    `|y - ginv(X'w/scale)|`` for the samples where 
    ``|(y - (X'w/scale))| > epsilon``, where w is to be optimized. 
    The parameter scale simply serves as a preconditioner to achieve numerical
    stability. Note that this does not take into account the fact that 
    the different features of X may be of different scales.
    Parameters
    ----------
    epsilon : float, default 1.0
        The parameter epsilon controls the number of samples that should be
        classified as outliers. 
    max_iter : int, default 100
        Maximum number of iterations that
        ``scipy.optimize.minimize(method="L-BFGS-B")`` should run for.
    fit_intercept : bool, default True
        Whether or not to fit the intercept. This can be set to False
        if the data is already centered around the origin.
    tol : float, default 1e-5
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n}`` <= ``tol``
        where pg_i is the i-th component of the projected gradient.
    scale : float, default 10.0
        Preconditioner for better numerical stability.
    link_dict : dictionary, default {'g':_log,'ginv':_loginv,'ginvp':_loginvp}         
    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Features got by optimizing the generalized Huber loss.
    intercept_ : float
        Bias.
    n_iter_ : int
        Number of iterations that
        ``scipy.optimize.minimize(method="L-BFGS-B")`` has run for.
        .. versionchanged:: 0.20
            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import HuberRegressor
    >>> from sklearn.datasets import make_regression
    >>> rng = np.random.RandomState(0)
    >>> X, y, coef = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
    >>> X[:4] = rng.uniform(10, 20, (4, 2))
    >>> y[:4] = rng.uniform(10, 20, 4)
    >>> ghuber = GeneralizedHuberRegressor().fit(X, y)
    >>> ghuber.score(X, y)
    0.094...
    >>> ghuber.predict(X[:1,])
    array([38.0325...])
    >>> huber = HuberRegressor().fit(X, y)
    >>> huber.score(X, y)
    -7.2846...
    >>> huber.predict(X[:1,])
    array([806.7200...])
    >>> print("True coefficients:", coef)
    True coefficients: [20.4923...  34.1698...]
    >>> print("Generalized Huber coefficients:", ghuber.coef_)
    Generalized Huber coefficients: [0.0468... 0.3324...]
    >>> print("Huber Regression coefficients:", huber.coef_)
    Huber Regression coefficients: [17.7906...  31.0106...]
    References
    ----------
    .. [1] Damian Draxler, 
    https://towardsdatascience.com/generalized-huber-regression-505afaff24c
    """    
    def __init__(self,epsilon=1.0,max_iter=100,tol=1e-5, scale=10,
                 fit_intercept=True, link_dict={'g':_log,'ginv':_loginv,'ginvp':_loginvp}):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale        
        self.fit_intercept = fit_intercept
        self.link_dict = link_dict

    def fit(self, X, y):
        
        if self.epsilon < 0.0:
            raise ValueError(
                "epsilon should be greater than or equal to 0.0, got %f"
                % self.epsilon)
        
        if len(X.shape)==1:
            raise ValueError("Expected 2D array, got 1D array instead:%s \n"            
                 "Reshape your data either using array.reshape(-1, 1) if your "
                 "data has a single feature or array.reshape(1, -1) if it " 
                 "contains a single sample."% X)
            
        if len(y.shape)==2:
            print("DataConversionWarning: A column-vector y was passed when "
                  "a 1d array was expected. Please change the shape of y to "
                  "(n_samples, ), for example using ravel().")
            y = y.ravel()
            
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

    def score(self, X, y):
        if len(y.shape)==2:
            y = y.ravel()    
        y_pred = self.predict(X,y)
        u = ((y - y_pred)**2).sum()
        v = ((y - y.mean())**2).sum()
        return (1 - u/v)