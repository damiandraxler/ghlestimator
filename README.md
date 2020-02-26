# ghlestimator

Linear Generalized Huber Regressor compatible with scikit-learn. A detailed 
explanation of the underlying generalized Huber objective function can be found
[here](https://towardsdatascience.com/generalized-huber-regression-505afaff24c).

The Generalized Huber Regressor depends on the definition of a invertible link function ``g`` and optimizes a term proportional to 
``(y - ginv(X'w))**2`` for samples where 
``|g(y) - (X'w)| <= epsilon`` and a term proportional to 
``|y - ginv(X'w)|`` for samples where 
``|g(y) - (X'w)| > epsilon``, where w is to be optimized and ``ginv`` denotes the inverse of ``g``.  

Parameters
----------

    class GeneralizedHuberRegressor(epsilon=1.0,max_iter=100,tol=1e-5, scale=10,
    fit_intercept=True, link_dict={'g':_log,'ginv':_loginv,'ginvp':_loginvp})

**epsilon : float, default 1.0**

    The parameter epsilon defines the crossover between the rmse type of loss 
    and the mae type of loss.  
**max_iter : int, default 100**

    Maximum number of iterations that
    scipy.optimize.minimize(method="L-BFGS-B") should run for.
**fit_intercept : bool, default True**

    Whether or not to fit the intercept.
**tol : float, default 1e-5**

    The iteration will stop when max{|proj g_i | i = 1, ..., n} <= tol
    where pg_i is the i-th component of the projected gradient.
**scale : float, default 10.0**

    Preconditioner for better numerical stability. Input array is internally 
    divided by scale.
**link_dict : dictionary, default {'g':_log,'ginv':_loginv,'ginvp':_loginvp}**

    The link function 'g', it's inverse 'ginv' and the derivative of the 
    latter 'ginvp' have to be specified as callables. 
    The default link function is g(x) = sign(x)log(1+|x|).              
Attributes
----------
**coef_ : array, shape (n_features,)**

    Fitted coefficients got by optimizing the generalized Huber loss.
**intercept_ : float**
    
    The bias.
**n_iter_ : int**
    
    Number of iterations that
    scipy.optimize.minimize(method="L-BFGS-B") has run for.
Methods
----------
**fit(self, X, y)**

    Fit the model to the given training features X and target y both given as 
    ndarrays.

**predict(self, X)**

    Predict using the fitted linear model.

**score(self, X, y)**

    Return the coefficient of determination R^2 of the prediction.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ghlestimator.

```bash
pip install ghlestimator
```

## Usage

```python
import ghlestimator as ghl

ghl = GeneralizedHuberRegressor() # initializes default ghl estimator 
ghl.fit(X, y) # fit on features X and target y
ghl.score(X, y) # compute the R^2 score
ghl.predict(X) # make pedictions
``` 

## License
[MIT](https://choosealicense.com/licenses/mit/)