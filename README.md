# ghl-estimator

Linear generalized Huber estimator compatible with scikit-learn. A detailed 
explanation of the underlying generalized Huber objective function can be found
[here](https://towardsdatascience.com/generalized-huber-regression-505afaff24c) 
and in [1].

[1]: https://towardsdatascience.com/generalized-huber-regression-505afaff24c

Linear regression model that is robust to outliers allows for a 
link function and is compatible with scikit-learn.
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
----------
.. [1] Damian Draxler, 
https://towardsdatascience.com/generalized-huber-regression-505afaff24c


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ghl-estimator.

```bash
pip install ghl-estimator
```

## Usage

```python
import ghl-estimator as ghl

ghl = GeneralizedHuberRegressor() # initializes default ghl estimator 
ghl.fit(X, y) # fit on features X and target y
ghl.score(X, y) # compute the R^2 score
ghl.predict(X) # make pedictions
``` 

## License
[MIT](https://choosealicense.com/licenses/mit/)
