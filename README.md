# ghl-estimator

Linear generalized Huber estimator compatible with scikit-learn. A detailed 
explanation of the underlying generalized Huber objective function can be found
[here](https://towardsdatascience.com/generalized-huber-regression-505afaff24c).


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
