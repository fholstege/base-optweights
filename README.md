# Estimating optimal weights in the presence of sub-population shift
This package implements the optimization procedure suggested in the paper _Optimizing importance weighting in the presence of sub-population shifts_. You can obtain weights for any dataset, in combination with an skLearn LogisticRegression model (see below).

You can obtain optimised weights for any problems for which you have the following:
- Groups (_g_)
- Independent variables (_X_)
- Outcome variable (_y_)




## Using optimal weights with SKLearn 

```python 

# import the weight_searcher object
from optweights.weight_searcher import WeightSearcher

# import the logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# import numpy
import numpy as np

# create some arbitrary data
n, d, k = 2000, 100, 2
X, y = make_classification(
    n_samples=n,
    n_features=d,
    n_classes=k,
    random_state=42,
)
g = np.random.binomial(1, 0.5, size=n) + 1
y, g = y.reshape(-1, 1), g.reshape(-1, 1)

# make a train/validation split for the data
n_train = int(n * 0.8)
X_train, y_train, g_train = X[:n_train], y[:n_train], g[:n_train]
X_val, y_val, g_val = X[n_train:], y[n_train:], g[n_train:]

# create a logistic regression model
model_param  = {'max_iter': 100,
                'penalty': 'l1',
                'C': 1,
                'solver': 'liblinear',
                'tol': 1e-4,
                'verbose': 0,
                'random_state': 0,
                'fit_intercept': True, 
                'warm_start': False}
logreg = LogisticRegression(**model_param)


# Define the probability of each group in the distribution of interest
# This is a case where we have two groups, and each group is given equal weight
p_ood = {1: 0.5, 2: 0.5}

# create a weight searcher object
ws = WeightSearcher(X_train, y_train, g_train, X_val, y_val, g_val, # define the X, y, g for both train/val
                        p_ood=p_ood,                                 # define the distribution of interest
                        sklearn_model=logreg                         # define the sklearn model (optional)
                     )

# define the arguments for the optimization
T = 100             # the number of steps
lr = 0.1            # the learning rate
momentum = 0.5      # the momentum parameter - higher is more momentum

# optimize the weights
p_hat =  ws.optimize_weights(T,  lr,  momentum)

# get the weights for the training set - these can then be used subsequently for an estimator. 
w_train = ws.return_weights(p_hat, g_train)
```




# Installation
Python 3.9 or later is required. 


```console
pip install optweights

``` 



