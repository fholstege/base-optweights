# Estimating optimal weights in the presence of sub-population shift



This package implements the optimization procedure suggested in the paper _Optimizing importance weighting in the presence of sub-population shifts_




```python 

# import the weight_searcher object
from optweights import weigh_searcher

# import helper function 
from optweights.utils import get_p_dict

# import the logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression


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
ws = weight_searcher(X_train, y_train, g_train, X_val, y_val, g_val, # define the X, y, g for both train/val
                        p_ood=p_ood,                                 # define the distribution of interest
                        sklearn_model=logreg                         # define the sklearn model for which to optimize weights (optional)
                     )

# define the arguments for the optimization
T = 100             # the number of steps
lr = 0.1            # the learning rate
momentum = 0.5      # the momentum parameter - higher is more momentum

# optimize the weights
p_hat =  ws.optimize_weights(T,  lr,  momentum)

# get the weights for the training set
w_train = ws.return_weights(p_hat, g_train)

```




# Installation
Python 3.9 or later is required. 




