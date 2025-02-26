# Estimating optimal weights in the presence of sub-population shift
This package implements the optimization procedure suggested in the paper _Optimizing importance weighting in the presence of sub-population shifts_. You can obtain weights for any dataset, in combination with an skLearn LogisticRegression model (see below). For now, it only supports a binary outcome variable. 

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




## Installation
Python 3.9.12 or later is required. 


```console
pip install optweights
``` 

# Reproducing the results from the paper

*Datasets*: The Waterbirds dataset is made up of the places dataset ([link](http://places.csail.mit.edu)) and Caltech-UCSD Birds-200-2011 (CUB) dataset ([link](https://www.vision.caltech.edu/datasets/cub_200_2011/)). The celebA dataset can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and the multiNLI dataset [here](https://gluebenchmark.com/tasks). In order to reproduce the results, save these in the respective folders, e.g. data/original/{dataset}. Then, run the set_{dataset}.py and clean_{dataset}.py files, which puts the datasets in data/Cleaned/{dataset}. After finetuning a model, save the embeddings in data/embeddings/{dataset} using the create_embeddings.py file. 

Below we give an example how to generate the results for a particular dataset (WB) and group-weighted ERM (GW-ERM) on the last-layer embeddings of a model trained with no early stopping, batch_size=32, no data augmentation, and a penalty strength of 100 for the logistic regression.

```console
python generate_result_standard.py \
    --dataset WB \
    --early_stopping False \
    --batch_size 32 \
    --data_augmentation False \
    --seed 1 \
    --penalty_strength 100 \
    --method GW-ERM 
```

Below we run the same method for the same embeddings, but with optimized weights. We use the standard parameters that were also used in the paper. 

```console
python generate_result_opt.py \
    --dataset WB \
    --early_stopping False \
    --batch_size 32 \
    --data_augmentation False \
    --seed 1 \
    --penalty_strength 100 \
    --method WS:GW-ERM \
    --T 200 \
    --lr 0.1 \
    --momentum 0.5 \
    --patience 200 
```







