

# import the add_up function from optweights
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from optweights.weight_searcher import weight_searcher
from optweights.data import Toy


def test_weight_searcher_helpers():

    # create a dict with entries more than 1 or lower than 0
    p_dict = {1: 0.25, 2: 1.25, 3: -0.25, 4: 0.75}

    # number of datapoints to be created
    n_train = 1000
    n_val = 1000

    # set probability of group in training, save in dict
    ptr = 0.9
    pte = 0.5

    # set the other parameters
    beta_1 = 0
    beta_0 = 0
    sigma_1 = 1
    sigma_0 = 1
    gamma = 1
    a_1 = 1
    a_0 = 0
    d=10
    mu = np.zeros(d)


    # instantiate obj. for training and validation + test
    toy_data_tr = Toy(n=n_train, p=ptr, beta_1=beta_1, beta_0=beta_0, sigma_1=sigma_1, sigma_0=sigma_0,  mu=mu, gamma=gamma, a_0=a_1, a_1=a_0, d=d)
    toy_data_val = Toy(n=n_val, p=ptr, beta_1=beta_1, beta_0=beta_0, sigma_1=sigma_1, sigma_0=sigma_0,  mu=mu, gamma=gamma, a_0=a_1, a_1=a_0, d=d)
    toy_data_te = Toy(n=n_val, p=pte, beta_1=beta_1, beta_0=beta_0, sigma_1=sigma_1, sigma_0=sigma_0,  mu=mu, gamma=gamma, a_0=a_1, a_1=a_0, d=d)

    # create training, validation and test data
    X_train, y_train, g_train = toy_data_tr.dgp_mv(logistic=True)
    X_val, y_val, g_val = toy_data_val.dgp_mv(logistic=True)
    X_te, y_te, g_te = toy_data_te.dgp_mv(logistic=True)

    # create a logistic regression model
    model_param  = {'max_iter': 100,
                    'penalty': 'l1',
                    'C': 1,
                    'solver': 'liblinear',
                    'tol': 1e-4,
                    'verbose': 0,
                    'random_state': 0,
                    'fit_intercept': True}
                
    logreg = LogisticRegression(**model_param)
    p_ood = {1: 0.5, 2: 0.5}

    # create a weight searcher object
    # sklearn_model, X_train, y_train, g_train, X_val, y_val, g_val, p_ood=None, GDRO=False, weight_rounding=4, p_min=10e-4):
    ws = weight_searcher(logreg, X_train, y_train, g_train, X_val, y_val, g_val, p_ood, GDRO=False, weight_rounding=4, p_min=10e-4)

    # clip the p_dict
    p_dict_clipped = ws.clip_p_dict_per_group(p_dict, p_min=0.0, p_max=1.0)

    # normalize the p_dict
    p_dict_normalized = ws.normalize_p_dict(p_dict_clipped)

    # check if the sum of the values in the p_dict_normalized is 1
    assert np.sum(list(p_dict_normalized.values())) == 1

    # check if the values in the p_dict_normalized are between 0 and 1
    assert all([0 <= val <= 1 for val in list(p_dict_normalized.values())])



    


  

   

# if main is run, run the tests
if __name__ == "__main__":
    test_weight_searcher_helpers()
    