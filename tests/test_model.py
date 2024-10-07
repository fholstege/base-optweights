

# import the add_up function from optweights
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from optweights.data import Toy
from optweights.model import model
from optweights.weights import weights


def test_weight_searcher_helpers():

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
    
    # create an sklearn model
    logreg = LogisticRegression(**model_param)
    p_ood = {1: 0.5, 2: 0.5}
    p_train = {1: 1-ptr, 2: ptr}

    # create a model obj with the weights obj
    weights_obj = weights(p_ood, p_train)
    model_obj = model( weights_obj,logreg, add_intercept=True)

  
    # check; when fitting the model, do we get a Beta that has the correct shape
    model_obj.fit_model(X_train, y_train, g_train)
    assert model_obj.Beta.shape == (d+1, 1)

    # check; when assigning the weights in fit_model, do we get the correct weights?
    correct_w = {1: 0.5/0.1, 2: 0.5/0.9}
    w_dict = model_obj.weights_obj.set_weights_per_group(normalize=False)
    # check if the same at 4 decimals
    assert np.allclose(list(w_dict.values()), list(correct_w.values()), atol=10**-4)

    # check; can we reset the weights of the model?
    p_alt = {1: 0.25, 2:0.75}
    model_obj.reset_weights(p_alt)
    assert model_obj.weights_obj.p_w == p_alt





    


  

   

# if main is run, run the tests
if __name__ == "__main__":
    test_weight_searcher_helpers()
    