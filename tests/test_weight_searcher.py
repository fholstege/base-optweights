

# import the add_up function from optweights
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from optweights.weight_searcher import weight_searcher
from optweights.data import Toy
from optweights.helpers import set_seed
import torch
from torch.autograd.functional import jacobian
import sys

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


def test_augmented_loss_grad():
    # Set random seed for reproducibility
    set_seed(0)
    eps = 1e-4

    # Generate random data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    g = np.random.randint(1, 3, 100)
    Beta = np.random.randn(6)  # 5 features + 1 intercept
   
    # from the weight_searcher class, get calc_augmented_loss
    calc_grad_augmented_loss_func = weight_searcher.calc_grad_augmented_loss
    grad_numpy = calc_grad_augmented_loss_func(X, Beta, y, g, subsample_weights=None, eps=1e-6)

    # Calculate gradient using PyTorch
    X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y, dtype=torch.float32)
    g_torch = torch.tensor(g, dtype=torch.float32)
    Beta_torch = torch.tensor(Beta, dtype=torch.float32, requires_grad=True)

    # Define the augmented loss function with L1 and L2 regularization
    def augmented_loss_with_reg(beta):

        # add intercept to X
        X_with_intercept = torch.cat([torch.ones(X_torch.shape[0], 1), X_torch], dim=1)

        # get the data for g==1 and g==2
        X_1 = X_with_intercept[g_torch == 1, :]
        y_1 = y_torch[g_torch == 1]
        X_2 = X_with_intercept[g_torch == 2, :]
        y_2 = y_torch[g_torch == 2]

        # calculate the loss for g==1 and g==2
        output_1 = torch.sigmoid(X_1 @ beta)
        output_2 = torch.sigmoid(X_2 @ beta)

        # calculate the loss for g==1 and g==2
        bce_loss_1 = torch.nn.functional.binary_cross_entropy(output_1, y_1, reduction='mean')
        bce_loss_2 = torch.nn.functional.binary_cross_entropy(output_2, y_2, reduction='mean')

        return  (bce_loss_1 - bce_loss_2)


    # Compute the Jacobian
    grad_torch = jacobian(augmented_loss_with_reg, Beta_torch).squeeze()
    
    # Convert numpy gradient to PyTorch tensor for comparison
    grad_numpy_tensor = torch.tensor(grad_numpy, dtype=torch.float32).squeeze()

    # Compare gradients
    torch.testing.assert_close(grad_numpy_tensor, grad_torch, rtol=eps, atol=eps,
                                msg="Gradients from numpy and PyTorch do not match")
        


    
def test_BCE_grad():
    # Set random seed for reproducibility
    set_seed(0)
    eps = 1e-4

    # Generate random data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    Beta = np.random.randn(6)  # 5 features + 1 intercept
    l1_penalty = 0.01
    l2_penalty = 0.01

    # from the weight_searcher class, get calc_grad_BCE
    calc_grad_func = weight_searcher.calc_grad_BCE
    grad_numpy = calc_grad_func(X, Beta, y, l1_penalty, l2_penalty)

    # Calculate gradient using PyTorch
    X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y, dtype=torch.float32)
    Beta_torch = torch.tensor(Beta, dtype=torch.float32, requires_grad=True)

    # Define the BCE loss function with L1 and L2 regularization
    def bce_loss_with_reg(beta):
        X_with_intercept = torch.cat([torch.ones(X_torch.shape[0], 1), X_torch], dim=1)
        output = torch.sigmoid(X_with_intercept @ beta)
        bce_loss = torch.nn.functional.binary_cross_entropy(output, y_torch, reduction='mean')
        l1_reg = l1_penalty * torch.norm(beta, 1)
        l2_reg = l2_penalty * torch.norm(beta, 2)**2
        return bce_loss + ((l1_reg + l2_reg) / X_torch.shape[0])

    # Compute the Jacobian
    grad_torch = jacobian(bce_loss_with_reg, Beta_torch).squeeze()

    # Convert numpy gradient to PyTorch tensor for comparison
    grad_numpy_tensor = torch.tensor(grad_numpy, dtype=torch.float32)

    # Compare gradients
    torch.testing.assert_close(grad_numpy_tensor, grad_torch, rtol=eps, atol=eps,
                                msg="Gradients from numpy and PyTorch do not match")


  

   

# if main is run, run the tests
if __name__ == "__main__":
    test_weight_searcher_helpers()
    test_BCE_grad()
    #test_augmented_loss_grad()
    