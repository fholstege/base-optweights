

# import the add_up function from optweights
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from optweights.weight_searcher import weight_searcher
from optweights.weights import weights
from optweights.data import Toy
from optweights.helpers import set_seed, get_p_dict, clip_p_dict_per_group, normalize_p_dict
import torch
from torch.autograd.functional import jacobian

import sys

def test_weight_searcher_helpers():

    # create a dict with entries more than 1 or lower than 0
    p_dict = {1: 0.25, 2: 1.25, 3: -0.25, 4: 0.75}

    # clip the p_dict
    p_dict_clipped = clip_p_dict_per_group(p_dict, p_min=0.0, p_max=1.0)

    # normalize the p_dict
    p_dict_normalized = normalize_p_dict(p_dict_clipped)

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
    y = np.random.randint(0, 2, 100).reshape(-1, 1)
    g = np.random.randint(1, 3, 100)
    Beta = np.random.randn(6).reshape(-1,1)  # 5 features + 1 intercept
    l1_penalty = 0.01
    l2_penalty = 0.01

    # from the weight_searcher class, get calc_grad_BCE
    calc_grad_func = weight_searcher.calc_grad_BCE
    p_train = get_p_dict(g)
    weight_obj_val = weights(p_w={1: 0.5, 2: 0.5}, p_train=p_train)
    w = weight_obj_val.assign_weights(g)
    grad_numpy = calc_grad_func(X, Beta, y, l1_penalty, l2_penalty, w)

    # Calculate gradient using PyTorch
    X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y, dtype=torch.float32)
    Beta_torch = torch.tensor(Beta, dtype=torch.float32, requires_grad=True)

    # Define the BCE loss function with L1 and L2 regularization
    def bce_loss_with_reg(beta):

        X_with_intercept = torch.cat([torch.ones(X_torch.shape[0], 1), X_torch], dim=1)
        output = torch.sigmoid(X_with_intercept @ beta)
        w_torch = torch.tensor(w, dtype=torch.float32).unsqueeze(1)
        bce_loss = torch.nn.functional.binary_cross_entropy(output, y_torch, reduction='mean', weight=w_torch)
        l1_reg = l1_penalty * torch.norm(beta, 1)
        l2_reg = l2_penalty * torch.norm(beta, 2)**2
        return bce_loss + ((l1_reg + l2_reg) / X_torch.shape[0])

    # Compute the Jacobian
    grad_torch = jacobian(bce_loss_with_reg, Beta_torch).squeeze()

    # Convert numpy gradient to PyTorch tensor for comparison
    grad_numpy_tensor = torch.tensor(grad_numpy, dtype=torch.float32).squeeze()

    # Compare gradients
    torch.testing.assert_close(grad_numpy_tensor, grad_torch, rtol=eps, atol=eps,
                                msg="Gradients from numpy and PyTorch do not match")


  

   

# if main is run, run the tests
if __name__ == "__main__":
    test_weight_searcher_helpers()
    test_BCE_grad()
    test_augmented_loss_grad()
    