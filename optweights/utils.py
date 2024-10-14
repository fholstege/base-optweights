

import numpy as np
import random


def set_seed(seed):
    """
    Sets the seed for the random number generators
    """
    random.seed(seed)
    np.random.seed(seed)

    

def fast_xtdx( X, diag):
        """
        Compute X.T * D * X where D is diagonal

        Arguments:
            X: ndarray of shape (n, d)
            diag: ndarray of shape (n,) representing the diagonal of D
        
        Returns:
            result: ndarray of shape (d, d) representing the result of the computation
        """
        # Element-wise multiply X by d
        DX = X * diag[:, np.newaxis]
        
        # Compute the final result
        result = X.T @ DX
        
        return result


def get_p_dict(g):
    """
    Retrieve the proportion of each group 

    Arguments:
        g: ndarray of shape (n,) representing the group labels
    
    Returns:
        p_dict: dictionary with the proportion of each group

    """
    
    # get the unique values of g
    unique_g = np.unique(g)
    
    # get the proportion of each group
    p_dict = {int(group): float(np.mean(g == group)) for group in unique_g}
    
    return p_dict




# calculate standard weights 
def calc_subsample_ood_weights(p_train, n_train):
    """
    For each group, calculate the fraction of the original data that should be used to reflect a distribution of equal size for each group

    Arguments:
        p_train: dictionary with the proportion of each group in the training data
        n_train: number of samples in the training data
    
    Returns:
        p_ood: dictionary with the fraction of the original data that should be used for each group
    """

    # get the n_g for each group
    n_g = {}
    for key in p_train.keys():
        n_g[key] = int(np.ceil(p_train[key]*n_train).item())

    # get the n_g for the smallest group
    n_s = min(n_g.values())

    # for each group, calculate the weights
    p_ood = {g: n_s/n_g[g] for g in n_g.keys()}

    return p_ood


def round_p_dict(p, weight_rounding):
    """
    Round the proportion of each group to the specified number of decimal places

    Arguments:
        p: dictionary with the proportion of each group
        weight_rounding: int,number of decimal places to round to
    
    Returns:
        p: dictionary with the proportion of each group rounded to the specified number of decimal places
    """

    # round each entry in p to the specified number of decimal places
    p = {g: round(p[g],weight_rounding) for g in p.keys()}

    return p

def clip_p_dict_per_group( p, p_min, p_max):
    """
    Clip the proportion of each group to be between p_min and p_max

    Arguments:
        p: dictionary with the proportion of each group
        p_min: dictionary with the minimum proportion of each group
        p_max: dictionary with the maximum proportion of each group
    
    Returns:
        p: dictionary with the proportion of each group clipped to be between p_min and p_max

    """
    

    # check; if p_min is a float,  turn to dict and apply to all groups
    if type(p_min) == float:
        p_min = {g: p_min for g in p.keys()}
    
    # check; if p_max is a float, turn to dict and apply to all groups
    if type(p_max) == float:
        p_max = {g: p_max for g in p.keys()}

    # clip each entry in p to be higher than min_p, lower than max_p
    p = {g: min(p_max[g], max(p_min[g], p[g])) for g in p.keys()}

    return p

def normalize_p_dict( p):
    """
    Normalize the proportion of each group to sum to 1
    
    Arguments:
        p: dictionary with the proportion of each group
    
    Returns:
        p: dictionary with the proportion of each group normalized to sum to 1
    """

    # normalize each entry in p to sum to 1
    p_sum = sum(p.values())
    p = {g: p[g] / p_sum for g in p.keys()}

    return p


def get_q(loss_g, eta, eps=10**-5):
    """
    Get the q for a group g, where the q is the exponential of the loss multiplied by the learning rate

    Arguments:
        loss_g: float, loss for group g
        eta: float, learning rate
        eps: float, small number to avoid numerical instability
    
    Returns:
        q: float, q for group g

    """

    q = np.exp(eta *(loss_g + eps))

    return q
    

def update_DRO_weights(q, loss_dict,  eta_q, C=0.0, n_dict=None, p_min=0.0, p_max=1.0): 
    """
    Update the weights for each group using the DRO update rule

    Arguments:
        q: dictionary with the weights for each group
        loss_dict: dictionary with the loss for each group
        eta_q: float, learning rate
        C: float, regularization parameter
        n_dict: dictionary with the number of samples for each group
        p_min: float, minimum weight for each group
        p_max: float, maximum weight for each group
    
    Returns:
        q_updated: dictionary with the updated weights for each group
    """
    
    # get the groups in the loss dict
    groups_loss = list(loss_dict.keys())

    # go through each group and update the weights
    for g in groups_loss:
        
        # Get the loss for group g
        loss_g = loss_dict[g]

        if n_dict is not None:
            n_g = n_dict[g]
            regularizer = C/np.sqrt(n_g)
        else:
            regularizer = 0.0
        

        # Get the  q for group g
        q_g = get_q(loss_g, eta_q) + regularizer

        # Update the weights for group g
        q[g] *= q_g
    
    # Normalize the weights
    q_normalized  = normalize_p_dict(q)

    # clip the p_normalized to be between p_min and p_max
    q_clipped = clip_p_dict_per_group(q_normalized, p_min, p_max)

    # then normalize the weights again
    q_updated = normalize_p_dict(q_clipped)

    return q_updated



