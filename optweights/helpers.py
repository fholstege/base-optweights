

import numpy as np
import random


def set_seed(seed):
    """
    Sets the seed for the random number generators
    """
    random.seed(seed)
    np.random.seed(seed)

    

def str_to_bool(text):
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    

def fast_xtdx( X, diag):
        """
        Compute X.T * D * X where D is diagonal
        X: ndarray of shape (n, d)
        d: ndarray of shape (n,) representing the diagonal of D
        """
        # Element-wise multiply X by d
        DX = X * diag[:, np.newaxis]

        
        # Compute the final result
        result = X.T @ DX
        
        
        return result


def get_p_dict(g):
    
    # get the unique values of g
    unique_g = np.unique(g)
    
    # get the proportion of each group
    p_dict = {int(group): float(np.mean(g == group)) for group in unique_g}
    
    return p_dict




# calculate standard weights 
def calc_subsample_ood_weights(p_train, n_train):


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

        # round each entry in p to the specified number of decimal places
        p = {g: round(p[g],weight_rounding) for g in p.keys()}

        return p

def clip_p_dict_per_group( p, p_min, p_max):

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

    # normalize each entry in p to sum to 1
    p_sum = sum(p.values())
    p = {g: p[g] / p_sum for g in p.keys()}

    return p


def get_q(loss_g, eta, eps=10**-5):

        q = np.exp(eta *(loss_g + eps))

        return q
    

def update_DRO_weights(q, loss_dict,  eta_q, C=0.0, n_dict=None, p_min=0.0, p_max=1.0): 
    
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
        q_fin = normalize_p_dict(q_clipped)

        return q_fin




def create_key(dataset, method, penalty_strength, batch_size_model, augmentation_data, early_stopping_model, seed, solver, tol, fraction_original_data, val_fit,  lambda_JTT=None, C_GDRO=None, eta_param_GDRO=None, eta_q_GDRO=None ):
    key = '{}_METHOD_{}_PENALTY_{}_BS_{}_AD_{}_ES_{}_SEED_{}_SOLVER_{}_TOL_{}_FRACTION_{}_VAL_FIT_{}'.format(dataset, method, penalty_strength, batch_size_model, augmentation_data, early_stopping_model, seed, solver, tol, fraction_original_data, val_fit)

    if method == 'JTT':
         key += '_LAMBDA_{}'.format(lambda_JTT)
    elif method == 'GDRO':
         key += 'C_GDRO_{}_ETA_PARAM_GDRO_{}_ETA_Q_GDRO_{}'.format(C_GDRO, eta_param_GDRO, eta_q_GDRO)

    return key
    


def get_fraction_original_data(fraction, X, y, g, seed):
    """
     Sample a fraction of the original data
    """
     
    # get the number of samples for the data
    n = X.shape[0]

    # get the number of samples to keep
    n_tilde = int(np.ceil(n*fraction))

    # shuffle the indices
    indices = np.arange(n)
    set_seed(seed)
    shuffled_indices = np.random.permutation(indices)

    # get the subsample
    indices = shuffled_indices[:n_tilde]
    X_tilde = X[indices, :]
    y_tilde = y[indices]
    g_tilde = g[indices]

    return X_tilde, y_tilde, g_tilde, indices

def split_data_in_train_val(fraction, X, y, g, seed):
     
    # we now need to split the validation data into two parts
    # take all indeces, shuffle them
    set_seed(seed)
    indeces = np.arange(X.shape[0])
    np.random.shuffle(indeces)

    # select up to x% of the data for training
    n_frac = int(fraction*X.shape[0])
    indeces_1 = indeces[:n_frac]
    indeces_2 = indeces[n_frac:]

    # X_train, y_train, g_train become the first half, X_val, y_val, g_val become the second half
    X_train = X[indeces_1]
    y_train = y[indeces_1]
    g_train = g[indeces_1]
    X_val = X[indeces_2]
    y_val = y[indeces_2]
    g_val = g[indeces_2]

    return X_train, y_train, g_train, X_val, y_val, g_val
