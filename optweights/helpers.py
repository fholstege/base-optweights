

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

def get_q(loss_g, eta, eps=10**-5):

        q = np.exp(eta *(loss_g + eps))

        return q

def update_DRO_weights(self, p, loss_dict,  eta_q, C=0.0, n_dict=None): 
       
        # get the groups in the loss dict
        groups_loss = list(loss_dict.keys())
    
        # go through each group and update the weights
        i=0
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
            p[i] = p[i] * q_g
            i += 1
        
        # Normalize the weights
        p_normalized = p / np.sum(p)

        # clip the p_normalized to be between p_min and p_max
        p_clipped = np.clip(p_normalized, self.p_min, self.p_max)

        # then normalize the weights again
        p = p_clipped / np.sum(p_clipped)

        # return the probability weights as a dictionary
        p_dict = {g: p[i].item() for i, g in enumerate(self.groups)}

        return p_dict

