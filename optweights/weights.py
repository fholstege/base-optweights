import numpy as np

class weights():
    """
    Class for setting weights for each group in the training/validation data.
    """

    def __init__(self, p_w, p_train, weighted_loss_weights=True):
        self.p_w = p_w
        self.p_train = p_train
        self.groups = list(p_train.keys())
        self.n_groups = len(self.groups)

        # which type of weights to use
        self.weighted_loss_weights = weighted_loss_weights

        # set the weights for each group
        self.weights_dict = self.set_weights_per_group()

    def reset_weights(self, p_w):
        """
        Reset the weights for each group.
        """

        # set the new weights
        self.p_w = p_w
        self.weights_dict = self.set_weights_per_group()
        print('self.weights_dict', self.weights_dict)



    def set_weights_per_group(self, normalize=False):
        """
        Set the weights for each group of the training data.
        w_g = p_w[g]/p_train[g], where p_train[g] is the proportion of the training data in group g and p_w[g] is the desired proportion for group g.

        Parameters:
            p_train: dict
                The proportion of the training data in each group. Each key is an integer representing the group and each value is a float representing the proportion of the training data in that group.
            p_w: dict
                The desired proportion of the weights for each group. Each key is an integer representing the group and each value is a float representing the desired proportion of the weights for that group.
            normalize: bool
                If True, normalize the weights so that they sum to 1. Default is True.
        """

        weights = {}
        if self.p_w is None:
            weights = {g: 1 for g in self.p_train.keys()}
        else:
            for g in self.p_train.keys():
                if self.weighted_loss_weights:
                    weights[g] = self.p_w[g]/self.p_train[g]
                else:
                    weights[g] = self.p_w[g]
        
        if normalize:
            total = sum(weights.values())
            for g in weights.keys():
                weights[g] = weights[g]/total
        
        return weights
    

    def assign_weights(self, g):
        """
        Assign the weights for a specific group.

        Parameters:
            g: np.array with shape (n,)

        """

        # lambda function to assign the weights
        assign_weights = lambda x: self.weights_dict[x]

        # assign the weights
        weights = np.vectorize(assign_weights)(g)

        # if not 1d array, reshape
        if len(weights.shape) > 1:
            weights = weights.reshape(-1)

        return weights




