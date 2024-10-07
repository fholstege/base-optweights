

# import the add_up function from optweights
import pytest
from optweights.weights import weights
import numpy as np


def test_weight_conversion():

    # example p_train and p_w
    p_train = {1:0.74, 2:0.01, 3:0.05, 4:0.2}
    p_w = {1:0.25, 2:0.25, 3:0.25, 4:0.25}

    # create a weights object
    weights_obj = weights(p_w, p_train)
    weights_dict = weights_obj.set_weights_per_group(normalize=False)

    # The weights to be are the desired weights
    weights_to_be = {1: 0.25/0.74, 2: 0.25/0.01, 3: 0.25/0.05, 4: 0.25/0.2}

    assert weights_dict == weights_to_be

def test_assign_weights():

    # example p_train and p_w
    p_train = {1:0.74, 2:0.01, 3:0.05, 4:0.2}
    p_w = {1:0.25, 2:0.25, 3:0.25, 4:0.25}

    # create a weights object
    weights_obj = weights(p_w, p_train)
    weights_dict = weights_obj.set_weights_per_group(normalize=False)

    # create a group array
    g = [1, 2, 3, 4, 1, 2, 3, 4]

    # assign the weights
    w = weights_obj.assign_weights(g)

    # check if the weights are assigned correctly
    w_to_be = [0.25/0.74, 0.25/0.01, 0.25/0.05, 0.25/0.2, 0.25/0.74, 0.25/0.01, 0.25/0.05, 0.25/0.2]

    # check if all elements are the same for the same index
    assert np.all(w == w_to_be)

   

# if main is run, run the tests
if __name__ == "__main__":
    test_weight_conversion()
    test_assign_weights()
    