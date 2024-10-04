

# import the add_up function from optweights
import pytest
from optweights.weights import weights



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


# if main is run, run the tests
if __name__ == "__main__":
    test_weight_conversion()
    