

# import the add_up function from optweights
from optweights.helpers import *
import numpy as np

def test_fast_xtdx():

    # create an example X and diag
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print('X:', X.shape)

    # create a diagonal matrix
    diag = np.array([1, 1, 1])

    # compute the result
    result = fast_xtdx(X, diag)

    # compute the expected result
    expected_result = X.T @ X

    # check if the results are the same
    assert np.all(result == expected_result)

def test_get_p_dict():

    # create an example group array
    g = np.array([1, 2, 3, 4, 1, 2, 3, 4])

    # get the proportion of each group
    p_dict = get_p_dict(g)

    # check if the proportions are correct
    assert p_dict == {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}


def test_get_q():

    # create an example loss_g and eta
    loss_g = 0.5
    eta = 0.1
    eps = 10**-5

    # compute the q
    q = get_q(loss_g, eta, eps)

    # compute the correct q
    correct = np.exp(eta *(loss_g + eps))

    # check if the q is correct
    assert  q == correct

   

# if main is run, run the tests
if __name__ == "__main__":
    test_fast_xtdx()
    test_get_p_dict()
    test_get_q()
    