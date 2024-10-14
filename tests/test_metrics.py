

# import the add_up function from optweights
from optweights.metrics import calc_BCE
from sklearn.metrics import log_loss
import numpy as np
import sys

def test_calc_BCE():

    # define the y and y_pred
    y = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    # use the log_loss function of sklearn
    loss_sk = log_loss(y, y_pred)

    # use the calc_BCE function
    loss_own = calc_BCE(y, y_pred)

    # check if the losses are the same
    assert np.allclose(loss_sk, loss_own)



   

# if main is run, run the tests
if __name__ == "__main__":
    test_calc_BCE()
    