

# import the add_up function from optweights
from optweights.metrics import calc_BCE, calc_loss_for_model
from optweights.model import model
from optweights.weights import weights
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
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


def test_calc_loss_for_model():

    # define the y and y_pred
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 0, 1, 0])
    g = np.array([1, 2, 1, 2])

    # create a model
    logreg = LogisticRegression()
    weights_obj = weights({1: 0.1, 2: 0.9}, {1: 0.1, 2: 0.9})
    model_obj = model( weights_obj, logreg, add_intercept=True)
    model_obj.fit_model(X, y, g)
    y_pred = model_obj.predict(X, type_pred='probabilities')
    

    # calculate the loss
    loss = calc_loss_for_model(model_obj, calc_BCE, X, y, g, weights_obj=None)

    # check if the loss is the same as the loss from the log_loss function
    assert np.allclose(loss, calc_BCE(y, y_pred))



    


   

# if main is run, run the tests
if __name__ == "__main__":
    test_calc_BCE()
    test_calc_loss_for_model()
    