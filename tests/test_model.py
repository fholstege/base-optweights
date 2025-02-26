# import the add_up function from optweights
import numpy as np
from sklearn.linear_model import LogisticRegression
from optweights.model import Model
from optweights.weights import Weights



def test_model_fit():

    # set probability of group in training, save in dict
    ptr = 0.9

    # create a logistic regression model
    model_param  = {'max_iter': 100,
                    'penalty': 'l1',
                    'C': 1,
                    'solver': 'liblinear',
                    'tol': 1e-4,
                    'verbose': 0,
                    'random_state': 0,
                    'fit_intercept': True}
    
    # create an sklearn model
    logreg = LogisticRegression(**model_param)
    p_ood = {1: 0.5, 2: 0.5}
    p_train = {1: 1-ptr, 2: ptr}

    # create a model obj with the weights obj
    weights_obj = Weights(p_ood, p_train)
    model_obj = Model(weights_obj,logreg, add_intercept=True)

    # fit the model
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([1, 0, 1, 0])
    g_train = np.array([1, 2, 1, 2])
    d = X_train.shape[1]
    model_obj.fit(X_train, y_train, g_train)

    # check; when fitting the model, do we get a Beta that has the correct shape
    Beta = model_obj.get_Beta()
    assert Beta.shape == (d+1, 1)

    # check; when predicting the model, do we get a prediction that has the correct shape
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_pred = model_obj.predict(X_test, type_pred='probabilities')
    assert y_pred.shape == (X_test.shape[0], 1)

    # check; when assigning the weights in fit, do we get the correct weights?
    correct_w = {1: 0.5/0.1, 2: 0.5/0.9}
    w_dict = model_obj.weights_obj.set_weights_per_group(normalize=False)
    # check if the same at 4 decimals
    assert np.allclose(list(w_dict.values()), list(correct_w.values()), atol=10**-4)

    # check; can we reset the weights of the model?
    p_alt = {1: 0.25, 2:0.75}
    model_obj.reset_weights(p_alt)
    assert model_obj.weights_obj.p_w == p_alt


   

# if main is run, run the tests
if __name__ == "__main__":
    test_model_fit()
