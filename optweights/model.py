

# standard libraries
import numpy as np

# for autograd
# for linear/logistic regression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error as mse

# wrapper for an Sklearn model. 
class model():
    """
    Takes in a weights object and a model object
    Wrapper to fit the model with the weights for sklearn models

    """

    def __init__(self, weights_obj, sklearn_model, add_intercept=True):
        """
        Parameters:
            weights_obj: weights object
                The object that contains the weights for each group in the training data.
            sklearn_model: sklearn model
                The sklearn model to be fit to the data.
            add_intercept: bool
                If True, add an intercept to the data. Default is False.
        """

        self.weights_obj = weights_obj
        self.base_model = sklearn_model
        self.add_intercept = add_intercept

        # get the penalty type from the sklearn model
        self.penalty_type = self.base_model.penalty

        # if l1 penalty, set self.l1_penalty to C
        if self.penalty_type == 'l1':
            self.l1_penalty = 1/self.base_model.C
        else:
            self.l1_penalty = 0
        
        # if l2 penalty, set self.l2_penalty to C
        if self.penalty_type == 'l2':
            self.l2_penalty = 1/self.base_model.C
        else:
            self.l2_penalty = 0
        

    def reset_weights(self, p_w):
        """
        Reset the weights object
        """
        self.weights_obj.reset_weights(p_w)
    
    def fit_model(self, X, y, g):
        """
        Based on the weights object, fit the model to the data.
        """

        # get the weights for the group
        w = self.weights_obj.assign_weights(g)

        # check - if shape[1] of y is 1, turn to (n,)
        if len(y.shape) == 2:
            y = y.reshape(-1)

        # fit the model
        self.base_model.fit(X, y, sample_weight=w)

        # get coef, intercept
        coef = self.base_model.coef_
        intercept = self.base_model.intercept_

        # combine coef and intercept in Beta
        self.Beta = np.concatenate((intercept, coef[0])).reshape(-1, 1)

    
    def predict(self, X, type_pred='linear'):
        """
        Make predictions based on the model
        """

        # add an intercept to the data
        if self.add_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # make predictions
        if type_pred == 'linear':
            pred = np.matmul(X, self.Beta)
        elif type_pred == 'probabilities':
            pred = 1/(1 + np.exp(-np.matmul(X, self.Beta)))
        elif type_pred == 'class':
            pred = np.round(1/(1 + np.exp(-np.matmul(X, self.Beta))))
        else:
            raise ValueError('type_pred should be linear, probabilities or class')
  
        return pred