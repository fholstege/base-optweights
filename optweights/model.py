

# standard libraries
import numpy as np
import sys

# for setting the seed
from optweights.helpers import set_seed, update_DRO_weights
from optweights.metrics import calc_BCE, calc_worst_group_loss

# for linear/logistic regression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error as mse

# wrapper for an Sklearn model. 
class model():
    """
    Takes in a weights object and a model object
    Wrapper to fit the model with the weights for sklearn models

    """

    def __init__(self, weights_obj, sklearn_model, add_intercept=True, subsampler=False, verbose=False, k_subsamples=1):
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

        
        # if subsampler, then fit via subsampling
        self.subsampler = subsampler
        self.k_subsamples = k_subsamples

        # do a check; if self.subsampler, then in the weights_obj, the self.weighted_loss_weights should be False
        if self.subsampler:
            if self.weights_obj.weighted_loss_weights:
                raise ValueError('If subsampler is True, then the weights object should have weighted_loss_weights set to False')

        # verbose
        self.verbose = verbose
    

    def get_subsample_groups(self, X, y, g, seed):
        """
        Creates subsample of original sample
        Selects p_g * n_g unique samples from group g
        """
        

        # get the groups
        groups = np.unique(g)

        # loop over each group, create dict with per group: indeces, and size
        group_dict = {}
        for group in groups:
            group_dict[group] = {}
            group_dict[group]['i'] = np.where(g == group)[0]
            group_dict[group]['n'] = len(group_dict[group]['i'])
        

        # now, get the subsample per group, each of size n_tilde
        i_sample = []
        for group in groups:
            i_group = group_dict[group]['i']
            n_g = group_dict[group]['n']
            m_g = int(np.ceil(self.weights_obj.weights_dict[group]*n_g).item())
           
            if self.verbose:
                print('Sampling without replacement for group {}, sampling proportion: {}, a total of {}'.format(group, m_g/n_g, m_g))
            
            # first, shuffle the i_group based on the seed
            set_seed(seed)
            i_group_shuffled = np.random.permutation(i_group)
            
            # second, select the first m_g indeces
            i_sample_group = i_group_shuffled[:m_g]

            # add to the list
            i_sample.append(i_sample_group)
        
        
        # now, combine the indeces
        i_sample = np.concatenate(i_sample)
        self.m = len(i_sample)
        if self.verbose:
            print('Size of subsample: {}'.format(len(i_sample)))
        
        # get the subsample
        X_tilde = X[i_sample, :]
        y_tilde = y[i_sample]
  

        return X_tilde, y_tilde, i_sample
        

    def reset_weights(self, p_w):
        """
        Reset the weights object
        """
        self.weights_obj.reset_weights(p_w)

    
    def get_Beta(self, base_model):
        """
        Combine the coef and intercept in Beta
        """
        coef = base_model.coef_
        intercept = base_model.intercept_
        return np.concatenate((intercept, coef[0])).reshape(-1, 1)
    
    def fit_model(self, X, y, g, seed=0):
        """
        Based on the weights object, fit the model to the data.
        """

        # check; if shape[1] of y is 1, turn to (n,)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        # if subsampler, then fit via subsampling
        if self.subsampler:

            # create k subsamples
            list_Beta = []
            for i in range(self.k_subsamples):

                # get the subsample
                X_tilde_i, y_tilde_i, _ = self.get_subsample_groups(X, y, g, seed=seed+i)
                self.base_model.fit(X_tilde_i, y_tilde_i)

                # get the Beta, add to list
                Beta_i = self.get_Beta(self.base_model)
                list_Beta.append(Beta_i)


            # if list > 1, get the mean
            if len(list_Beta) > 1:
                self.Beta = np.mean(np.array(list_Beta), axis=0)
            else:
                self.Beta = list_Beta[0]

        else:

            # get the weights for the group
            w = self.weights_obj.assign_weights(g)

            # check - if shape[1] of y is 1, turn to (n,)
            if len(y.shape) == 2:
                y = y.reshape(-1)

            # fit the model
            self.base_model.fit(X, y, sample_weight=w)
            self.m = X.shape[0]

            # get the Beta
            self.Beta = self.get_Beta(self.base_model)

    
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
    
    def gradient_step_DRO(self, X_b, y_b, g_b, eta_param):

        """
        Run a single gradient step for the sklearn model
        """

        # get the weights
        w_b = self.weights_obj.assign_weights(g_b)

        # set the learning rate
        self.base_model.learning_rate = 'constant'
        self.base_model.eta0 = eta_param

        # use the partial fit method of the logreg object
        self.base_model.partial_fit(X_b, y_b, sample_weight=w_b, classes=np.unique(y_b))

        # turn the coef_ and intercept_ into Beta
        self.Beta = self.get_Beta(self.base_model)
    


    def optimize_GDRO_via_SGD(self, X_train, y_train, g_train, X_val, y_val, g_val, T, batch_size, eta_param, eta_q, C=0.0, early_stopping=False, patience=1):
        """
        Run the GDRO algorithm for the model
        """

        # get the number of groups in the training data
        groups = np.unique(g_train)

        # get the initial q_t
        q_t ={int(group): 1/len(groups) for group in groups}

        # set the initial weights
        self.reset_weights(q_t)

        # define the n_dict; this is the number of observations in each group
        n_dict = {int(group): np.sum(g_train == group) for group in groups}

        # save the best worst-case loss
        best_worst_group_loss = np.inf

        # if the sklearn model uses the log loss, define this
        if self.base_model.loss == 'log_loss':
            loss_fn = calc_BCE

        # loop over T epochs
        for t in range(T):
            
            # shuffle the indeces
            indices = np.arange(X_train.shape[0])
            shuffled_indices = np.random.permutation(indices)

            # loop over the batches, including the last batch
            for i in range(0, len(shuffled_indices), batch_size):

                # get the final index
                last_batch_index = min(i+batch_size, len(shuffled_indices))

                # get the batch indices
                batch_indices = shuffled_indices[i:last_batch_index]

                # get the batch
                X_b, y_b, g_b = X_train[batch_indices, :], y_train[batch_indices], g_train[batch_indices]

                # update the parameters
                self.gradient_step_DRO(X_b, y_b.squeeze(-1), g_b, eta_param)

                # get loss per group for the batch
                _, loss_dict = calc_worst_group_loss(self, loss_fn, X_b, y_b, g_b)
                
                # based on the loss, update the q_t
                q_t = update_DRO_weights(q_t, loss_dict,  eta_q, C=C, n_dict=n_dict)

                # reset the p_weights of the model
                self.reset_weights(q_t)

            # after completing the first epoch, get the param
            if t == 0:
                best_Beta = self.Beta
                best_t = t
            
            # after completing an epoch, get the loss for the entire dataset
            worst_group_loss_val, _ = calc_worst_group_loss(self, loss_fn, X_val, y_val, g_val)
                    

            # print the following stats
            print('At epoch {}, the worst group loss on the validation set is {} '.format(t, worst_group_loss_val))

            if early_stopping and worst_group_loss_val >= best_worst_group_loss:
                patience -= 1
                if patience == 0:
                    print('Early stopping at epoch {}'.format(t))
                    break
            
            # if loss is better, save the parameters
            if worst_group_loss_val < best_worst_group_loss:
                print('New best worst group loss found at epoch {}'.format(t))
                best_worst_group_loss = worst_group_loss_val
                best_Beta = self.Beta
                best_t = t
        print('Best worst group loss found at epoch {}'.format(best_t))
        return best_Beta, best_worst_group_loss, best_t