
import sys
import numpy as np
from optweights.helpers import fast_xtdx, get_p_dict, set_seed, update_DRO_weights
from optweights.model import model
from optweights.metrics import calc_loss_for_model, calc_worst_group_loss
from optweights.weights import weights
from sklearn.metrics import log_loss, mean_squared_error
import copy
import time

class weight_searcher():
    def __init__(self, sklearn_model, X_train, y_train, g_train, X_val, y_val, g_val, p_ood=None, GDRO=False, weight_rounding=4, p_min=10e-4):

        # set the attributes
        self.sklearn_model = sklearn_model
        self.X_train = X_train
        self.y_train = y_train
        self.g_train = g_train
        self.X_val = X_val
        self.y_val = y_val
        self.g_val = g_val
        self.GDRO = GDRO
        self.weight_rounding = weight_rounding
        self.p_min = p_min
        

        # given the g_train, g_val, calculate the p_train, p_val
        self.p_train = get_p_dict(g_train)
        self.p_val = get_p_dict(g_val)
        self.groups = list(self.p_train.keys())
        self.G = len(self.groups)

        # initialize the weights object for train, val
        self.weights_obj_tr = weights(p_w=None, p_train=self.p_train)
        self.weights_obj_val = weights(p_w=p_ood, p_train=self.p_val)

        # initialize the model object
        self.model = model(weights_obj=self.weights_obj_tr, sklearn_model=self.sklearn_model)

        # check: are the groups in p_train the same as in p_val?
        if set(self.p_train.keys()) != set(self.p_val.keys()):
            KeyError('The groups in p_train are not the same as in p_val')

        # based on the model class, define the loss function
        # if sklearn.linear_model.LogisticRegression, use the log_loss
        if self.sklearn_model.__class__.__name__ == 'LogisticRegression':
            # from sklearn use log_loss
            self.loss = log_loss
        elif self.sklearn_model.__class__.__name__ == 'LinearRegression':
            self.loss = mean_squared_error
        else:
            sys.exit('The model class is not supported. Please use sklearn.linear_model.LogisticRegression or sklearn.linear_model.LinearRegression')

    def round_p_dict(self, p):

        # round each entry in p to the specified number of decimal places
        p = {g: round(p[g],self.weight_rounding) for g in p.keys()}

        return p

    def clip_p_dict_per_group(self, p, p_min, p_max):

            # check; if p_min is a float,  turn to dict and apply to all groups
            if type(p_min) == float:
                p_min = {g: p_min for g in p.keys()}
            
            # check; if p_max is a float, turn to dict and apply to all groups
            if type(p_max) == float:
                p_max = {g: p_max for g in p.keys()}

            # clip each entry in p to be higher than min_p, lower than max_p
            p = {g: min(p_max[g], max(p_min[g], p[g])) for g in p.keys()}

            return p
    
    def normalize_p_dict(self, p):

        # normalize each entry in p to sum to 1
        p_sum = sum(p.values())
        p = {g: p[g] / p_sum for g in p.keys()}

        return p
    

    def calc_Hessian_weighted_logistic_loss(self, X,  w, Beta, l1_penalty, l2_penalty, eps=1e-6, divide_by_n=True):
        """
        Calculate the Hessian of the logistic loss function
        """

        # add the intercept to X, if the column dim of X is one less than the row dim of Beta
        if (X.shape[1]+1) == Beta.shape[0]:
            X =  np.c_[np.ones(X.shape[0]), X]
        
        # create a diagonal matrix with inputs sigmoid(x_i^T beta) * (1 - sigmoid(x_i^T beta))
        X_t_Beta = np.matmul(X, Beta).squeeze()
        sigmoid_X_t_Beta = 1/(1 + np.exp(-X_t_Beta))
       
        # calculate the diagonal matrix
        diag_H = ((sigmoid_X_t_Beta * (1 - sigmoid_X_t_Beta)) * w )
        
        # calculate the Hessian
        H = fast_xtdx(X, diag_H)

        # divide by the number of samples
        if divide_by_n:
            H /= X.shape[0]

        # add the l_2 penalty
        if l2_penalty>0:
            added_term = np.eye(H.shape[0])*l2_penalty *2

            # add the term
            if divide_by_n:
                H+= (added_term/X.shape[0])
            else:
                H += added_term

        elif l1_penalty>0:
            # the following is an approximation of the derivative of the l_1 penalty
            beta_squared = (Beta**2).squeeze()
            H_l_1_approx =   eps/((beta_squared + eps)**(3/2))
            H_l_1_approx_diag = np.diag(H_l_1_approx)*l1_penalty
            added_term = H_l_1_approx_diag/X.shape[0]

            # add the term
            if divide_by_n:
                H += (added_term/X.shape[0])
            else:
                H += added_term

        return H

    @classmethod
    def calc_grad_augmented_loss(self, X, Beta, y,  g, subsample_weights, eps=1e-6):

        # first, calculate the gradient of the loss for each group
        groups = np.unique(g)

        # First, calculat the grad per group
        grad = np.zeros((X.shape[1]+1, len(groups)))

        # go over each group, except the last one
        for i, group in enumerate(groups):
            # get the indices for the group
            indices =(g == group).squeeze()
            X_group = X[indices, :]
            y_group = y[indices]

            # calculate the gradient
            grad_group = self.calc_grad_BCE(X_group, Beta, y_group, 0, 0, eps=eps)

            # add to the grad
            grad[:, i] = grad_group.squeeze()
        
        # if not subsample weights, deduct the grad of the last group from the other groups
        if not subsample_weights:
            # first, deduct the grad of the last group from the other groups
            grad_G = grad[:, -1].reshape(-1, 1)
            grad[:, :-1] -= grad_G
            
            # second, remove the last col. (the grad of the last group)
            grad = grad[:, :-1]
        else:
            # multiply the grad with the factor
            grad *= self.m/X.shape[0]
        
        return grad
        

    
    @classmethod
    def calc_grad_BCE(self, X, Beta,  y, l1_penalty, l2_penalty, w=None, eps=1e-6, divide_by_n=True):
        """
        Calculate the gradient of the BCE
        """

        # add the intercept to X, if the column dim of X is one less than the row dim of Beta
        if (X.shape[1]+1) == Beta.shape[0]:
            X =  np.c_[np.ones(X.shape[0]), X]
        
        
        # create a diagonal matrix with inputs sigmoid(x_i^T beta) * (1 - sigmoid(x_i^T beta))
        X_t_Beta = np.matmul(X, Beta)
        sigmoid_X_t_Beta = 1/(1 + np.exp(-X_t_Beta))
               

        # calculate the gradient in two steps:
        # 1. first term: (sigmoid - y)
        # 2. second term: multiply X^T with the first term
        if w is not None:
            # check the shape of w
            if len(w.shape) == 1:
                w = w.reshape(-1, 1)
                
            # multiply element wise with w
            weighted_sigmoid_X_t_Beta = (sigmoid_X_t_Beta - y) * w
            grad =np.matmul(X.T, weighted_sigmoid_X_t_Beta)
        else:
            grad =np.matmul(X.T, (sigmoid_X_t_Beta - y))
        # divide by the number of samples
        if divide_by_n:
            grad /= X.shape[0]

        # add the l_2 penalty
        if l2_penalty>0:
            added_term = 2 * l2_penalty * Beta

            # add the term
            if divide_by_n:
                added_term /= X.shape[0]
            grad += added_term

        elif l1_penalty>0:
            # the following is an approximation of the derivative of the l_1 penalty
            beta_squared = (Beta**2)
            sqrt_beta_squared = np.sqrt(beta_squared + eps)
            added_term =  ((Beta / sqrt_beta_squared) * l1_penalty)
            
            # add the term
            if divide_by_n:
                added_term /= X.shape[0]
            grad += added_term

        return grad
    

    
    def weight_grad_via_ift(self, model, p, X_train, y_train, g_train, X_val, y_val, g_val, weights_obj_val, eps=1e-6,   subsample_weights=False):

        # create a copy of the starting weights
        groups = list(p.keys())
        last_group = self.G

        
        # get the w_train, w_val
        w_train = model.weights_obj.assign_weights(g_train)
        w_val = weights_obj_val.assign_weights(g_val)

        # if the weights are subsampled, calculate the m - e.g. how many 
        if subsample_weights:
            self.m = model.m

        # calculate the hessian
        H = self.calc_Hessian_weighted_logistic_loss(X_train, w_train, model.Beta, model.l1_penalty, model.l2_penalty, eps=1e-6)

        # use multiplication factor if subsample_weights
        if subsample_weights:
            factor = model.X_train.shape[0]/self.m
            H *= factor

       

        # ensure the Hessian is positive definite
        H += np.eye(H.shape[0])*eps

        # if the d > n, use the moores-penrose inverse
        if H.shape[0] > H.shape[1]:
            H_inv = np.linalg.pinv(H)
        else:
            H_inv =np.linalg.inv(H)

        # Calculate the gradient of the augmented loss with respect to the parameters
        J_augmented_w = self.calc_grad_augmented_loss(X_train, model.Beta, y_train, g_train, subsample_weights=subsample_weights, eps=1e-4)
       
        # third, calc the jacobian with respect to the weighted validation loss - take the average over the validation set
        J_val_w = self.calc_grad_BCE(X_val, model.Beta, y_val, 0, 0, w=w_val, eps=1e-4, divide_by_n=True)
        
        # calculate the derivative of the parameters with respect to w
        partial_deriv_param_w = np.matmul(-H_inv,  J_augmented_w)
       
        # now, calculate the derivative of the validation loss with respect to w
        grad_ift = np.matmul(J_val_w.T, partial_deriv_param_w)
       
        # now, calculate the derivative
        if subsample_weights:
            grad_ift_dict =  {g: grad_ift[i].item() for i, g in enumerate(groups)}

        # for the last group, sum the changes in the other groups and taking the negative
        else:

            # turn into a dictionary
            grad_ift_dict = {g:grad_ift[g-1].item() for g in groups[:-1]}

            # calculate the change for the last group
            grad_last_group = -np.sum([grad_ift_dict[g] for g in groups[:-1]]).item()

            # set the change for the last group based on change in all other groups
            grad_ift_dict[last_group] = grad_last_group
        
      

        return grad_ift_dict

    def optimize_weights(self, start_p, T,  lr,  momentum, eps=0, patience=None,  save_trajectory=False,  verbose=True,  eta_q=0.1, decay=0.9, lr_schedule='constant',stable_exp=True, p_min=10e-4, subsample_weights=False, lock_in_p_g = None, seed=0):
        """
        Optimize the weights using exponentiated gradient descent
        
        Parameters:
            T: int, number of iterations
            lr: float, learning rate
            momentum: float, momentum
            eps: float, threshold for the stopping criterion
            patience: int, number of iterations to wait for improvement
            save_trajectory: bool, if True, save the trajectory of the weights
            verbose: bool, if True, print the loss at each iteration
            eta_q: float,used for optimizing worst-group loss of the weights
            lr_schedule: str, learning rate schedule
            stable_exp: bool, if True, use stable exponentiation
            p_min: float, minimum value for the weights
            subsample_weights: bool, if True, use subsample weights
            lock_in_p_g: int, if not None, lock in the weights for group g
        """

        # Check: are the groups in start_p the same as in p_train?
        if set(start_p.keys()) != set(self.p_train.keys()):
            KeyError('The groups in start_p are not the same as in p_train')

        # Check: are the entries in start_p floats?
        if not all(isinstance(value, float) for value in start_p.values()):
            TypeError('The values in start_p are not floats')
        
        # Initialize the gradient and the current p
        grad = dict.fromkeys(start_p, 999)
        p_t = self.round_p_dict(copy.deepcopy(start_p))

        # if the trajectory is saved, initialize the trajectory
        if save_trajectory:

            # save the p trajectory
            p_at_t_traj = np.zeros((T, self.G))
            p_at_t_traj[0] = np.array(list(p_t.values()))
            
            # save the loss trajectory
            loss_at_t_traj = np.zeros(T-1)

        # check if momentum is not None
        if momentum is not None:
            prev_update = dict.fromkeys(self.groups, 0)
        
        # initialize the iteration
        t = 0
        best_loss = np.inf
        best_p = start_p
        stop_GD = False
        patience_count = patience

        # create weight obj. for initial weights, set the weights in the model
        self.model.reset_weights(p_t)
        self.model.fit_model(self.X_train, self.y_train, self.g_train)

         # if GDRO, save the p_at_t
        if self.GDRO:
            q_t = {g: 1/self.G for g in self.groups}
            best_worst_group_loss = np.inf


        # start the gradient descent
        while not stop_GD and (t < T):

             # calculate the loss at the current weights, using the validation data and the validation weights
            loss_t = calc_loss_for_model(self.model, self.loss, self.X_val, self.y_val, self.g_val, weights_obj = self.weights_obj_val, type_pred='probabilities')

            # if GDRO, calculate the worst group loss
            if self.GDRO:
                worst_group_loss_t, loss_per_group_t = calc_worst_group_loss(self.model, self.loss, self.X_val, self.y_val, self.g_val)
                

             # save the loss at t
            if save_trajectory:
                loss_at_t_traj[t-1] = loss_t

            # if GDRO, this is done based on the worst group loss
            if self.GDRO:
                if worst_group_loss_t < best_worst_group_loss:
                    best_worst_group_loss = worst_group_loss_t
                    patience_count = patience
                    best_p = p_t.copy()
                else:
                    patience_count -= 1
            
            # if not GDRO, this is done based on the overall loss
            else:
                # check if the loss is less than the best loss minus eps
                if loss_t < (best_loss - eps):
                    best_loss = loss_t
                    patience_count = patience
                    best_p = p_t.copy()
                else:
                    patience_count -= 1
            
            # check if the patience count is 0
            if patience_count == 0:
                stop_GD = True
            

            # if GDRO, change the weights based on the loss
            if self.GDRO:

                # update the weights
                p_GDRO_t =  update_DRO_weights(p_GDRO_t, loss_per_group_t, eta_q)

                # set the weights for the validation set
                self.weights_obj_val.reset_weights(p_w=p_GDRO_t)

                # set the weights for the validation data
                print('The GDRO probabilities are updated to {}, based on this loss per group: {}'.format(p_GDRO_t, loss_per_group_t))

            # calculate the grad
            grad = self.weight_grad_via_ift(self.model, p_t, self.X_train, self.y_train, self.g_train, self.X_val, self.y_val, self.g_val, self.weights_obj_val, eps=1e-6,   subsample_weights=subsample_weights)

            # provide information about the process
            if verbose:
                if self.GDRO:
                    loss_format = worst_group_loss_t
                else:
                    loss_format = loss_t
                print('At step {}, the loss is {}, we have {} patience left, and the probabilities are {}, which sum to {} with gradients {}.'.format(t, loss_format, patience_count, p_t,  sum(p_t.values()), grad))
            
            # make a copy of the weights
            p_t_plus_1 = p_t.copy()

            # determine the learning rate at time t
            if lr_schedule == 'constant':
                lr_t = lr
            elif lr_schedule == 'exponential':
                lr_t = lr * np.exp(-decay*t)
                #print('The learning rate at time {} is {}'.format(t, lr_t))
            elif lr_schedule == 'linear':
                lr_t = lr * decay
                #print('The learning rate at time {} is {}'.format(t, lr_t))
            else:
                Exception('The learning rate schedule is not recognized')
            

            # calculate the updates
            updates = dict.fromkeys(self.groups, 0)

            # update the weights per group
            for g in self.groups:
                # get the grad
                update =  (grad[g])
                
                # if locked in, do not update
                if g == lock_in_p_g and lock_in_p_g is not None:
                    continue
                
                # check if momentum is not None
                if momentum is not None:
                    update = (1-momentum)*update + (momentum * prev_update[g])
                    
                    # save the update
                    prev_update[g] = update
              
                # add to dict of updates via the learning rate
                updates[g] = -lr_t*update

            # if stable, then deduct the max update
            if stable_exp:
                max_update = max([updates[g] for g in self.groups])
                updates = {g: updates[g] - max_update for g in self.groups}
            
            
            # update the p
            for g in self.groups:
                p_t_plus_1[g] = (p_t_plus_1[g] * np.exp(updates[g])).item()


            # round the p_at_t to the specified number of decimal places
            p_t= self.round_p_dict(p_t_plus_1)

            # clip the p_at_t
            p_t =  self.clip_p_dict_per_group(p_t, p_min=self.p_min, p_max=1.0)

            # if normalize, clip again
            if not subsample_weights:
                p_t = self.normalize_p_dict(p_t)


            # after the p_at_t is determined, update the model
            time_before = time.time()
            self.model.reset_weights(p_w=p_t)
            self.model.fit_model(self.X_train, self.y_train, self.g_train)
            print('The model is updated in {} seconds'.format(time.time()-time_before))


            # save the trajectory if needed
            if save_trajectory:
                p_at_t_traj[t] = np.array(list(p_t.values()))
            t += 1
        
        # return the best p
        if self.GDRO:
            print('Returning the p={}, for which loss is {}'.format(best_p, best_worst_group_loss))
        else:
            print('Returning the p={}, for which loss is {}'.format(best_p, best_loss))

        # return the weight
        if save_trajectory:
            return best_p, p_at_t_traj[:t-1], loss_at_t_traj[:t-1]
        else:
            return best_p

                


