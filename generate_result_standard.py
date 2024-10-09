


# import the add_up function from optweights
from optweights.weights import weights
from optweights.data import WB, CelebA
from optweights.model import model
from optweights.weight_searcher import weight_searcher
from optweights.helpers import calc_subsample_ood_weights, get_p_dict, set_seed, str_to_bool
from optweights.metrics import calc_worst_and_weighted_acc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from matplotlib import pyplot as plt
import sys
import argparse
import numpy as np


def main(dataset, early_stopping, batch_size, data_augmentation, seed, penalty_strength, method,  solver, tol=1e-4, max_iter=100):

    # define the data object
    if dataset == 'WB':
        data_obj = WB()
    elif dataset == 'CelebA':
        data_obj = CelebA()
   
    # load the data
    X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test= data_obj.load_embeddings( early_stopping, batch_size, data_augmentation, seed)
    
    # create an sklearn model
    if method !='GDRO':
         # create a logistic regression model
        model_param  = {'max_iter': max_iter,
                    'penalty': 'l1',
                    'C': 1/penalty_strength,
                    'solver': solver,
                    'tol': tol,
                    'verbose': 0,
                    'random_state': seed,
                    'fit_intercept': True, 
                    'warm_start': False}
        logreg = LogisticRegression(**model_param)
    else:
        # create a logistic regression model, via the SGDClassifier class
        model_param  = {'loss': 'log_loss',
                    'penalty': 'l1',
                    'l1_ratio':1,
                    'alpha': penalty_strength,
                    'tol': tol,
                    'verbose': 0,
                    'random_state': seed,
                    'fit_intercept': True, 
                    'warm_start': False}
        logreg = SGDClassifier(**model_param)

    # define the p_ood
    p_ood = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
  
    # define the p_train
    p_train = get_p_dict(g_train)
    n_train = len(g_train)
    
    # define the type of optimizer
    if method == 'SUBG':
        subsample_weights= True
        k_subsamples = 1
    elif method == 'DFR':
        subsample_weights= True
        k_subsamples = 10
    elif method == 'GW-ERM' or method == 'GDRO':
        subsample_weights= False
        k_subsamples = 1

    # define the p_standard
    if subsample_weights:
        # count how often each group occurs, determine the weights
        p_standard = calc_subsample_ood_weights(p_train, n_train)
    else:
        p_standard = p_ood 

    # define the model for the standard weights
    weights_obj_standard = weights(p_standard, p_train, weighted_loss_weights=not subsample_weights)
    logreg_standard_weights = model(weights_obj_standard, logreg, add_intercept=True, subsampler= subsample_weights, k_subsamples=k_subsamples)

    if method == 'GDRO': 
        Beta, _, _ = logreg_standard_weights.optimize_GDRO_via_SGD(X_train, y_train, g_train, X_val, y_val, g_val, T=100, batch_size=X_train.shape[0], eta_param=1e-5, eta_q=0.1, C=0, early_stopping=False)
        logreg_standard_weights.Beta = Beta
    else:
        logreg_standard_weights.fit_model(X_train, y_train, g_train, seed=seed)

    ## First, measure the loss on the validation and test data for the standard weights
    y_val_pred_standard= logreg_standard_weights.predict(X_val, type_pred='class')
    wg_val_standard, weighted_acc_val_standard = calc_worst_and_weighted_acc(y_val, y_val_pred_standard, g_val)

    # measure the loss on test data
    y_test_pred_standard= logreg_standard_weights.predict(X_test, type_pred='class')
    wg_test_standard, weighted_acc_test_standard = calc_worst_and_weighted_acc(y_test, y_test_pred_standard, g_test)

    print('---------------------------------------------------')
    print('--- Results on validation data (Standard weights) ---')
    print('Weighted accuracy on val data: ', weighted_acc_val_standard)
    print('Worst group on val data: ', wg_val_standard)

    print('--- Results on test data---')
    print('Weighted accuracy on test data: ', weighted_acc_test_standard)
    print('Worst group on test data: ', wg_test_standard)
    print('---------------------------------------------------')


# if main is run, run the tests
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for the experiment')
    parser.add_argument('--dataset', type=str, default='WB', help='Dataset to use')
    parser.add_argument('--early_stopping', type=str, default='true', help='Use early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_augmentation', type=str, default='false', help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--penalty_strength', type=float, default=1, help='Penalty strength')
    parser.add_argument('--method', type=str, default='GW-ERM', help='Method to use')
    parser.add_argument('--solver', type=str, default='liblinear', help='Solver')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations')
    args = parser.parse_args()

    # turn str to bool for relevant arguments
    args.early_stopping = str_to_bool(args.early_stopping)
    args.data_augmentation = str_to_bool(args.data_augmentation)

    # run the main function
    main(args.dataset, args.early_stopping, args.batch_size, args.data_augmentation, args.seed,
          args.penalty_strength, args.method, args.solver, args.tol, args.max_iter)

   
    


