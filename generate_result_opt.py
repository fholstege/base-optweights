
# import the add_up function from optweights
from optweights.weights import weights
from optweights.data import WB, CelebA
from optweights.model import model
from optweights.weight_searcher import weight_searcher
from optweights.helpers import calc_subsample_ood_weights, get_p_dict, set_seed, str_to_bool
from optweights.metrics import calc_worst_and_weighted_acc
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import sys
import argparse
import numpy as np


def main(dataset, early_stopping, batch_size, data_augmentation, seed, penalty_strength, method, GDRO, T, lr, momentum, patience, lr_schedule, decay, solver, verbose, stable_exp=False, tol=1e-4, max_iter=100):

    # define the data object
    if dataset == 'WB':
        data_obj = WB()
    elif dataset == 'CelebA':
        data_obj = CelebA()
   
    # load the data
    X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test= data_obj.load_embeddings( early_stopping, batch_size, data_augmentation, seed)

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
    
    # create an sklearn model
    logreg = LogisticRegression(**model_param)

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
    elif method == 'GW-ERM':
        subsample_weights= False
        k_subsamples = 1

    # define the method name if GW-ERM with GDRO= True
    if method == 'GW-ERM' and GDRO:
        method = 'GDRO'


    # define the p_standard
    if subsample_weights:
        # count how often each group occurs, determine the weights
        p_standard = calc_subsample_ood_weights(p_train, n_train)
    else:
        p_standard = p_ood 
   
    # create a weight searcher object
    set_seed(seed)
    ws = weight_searcher(logreg, X_train, y_train, g_train, X_val, y_val, g_val, p_ood,
                          GDRO=GDRO, weight_rounding=4, p_min=10e-4, subsample_weights=subsample_weights, k_subsamples=k_subsamples)
    

    # get the group with the lowest number of samples, use if subsample_weights=True
    min_group = min([len(g_train[g_train==group]) for group in np.unique(g_train)])
    lock_in_p_g = min_group if subsample_weights else None

    # optimize the weights
    p_hat =  ws.optimize_weights( p_standard, T,  lr,  momentum, patience=patience,    
                                  verbose=verbose,  lr_schedule=lr_schedule,stable_exp=stable_exp, lock_in_p_g = lock_in_p_g,
                                  save_trajectory=False, decay=decay)
    
    # define the model with p_hat
    weights_obj_opt = weights(p_hat, p_train, weighted_loss_weights=not subsample_weights)
    logreg_opt_weights = model(weights_obj_opt, logreg, add_intercept=True, subsampler=subsample_weights, k_subsamples=k_subsamples)
    logreg_opt_weights.fit_model(X_train, y_train, g_train, seed=seed)

    # measure the loss on the validation data for the optimized weights
    y_val_pred= logreg_opt_weights.predict(X_val, type_pred='class')
    wg_val, weighted_acc_val = calc_worst_and_weighted_acc(y_val, y_val_pred, g_val)

    # measure the loss on test data
    y_test_pred= logreg_opt_weights.predict(X_test, type_pred='class')
    wg_test, weighted_acc_test = calc_worst_and_weighted_acc(y_test, y_test_pred, g_test)


    print('---------------------------------------------------')
    print('--- Results on validation data (Optimized weights) ---')
    print('Weighted accuracy on val data: ', weighted_acc_val)
    print('Worst group on val data: ', wg_val)

    print('--- Results on test data---')
    print('Weighted accuracy on test data: ', weighted_acc_test)
    print('Worst group on test data: ', wg_test)
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
    parser.add_argument('--GDRO', type=str, default='false', help='Use GDRO')
    parser.add_argument('--T', type=int, default=300, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum')
    parser.add_argument('--patience', type=int, default=300, help='Patience')
    parser.add_argument('--lr_schedule', type=str, default='constant', help='Learning rate schedule')
    parser.add_argument('--decay', type=float, default=0.99, help='Decay')
    parser.add_argument('--solver', type=str, default='liblinear', help='Solver')
    parser.add_argument('--verbose', type=str, default='true', help='Verbose')
    parser.add_argument('--stable_exp', type=bool, default=False, help='Stable exp')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations')
    args = parser.parse_args()

    # turn str to bool for relevant arguments
    args.early_stopping = str_to_bool(args.early_stopping)
    args.data_augmentation = str_to_bool(args.data_augmentation)
    args.GDRO = str_to_bool(args.GDRO)
    args.verbose = str_to_bool(args.verbose)

    # run the main function
    main(args.dataset, args.early_stopping, args.batch_size, args.data_augmentation, args.seed,
          args.penalty_strength, args.method, args.GDRO, args.T, args.lr, args.momentum, args.patience, args.lr_schedule, args.decay, 
          args.solver, args.verbose, args.stable_exp, args.tol, args.max_iter)

   
    
