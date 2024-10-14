


# import the add_up function from optweights
from optweights.weights import weights
from optweights.data import WB, CelebA, multiNLI
from optweights.model import model, GDRO_model, JTT_model
from optweights.weight_searcher import weight_searcher
from optweights.helpers import calc_subsample_ood_weights, get_p_dict, set_seed, str_to_bool, create_key, get_fraction_original_data, split_data_in_train_val
from optweights.metrics import calc_worst_and_weighted_acc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from matplotlib import pyplot as plt
import sys, os
import argparse
import numpy as np
import pandas as pd


def main(dataset, early_stopping, batch_size, data_augmentation, seed, penalty_strength, method,  solver, tol, val_fit=False, fraction_original_data=1, lambda_JTT=1, T_GDRO=100, C_GDRO=0, eta_param_GDRO=1e-5, eta_q_GDRO=0.1,  max_iter=100, frac_val_data=0.5, save=False, result_folder='results/param_search/'):

    # define the data object
    if dataset == 'WB':
        data_obj = WB()
    elif dataset == 'CelebA':
        data_obj = CelebA()
    elif dataset == 'multiNLI':
        data_obj = multiNLI()
   
    # load the data
    X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test= data_obj.load_embeddings( early_stopping, batch_size, data_augmentation, seed)

    # change fraction of original data
    if fraction_original_data < 1.0:
        X_train, y_train, g_train, indices_train = get_fraction_original_data( fraction_original_data, X_train, y_train, g_train, seed)
        X_val, y_val, g_val, indices_val = get_fraction_original_data(fraction_original_data,X_val, y_val, g_val, seed)
        print('Fraction of original data used: {}'.format(fraction_original_data))
    
    # if val_fit, use half of the validation data for training
    if val_fit:
        # save the original data
        X_val_orig, y_val_orig, g_val_orig = X_val, y_val, g_val

        # split the validation data in two
        X_train, y_train, g_train, X_val, y_val, g_val = split_data_in_train_val(frac_val_data, X_val, y_val, g_val, seed)
       

    
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
        # create a logistic regression model, via the SGDClassifier class in order to use the optimize_GDRO_via_SGD method
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
    elif method == 'GW-ERM' or method == 'GDRO' or method == 'JTT':
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
    

    # fit the model
    if method == 'GDRO': 
        logreg_standard_weights = GDRO_model(weights_obj_standard, logreg, add_intercept=True)
        Beta, _, _ = logreg_standard_weights.optimize_GDRO_via_SGD(X_train, y_train, g_train, X_val, y_val, g_val, T=T_GDRO, batch_size=X_train.shape[0], eta_param=eta_param_GDRO, eta_q=eta_q_GDRO, C=C_GDRO, early_stopping=False)
        logreg_standard_weights.Beta = Beta
    elif method == 'JTT':
        logreg_standard_weights = JTT_model(weights_obj_standard, logreg, add_intercept=True)

        # load the predictions from the JTT model
        y_hat_class_JTT_train, _,_ = data_obj.load_JTT_pred(seed)
        logreg_standard_weights.fit_model(X_train, y_train, y_hat_class_JTT_train, 0.5, lambda_JTT)

    else:
        logreg_standard_weights = model(weights_obj_standard, logreg, add_intercept=True, subsampler= subsample_weights, k_subsamples=k_subsamples)
        logreg_standard_weights.fit_model(X_train, y_train, g_train)

    # measure the loss on the validation and test data for the standard weights
    y_val_pred_standard= logreg_standard_weights.predict(X_val, type_pred='class')
    wg_val_standard, weighted_acc_val_standard = calc_worst_and_weighted_acc(y_val, y_val_pred_standard, g_val)

    # if val_fit, fit the whole model on the validation data
    if val_fit:
        # give warning if the method is GDRO
        if method == 'GDRO':
            print('Note: GDRO is not fitted on the validation data, since then cannot be stopped if worst-group loss degrades on the validation data')
        else:
            # fit the model on the whole validation data
            logreg_standard_weights.fit_model(X_val_orig, y_val_orig, g_val_orig)



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

    # save the results in a dict
    result = {'dataset': dataset,
            'early_stopping': early_stopping,
            'batch_size': batch_size,
            'data_augmentation': data_augmentation,
            'method': method,
            'seed': seed,
            'penalty_strength': penalty_strength,
            'weighted_acc_val': weighted_acc_val_standard,
            'wg_val': wg_val_standard,
            'weighted_acc_test': weighted_acc_test_standard,
            'wg_test': wg_test_standard}
    
    # from model param: remove the C key, replace with penalty_strength
    model_param.pop('C', None)
    model_param['penalty_strength'] = penalty_strength
    
    # create a key for the result
    key = create_key(dataset, method, penalty_strength, batch_size, data_augmentation, early_stopping, seed, solver, tol, fraction_original_data, val_fit, lambda_JTT, C_GDRO, eta_param_GDRO, eta_q_GDRO)

    # combine with the model param
    result_param_key = {**result, **model_param, 'key':key}


    # save the results
    if save:

        # the result file is...
        result_file = 'results_standard_{}.csv'.format(dataset)

        # turn the dict into a pandas dataframe
        result_table = pd.DataFrame(result_param_key, index=[0])

        # check if the file exists
        if os.path.exists(result_folder + '/' + result_file):

            # load the main file
            main_table = pd.read_csv(result_folder + '/' + result_file)

            # concat in such a way that if there are new columns, they are added
            # where the empty values are filled with NaN
            main_table = pd.concat([main_table, result_table], axis=0, sort=False)

            # check: entries with the same key? - if so, remove the old one
            main_table = main_table.drop_duplicates(subset='key', keep='last')

            # save the main file
            main_table.to_csv(result_folder + '/' + result_file, index=False)

        else:
            # save the result table
            result_table.to_csv(result_folder + '/' + result_file, index=False)



    return result_param_key, logreg_standard_weights




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
    parser.add_argument('--fraction_original_data', type=float, default=1, help='Fraction of original data')
    parser.add_argument('--val_fit', type=str, default='false', help='Fit the model on the validation data')
    parser.add_argument('--lambda_JTT', type=float, default=1, help='Lambda for JTT')
    parser.add_argument('--T_GDRO', type=int, default=100, help='T for GDRO')
    parser.add_argument('--C_GDRO', type=float, default=0, help='C for GDRO')
    parser.add_argument('--eta_param_GDRO', type=float, default=1e-5, help='Eta param for GDRO')
    parser.add_argument('--eta_q_GDRO', type=float, default=0.1, help='Eta q for GDRO')
    parser.add_argument('--save', type=str, default='false', help='Save the results')
    parser.add_argument('--result_folder', type=str, default='results/param_search', help='Document to save the results')
    args = parser.parse_args()

    # turn str to bool for relevant arguments
    args.early_stopping = str_to_bool(args.early_stopping)
    args.data_augmentation = str_to_bool(args.data_augmentation)
    args.save = str_to_bool(args.save)
    args.val_fit = str_to_bool(args.val_fit)

    # run the main function
    main(args.dataset, args.early_stopping, args.batch_size, args.data_augmentation, args.seed,
          args.penalty_strength, args.method, args.solver,args.tol, args.val_fit,
          fraction_original_data=args.fraction_original_data, lambda_JTT=args.lambda_JTT, T_GDRO=args.T_GDRO, C_GDRO=args.C_GDRO, eta_param_GDRO=args.eta_param_GDRO, eta_q_GDRO=args.eta_q_GDRO,
           max_iter = args.max_iter, save = args.save, result_folder = args.result_folder)

   
    


