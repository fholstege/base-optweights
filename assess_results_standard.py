



# import helper function
from helpers import  str_to_bool, select_df_for_hyper_param_search, get_best_hyper_param_method

# import functions from other libraries
import  os, sys
import argparse
import pandas as pd
import time

# import functions to get the best hyper-parameters and results
from generate_result_standard import main as main_standard



def main(dataset, early_stopping, batch_size, data_augmentation, seeds,  method, param_selection_method,  solver, tol,  max_iter, fraction_original_data, save, result_folder):
    """
    Run generate the result for optimal weights, using the hyper-parameters of the standard methods
    """


    # create a dictionary to store the results
    n_seeds = len(seeds)
    result_table = {i: {} for i in range(n_seeds)}


    # set the other necessary parameters. If method == DFR, val_fit = True
    if method == 'DFR':
        val_fit = True
    else:
        val_fit = False

    # loop over the seeds
    i = 0
    for seed in seeds:


        # get the params for the current combination
        print('Results for seed {}/{}'.format(i+1, n_seeds))
            
        # if param_selection_method is 'from standard', get the parameters from the standard method
        if param_selection_method == 'param search':

            # load the results of the standard method
            result_file = 'results/param_search/results_param_search_{}.csv'.format(dataset)
            result_table_standard = pd.read_csv(result_file)

            # select the rows which share key features of the data and method
            result_table_standard_selected = select_df_for_hyper_param_search(result_table_standard, dataset, method, early_stopping, batch_size, data_augmentation, seed, solver, tol, fraction_original_data, val_fit, max_iter)
            best_param_dict = get_best_hyper_param_method(method, result_table_standard_selected)

            # get the best parameters
            penalty_strength = best_param_dict['penalty_strength']

            # get the best param for GDRO
            if method == 'GDRO':
                C_GDRO = best_param_dict['C_GDRO']
                eta_param_GDRO = best_param_dict['eta_param_GDRO']
            else:
                C_GDRO = None
                eta_param_GDRO = None
            
            # get the best lambda_JTT
            if method == 'JTT':
                lambda_JTT = best_param_dict['lambda_JTT']
            else:
                lambda_JTT = None
            print('Best parameters: penalty_strength: {}, C_GDRO: {}, eta_param_GDRO: {}, lambda_JTT: {}'.format(penalty_strength, C_GDRO, eta_param_GDRO, lambda_JTT))
        
        # run the main function
        result_param_key, _ = main_standard(dataset, early_stopping, batch_size, data_augmentation, seed, 
                                            penalty_strength=penalty_strength, 
                                            lambda_JTT=lambda_JTT, 
                                            C_GDRO=C_GDRO,
                                            eta_param_GDRO=eta_param_GDRO,
                                            method = method,
                                                solver = solver,
                                                tol = tol,
                                                max_iter = max_iter,
                                                val_fit = val_fit,
                                                fraction_original_data=fraction_original_data,
                                                save = False,
                                                result_folder = None)
            
        # get the key
        key = result_param_key['key']

        # now, create an entry for the result_table
        entry = {'dataset': dataset,
                    'early_stopping': early_stopping,
                    'batch_size': batch_size,
                    'data_augmentation': data_augmentation,
                    'method': method,
                    'seed': seed,
                        'fraction_original_data': fraction_original_data,
                    'val_fit': val_fit,
                    'wg_val': result_param_key['wg_val'],
                    'wg_test': result_param_key['wg_test'],
                    'weighted_acc_val': result_param_key['weighted_acc_val'],
                    'weighted_acc_test': result_param_key['weighted_acc_test'],
                    'penalty_strength': penalty_strength,
                    'lambda_JTT': lambda_JTT,
                    'C_GDRO': C_GDRO,
                    'eta_param_GDRO': eta_param_GDRO,
                    'solver': solver,
                    'tol': tol,
                    'max_iter': max_iter,
                    'key': key}
        

        # save the result in the result_table
        result_table[i] = entry
        i += 1

    # save the results
    if save:

        ## first, save the results
            
        # the result file is...
        result_file = 'results_standard_{}.csv'.format(dataset)

        # turn the dict into a pandas dataframe
        result_table_df =pd.DataFrame.from_dict(result_table, orient='index')

        # add the time
        result_table_df['time'] = time.ctime()
     
        # check if the file exists
        if os.path.exists(result_folder + '/' + result_file):

            # load the main file
            main_table_df = pd.read_csv(result_folder + '/' + result_file)

            # concat in such a way that if there are new columns, they are added
            # where the empty values are filled with NaN
            main_table_df = pd.concat([main_table_df, result_table_df], axis=0, sort=False)

            # check: entries with the same key? - if so, remove the old one
            main_table_df = main_table_df.drop_duplicates(subset='key', keep='last')

            # save the main file
            main_table_df.to_csv(result_folder + '/' + result_file, index=False)

        else:
            # save the result table
            result_table_df.to_csv(result_folder + '/' + result_file, index=False)
      
            


    

# if main is run, run the tests
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Arguments for the experiment')
    parser.add_argument('--dataset', type=str, default='WB', help='Dataset to use')
    parser.add_argument('--early_stopping', type=str, default='true', help='Use early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_augmentation', type=str, default='false', help='Use data augmentation')
    parser.add_argument('--seeds', type=str, default='1', help='Seed')
    parser.add_argument('--method', type=str, default='GW-ERM', help='Method to use')
    parser.add_argument('--param_selection_method', type=str, default='from standard', help='Method to select the hyper-parameters')
    parser.add_argument('--solver', type=str, default='liblinear', help='Solver')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations')
    parser.add_argument('--fraction_original_data', type=float, default=1, help='Fraction of original data to use')
    parser.add_argument('--save', type=str, default='false', help='Save the results')
    parser.add_argument('--result_folder', type=str, default='results/param_search', help='Document to save the results')
   
    args = parser.parse_args()

    # turn str to bool for relevant arguments
    args.early_stopping = str_to_bool(args.early_stopping)
    args.data_augmentation = str_to_bool(args.data_augmentation)
    args.save = str_to_bool(args.save)

        # convert seed to list of ints
    args.seeds = [int(seed) for seed in args.seeds.split('-')]



    # run the main function
    main(args.dataset, args.early_stopping, args.batch_size, args.data_augmentation, args.seeds, args.method, args.param_selection_method,
            args.solver, args.tol, args.max_iter, args.fraction_original_data, args.save, args.result_folder)

