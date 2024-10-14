


# import the add_up function from optweights
from optweights.weights import weights
from data import WB, CelebA, multiNLI
from optweights.model import model
from optweights.weight_searcher import weight_searcher
from optweights.helpers import calc_subsample_ood_weights, get_p_dict, set_seed, str_to_bool, create_key, get_fraction_original_data, split_data_in_train_val
from optweights.metrics import calc_worst_and_weighted_acc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from matplotlib import pyplot as plt
import sys, os
import argparse
import numpy as np
import pandas as pd
from generate_result_standard import main as main_standard
import time
import pickle

def main(dataset, early_stopping, batch_size, data_augmentation, seeds,  method,  solver, tol, penalty_strengths, lambdas_JTT, Cs_GDRO, etas_param_GDRO, max_iter, fraction_original_data, save, result_folder):
    """
    Run the parameter search for the standard methods
    """

    # first, we need to create a set of param combinations based on the method and the penalty strength/lambda_JTT/C_GDRO/eta_param_GDRO
    if method == 'GW-ERM' or method == 'SUBG' or method == 'DFR':
        
        # create parameter combinations of - penalty strength, lambda_JTT, C_GDRO, eta_param_GDRO
        param_combinations = [{'penalty_strength':penalty_strength} for penalty_strength in penalty_strengths ]

    elif method == 'JTT':

        # create parameter combinations of - penalty strength, lambda_JTT, C_GDRO, eta_param_GDRO
        param_combinations = [{'penalty_strength': penalty_strength, 'lambda_JTT':lambda_JTT} for lambda_JTT in lambdas_JTT for penalty_strength in penalty_strengths]

    elif method == 'GDRO':
            
        # create parameter combinations of - penalty strength, lambda_JTT, C_GDRO, eta_param_GDRO
        param_combinations = [{'penalty_strength': penalty_strength, 'C_GDRO':C_GDRO, 'eta_param_GDRO':eta_param_GDRO} for C_GDRO in Cs_GDRO for eta_param_GDRO in etas_param_GDRO for penalty_strength in penalty_strengths]

    # define the dictionary to store the results - count n combinations for n_seeds
    n_combinations = len(param_combinations)
    n_seeds = len(seeds)
    result_table = {i: {} for i in range(n_combinations* n_seeds)}
    models ={}

    # loop over the combinations
    i = 0
    j=0
    for param_combination in param_combinations:
       

        # loop over the seeds
        for seed in seeds:

            # get the params for the current combination
            print('Running combination {}/{} for seed {}/{}'.format(j+1, n_combinations, seed, n_seeds))
            print('Params: {}'.format(param_combination))

            # if penalty strength is in the param_combination, set the penalty strength
            if 'penalty_strength' in param_combination:
                penalty_strength_combination = param_combination['penalty_strength']
            else:
                # raise an error
                raise ValueError('Penalty strength must be in the param_combination')
            
            # if lambda_JTT is in the param_combination, set the lambda_JTT
            if 'lambda_JTT' in param_combination:
                lambda_JTT_combination = param_combination['lambda_JTT']
            else:
                # set the standard 
                lambda_JTT_combination = None

            # if C_GDRO is in the param_combination, set the C_GDRO
            if 'C_GDRO' in param_combination:
                C_GDRO_combination = param_combination['C_GDRO']
            else:
                # set the standard 
                C_GDRO_combination = None

            # if eta_param_GDRO is in the param_combination, set the eta_param_GDRO
            if 'eta_param_GDRO' in param_combination:
                eta_param_GDRO_combination = param_combination['eta_param_GDRO']
            else:
                # set the standard 
                eta_param_GDRO_combination = None

            
            # set the other necessary parameters. If method == DFR, val_fit = True
            if method == 'DFR':
                val_fit = True
            else:
                val_fit = False
                
            # run the main function
            result_param_key, logreg_model_standard = main_standard(dataset, early_stopping, batch_size, data_augmentation, seed, 
                                            penalty_strength=penalty_strength_combination, 
                                            lambda_JTT=lambda_JTT_combination, 
                                            C_GDRO=C_GDRO_combination,
                                            eta_param_GDRO=eta_param_GDRO_combination,
                                            method = method,
                                                solver = solver,
                                                tol = tol,
                                                max_iter = max_iter,
                                                val_fit = val_fit,
                                                fraction_original_data=fraction_original_data,
                                                save = False)
            
            # get the key
            key = result_param_key['key']

            # save the result in the result_table
            result_table[i] = result_param_key

            # save the model via the key
            models[key] = logreg_model_standard

            # proceed
            i += 1
        j+=1
    

      
   
    # save the results
    if save:

        ## first, save the results
            
        # the result file is...
        result_file = 'results_param_search_{}.csv'.format(dataset)

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
        

        ## second, save the models in the same directory
        # create a directory in the folder called 'models'
        model_folder = result_folder + '/models'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        # save the models
        for key, model in models.items():

            # save the model as a pickle file
            with open(model_folder + '/{}.pkl'.format(key), 'wb') as f:
                pickle.dump(model, f)
           
            


    

# if main is run, run the tests
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for the experiment')
    parser.add_argument('--dataset', type=str, default='WB', help='Dataset to use')
    parser.add_argument('--early_stopping', type=str, default='true', help='Use early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_augmentation', type=str, default='false', help='Use data augmentation')
    parser.add_argument('--seeds', type=str, default='1', help='Seed')
    parser.add_argument('--method', type=str, default='GW-ERM', help='Method to use')
    parser.add_argument('--solver', type=str, default='liblinear', help='Solver')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations')
    parser.add_argument('--save', type=str, default='false', help='Save the results')
    parser.add_argument('--result_folder', type=str, default='results/param_search', help='Document to save the results')
    parser.add_argument('--penalty_strengths', type=str, default='1', help='Penalty strengths to use')
    parser.add_argument('--Cs_GDRO', type=str, default='0', help='C for GDRO')
    parser.add_argument('--etas_param_GDRO', type=str, default='0.00001', help='eta_param for GDRO')
    parser.add_argument('--lambdas_JTT', type=str, default='1', help='lambda_JTT')
    parser.add_argument('--fraction_original_data', type=float, default=1, help='Fraction of original data to use')

    args = parser.parse_args()

    # turn str to bool for relevant arguments
    args.early_stopping = str_to_bool(args.early_stopping)
    args.data_augmentation = str_to_bool(args.data_augmentation)
    args.save = str_to_bool(args.save)

        # convert seed to list of ints
    args.seeds = [int(seed) for seed in args.seeds.split('-')]

    # turn the penalty strength into a list of floats
    args.penalty_strengths = [float(penalty_strength) for penalty_strength in args.penalty_strengths.split('-')]

    # turn the lambda_JTT into a list of floats
    args.lambdas_JTT = [float(lambda_JTT) for lambda_JTT in args.lambdas_JTT.split('-')]

    # turn the C_GDRO into a list of floats
    args.Cs_GDRO = [float(C_GDRO) for C_GDRO in args.Cs_GDRO.split('-')]

    # turn the eta_param_GDRO into a list of floats
    args.etas_param_GDRO = [float(eta_param_GDRO) for eta_param_GDRO in args.etas_param_GDRO.split('-')]



    # run the main function
    main(args.dataset, args.early_stopping, args.batch_size, args.data_augmentation, args.seeds,
          args.method, args.solver,args.tol, 
          penalty_strengths = args.penalty_strengths, 
          lambdas_JTT=args.lambdas_JTT,
          Cs_GDRO=args.Cs_GDRO, 
          etas_param_GDRO=args.etas_param_GDRO,
          max_iter = args.max_iter, 
          fraction_original_data = args.fraction_original_data,
          save = args.save, result_folder = args.result_folder)

   
    


