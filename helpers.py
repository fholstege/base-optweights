import numpy as np




def str_to_bool(text):
    """
    turn string to boolean
    """
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    


def create_key(dataset, method, penalty_strength, batch_size_model, augmentation_data, early_stopping_model, seed, solver, tol, fraction_original_data, val_fit,  lambda_JTT=None, C_GDRO=None, eta_param_GDRO=None, eta_q_GDRO=None ):
    key = '{}_METHOD_{}_PENALTY_{}_BS_{}_AD_{}_ES_{}_SEED_{}_SOLVER_{}_TOL_{}_FRACTION_{}_VAL_FIT_{}'.format(dataset, method, penalty_strength, batch_size_model, augmentation_data, early_stopping_model, seed, solver, tol, fraction_original_data, val_fit)

    if method == 'JTT':
         key += '_LAMBDA_{}'.format(lambda_JTT)
    elif method == 'GDRO':
         key += 'C_GDRO_{}_ETA_PARAM_GDRO_{}_ETA_Q_GDRO_{}'.format(C_GDRO, eta_param_GDRO, eta_q_GDRO)

    return key
    


def get_fraction_original_data(fraction, X, y, g, seed):
    """
     Sample a fraction of the original data
    """
     
    # get the number of samples for the data
    n = X.shape[0]

    # get the number of samples to keep
    n_tilde = int(np.ceil(n*fraction))

    # shuffle the indices
    indices = np.arange(n)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(indices)

    # get the subsample
    indices = shuffled_indices[:n_tilde]
    X_tilde = X[indices, :]
    y_tilde = y[indices]
    g_tilde = g[indices]

    return X_tilde, y_tilde, g_tilde, indices

def split_data_in_train_val(fraction, X, y, g, seed):
     
    # we now need to split the validation data into two parts
    # take all indeces, shuffle them
    np.random.seed(seed)
    indeces = np.arange(X.shape[0])
    np.random.shuffle(indeces)

    # select up to x% of the data for training
    n_frac = int(fraction*X.shape[0])
    indeces_1 = indeces[:n_frac]
    indeces_2 = indeces[n_frac:]

    # X_train, y_train, g_train become the first half, X_val, y_val, g_val become the second half
    X_train = X[indeces_1]
    y_train = y[indeces_1]
    g_train = g[indeces_1]
    X_val = X[indeces_2]
    y_val = y[indeces_2]
    g_val = g[indeces_2]

    return X_train, y_train, g_train, X_val, y_val, g_val



def select_df_for_hyper_param_search(df, dataset, method, early_stopping, batch_size, data_augmentation, seed,  solver, tol, fraction_original_data, val_fit, max_iter):

    # select the rows which share the arguments of the function
    df_selected = df[(df['dataset'] == dataset) & (df['method'] == method) & (df['early_stopping'] == early_stopping) & (df['batch_size'] == batch_size) & (df['data_augmentation'] == data_augmentation) & (df['seed'] == seed) & (df['solver'] == solver) & (df['tol'] == tol) & (df['fraction_original_data'] == fraction_original_data) & (df['val_fit'] == val_fit) & (df['max_iter'] == max_iter)]

    return df_selected



def get_best_hyper_param_method(method, param_search_df, selection_criteria='wg_val'):
    """
    param_search_df: dataframe with the hyper-parameter search results for the respective dataset
    """

    # Based on the method, we will loop over different hyper-parameters, and get all results
    if method == 'GW-ERM' or method == 'SUBG' or method == 'DFR':
        param_to_search = ['penalty_strength']
    elif method == 'GDRO':
        param_to_search = ['penalty_strength', 'C_GDRO', 'eta_param_GDRO']
    elif method == 'JTT':
        param_to_search = ['penalty_strength', 'lambda_JTT']

    # Based on the param to search, select the best model based on the selection criteria
    cols = param_to_search + [selection_criteria] + ['key'] # columns for the selection
    param_search_df_param_to_search = param_search_df[cols]
    
    # if worst group accuracy is the selection criteria, we select the model with the lowest value
    if selection_criteria == 'wg_val':
        # multiply the column by -1, so we can select the minimum value
        param_search_df_param_to_search.loc[:, 'wg_val'] = param_search_df_param_to_search.loc[:, 'wg_val']*-1

    # select the minimum value
    min_idx =param_search_df_param_to_search[selection_criteria]==param_search_df_param_to_search[selection_criteria].min()
    best_param = param_search_df_param_to_search.loc[min_idx, :]

    # reshape the best_param
    if len(best_param.shape) ==1:
        best_param = best_param.unsqueeze(-1).T

    # check if more than one row in best_param
    if best_param.shape[0] > 1:
        if param_to_search[0] == 'penalty_strength':

            # sort on penalty strength, and select the last row
            best_param = best_param.sort_values(by='penalty_strength', ascending=True)
            best_param = best_param.iloc[-1]
            print('More than one best hyper-parameter found, selected: {}'.format(best_param))  
    else:
        best_param = best_param.iloc[0]

    
    # instead of a pd.dataframe row, return as a dictionary with keys param_to_search
    best_param_dict = best_param[param_to_search].to_dict()

    return best_param_dict