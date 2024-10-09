
import numpy as np
import torch
import sys

class data():

    def __init__(self):
        pass

    def demean_X(self, X_train, X_val, X_test):
        """
        Demean the data
        """

        # estimate the mean based on the training data
        X_train_mean = X_train.mean(dim = 0)  

        # demean the data based on the mean estimated from the training data 
        X_train_demeaned = X_train - X_train_mean
        X_val_demeaned = X_val - X_train_mean
        X_test_demeaned = X_test - X_train_mean

     
        return X_train_demeaned, X_val_demeaned, X_test_demeaned

    
    def unit_stdev_X(self, X_train, X_val, X_test):
        """
        Set the standard deviation of each column in X to 1
        """

        # estimate the standard deviation based on the training data
        X_train_std = X_train.std(dim = 0)
        
        # standardize the data
        X_train_standardized = X_train/X_train_std
        X_val_standardized = X_val/X_train_std
        X_test_standardized = X_test/X_train_std

        return X_train_standardized, X_val_standardized, X_test_standardized


    
    def demean_scale_X(self, X_train, X_val, X_test):
        """
        Demean and scale the data
        """

        # demean the data
        X_train_demeaned, X_val_demeaned, X_test_demeaned = self.demean_X(X_train, X_val, X_test)

        # scale the data
        X_train_scaled, X_val_scaled, X_test_scaled = self.unit_stdev_X(X_train_demeaned, X_val_demeaned, X_test_demeaned)

        return X_train_scaled, X_val_scaled, X_test_scaled
    

    def prep_embeddings(self, data):

        # get the data from X
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']

        # get the data from y
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']

        # get the data from g
        g_train = data['g_train']
        g_val = data['g_val']
        g_test = data['g_test']

        # demean and scale the data
        X_train, X_val, X_test = self.demean_scale_X(X_train, X_val, X_test)

        # if not numpy, turn to numpy
        if not isinstance(X_train, np.ndarray):
            X_train, X_val, X_test = X_train.cpu().numpy(), X_val.cpu().numpy(), X_test.cpu().numpy()
        if not isinstance(y_train, np.ndarray):
            y_train, y_val, y_test = y_train.cpu().numpy(), y_val.cpu().numpy(), y_test.cpu().numpy()
        if not isinstance(g_train, np.ndarray):
            g_train, g_val, g_test = g_train.cpu().numpy(), g_val.cpu().numpy(), g_test.cpu().numpy()

        return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test
    
class CelebA(data):

    def load_embeddings(self, early_stopping, batch_size, data_augmentation, seed):
        """
        Load the embeddings
        """
        
        folder_embeddings = 'data/embeddings/CelebA/param_set_ES_{}_BS_{}_DA_{}/CelebA_model_seed_{}/'.format(early_stopping, batch_size, data_augmentation, seed)

        # load the .pt files
        data = torch.load(folder_embeddings + '/data.pt')

        # prepare the embeddings
        X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = self.prep_embeddings(data)

        return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test




class WB(data):

    def load_embeddings(self, early_stopping, batch_size, data_augmentation, seed):
        """
        Load the embeddings
        """

        # define the folder
        folder_embeddings = 'data/embeddings/WB/param_set_ES_{}_BS_{}_DA_{}/WB_model_seed_{}/'.format(early_stopping, batch_size, data_augmentation, seed)

        # load the .pt files
        data = torch.load(folder_embeddings + '/data.pt')

        # prepare the embeddings
        X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = self.prep_embeddings(data)

        return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test
        
        
class Toy(data):


    def __init__(self,  n, p, beta_1, beta_0, sigma_1, sigma_0, mu, gamma, a_0, a_1, d):
        super().__init__()

        # set the parameters of the Toy data
        self.n = n
        self.p = p
        self.beta_1 = beta_1
        self.beta_0 = beta_0
        self.sigma_1 = sigma_1
        self.sigma_0 = sigma_0
        self.mu = mu
        self.gamma = gamma
        self.a_0 = a_0
        self.a_1 = a_1
        self.d = d
        self.n_0 = int(np.round(n * (1 - p)))
        self.n_1 = n - self.n_0

        

    def dgp_mv(self, logistic=False):
        

        # generate multivariate x with mean mu and covariance matrix based on gamma
        Sigma = np.diag(np.ones(self.d)) * self.gamma
        X = np.random.multivariate_normal(self.mu, Sigma, self.n)

        # create n obs. for g
        g = np.zeros(self.n)

        # set n_1 random obs. to 1
        i_1 = np.random.choice(self.n, self.n_1, replace=False)
        g[i_1] = 1

        # create empty y
        y = np.zeros(self.n)

        # if g = 1, generate y with mean beta_1*x + a and variance sigma_1
        X_1 = X[g == 1]
        eps_1 = np.random.normal(0, np.sqrt(self.sigma_1), self.n_1).reshape(-1, 1)
        B_1 = np.zeros((self.d, 1))
        B_1[0] = self.beta_1


        # if logistic, first generate the p(y=1)
        if logistic:
            p = 1/(1 + np.exp(-np.matmul(X_1, B_1) - self.a_1))
            y_1 = np.random.binomial(1, p, size=(self.n_1, 1))
    
        else:
            y_1 = np.matmul(X_1, B_1) + self.a_1 + eps_1
        
        y[g == 1] = y_1.squeeze(-1)

        # if g = 0, generate y with mean beta_0*x and variance sigma_2
        X_0 = X[g == 0]
        eps_0 = np.random.normal(0, np.sqrt(self.sigma_0), self.n_0).reshape(-1, 1)
        B_0 =  np.zeros((self.d, 1))
        B_0[0] = self.beta_0

        # if logistic, first generate the p(y=1)
        if logistic:
            p_0 = 1/(1 + np.exp(-np.matmul(X_0, B_0) - self.a_0))
            y_0 = np.random.binomial(1, p_0, size=(self.n_0, 1))
        else:
            y_0 = np.matmul(X_0, B_0) + self.a_0 + eps_0
        y[g == 0] = y_0.squeeze(-1)

        # expand the dim of x, y, g
        y, g = np.expand_dims(y, -1), np.expand_dims(g, -1)

        # save the parameters
        self.B_1 = B_1
        self.B_0 = B_0

        # transform g variable - add 1, turn to int
        g = g + 1
        g = g.astype(int)

        return X, y, g


