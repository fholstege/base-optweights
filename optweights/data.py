
import numpy as np


class data():

    def __init__(self):
        pass




class Toy(data):


    def __init__(self,  n, p, beta_1, beta_0, sigma_1, sigma_0, mu, gamma, a_0, a_1, d=1):
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

        

    def dgp_mv(self, add_index_group=True, logistic=False):
        

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

        if add_index_group:
            g += 1

        return X, y, g


