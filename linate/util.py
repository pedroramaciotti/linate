# take two dataframes a point per row and compute
# euclidean distance between each pairs of rows

import pandas as pd

import numpy as np

from scipy.stats import pearsonr

def compute_euclidean_distance(X, Y):

    if X is None:
        raise ValueError('First parameter cannot be none')

    if Y is None:
        raise ValueError('Second parameter cannot be none')

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if not isinstance(X, np.ndarray):
        raise ValueError('First parameter must be a dataframe or a numpy array.')

    if not isinstance(Y, np.ndarray):
        raise ValueError('Second parameter must be a dataframe or a numpy array.')

    if len(X.shape) != len(Y.shape):
        raise ValueError('The two matrices are not compatible.') 

    # X and Y should have the same dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Dimensions of matrices should have the same shape.') 

    if X.shape[1] != Y.shape[1]:
        raise ValueError('Dimensions of matrices should have the same shape.') 

    # compute euclidean norm of each row
    l2 = np.sqrt(np.power(X - Y, 2.0).sum(axis = 1))
    #agg_l2 = np.sqrt(np.power(l2, 2.0).sum()) / float(len(l2))

    return(l2)

def compute_correlation_coefficient(X, Y):

    if X is None:
        raise ValueError('First parameter cannot be none')

    if Y is None:
        raise ValueError('Second parameter cannot be none')

    if not isinstance(X, pd.DataFrame):
        raise ValueError('First parameter must be a dataframe.')

    if not isinstance(Y, pd.DataFrame):
        raise ValueError('Second parameter must be a dataframe.')

    # X and Y should have the same dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Dataframes should have the same number of rows.') 

    X_dim_alias = ['d%d' % (d + 1) for d in range(len(X.columns))]

    corr = pd.DataFrame(index = X_dim_alias, columns = Y.columns)
    corp = pd.DataFrame(index = X_dim_alias, columns = Y.columns)

    for x_d_indx in range(len(X.columns)):
        for y_d in Y.columns:
            corr.loc[X_dim_alias[x_d_indx], y_d], corp.loc[X_dim_alias[x_d_indx], y_d] = pearsonr(X[X.columns[x_d_indx]], Y[y_d])

    return(corr, corp)
