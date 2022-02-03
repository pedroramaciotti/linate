"""LINATE module 2: Correspondece Analsis + dimension matching"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

import os.path

import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix

import prince

class LINATE(BaseEstimator, TransformerMixin): 

    def __init__(self, n_components = 2, out_degree_threshold = None, in_degree_threshold = None,
            n_iter = 10, copy = True, check_input = True, random_state = None, target_col_name = 'target',
            source_col_name = 'source', multiplicity_col_name = 'multiplicity', engine = 'auto'):

        self.random_state = random_state
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.engine = engine 
        self.out_degree_threshold = out_degree_threshold
        self.in_degree_threshold = in_degree_threshold
        self.source_col_name = source_col_name
        self.target_col_name = target_col_name
        self.multiplicity_col_name = multiplicity_col_name

    def fit(self, X, y = None):
        #self.random_state_ = check_random_state(self.random_state)

        # load data 
        if not isinstance(X, str):
            raise TypeError('X should be of type string...')

        if not os.path.exists(X):
            raise BaseException('Twitter network file does not exist...') 

        in_nd_thrshold = self.in_degree_threshold
        if in_nd_thrshold is None:
            in_nd_thrshold = 0
        out_nd_thrshold = self.out_degree_threshold
        if out_nd_thrshold is None:
            out_nd_thrshold = 0

        ntwrk_df = pd.read_csv(X)
        if len(ntwrk_df.columns) < 2:
            raise BaseException('Twitter network file should have at least a \'source\' and \'target\' attribute') 
        if (ntwrk_df.columns[0] != self.target_col_name) and (ntwrk_df.columns[1] != self.target_col_name):
            raise BaseException('Twitter network file one of first two columns should be the \'target\' attribute') 
        if (ntwrk_df.columns[0] != self.source_col_name) and (ntwrk_df.columns[1] != self.source_col_name):
            raise BaseException('Twitter network file one of first two columns should be the \'source\' attribute') 
        ntwrk_df.dropna(subset = [self.target_col_name, self.source_col_name], inplace = True)

        #print(ntwrk_df.shape)
        if in_nd_thrshold > 0:
            degree_per_target = ntwrk_df.groupby(self.target_col_name).count()
            if self.multiplicity_col_name in degree_per_target.columns:
                degree_per_target.drop(self.multiplicity_col_name, axis = 1, inplace = True)
            degree_per_target = degree_per_target[degree_per_target > in_nd_thrshold].dropna().reset_index()
            degree_per_target.drop(self.source_col_name, axis = 1, inplace = True)
            ntwrk_df = pd.merge(ntwrk_df, degree_per_target, on = [self.target_col_name], how = 'inner')
        #print(ntwrk_df.shape)

        if out_nd_thrshold > 0:
            degree_per_source = ntwrk_df.groupby(self.source_col_name).count()
            if self.multiplicity_col_name in degree_per_source.columns:
                degree_per_source.drop(self.multiplicity_col_name, axis = 1, inplace = True)
            degree_per_source = degree_per_source[degree_per_source > out_nd_thrshold].dropna().reset_index()
            degree_per_source.drop(self.target_col_name, axis = 1, inplace = True)
            ntwrk_df = pd.merge(ntwrk_df, degree_per_source, on = [self.source_col_name], how = 'inner') 

        # TODO: what should we be doing witht he multiplicity column?
        ntwrk_df.drop(self.multiplicity_col_name, axis = 1, inplace = True)
        #print(ntwrk_df.shape)

        n_i, r = ntwrk_df[self.source_col_name].factorize()
        self.source_users_no_ = len(np.unique(n_i))
        self.column_ids_ = r.values
        #print(source_users_no, len(n_i), len(r))
        n_j, c = ntwrk_df[self.target_col_name].factorize()
        assert len(n_i) == len(n_j)
        self.target_users_no_ = len(np.unique(n_j))
        self.row_ids_ = c.values
        #print(target_users_no, len(n_j), len(c))
        network_edge_no = len(n_i)
        n_in_j, tups = pd.factorize(list(zip(n_j, n_i)))
        ntwrk_csr = csr_matrix((np.bincount(n_in_j), tuple(zip(*tups))))
        # TODO : add delayed sparse?
        ntwrk_np = ntwrk_csr.toarray()
        #print(ntwrk_np.shape)

        ca_model = ca_randomized = prince.CA(n_components = self.n_components, n_iter = self.n_iter,
                copy = self.copy, check_input = self.check_input, engine = self.engine, random_state = self.random_state)
        ca_model.fit(ntwrk_np)

        self.ca_row_coordinates_ = ca_model.row_coordinates(ntwrk_np) # pandas data frame
        self.ca_column_coordinates_ = ca_model.column_coordinates(ntwrk_np) # pandas data frame
        self.eigenvalues_ = ca_model.eigenvalues_  # list
        self.total_inertia_ = ca_model.total_inertia_  # numpy.float64
        self.explained_inertia_ = ca_model.explained_inertia_ # list

        return self
