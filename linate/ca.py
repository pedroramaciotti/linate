"""LINATE module 1: Correspondece Analysis """

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

import os.path

import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix

# the user can specify the CA computation library to use
from importlib import import_module

class CA(BaseEstimator, TransformerMixin): 

    default_ca_engines = ['sklearn', 'auto', 'fbpca']  # by default use the 'prince' code for CA computation

    def __init__(self, n_components = 2, n_iter = 10, copy = True, check_input = True, random_state = None,
            engine = 'auto', in_degree_threshold = None, out_degree_threshold = None):

        self.random_state = random_state

        self.check_input = check_input    # sklearn valid input check
        self.copy = copy                  # make a copy of the array

        self.n_components = n_components    # number of CA coordinates
        self.n_iter = n_iter                # number of iteration in SVD computation

        self.engine = engine 
        print('Using CA engine:', self.engine)
        self.ca_module_name = 'prince'
        if self.engine not in self.default_ca_engines:
            self.ca_module_name = self.engine

        try:
            self.ca_module = import_module(self.ca_module_name)
        except ModuleNotFoundError:
            raise ValueError(self.ca_module_name 
                    + ' module is not installed; please install and make it visible if you want to use it')

        self.in_degree_threshold = in_degree_threshold # nodes that are followed by less than this number
                                                       # (in the original graph) are taken out of the network
        self.out_degree_threshold = out_degree_threshold # nodes that follow less than this number
                                                       # (in the original graph) are taken out of the network

    def fit(self, X, y = None):
        return self

    def load_input_from_file(self, path_to_network_data, network_file_header_names = None):

        # check that a network file is provided
        if path_to_network_data is None:
            raise ValueError('Network file name is not provided.')

        # check network file exists
        if not os.path.isfile(path_to_network_data):
            raise ValueError('Network file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_to_network_data, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Network file has to have at least two columns.')

        # sanity checks in header
        if network_file_header_names is not None:
            if network_file_header_names['source'] not in header_df.columns:
                raise ValueError('Network file has to have a ' + network_file_header_names['source'] + ' column.')
            if network_file_header_names['target'] not in header_df.columns:
                raise ValueError('Network file has to have a ' + network_file_header_names['target'] + ' column.')

        # load network data
        input_df = None
        if network_file_header_names is None:
            if column_no == 2:
                input_df = pd.read_csv(path_to_network_data, header = None,
                        dtype = {0:str, 1:str}).rename(columns = {0:'source', 1:'target'})
            else:
                input_df = pd.read_csv(path_to_network_data, header = None,
                        dtype = {0:str, 1:str}).rename(columns = {0:'source', 1:'target', 2:'multiplicity'})
        else:
            input_df = pd.read_csv(path_to_network_data, dtype = {network_file_header_names['source']:str,
                network_file_header_names['target']:str}).rename(columns = {network_file_header_names['source']:'source', 
                    network_file_header_names['target']:'target'})

        input_df = self.__check_input_and_convert_to_matrix(input_df)

        return(input_df)

    def __check_input_and_convert_to_matrix(self, input_df):

        # first perform validity checks over the input 
        if not isinstance(input_df, pd.DataFrame):
            raise ValueError('Input should be a pandas dataframe.')

        if 'source' not in input_df.columns:
            raise ValueError('Input dataframe should have a source column.')

        if 'target' not in input_df.columns:
            raise ValueError('Input dataframe should have a target column.')

        # remove NAs from input data
        input_df.dropna(subset = ['source', 'target'], inplace = True)

        # convert to 'str'
        input_df['source'] = input_df['source'].astype(str)
        input_df['target'] = input_df['target'].astype(str)

        # the file should either half repeated edges or a multiplicity column but not both
        has_more_columns = True if input_df.columns.size > 2 else False
        has_repeated_edges = True if input_df.duplicated(subset = ['source', 'target']).sum() > 0 else False
        if has_more_columns and has_repeated_edges:
            raise ValueError('There cannot be repeated edges AND a 3rd column with edge multiplicities.')

        # if there is a third column, it must containt integers
        if has_more_columns:
            if 'multiplicity' not in input_df.columns:
                raise ValueError('Input dataframe should have a multiplicity column.')
            input_df['multiplicity'] = input_df['multiplicity'].astype(int) # will fail if missing element, or cannot convert

        # remove nodes with small degree if needed
        degree_per_target = None
        if self.in_degree_threshold is not None:
            degree_per_target = input_df.groupby('target').count()

        degree_per_source = None
        if self.out_degree_threshold is not None:
            degree_per_source = input_df.groupby('source').count()

        if degree_per_target is not None:
            if 'multiplicity' in degree_per_target.columns:
                degree_per_target.drop('multiplicity', axis = 1, inplace = True)
            degree_per_target = degree_per_target[degree_per_target >= self.in_degree_threshold].dropna().reset_index()
            degree_per_target.drop('source', axis = 1, inplace = True)
            input_df = pd.merge(input_df, degree_per_target, on = ['target'], how = 'inner')

        if degree_per_source is not None:
            if 'multiplicity' in degree_per_source.columns:
                degree_per_source.drop('multiplicity', axis = 1, inplace = True)
            degree_per_source = degree_per_source[degree_per_source >= self.out_degree_threshold].dropna().reset_index()
            degree_per_source.drop('target', axis = 1, inplace = True)
            input_df = pd.merge(input_df, degree_per_source, on = ['source'], how = 'inner')

        # checking if final network is bipartite:
        self.is_bipartite_ = np.intersect1d(input_df['source'], input_df['target']).size == 0
        print('Bipartite graph: ', self.is_bipartite_)

        # and then assemble the matrices to be fed to CA
        ntwrk_df = input_df[['source', 'target']]

        n_i, r = ntwrk_df['target'].factorize()
        source_users_no_ = len(np.unique(n_i))
        self.column_ids_ = r.values
        n_j, c = ntwrk_df['source'].factorize()
        assert len(n_i) == len(n_j)
        target_users_no_ = len(np.unique(n_j))
        self.row_ids_ = c.values
        network_edge_no = len(n_i)
        n_in_j, tups = pd.factorize(list(zip(n_j, n_i)))
        ntwrk_csr = csr_matrix((np.bincount(n_in_j), tuple(zip(*tups)))) # COO might be faster

        if self.engine in self.default_ca_engines:
            ntwrk_np = ntwrk_csr.toarray()
            return (ntwrk_np)

        return (ntwrk_csr)
