"""LINATE module 1: Correspondece Analysis """

from sklearn.base import BaseEstimator, TransformerMixin

import os.path

import pandas as pd

import numpy as np

class AffineTransformation(BaseEstimator, TransformerMixin): 

    def __init__(self, N = None, random_state = None):

        self.random_state = random_state

        # number of latent ideological dimensions to be considered
        self.N = N # default : None --> P (number of groups) - 1

    def fit(self, X, Y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('\'X\' parameter must be a pandas dataframe')

        if 'entity' not in X.columns:
            raise ValueError('\'X\' has to have an \'entity\' column')

        if not isinstance(Y, pd.DataFrame):
            raise ValueError('\'Y\' parameter must be a pandas dataframe')

        if 'entity' not in Y.columns:
            raise ValueError('\'Y\' has to have an \'entity\' column')

        # also keep only the groups that exist in both datasets
        ga_merge_df = pd.merge(X, Y, on = 'entity', how = 'inner')
        X = X[X['entity'].isin(ga_merge_df.entity.unique())]
        Y = Y[Y['entity'].isin(ga_merge_df.entity.unique())]

        # finally fit an affine transformation to map X --> Y

        # first sort X and Y by entity (so as to have the corresponding mapping in the same rows)
        X = X.sort_values('entity', ascending = True)
        Y = Y.sort_values('entity', ascending = True)

        # convert Y to Y_tilda
        Y_df = Y.drop('entity', axis = 1, inplace = False)
        Y_np = Y_df.to_numpy().T
        ones_np = np.ones((Y_np.shape[1],), dtype = float)
        Y_tilda_np = np.append(Y_np, [ones_np], axis = 0)
        #print(Y_np.shape, Y_tilda_np.shape)

        # convert X to X_tilda
        X_df = X.drop('entity', axis = 1, inplace = False)
        X_np = X_df.to_numpy()
        if self.N is None:
            self.employed_N_ = X_np.shape[0] - 1
        else:
            self.employed_N_ = N
        X_np = X_np[:, :self.employed_N_]
        X_np = X_np.T
        ones_np = np.ones((X_np.shape[1],), dtype = float)
        X_tilda_np = np.append(X_np, [ones_np], axis = 0)
        #print(X_tilda_np.shape)

        # finally compute T_tilda_aff
        T_tilda_aff_np_1 = np.matmul(Y_tilda_np, X_tilda_np.T)
        T_tilda_aff_np_2 = np.matmul(X_tilda_np, X_tilda_np.T)
        T_tilda_aff_np_3 = np.linalg.inv(T_tilda_aff_np_2)
        self.T_tilda_aff_np_ = np.matmul(T_tilda_aff_np_1, T_tilda_aff_np_3)
        #print(self.T_tilda_aff_np.shape)

        return self

    def transform(self, X):
        return (X)

    def get_params(self, deep = True):
        return {'random_state': self.random_state, 
                'N': self.N}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def score(self, X, y):
        return 1

    def load_attitudinal_coordinates_from_file(self, path_attitudinal_reference_data,
            attitudinal_reference_data_header_names = None):

        # check if attitudinal reference data file exists
        if not os.path.isfile(path_attitudinal_reference_data):
            raise ValueError('Attitudinal reference data file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_attitudinal_reference_data, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Attitudinal reference data file has to have at least two columns.')

        if attitudinal_reference_data_header_names is not None:
            if attitudinal_reference_data_header_names['entity'] not in header_df.columns:
                raise ValueError('Attitudinal reference data file has to have a '
                        + attitudinal_reference_data_header_names['entity'] + ' column.')

        # load attitudinal reference data
        attitudinal_reference_data_df = None
        if attitudinal_reference_data_header_names is None:
            attitudinal_reference_data_df = pd.read_csv(path_attitudinal_reference_data,
                    header = None).rename(columns = {0:'entity'})
        else:
            attitudinal_reference_data_df = pd.read_csv(path_attitudinal_reference_data).rename(columns
                    = {attitudinal_reference_data_header_names['entity']:'entity'})
            if 'dimensions' in attitudinal_reference_data_header_names.keys():
                cols = attitudinal_reference_data_header_names['dimensions'].copy()
                cols.append('entity')
                attitudinal_reference_data_df = attitudinal_reference_data_df[cols]

        # exclude groups with a NaN in any of the dimensions (or group)
        attitudinal_reference_data_df.dropna(inplace = True)
        attitudinal_reference_data_df['entity'] = attitudinal_reference_data_df['entity'].astype(str)

        return (attitudinal_reference_data_df)

    def load_node_ca_coordinates_from_file(self, path_node_ca_coordinates, node_ca_coordinates_header_names = None):

        # check if node ca coordinates file exists
        if not os.path.isfile(path_node_ca_coordinates):
            raise ValueError('Node CA coordinates data file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_node_ca_coordinates, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Node CA coordinates data file has to have at least two columns.')
        #
        if node_ca_coordinates_header_names is not None:
            if node_ca_coordinates_header_names['entity'] not in header_df.columns:
                raise ValueError('Node CA cordinates data file has to have a '
                        + node_ca_coordinates_header_names['entity'] + ' column.')

        # load node CA coordinates data
        node_ca_coordinates_df = None
        if node_ca_coordinates_header_names is None:
            node_ca_coordinates_df = pd.read_csv(path_node_ca_coordinates,
                    header = None).rename(columns = {0:'entity'})
        else:
            node_ca_coordinates_df = pd.read_csv(path_node_ca_coordinates).rename(columns =
                    {node_ca_coordinates_header_names['entity']:'entity'})

        # exclude nodes with a NaN in any of the dimensions (or group)
        node_ca_coordinates_df.dropna(inplace = True)
        node_ca_coordinates_df['entity'] = node_ca_coordinates_df['entity'].astype(str)

        return(node_ca_coordinates_df)

    def load_node_to_group_mapping_from_file(self, path_node_to_group_mapping,
            node_to_group_mapping_header_names = None):

            # check if node group file exists
            if not os.path.isfile(path_node_to_group_mapping):
                raise ValueError('Node group data file does not exist.')

            # handles node group files with or without header
            header_df = pd.read_csv(path_node_to_group_mapping, nrows = 0)
            column_no = len(header_df.columns)
            if column_no < 2:
                raise ValueError('Node group data file has to have at least two columns.')

            if node_to_group_mapping_header_names is not None:
                if node_to_group_mapping_header_names['group'] not in header_df.columns:
                    raise ValueError('Node group data file has to have a '
                            + node_to_group_mapping_header_names['group'] + ' column.')
            #
            if node_to_group_mapping_header_names['entity'] not in header_df.columns:
                raise ValueError('Node group data file has to have a '
                        + node_to_group_mapping_header_names['entity'] + ' column.')

            # load node group data
            node_to_group_data_df = None
            if node_to_group_mapping_header_names is None:
                node_to_group_data_df = pd.read_csv(path_node_to_group_mapping, header
                        = None).rename(columns = {0:'entity', 1:'group'})
            else:
                node_to_group_data_df = pd.read_csv(path_node_to_group_mapping).rename(columns =
                        {node_to_group_mapping_header_names['group']:'group',
                            node_to_group_mapping_header_names['entity']:'entity'})

            # maintain only entity and group columns
            node_to_group_data_df = node_to_group_data_df[['entity', 'group']]
            node_to_group_data_df.dropna(inplace = True)
            node_to_group_data_df['entity'] = node_to_group_data_df['entity'].astype(str)
            node_to_group_data_df['group'] = node_to_group_data_df['group'].astype(str)

            # exclude rows with a NaN in any of the columns
            node_to_group_data_df.dropna(inplace = True)

            return(node_to_group_data_df)

    def convert_to_group_ca_coordinates(self, node_ca_coordinates_df, # will crash if dataframes not in right format
            node_to_group_data_df, group_ca_agg_fun = None):

        if not isinstance(node_ca_coordinates_df, pd.DataFrame):
            raise ValueError('\'ca_coordinates_df\' parameter must be a pandas dataframe')

        if not isinstance(node_to_group_data_df, pd.DataFrame):
            raise ValueError('\'node_to_group_data_df\' parameter must be a pandas dataframe')

        if 'entity' not in node_ca_coordinates_df.columns:
            raise ValueError('\'ca_coordinates_df\' has to have an \'entity\' column')

        if 'entity' not in node_to_group_data_df.columns:
            raise ValueError('\'node_to_group_data_df\' has to have an \'entity\' column')

        if 'group' not in node_to_group_data_df.columns:
            raise ValueError('\'node_to_group_data_df\' has to have an \'entity\' column')

        # add group information to the CA coordinates
        node_group_ca_coordinates_df = pd.merge(node_ca_coordinates_df, node_to_group_data_df, on = 'entity')
        node_group_ca_coordinates_df.drop('entity', axis = 1, inplace = True)

        # create ca coordinates aggregates : user can define custom (columnwise) aggregate
        entity_ca_coordinates_df = None
        if group_ca_agg_fun is None:
            entity_ca_coordinates_df = node_group_ca_coordinates_df.groupby(['group']).agg('mean').reset_index()
        else:
            entity_ca_coordinates_df = node_group_ca_coordinates_df.groupby(['group']).agg(group_ca_agg_fun).reset_index()

        entity_ca_coordinates_df.rename(columns = {'group': 'entity'}, inplace = True)

        return(entity_ca_coordinates_df)

    def save_affine_transformation(self, path_to_affine_transformation_file):
        np.savetxt(path_to_affine_transformation_file, self.T_tilda_aff_np_, delimiter = ",")
