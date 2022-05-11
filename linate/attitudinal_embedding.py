""" LINATE module: Compute Attitudinal Embedding """

from sklearn.base import BaseEstimator, TransformerMixin

import os.path

import pandas as pd

import numpy as np

class AttitudinalEmbedding(BaseEstimator, TransformerMixin): 

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

        print('Groups: ', Y['entity'].values)
        print('Y columns: ', len(Y.columns), Y.columns)

        # finally fit an affine transformation to map X --> Y

        # first sort X and Y by entity (so as to have the corresponding mapping in the same rows)
        X = X.sort_values('entity', ascending = True)
        Y = Y.sort_values('entity', ascending = True)

        # convert Y to Y_tilda
        Y_df = Y.drop('entity', axis = 1, inplace = False)
        self.Y_columns = Y_df.columns.tolist()
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
            self.employed_N_ = self.N
        X_np = X_np[:, :self.employed_N_]
        self.X_columns = X_df.columns.tolist() 
        if self.employed_N_ < len(self.X_columns):
            self.X_columns = self.X_columns[:self.employed_N_]
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

        entitiy_col = None
        if isinstance(X, pd.DataFrame): # check input and convert to matrix
            if 'entity' not in X.columns:
                raise ValueError('Input dataframe has to have an \'entity\' column.')

            entitiy_col = X['entity'].values
            X.drop('entity', axis = 1, inplace = True)

            for c in X.columns:
                X[c] = X[c].astype(float)

            X = X.to_numpy()

        try:
            X_np = X[:, :self.employed_N_]
            X_np = X_np.T
            ones_np = np.ones((X_np.shape[1],), dtype = float)
            X_tilda_np = np.append(X_np, [ones_np], axis = 0)

            if self.T_tilda_aff_np_.shape[1] != X_tilda_np.shape[0]:
                raise ValueError('Wrong input dimensions')

            Y_tilda_np = np.matmul(self.T_tilda_aff_np_, X_tilda_np)
            #print(Y_tilda_np.shape)
            Y_tilda_np = Y_tilda_np[:-1]
            Y_tilda_np = Y_tilda_np.T

            Y = pd.DataFrame(Y_tilda_np, columns = self.Y_columns)
            if entitiy_col is not None:
                cols = Y.columns
                cols = cols.insert(0, 'entity')
                Y['entity'] = entitiy_col
                Y = Y[cols]

        except AttributeError:
            raise AttributeError('Transformation parameters have not been computed.')

        return (Y)

    def get_params(self, deep = True):
        return {'random_state': self.random_state, 
                'N': self.N}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def score(self, X, y):
        return 1

    def load_attitudinal_referential_coordinates_from_file(self, path_attitudinal_reference_data,
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

    def load_ideological_embedding_from_file(self, path_ideological_embedding, ideological_embedding_header_names = None):

        # check if ideological embedding file exists
        if not os.path.isfile(path_ideological_embedding):
            raise ValueError('Ideological embedding data file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_ideological_embedding, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Ideological embedding data file has to have at least two columns.')
        #
        if ideological_embedding_header_names is not None:
            if ideological_embedding_header_names['entity'] not in header_df.columns:
                raise ValueError('Ideological embedding data file has to have a '
                        + ideological_embedding_header_names['entity'] + ' column.')

        # load ideological embeddings
        ideological_embedding_df = None
        if ideological_embedding_header_names is None:
            ideological_embedding_df = pd.read_csv(path_ideological_embedding,
                    header = None).rename(columns = {0:'entity'})
        else:
            ideological_embedding_df = pd.read_csv(path_ideological_embedding).rename(columns =
                    {ideological_embedding_header_names['entity']:'entity'})

        # exclude nodes with a NaN in any of the dimensions (or group)
        ideological_embedding_df.dropna(inplace = True)
        ideological_embedding_df['entity'] = ideological_embedding_df['entity'].astype(str)

        if ideological_embedding_header_names is not None:
            if 'dimensions' in ideological_embedding_header_names.keys():
                ideological_embedding_df = ideological_embedding_df[ideological_embedding_header_names['dimensions']]

        return(ideological_embedding_df)

    def load_entity_to_group_mapping_from_file(self, path_entity_to_group_mapping,
            entity_to_group_mapping_header_names = None):

            # check if entity to group file exists
            if not os.path.isfile(path_entity_to_group_mapping):
                raise ValueError('Entity to group data file does not exist.')

            # handles entity to group files with or without header
            header_df = pd.read_csv(path_entity_to_group_mapping, nrows = 0)
            column_no = len(header_df.columns)
            if column_no < 2:
                raise ValueError('Entity to group data file has to have at least two columns.')

            if entity_to_group_mapping_header_names is not None:
                if entity_to_group_mapping_header_names['group'] not in header_df.columns:
                    raise ValueError('Entity to group data file has to have a '
                            + entity_to_group_mapping_header_names['group'] + ' column.')

                    if entity_to_group_mapping_header_names['entity'] not in header_df.columns:
                        raise ValueError('Entity to group data file has to have a '
                                + entity_to_group_mapping_header_names['entity'] + ' column.')

            # load entity to group data
            entity_to_group_data_df = None
            if entity_to_group_mapping_header_names is None:
                entity_to_group_data_df = pd.read_csv(path_entity_to_group_mapping, header
                        = None).rename(columns = {0:'entity', 1:'group'})
            else:
                entity_to_group_data_df = pd.read_csv(path_entity_to_group_mapping).rename(columns =
                        {entity_to_group_mapping_header_names['group']:'group',
                            entity_to_group_mapping_header_names['entity']:'entity'})

            # maintain only entity and group columns
            entity_to_group_data_df = entity_to_group_data_df[['entity', 'group']]
            entity_to_group_data_df.dropna(inplace = True)
            entity_to_group_data_df['entity'] = entity_to_group_data_df['entity'].astype(str)
            entity_to_group_data_df['group'] = entity_to_group_data_df['group'].astype(str)

            # exclude rows with a NaN in any of the columns
            entity_to_group_data_df.dropna(inplace = True)

            # check that each entity belongs to only 1 group
            has_entities_in_more_than_one_group = entity_to_group_data_df.groupby(['entity']).size().max() > 1
            if has_entities_in_more_than_one_group:
                raise ValueError('Entities should belong to a single group.')

            return(entity_to_group_data_df)

    def convert_to_group_ideological_embedding(self, ideological_embedding_df, entity_to_group_data_df,
            entity_to_group_agg_fun = None):

        if not isinstance(ideological_embedding_df, pd.DataFrame):
            raise ValueError('\'ideological_embedding_df\' parameter must be a pandas dataframe')

        if not isinstance(entity_to_group_data_df, pd.DataFrame):
            raise ValueError('\'entity_to_group_data_df\' parameter must be a pandas dataframe')

        if 'entity' not in ideological_embedding_df.columns:
            raise ValueError('\'ideological_embedding_df\' has to have an \'entity\' column')

        if 'entity' not in entity_to_group_data_df.columns:
            raise ValueError('\'entity_to_group_data_df\' has to have an \'entity\' column')

        if 'group' not in entity_to_group_data_df.columns:
            raise ValueError('\'entity_to_group_data_df\' has to have an \'group\' column')

        # add group information to the ideological embeddings
        entity_group_ideological_embedding_df = pd.merge(ideological_embedding_df, entity_to_group_data_df, on = 'entity')
        entity_group_ideological_embedding_df.drop('entity', axis = 1, inplace = True)

        # create ideological embeddings aggregates : user can define custom (columnwise) aggregate
        entity_ideological_embedding_df = None
        if entity_to_group_agg_fun is None:
            entity_ideological_embedding_df = entity_group_ideological_embedding_df.groupby(['group']).agg('mean').reset_index()
        else:
            entity_ideological_embedding_df = \
                    entity_group_ideological_embedding_df.groupby(['group']) .agg(entity_to_group_agg_fun).reset_index()

        entity_ideological_embedding_df.rename(columns = {'group': 'entity'}, inplace = True)

        return(entity_ideological_embedding_df)

    def save_transformation_parameters(self, path_to_transformation_parameters_file):
        try:
            at_df_index = self.Y_columns.copy()  # metadata
            at_df_index.append('plus_one_column')
            at_df_columns = self.X_columns.copy()
            at_df_columns.append('plus_one_row')

            at_df = pd.DataFrame(self.T_tilda_aff_np_, columns = at_df_columns)
            at_df.index = at_df_index
            #at_df.index.name = ''
            at_df.to_csv(path_to_transformation_parameters_file)
        except AttributeError:
            raise AttributeError('Transformation parameters have not been computed.')
