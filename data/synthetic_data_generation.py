# Process:
#    1. generate data in ideological space
#    2. transform the data into attitudinal space
#    3. generate social graph

import numpy as np

import os

import csv

from scipy.stats import multivariate_normal as gaussian

import pandas as pd

from scipy.special import expit

from scipy.stats import bernoulli

######################################################
# Generating synthetic data in ideological space :   #
######################################################

def load_gaussian_mixture_model_mu_and_cov_from_files(mu_filename, cov_filename):
    if mu_filename is None:
        raise ValueError('\'MU\' filename must be provided')

    if not os.path.isfile(mu_filename):
        raise ValueError('\'MU\' filename does not exist')

    if cov_filename is None:
        raise ValueError('\'COV\' filename must be provided')

    if not os.path.isfile(mu_filename):
        raise ValueError('\'COV\' filename does not exist')

    mus = []
    mu_dim = None
    with open(mu_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for mu in csv_reader:
            if len(mu) > 0: 
                if mu_dim is None:
                    mu_dim = len(mu)
                else:
                    if len(mu) != mu_dim:
                        raise ValueError('\'MU\' filename contains inconsistent \'MU\' vectors')
                mus.append(list(np.float_(mu)))
    #print(mus)

    covs = []
    with open(cov_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        cov_m = []
        for cov in csv_reader:
            if len(cov) == 0:
                if cov_m:
                    if len(cov_m) != mu_dim:
                        raise ValueError('\'COV\' filename contains inconsistent \'COV\' matrices')
                    covs.append(cov_m)
                cov_m = []
            else:
                if len(cov) != mu_dim:
                    raise ValueError('\'COV\' filename contains inconsistent \'COV\' matrices')
                cov_m.append(list(np.float_(cov)))
        if cov_m:
            if len(cov_m) != mu_dim:
                raise ValueError('\'COV\' filename contains inconsistent \'COV\' matrices')
            covs.append(cov_m)

    if len(mus) != len(covs):
        raise ValueError('Number of \'MU\' vectors and \'COV\' matrices should be equal')

    mixmod = {}

    for i in range(len(mus)):
        mixmod[i] = {'mu': mus[i], 'cov': covs[i]}

    return(mixmod)

# returns generated entity dimensions as well as groups where those entities belong (it can be a single group)
def generate_entities_in_idelogical_space(N, gaussian_mixture_model_parameters, entity_id_prefix = 'n_',
        produce_group_dimensions = True, random_state = None):
    if N is None:
        raise ValueError('Total number of entities to be generated must be provided')

    if gaussian_mixture_model_parameters is None:
        raise ValueError('Mixture model parameters must be provided')

    entities = []
    entity_group_ids = []
    entity_indices = []
    entity_ids = []
    entity_id = 0

    if produce_group_dimensions:
        g_no = len(gaussian_mixture_model_parameters)
        g_dim = len(gaussian_mixture_model_parameters[0]['mu'])
        entity_groups = np.empty([g_no, g_dim])
        entity_group_info = np.empty(g_no, dtype = int)

        g_indx = 0

    for k, component in gaussian_mixture_model_parameters.items():
        k_entities = gaussian.rvs(mean = component['mu'], cov = component['cov'], 
                size = int(N / len(gaussian_mixture_model_parameters)), random_state = random_state)

        k_entity_group = [k for i in range(0, len(k_entities))]

        for i in range(0, len(k_entities)):
                entity_indices.append(entity_id)
                entity_ids.append(entity_id_prefix + str(entity_id))
                entity_id = entity_id + 1

        entities.extend(k_entities.tolist())
        entity_group_ids.extend(k_entity_group)

        if produce_group_dimensions:
            entity_group_info[g_indx] = k
            entity_groups[g_indx] = np.mean(k_entities, axis = 0)
            g_indx = g_indx + 1

    entities = np.array(entities)

    entity_info = np.column_stack((entity_indices, entity_group_ids, entity_ids))
    #entity_info = entity_info.astype((object))

    if produce_group_dimensions:
        return (entities, entity_info, entity_groups, entity_group_info)

    return (entities, entity_info)

#######################################################################
# Creating unobservable attitudes from prescribed ideologies: A       #
#######################################################################

def load_augmented_transformation_from_file(transformation_filename):

    if transformation_filename is None:
        raise ValueError('Augmented transformation filename should be provided') 

    if not os.path.isfile(transformation_filename):
        raise ValueError('Augmented transformation file does not exist') 

    T_tilda_aff_np = np.loadtxt(transformation_filename, delimiter = ",")

    if (len(T_tilda_aff_np.shape)) != 2:
        raise ValueError('Augmented transformation matrix should have 2 dimensions') 

    if int(T_tilda_aff_np[T_tilda_aff_np.shape[0] - 1, T_tilda_aff_np.shape[1] - 1]) != 1:
        raise ValueError('Augmented transformation matrix does not have the right format') 

    if int(sum((T_tilda_aff_np[T_tilda_aff_np.shape[0] - 1]))) != 1:
        raise ValueError('Augmented transformation matrix does not have the right format') 

    return (T_tilda_aff_np)

# assume entity dimensions matrix is in (dimensions x entity) format
def transform_entity_dimensions_to_new_space(entity_dimensions, T_tilda_aff, 
        produce_group_dimensions = True, entity_dimensions_info = None,
        introduce_standard_error = False, error_std = None):

    if not isinstance(entity_dimensions, np.ndarray):
        raise ValueError('Entity dimensions should be a numpy array') 

    ones_np = np.ones((entity_dimensions.shape[1],), dtype = float)
    entity_dimensions_tilda_np = np.append(entity_dimensions, [ones_np], axis = 0)

    if not isinstance(T_tilda_aff, np.ndarray):
        raise ValueError('Augmented transformation matrix should be a numpy array') 

    if T_tilda_aff.shape[1] != entity_dimensions_tilda_np.shape[0]:
        raise ValueError('Transformation and entity dimension matrices cannot be multiplied') 

    if introduce_standard_error == True:
        if error_std is None:
            raise ValueError('Need to provide std of standard error') 
        if error_std < 0.0:
            raise ValueError('Std of standard error should be non-negative') 

    transformed_entity_dimensions = np.matmul(T_tilda_aff, entity_dimensions_tilda_np)
    transformed_entity_dimensions = transformed_entity_dimensions[:-1]
    if introduce_standard_error == True:
        error = np.vstack([np.random.normal(loc = 0.0, scale = error_std, 
            size = (2, entity_dimensions.shape[1])), np.ones((entity_dimensions.shape[1]))])
        transformed_entity_dimensions = transformed_entity_dimensions + error
    transformed_entity_dimensions = transformed_entity_dimensions.T

    if produce_group_dimensions:
        #entity_groups[g_indx] = np.mean(k_entities, axis = 0)
        if entity_dimensions_info is None:
            raise ValueError('For group level dimensions the \'entity_dimentions_info \' parameter cannot be \'None\'.') 

        groups = entity_dimensions_info[1].astype(int)
        group_ids = np.unique(groups)

        transformed_entity_groups = np.empty([len(group_ids), 1 + transformed_entity_dimensions.shape[1]])
        eg_indx = 0
        for g in group_ids:
            g_indx = np.where(groups == g)
            group = transformed_entity_dimensions[g_indx]
            transformed_entity_groups[eg_indx, 0] = eg_indx
            transformed_entity_groups[eg_indx, 1:] = np.mean(group, axis = 0)
            eg_indx = eg_indx + 1

        return (transformed_entity_dimensions, transformed_entity_groups)

    return (transformed_entity_dimensions)

def load_transformation_from_file(transformation_filename):

    if transformation_filename is None:
        raise ValueError('Transformation filename should be provided') 

    if not os.path.isfile(transformation_filename):
        raise ValueError('Transformation file does not exist') 

    A_P_np = np.loadtxt(transformation_filename, delimiter = ",")

    if (len(A_P_np.shape)) != 2:
        raise ValueError('Transformation matrix should have 2 dimensions') 

    return (A_P_np)

# assume entity dimensions matrix is in (dimensions x entity) format
def transform_entity_dimensions_same_space(entity_dimensions, A_P,
        produce_group_dimensions = True, entity_dimensions_info = None):

    if not isinstance(entity_dimensions, np.ndarray):
        raise ValueError('Entity dimensions should be a numpy array') 

    if not isinstance(A_P, np.ndarray):
        raise ValueError('Transformation matrix should be a numpy array') 

    if A_P.shape[0] != A_P.shape[1]:
        raise ValueError('Transformation should not change space') 

    if A_P.shape[1] != entity_dimensions.shape[0]:
        raise ValueError('Transformation and entity dimension matrices cannot be multiplied') 

    transformed_entity_dimensions = np.matmul(A_P, entity_dimensions)
    transformed_entity_dimensions = transformed_entity_dimensions.T

    if produce_group_dimensions:
        #entity_groups[g_indx] = np.mean(k_entities, axis = 0)
        if entity_dimensions_info is None:
            raise ValueError('For group level dimensions the \'entity_dimentions_info \' parameter cannot be \'None\'.') 

        groups = entity_dimensions_info[1].astype(int)
        group_ids = np.unique(groups)

        transformed_entity_groups = np.empty([len(group_ids), 1 + transformed_entity_dimensions.shape[1]])
        eg_indx = 0
        for g in group_ids:
            g_indx = np.where(groups == g)
            group = transformed_entity_dimensions[g_indx]
            transformed_entity_groups[eg_indx, 0] = eg_indx
            transformed_entity_groups[eg_indx, 1:] = np.mean(group, axis = 0)
            eg_indx = eg_indx + 1

        return (transformed_entity_dimensions, transformed_entity_groups)

    return (transformed_entity_dimensions)

#############################################################################
# Computing the social graph using distances within a given space           #
#############################################################################

def compute_social_graph(source_indx, source_id, source_dimensions, target_indx, target_id,
        target_dimensions, random_state, alpha = 2, beta = 2, output_all_distances = False):
    # first distances in for all potential edges (pairs of source-target entities)
    graph_edges = pd.DataFrame(columns = ['source', 'target'])

    source_indx = source_indx.astype(int)
    target_indx = target_indx.astype(int)

    source_indx_list = []
    target_indx_list = []
    source_id_list = []
    target_id_list = []
    for i in source_indx:
        for j in target_indx:
            source_indx_list.append(int(source_indx[i]))
            target_indx_list.append(int(target_indx[j]))

            source_id_list.append(source_id[i])
            target_id_list.append(target_id[j])

    graph_edges['source'] = source_id_list
    graph_edges['target'] = target_id_list

    graph_edges['source_indx'] = source_indx_list
    graph_edges['target_indx'] = target_indx_list

    distances_list = []
    for _, row in graph_edges.iterrows():
        s_id = int(row['source_indx'])
        t_id = int(row['target_indx'])
        distances_list.append(np.linalg.norm(source_dimensions[s_id] - target_dimensions[t_id])) # Euclidean distance
    graph_edges['distance'] = distances_list

    # probability function for target-source connections based on distances
    prob = lambda d: expit(alpha - beta * d)

    # Computing an instance of the social graph G based on probability of edges
    graph_edges['probability'] = graph_edges['distance'].apply(prob)
    if output_all_distances:
        all_candidate_graph_edges = graph_edges.copy()

    graph_edges['selected'] = graph_edges['probability'].apply(lambda d: bernoulli.rvs(d, size = 1,
        random_state = random_state)[0])
    #print('Density = %0.5f'%(edges['selected'].sum()/edges.shape[0]))
    graph_edges = graph_edges[graph_edges['selected'] == 1]
    graph_edges.drop(['selected'], axis = 1, inplace = True)
    #graph_edges.drop(['source_indx'], axis = 1, inplace = True)
    #graph_edges.drop(['target_indx'], axis = 1, inplace = True)

    if output_all_distances:
        return(graph_edges, all_candidate_graph_edges)

    #print(graph_edges)
    return(graph_edges)

#######################################################################
# general utility functions
#######################################################################
def save_array_to_file(array, filename, format_specifier):
    if array is None:
        raise ValueError('First argument cannot be \'None\'')

    if filename is None:
        raise ValueError('Filename should be provided') 

    if not isinstance(array, np.ndarray):
        raise ValueError('First argument should be a numpy array') 

    np.savetxt(filename, array, delimiter = ",", fmt = format_specifier)

def load_array_from_file(filename, is_str = False):

    if filename is None:
        raise ValueError('Filename should be provided') 

    if not os.path.isfile(filename):
        raise ValueError('File does not exist')

    if is_str:
        array = np.loadtxt(filename, delimiter = ",", dtype = str)
    else:
        array = np.loadtxt(filename, delimiter = ",")

    return (array)

def save_dataframe_to_file(df, filename):
    if df is None:
        raise ValueError('First argument cannot be \'None\'')

    if filename is None:
        raise ValueError('Filename should be provided') 

    if not isinstance(df, pd.DataFrame):
        raise ValueError('First argument should be a dataframe')

    df.to_csv(filename, sep = ',', index = False)
