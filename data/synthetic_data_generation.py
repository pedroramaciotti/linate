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
def generate_entities_in_idelogical_space(N, gaussian_mixture_model_parameters,
        produce_group_dimensions = True, random_state = None):
    if N is None:
        raise ValueError('Total number of entities to be generated must be provided')

    if gaussian_mixture_model_parameters is None:
        raise ValueError('Mixture model parameters must be provided')

    entities = []
    entity_group_ids = []
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
                entity_ids.append(entity_id)
                entity_id = entity_id + 1

        entities.extend(k_entities.tolist())
        entity_group_ids.extend(k_entity_group)

        if produce_group_dimensions:
            entity_group_info[g_indx] = k
            entity_groups[g_indx] = np.mean(k_entities, axis = 0)
            g_indx = g_indx + 1

    entities = np.array(entities)
    entity_info = np.column_stack((entity_ids, entity_group_ids))
    entity_info = entity_info.astype(int)

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

# assume entity dimensions matrix is in (entity x dimensions) format
def transform_entity_dimensions_to_new_space(entity_dimensions, T_tilda_aff):

    if not isinstance(entity_dimensions, np.ndarray):
        raise ValueError('Entity dimensions should be a numpy array') 

    ones_np = np.ones((entity_dimensions.shape[1],), dtype = float)
    entity_dimensions_tilda_np = np.append(entity_dimensions, [ones_np], axis = 0)

    if not isinstance(T_tilda_aff, np.ndarray):
        raise ValueError('Augmented transformation matrix should be a numpy array') 

    if T_tilda_aff.shape[1] != entity_dimensions_tilda_np.shape[0]:
        raise ValueError('Transformation and entity dimension matrices cannot be multiplied') 

    transformed_entity_dimensions = np.matmul(T_tilda_aff, entity_dimensions_tilda_np)
    transformed_entity_dimensions = transformed_entity_dimensions[:-1]

    return (transformed_entity_dimensions.T)

#############################################################################
# Computing the social graph using distances within a given space           #
#############################################################################

def compute_social_graph(source_id, source_dimensions, target_id, target_dimensions,
        random_state, alpha = 2, beta = 2):
    # first distances in for all potential edges (pairs of source-target entities)
    graph_edges = pd.DataFrame(columns = ['source', 'target'])

    source_list = []
    target_list = []
    for i in source_id:
        for j in target_id:
            source_list.append(source_id[i])
            target_list.append(target_id[j])

    graph_edges['source'] = source_list
    graph_edges['target'] = target_list

    distances_list = []
    for _, row in graph_edges.iterrows():
        s_id = int(row['source'])
        t_id = int(row['target'])
        distances_list.append(np.linalg.norm(source_dimensions[s_id] - target_dimensions[t_id])) # Euclidean distance
    graph_edges['distance'] = distances_list

    # probability function for target-source connections based on distances
    prob = lambda d: expit(alpha-beta*d)

    # Computing an instance of the social graph G based on probability of edges
    graph_edges['probability'] = graph_edges['distance'].apply(prob)
    graph_edges['selected'] = graph_edges['probability'].apply(lambda d: bernoulli.rvs(d, size = 1,
        random_state = random_state)[0])
    #print('Density = %0.5f'%(edges['selected'].sum()/edges.shape[0]))
    graph_edges = graph_edges[graph_edges['selected'] == 1]
    graph_edges.drop(['selected'], axis = 1, inplace = True)

    #print(graph_edges)
    return(graph_edges)

#######################################################################
# general utility functions
#######################################################################
def save_array_to_file(array, filename, is_float = True):
    if array is None:
        raise ValueError('First argument cannot be \'None\'')

    if filename is None:
        raise ValueError('Filename should be provided') 

    if not isinstance(array, np.ndarray):
        raise ValueError('First argument should be a numpy array')

    if is_float:
        np.savetxt(filename, array, delimiter = ",", fmt = '%f')
    else:
        np.savetxt(filename, array, delimiter = ",", fmt = '%i')

def load_array_from_file(filename, is_float = True):

    if filename is None:
        raise ValueError('Filename should be provided') 

    if not os.path.isfile(filename):
        raise ValueError('File does not exist')

    array = np.loadtxt(filename, delimiter = ",")
    if not is_float:
        array = array.astype(int)

    return (array)
