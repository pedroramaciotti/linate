# Process:
#    1. generate data in ideological space

import numpy as np

import os

import csv

from scipy.stats import multivariate_normal as gaussian

import pandas as pd

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

def generate_entities_in_idelogical_space(N, gaussian_mixture_model_parameters, random_state = None):
    if N is None:
        raise ValueError('Total number of entities to be generated must be provided')

    if gaussian_mixture_model_parameters is None:
        raise ValueError('Mixture model parameters must be provided')

    entities = []
    #entity_labels = [] 
    for k, component in gaussian_mixture_model_parameters.items():
        k_entities = gaussian.rvs(mean = component['mu'], cov = component['cov'], 
                size = int(N / len(gaussian_mixture_model_parameters)), random_state = random_state)

        entities.extend(k_entities.tolist())
     #   phi_labels+=[k]*int(N_referential/len(ref_mixmod))
     #   phi_label_colors = [ref_label_colors[l] for l in phi_labels] 

    entities = np.array(entities)
    #print(entities)
    return (entities)

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

def load_entities_in_idelogical_space_from_file(entity_filename):

    if entity_filename is None:
        raise ValueError('Entity embedding filename should be provided') 

    if not os.path.isfile(entity_filename):
        raise ValueError('Entity embedding file does not exist')

    entities = np.loadtxt(entity_filename, delimiter = ",")
    return (entities)

#############################################################################
# Computing the social graph using distances within a given space           #
#############################################################################

def compute_social_graph():
    # first distances in for all potential edges (pairs of source-target entities)
    graph_edges = pd.DataFrame(columns = ['source', 'target'])
    print(graph_edges) 

    edges['source'] = list(range(f_P.shape[1])) * r_P.shape[1]

    #edges['distance'] = edges.apply(lambda row: np.sqrt((row['references_d_P_1'] - row['followers_d_P_1'])**2
    #    + (row['references_d_P_2'] - row['followers_d_P_2'])**2
    #    + (row['references_d_P_3'] - row['followers_d_P_3'])**2), axis = 1)

    #edges['references'] = list(itertools.chain.from_iterable(itertools.repeat(x,
    #@    f_P.shape[1]) for x in range(r_P.shape[1])))

#for dim in [1,2,3]:
#    edges['references_d_P_%d'%dim] = edges['references'].apply(lambda i: r_P[dim-1,i])
#    edges['followers_d_P_%d'%dim] = edges['followers'].apply(lambda i: f_P[dim-1,i])

#######################################################################
# general utility functions
def save_entities_dimensions_to_file(entity_dimensions, entity_filename):
    if entity_dimensions is None:
        raise ValueError('Entity dimensions should be provided')

    if entity_filename is None:
        raise ValueError('Entity filename should be provided') 

    if not isinstance(entity_dimensions, np.ndarray):
        raise ValueError('Entity dimensions should be a numpy array')

    np.savetxt(entity_filename, entity_dimensions, delimiter = ",")
