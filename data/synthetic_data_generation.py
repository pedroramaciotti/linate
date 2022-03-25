# generating
#    1. data in ideological space

import numpy as np

import os

import csv

from scipy.stats import multivariate_normal as gaussian

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
    #print(covs)

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

    entities = np.zeros((2,0))
    entity_labels = [] 
    for k, component in gaussian_mixture_model_parameters.items():
        entities = np.hstack([entities, gaussian.rvs(mean = component['mu'], cov = component['cov'], 
            size = int(N / len(gaussian_mixture_model_parameters)), random_state = random_state).T])
     #   phi_labels+=[k]*int(N_referential/len(ref_mixmod))
     #   phi_label_colors = [ref_label_colors[l] for l in phi_labels] 

    return (entities)

def save_entities_in_idelogical_space_to_file(entities, entity_filename):
    if entities is None:
        raise ValueError('Entity dimensions should be provided')

    if entity_filename is None:
        raise ValueError('Entity filename should be provided') 

    if not isinstance(entities, np.ndarray):
        raise ValueError('Entity dimensions should be a numpy array')

    np.savetxt(entity_filename, entities.T, delimiter = ",")

def main():

    random_state = None # for random number generation

    N_referential = 100
    N_followers = 1000

    ######################################################
    # Generating synthetic data in ideological space :   #
    ######################################################

    # reference users in ideological space
    ref_mixmod = load_gaussian_mixture_model_mu_and_cov_from_files('parameters/phi_mu.txt',
            'parameters/phi_cov.txt')
    #print(ref_mixmod)
    phi = generate_entities_in_idelogical_space(N_referential, ref_mixmod, random_state = random_state)
    save_entities_in_idelogical_space_to_file(phi, 'generated_data/phi.txt')

    fol_mixmod = load_gaussian_mixture_model_mu_and_cov_from_files('parameters/theta_mu.txt',
            'parameters/theta_cov.txt')
    #print(fol_mixmod)
    theta = generate_entities_in_idelogical_space(N_followers, fol_mixmod, random_state = random_state)
    save_entities_in_idelogical_space_to_file(theta, 'generated_data/theta.txt')

if __name__ == "__main__":
    main()
