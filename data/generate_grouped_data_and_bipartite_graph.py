# generate grouped data and a bipartite social graph

import synthetic_data_generation as gen

import numpy as np

import pandas as pd

import configparser

import sys

def main():

    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    random_state = None # for random number generation

    N_referential = int(params['data_gen']['N_referential'])
    N_followers = int(params['data_gen']['N_followers'])

    ######################################################
    # Generating synthetic data in ideological space :   #
    ######################################################

    # reference users in ideological space
    ref_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files(params['data_gen']['phi_mu'],
            params['data_gen']['phi_cov'])
    #print(ref_mixmod)
    phi, phi_info, phi_group, phi_group_info = gen.generate_entities_in_idelogical_space(N_referential,
            ref_mixmod, entity_id_prefix = 'r_', random_state = random_state)
    gen.save_array_to_file(phi, params['data_gen']['phi'], format_specifier = '%f')
    gen.save_array_to_file(phi_info, params['data_gen']['phi_info'], format_specifier = '%s')
    # also save it with a header for downstream processing
    phi_info_df = pd.DataFrame(phi_info, columns = ['index', 'group', 'id'])
    phi_info_df.to_csv(params['data_gen']['phi_info_header'], sep = ',', index = None)
    #
    gen.save_array_to_file(phi_group, params['data_gen']['phi_group'], format_specifier = '%f')
    gen.save_array_to_file(phi_group_info, params['data_gen']['phi_group_info'], format_specifier = '%i')

    # followers in ideological space
    fol_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files(params['data_gen']['theta_mu'],
            params['data_gen']['theta_cov'])
    #print(fol_mixmod)
    theta, theta_info = gen.generate_entities_in_idelogical_space(N_followers, fol_mixmod,
            entity_id_prefix = 'f_', random_state = random_state, produce_group_dimensions = False)
    gen.save_array_to_file(theta, params['data_gen']['theta'], format_specifier = '%f')
    gen.save_array_to_file(theta_info, params['data_gen']['theta_info'], format_specifier = '%s')

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    Tideo2att_tilde = gen.load_augmented_transformation_from_file(params['data_gen']['Tideo2att_tilde'])

    phi = gen.load_array_from_file(params['data_gen']['phi'])
    phi_info = gen.load_array_from_file(params['data_gen']['phi_info'], is_str = True)
    r, r_group = gen.transform_entity_dimensions_to_new_space(phi.T, Tideo2att_tilde,
            entity_dimensions_info = phi_info.T)
    #print(r)
    gen.save_array_to_file(r, params['data_gen']['r'], format_specifier = '%f')
    #
    r_group_df = pd.DataFrame(r_group)
    r_group_df.rename({0: 'group_id'}, axis = 1, inplace = True)
    r_group_df['group_id'] = r_group_df['group_id'].astype(int)
    r_group_df.to_csv(params['data_gen']['r_group'], sep = ',', index = None)

    theta = gen.load_array_from_file(params['data_gen']['theta'])
    f = gen.transform_entity_dimensions_to_new_space(theta.T, Tideo2att_tilde,
            produce_group_dimensions = False)
    gen.save_array_to_file(f, params['data_gen']['f'], format_specifier = '%f')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_info = gen.load_array_from_file(params['data_gen']['phi_info'], is_str = True)
    r = gen.load_array_from_file(params['data_gen']['r'])
    #print(r_info.shape, r.shape)

    f_info = gen.load_array_from_file(params['data_gen']['theta_info'], is_str = True)
    f = gen.load_array_from_file(params['data_gen']['f'])
    #print(f_info.shape, f.shape)

    alpha = int(params['data_gen']['alpha'])
    beta = int(params['data_gen']['beta'])
    social_graph_df, all_edges = gen.compute_social_graph(f_info.T[0], f_info.T[2], f, r_info.T[0],
            r_info.T[2], r, random_state, alpha, beta, output_all_distances = True)
    gen.save_dataframe_to_file(social_graph_df, params['data_gen']['social_graph'])
    gen.save_dataframe_to_file(all_edges, params['data_gen']['all_edges'])

if __name__ == "__main__":
    main()
