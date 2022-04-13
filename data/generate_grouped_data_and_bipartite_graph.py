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
    phi_group_df = pd.DataFrame(phi_group)
    phi_group_df.index.name = 'group_id'
    phi_group_df = phi_group_df.reset_index()
    phi_group_df.to_csv(params['data_gen']['phi_group'], sep = ',', index = None)
    #
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

    # loads the augmented transformation
    Tideo2att_tilde = gen.load_augmented_transformation_from_file(params['data_gen']['Tideo2att_tilde'])

    phi = gen.load_array_from_file(params['data_gen']['phi'])
    phi_info = gen.load_array_from_file(params['data_gen']['phi_info'], is_str = True)
    r, r_group = gen.transform_entity_dimensions_to_new_space(phi.T, Tideo2att_tilde,
            entity_dimensions_info = phi_info.T)
    #print(r[:5])
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

    # perform an additional transformation in the same space (can be I)
    A_P = gen.load_augmented_transformation_from_file(params['data_gen']['A_P'])

    r_P, r_P_group = gen.transform_entity_dimensions_same_space(r.T, A_P, entity_dimensions_info = phi_info.T)
    #print(r_P_group)
    gen.save_array_to_file(r_P, params['data_gen']['r_P'], format_specifier = '%f')
    #
    r_P_group_df = pd.DataFrame(r_P_group)
    r_P_group_df.rename({0: 'group_id'}, axis = 1, inplace = True)
    r_P_group_df['group_id'] = r_P_group_df['group_id'].astype(int)
    r_P_group_df.to_csv(params['data_gen']['r_P_group'], sep = ',', index = None)

    f_P = gen.transform_entity_dimensions_same_space(f.T, A_P, produce_group_dimensions = False)
    #print(r_P_group)
    gen.save_array_to_file(f_P, params['data_gen']['f_P'], format_specifier = '%f')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_P_info = gen.load_array_from_file(params['data_gen']['phi_info'], is_str = True)
    r_P = gen.load_array_from_file(params['data_gen']['r_P'])
    #print(r_P_info.shape, r_P.shape)

    f_P_info = gen.load_array_from_file(params['data_gen']['theta_info'], is_str = True)
    f_P = gen.load_array_from_file(params['data_gen']['f_P'])
    #print(f_P_info.shape, f_P.shape)

    alpha = int(params['data_gen']['alpha'])
    beta = int(params['data_gen']['beta'])
    social_graph_df, all_edges = gen.compute_social_graph(f_P_info.T[0], f_P_info.T[2], f_P, r_P_info.T[0],
            r_P_info.T[2], r_P, random_state, alpha, beta, output_all_distances = True)
    gen.save_dataframe_to_file(social_graph_df, params['data_gen']['social_graph'])
    gen.save_dataframe_to_file(all_edges, params['data_gen']['all_edges'])

if __name__ == "__main__":
    main()
