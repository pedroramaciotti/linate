# generate grouped data and a bipartite social graph

import synthetic_data_generation as gen

import numpy as np

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
            ref_mixmod, random_state = random_state)
    gen.save_array_to_file(phi, params['data_gen']['phi'])
    gen.save_array_to_file(phi_info, params['data_gen']['phi_info'], is_float = False)
    gen.save_array_to_file(phi_group, params['data_gen']['phi_group'])
    gen.save_array_to_file(phi_group_info, params['data_gen']['phi_group_info'], is_float = False)

    # followers in ideological space
    fol_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files(params['data_gen']['theta_mu'],
            params['data_gen']['theta_cov'])
    #print(fol_mixmod)
    theta, theta_info = gen.generate_entities_in_idelogical_space(N_followers, fol_mixmod,
            random_state = random_state, produce_group_dimensions = False)
    gen.save_array_to_file(theta, params['data_gen']['theta'])
    gen.save_array_to_file(theta_info, params['data_gen']['theta_info'], is_float = False)

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    Tideo2att_tilde = gen.load_augmented_transformation_from_file(params['data_gen']['Tideo2att_tilde'])

    phi = gen.load_array_from_file(params['data_gen']['phi'])
    r, r_group = gen.transform_entity_dimensions_to_new_space(phi.T, Tideo2att_tilde,
            entity_dimensions_info = phi_info.T)
    #print(r)
    gen.save_array_to_file(r, params['data_gen']['r'])
    gen.save_array_to_file(r_group, params['data_gen']['r_group'])

    theta = gen.load_array_from_file(params['data_gen']['theta'])
    f = gen.transform_entity_dimensions_to_new_space(theta.T, Tideo2att_tilde,
            produce_group_dimensions = False)
    gen.save_array_to_file(f, params['data_gen']['f'])

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_info = gen.load_array_from_file(params['data_gen']['phi_info'], is_float = False)
    r = gen.load_array_from_file(params['data_gen']['r'])
    #print(r_info.shape, r.shape)

    f_info = gen.load_array_from_file(params['data_gen']['theta_info'], is_float = False)
    f = gen.load_array_from_file(params['data_gen']['f'])
    #print(f_info.shape, f.shape)

    alpha = int(params['data_gen']['alpha'])
    beta = int(params['data_gen']['beta'])
    social_graph_df, all_edges = gen.compute_social_graph(f_info.T[0], f, r_info.T[0], r, random_state,
            alpha, beta, output_all_distances = True)
    gen.save_dataframe_to_file(social_graph_df, params['data_gen']['social_graph'])
    gen.save_dataframe_to_file(all_edges, params['data_gen']['all_edges'])

if __name__ == "__main__":
    main()
