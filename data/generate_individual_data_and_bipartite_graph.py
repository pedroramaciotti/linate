# generate grouped data and a bipartite social graph

import synthetic_data_generation as gen

import numpy as np

def main():

    random_state = None # for random number generation

    N_referential = 100
    N_followers = 1000

    ######################################################
    # Generating synthetic data in ideological space :   #
    ######################################################

    # reference users in ideological space
    ref_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files('parameters/phi_mu.txt',
            'parameters/phi_cov.txt')
    #print(ref_mixmod)
    phi_indvdl, phi_indvdl_info = gen.generate_entities_in_idelogical_space(N_referential,
            ref_mixmod, random_state = random_state, produce_group_dimensions = False)
    gen.save_array_to_file(phi_indvdl, 'generated_data/phi_indvdl.txt')
    gen.save_array_to_file(phi_indvdl_info, 'generated_data/phi_indvdl_info.txt', is_float = False)

    # followers in ideological space
    fol_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files('parameters/theta_mu.txt',
            'parameters/theta_cov.txt')
    #print(fol_mixmod)
    theta_indvdl, theta_indvdl_info = gen.generate_entities_in_idelogical_space(N_followers, fol_mixmod,
            random_state = random_state, produce_group_dimensions = False)
    gen.save_array_to_file(theta_indvdl, 'generated_data/theta_indvdl.txt')
    gen.save_array_to_file(theta_indvdl_info, 'generated_data/theta_indvdl_info.txt', is_float = False)

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    T_tilda_aff = gen.load_augmented_transformation_from_file('parameters/T_tilda_aff.txt')

    phi_indvdl = gen.load_array_from_file('generated_data/phi_indvdl.txt')
    r_indvdl = gen.transform_entity_dimensions_to_new_space(phi_indvdl.T, T_tilda_aff)
    #print(r_indvdl)
    gen.save_array_to_file(r_indvdl, 'generated_data/r_indvdl.txt')

    theta_indvdl = gen.load_array_from_file('generated_data/theta_indvdl.txt')
    f_indvdl = gen.transform_entity_dimensions_to_new_space(theta_indvdl.T, T_tilda_aff)
    gen.save_array_to_file(f_indvdl, 'generated_data/f_indvdl.txt')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_indvdl_info = gen.load_array_from_file('generated_data/phi_indvdl_info.txt', is_float = False)
    r_indvdl = gen.load_array_from_file('generated_data/r_indvdl.txt')
    #print(r_info.shape, r.shape)

    f_indvdl_info = gen.load_array_from_file('generated_data/theta_indvdl_info.txt', is_float = False)
    f_indvdl = gen.load_array_from_file('generated_data/f_indvdl.txt')
    #print(f_info.shape, f.shape)

    alpha = 2
    beta = 2
    social_graph = gen.compute_social_graph(f_indvdl_info.T[0], f_indvdl, r_indvdl_info.T[0],
            r_indvdl, random_state, alpha, beta)
    print(social_graph.shape)

if __name__ == "__main__":
    main()
