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
    phi, phi_info = gen.generate_entities_in_idelogical_space(N_referential, ref_mixmod, random_state = random_state)
    gen.save_array_to_file(phi, 'generated_data/phi.txt')
    gen.save_array_to_file(phi_info, 'generated_data/phi_info.txt', is_float = False)

    # followers in ideological space
    fol_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files('parameters/theta_mu.txt',
            'parameters/theta_cov.txt')
    #print(fol_mixmod)
    theta, theta_info = gen.generate_entities_in_idelogical_space(N_followers, fol_mixmod, random_state = random_state)
    gen.save_array_to_file(theta, 'generated_data/theta.txt')
    gen.save_array_to_file(theta_info, 'generated_data/theta_info.txt', is_float = False)

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    T_tilda_aff = gen.load_augmented_transformation_from_file('parameters/T_tilda_aff.txt')

    phi = gen.load_array_from_file('generated_data/phi.txt')
    r = gen.transform_entity_dimensions_to_new_space(phi.T, T_tilda_aff)
    #print(r)
    gen.save_array_to_file(r, 'generated_data/r.txt')

    theta = gen.load_array_from_file('generated_data/theta.txt')
    f = gen.transform_entity_dimensions_to_new_space(theta.T, T_tilda_aff)
    gen.save_array_to_file(f, 'generated_data/f.txt')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    r_info = gen.load_array_from_file('generated_data/phi_info.txt', is_float = False)
    r = gen.load_array_from_file('generated_data/r.txt')
    #print(r_info.shape, r.shape)

    f_info = gen.load_array_from_file('generated_data/theta_info.txt', is_float = False)
    f = gen.load_array_from_file('generated_data/f.txt')
    #print(f_info.shape, f.shape)

    alpha = 2
    beta = 2
    social_graph = gen.compute_social_graph(f_info.T[0], f, r_info.T[0], r, alpha, beta)
    print(social_graph.shape)

    # create aggregates here (i.e after node filtering)

if __name__ == "__main__":
    main()
