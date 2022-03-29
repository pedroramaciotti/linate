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
    phi = gen.generate_entities_in_idelogical_space(N_referential, ref_mixmod, random_state = random_state)
    gen.save_entities_dimensions_to_file(phi, 'generated_data/phi.txt')

    # followers in ideological space
    fol_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files('parameters/theta_mu.txt',
            'parameters/theta_cov.txt')
    #print(fol_mixmod)
    theta = gen.generate_entities_in_idelogical_space(N_followers, fol_mixmod, random_state = random_state)
    gen.save_entities_dimensions_to_file(theta, 'generated_data/theta.txt')

    #######################################################################
    # Creating unobservable attitudes from prescribed ideologies: A       #
    #######################################################################

    T_tilda_aff = gen.load_augmented_transformation_from_file('parameters/T_tilda_aff.txt')

    phi = gen.load_entities_in_idelogical_space_from_file('generated_data/phi.txt')
    r = gen.transform_entity_dimensions_to_new_space(phi.T, T_tilda_aff)
    #print(r)
    gen.save_entities_dimensions_to_file(r, 'generated_data/r.txt')

    theta = gen.load_entities_in_idelogical_space_from_file('generated_data/theta.txt')
    f = gen.transform_entity_dimensions_to_new_space(theta.T, T_tilda_aff)
    gen.save_entities_dimensions_to_file(f, 'generated_data/f.txt')

    #############################################################################
    # Computing the social graph using distances within a given space           #
    #############################################################################
    #gen.compute_social_graph()


if __name__ == "__main__":
    main()
